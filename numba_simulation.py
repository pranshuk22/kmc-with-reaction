import numpy as np
import pickle
import time
from itertools import product
from numba import njit

class Lattice:
    def __init__(self, size):
        self.size = size


class MovementMatrix:
    """
    Creates a site map and builds an event catalog encoded as integer arrays for Numba.
    """
    def __init__(self, n, mode, seed,
                 site_type_hop_rates,
                 site_type_ads_A_rates, site_type_des_A_rates,
                 site_type_ads_B_rates, site_type_des_B_rates,
                 site_type_react_A_to_B_rates, site_type_react_B_to_A_rates,
                 num_site_types=None, site_type_fractions=None):
        
        self.n = n
        self.directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        rng = np.random.default_rng(seed)

        # --- Generate site_map ---
        if mode == 'stripes':
            print("Generating 'stripes' site map...")
            if num_site_types is None:
                num_site_types = len(site_type_hop_rates)
            
            stripe_width = n // num_site_types
            self.site_map = np.zeros((n, n), dtype=np.int32)
            for stripe_index in range(num_site_types):
                x_start = stripe_index * stripe_width
                x_end = (x_start + stripe_width) if stripe_index < num_site_types - 1 else n
                self.site_map[:, x_start:x_end] = stripe_index
            
        elif mode == 'random':
            print("Generating 'random' site map...")
            if site_type_fractions is None:
                raise ValueError("site_type_fractions must be provided for 'random' mode")
            
            total_sites = n * n
            site_counts = [int(f * total_sites) for f in site_type_fractions]
            site_counts[-1] = total_sites - sum(site_counts[:-1])
            
            flat_map = np.concatenate([
                np.full(c, s, dtype=np.int32) for s, c in enumerate(site_counts)
            ])
            rng.shuffle(flat_map)
            self.site_map = flat_map.reshape((n, n))
        else:
            raise ValueError(f"Unknown LATTICE_MODE: '{mode}'")

        # --- Build event catalog as integer arrays ---
        # Event encoding: [event_type, x, y, dx, dy]
        # event_type: 0=hop, 1=in_A, 2=in_B, 3=out_A, 4=out_B, 5=react_A_to_B, 6=react_B_to_A
        
        events = []
        rates = []

        for x, y in product(range(n), range(n)):
            site_type = self.site_map[y, x]
            
            # Hopping events
            for dx, dy in self.directions:
                rate = site_type_hop_rates[site_type]
                if rate > 0:
                    events.append([0, x, y, dx, dy])
                    rates.append(rate)
            
            # Adsorption/Desorption/Reaction events
            event_configs = [
                (1, site_type_ads_A_rates[site_type]),      # in_A
                (2, site_type_ads_B_rates[site_type]),      # in_B
                (3, site_type_des_A_rates[site_type]),      # out_A
                (4, site_type_des_B_rates[site_type]),      # out_B
                (5, site_type_react_A_to_B_rates[site_type]),  # react_A_to_B
                (6, site_type_react_B_to_A_rates[site_type])   # react_B_to_A
            ]
            
            for event_type, rate in event_configs:
                if rate > 0:
                    events.append([event_type, x, y, 0, 0])
                    rates.append(rate)
        
        self.event_array = np.array(events, dtype=np.int32)
        self.rate_array = np.array(rates, dtype=np.float64)
        self.cumulative_rates = np.cumsum(self.rate_array)
        self.total_rate_val = self.cumulative_rates[-1] if len(self.cumulative_rates) > 0 else 0.0


@njit
def kmc_run_numba(event_array, cum_rates, total_rate, lattice_size, 
                  total_time, max_particles, seed, max_snapshots):
    """
    Numba-accelerated KMC simulation with full particle tracking.
    
    State representation:
    - particle_x, particle_y: position of each particle (x, y coords)
    - particle_type: 0=A, 1=B, -1=inactive
    - occupation_grid: particle ID at each site, -1=empty
    - trajectory storage: separate arrays for each particle
    """
    np.random.seed(seed)
    
    # Initialize particle state arrays
    particle_x = np.full(max_particles, -1, dtype=np.int32)
    particle_y = np.full(max_particles, -1, dtype=np.int32)
    particle_type = np.full(max_particles, -1, dtype=np.int32)  # 0=A, 1=B, -1=inactive
    occupation_grid = np.full((lattice_size, lattice_size), -1, dtype=np.int32)
    
    # Trajectory storage (fixed max length per particle)
    max_traj_len = 100000
    trajectories_x = np.full((max_particles, max_traj_len), -1, dtype=np.int32)
    trajectories_y = np.full((max_particles, max_traj_len), -1, dtype=np.int32)
    traj_lengths = np.zeros(max_particles, dtype=np.int32)
    
    # Snapshot storage
    snapshot_times = [0.0]
    snapshot_data_x = [np.copy(particle_x)]
    snapshot_data_y = [np.copy(particle_y)]
    snapshot_data_types = [np.copy(particle_type)]
    
    # Particle ID management
    active_count = 0
    available_pids = np.arange(max_particles, dtype=np.int32)
    num_available = max_particles
    
    # Time tracking
    t = 0.0
    last_snapshot_t = 0.0
    min_interval = total_time / max_snapshots if max_snapshots > 0 else float('inf')
    
    successful_events = 0
    failed_events = 0
    
    while t < total_time:
        if total_rate == 0:
            break
        
        # Select event
        r = np.random.random() * total_rate
        idx = np.searchsorted(cum_rates, r)
        ev = event_array[idx]
        
        event_type = ev[0]
        x, y = ev[1], ev[2]
        dx, dy = ev[3], ev[4]
        
        executed = False
        
        # Execute event based on type
        if event_type == 1:  # in_A (adsorption of A)
            if occupation_grid[y, x] == -1 and active_count < max_particles:
                if num_available > 0:
                    pid = available_pids[num_available - 1]
                    num_available -= 1
                else:
                    pid = active_count
                
                particle_x[pid] = x
                particle_y[pid] = y
                particle_type[pid] = 0  # Type A
                occupation_grid[y, x] = pid
                
                # Record trajectory
                trajectories_x[pid, 0] = x
                trajectories_y[pid, 0] = y
                traj_lengths[pid] = 1
                
                active_count += 1
                executed = True
        
        elif event_type == 2:  # in_B (adsorption of B)
            if occupation_grid[y, x] == -1 and active_count < max_particles:
                if num_available > 0:
                    pid = available_pids[num_available - 1]
                    num_available -= 1
                else:
                    pid = active_count
                
                particle_x[pid] = x
                particle_y[pid] = y
                particle_type[pid] = 1  # Type B
                occupation_grid[y, x] = pid
                
                trajectories_x[pid, 0] = x
                trajectories_y[pid, 0] = y
                traj_lengths[pid] = 1
                
                active_count += 1
                executed = True
        
        elif event_type == 3:  # out_A (desorption of A)
            if occupation_grid[y, x] != -1:
                pid = occupation_grid[y, x]
                if particle_type[pid] == 0:  # Is type A
                    particle_x[pid] = -1
                    particle_y[pid] = -1
                    particle_type[pid] = -1
                    occupation_grid[y, x] = -1
                    
                    if pid < max_particles:
                        available_pids[num_available] = pid
                        num_available += 1
                    
                    active_count -= 1
                    executed = True
        
        elif event_type == 4:  # out_B (desorption of B)
            if occupation_grid[y, x] != -1:
                pid = occupation_grid[y, x]
                if particle_type[pid] == 1:  # Is type B
                    particle_x[pid] = -1
                    particle_y[pid] = -1
                    particle_type[pid] = -1
                    occupation_grid[y, x] = -1
                    
                    if pid < max_particles:
                        available_pids[num_available] = pid
                        num_available += 1
                    
                    active_count -= 1
                    executed = True
        
        elif event_type == 0:  # hop
            if occupation_grid[y, x] != -1:
                pid = occupation_grid[y, x]
                dest_x = (x + dx) % lattice_size
                dest_y = (y + dy) % lattice_size
                
                if occupation_grid[dest_y, dest_x] == -1:
                    occupation_grid[y, x] = -1
                    occupation_grid[dest_y, dest_x] = pid
                    particle_x[pid] = dest_x
                    particle_y[pid] = dest_y
                    
                    # Record trajectory
                    traj_len = traj_lengths[pid]
                    if traj_len < max_traj_len:
                        trajectories_x[pid, traj_len] = dest_x
                        trajectories_y[pid, traj_len] = dest_y
                        traj_lengths[pid] += 1
                    
                    executed = True
        
        elif event_type == 5:  # react_A_to_B
            if occupation_grid[y, x] != -1:
                pid = occupation_grid[y, x]
                if particle_type[pid] == 0:  # Is type A
                    particle_type[pid] = 1  # Convert to B
                    executed = True
        
        elif event_type == 6:  # react_B_to_A
            if occupation_grid[y, x] != -1:
                pid = occupation_grid[y, x]
                if particle_type[pid] == 1:  # Is type B
                    particle_type[pid] = 0  # Convert to A
                    executed = True
        
        if executed:
            successful_events += 1
            dt = -np.log(1.0 - np.random.random()) / total_rate
            t += dt
            
            # Record snapshot periodically
            if t - last_snapshot_t >= min_interval:
                snapshot_times.append(t)
                snapshot_data_x.append(np.copy(particle_x))
                snapshot_data_y.append(np.copy(particle_y))
                snapshot_data_types.append(np.copy(particle_type))
                last_snapshot_t = t
        else:
            failed_events += 1
    
    # Final snapshot
    if t > snapshot_times[-1]:
        snapshot_times.append(t)
        snapshot_data_x.append(np.copy(particle_x))
        snapshot_data_y.append(np.copy(particle_y))
        snapshot_data_types.append(np.copy(particle_type))
    
    return (np.array(snapshot_times), 
            snapshot_data_x, snapshot_data_y, snapshot_data_types,
            trajectories_x, trajectories_y, traj_lengths,
            successful_events, failed_events)


class KMCSimulation:
    """Wrapper class that calls Numba-accelerated simulation."""
    
    def __init__(self, lattice, movement_matrix, max_unique_particles, total_time, seed):
        self.lattice = lattice
        self.mm = movement_matrix
        self.total_time = total_time
        self.max_particles = max_unique_particles
        self.seed = seed
    
    def run(self, max_snapshots=10000):
        """Runs the Numba-accelerated KMC simulation."""
        print("Running Numba-accelerated KMC simulation...")
        start = time.time()
        
        result = kmc_run_numba(
            self.mm.event_array,
            self.mm.cumulative_rates,
            self.mm.total_rate_val,
            self.lattice.size,
            self.total_time,
            self.max_particles,
            self.seed,
            max_snapshots
        )
        
        elapsed = time.time() - start
        
        (time_stamps, snap_x, snap_y, snap_types,
         traj_x, traj_y, traj_lens, 
         successful, failed) = result
        
        print(f"Simulation completed in {elapsed:.3f} seconds")
        print(f"Successful events: {successful}, Failed events: {failed}")
        
        # Convert back to dict format for compatibility
        trajectories = {}
        for pid in range(self.max_particles):
            if traj_lens[pid] > 0:
                traj = [(int(traj_x[pid, i]), int(traj_y[pid, i])) 
                        for i in range(traj_lens[pid])]
                trajectories[pid] = traj
        
        snapshots = []
        for i in range(len(time_stamps)):
            snap_dict = {}
            for pid in range(self.max_particles):
                if snap_types[i][pid] >= 0:
                    site = (int(snap_x[i][pid]), int(snap_y[i][pid]))
                    ptype = 'A' if snap_types[i][pid] == 0 else 'B'
                    snap_dict[site] = ptype
            snapshots.append(snap_dict)
        
        return trajectories, snapshots, list(time_stamps)


# Main execution block
if __name__ == "__main__":
    
    LATTICE_MODE = 'stripes'
    LATTICE_SIZE = 20
    TOTAL_TIME = 1000
    RANDOM_SEED = 42

    SITE_TYPE_HOP_RATES = [1.0, 1.0]
    SITE_TYPE_ADS_A_RATES = [0.1, 0.0]
    SITE_TYPE_DES_A_RATES = [0.001, 0.01]
    SITE_TYPE_ADS_B_RATES = [0.0, 0.0]
    SITE_TYPE_DES_B_RATES = [0.01, 0.1]
    SITE_TYPE_REACT_A_TO_B_RATES = [0.0, 0.1]
    SITE_TYPE_REACT_B_TO_A_RATES = [0.0, 0.0]

    NUM_SITE_TYPES = len(SITE_TYPE_HOP_RATES)
    SITE_TYPE_FRACTIONS = [0.5, 0.5]
    
    assert LATTICE_MODE in ['stripes', 'random']
    if LATTICE_MODE == 'random':
        assert len(SITE_TYPE_FRACTIONS) == NUM_SITE_TYPES
        assert np.isclose(sum(SITE_TYPE_FRACTIONS), 1.0)
    
    start_time = time.time()
    print(f"Starting KMC simulation (Mode: {LATTICE_MODE})...")
    
    lattice = Lattice(LATTICE_SIZE)
    
    movement = MovementMatrix(
        n=LATTICE_SIZE,
        mode=LATTICE_MODE,
        seed=RANDOM_SEED,
        site_type_hop_rates=SITE_TYPE_HOP_RATES,
        site_type_ads_A_rates=SITE_TYPE_ADS_A_RATES,
        site_type_des_A_rates=SITE_TYPE_DES_A_RATES,
        site_type_ads_B_rates=SITE_TYPE_ADS_B_RATES,
        site_type_des_B_rates=SITE_TYPE_DES_B_RATES,
        site_type_react_A_to_B_rates=SITE_TYPE_REACT_A_TO_B_RATES,
        site_type_react_B_to_A_rates=SITE_TYPE_REACT_B_TO_A_RATES,
        num_site_types=NUM_SITE_TYPES,
        site_type_fractions=SITE_TYPE_FRACTIONS
    )
    
    sim = KMCSimulation(lattice, movement, LATTICE_SIZE * LATTICE_SIZE, TOTAL_TIME, seed=RANDOM_SEED)
    
    trajectories, snapshots, time_stamps = sim.run()
    
    results = {
        'lattice_size': LATTICE_SIZE,
        'lattice_mode': LATTICE_MODE,
        'site_map': movement.site_map,
        'num_site_types': NUM_SITE_TYPES,
        'site_type_fractions': SITE_TYPE_FRACTIONS if LATTICE_MODE == 'random' else None,
        'total_time': TOTAL_TIME,
        'trajectories': trajectories,
        'snapshots': snapshots,
        'time_stamps': time_stamps,
        'params': {
            'site_type_hop_rates': SITE_TYPE_HOP_RATES,
            'site_type_ads_A_rates': SITE_TYPE_ADS_A_RATES,
            'site_type_des_A_rates': SITE_TYPE_DES_A_RATES,
            'site_type_ads_B_rates': SITE_TYPE_ADS_B_RATES,
            'site_type_des_B_rates': SITE_TYPE_DES_B_RATES,
            'site_type_react_A_to_B': SITE_TYPE_REACT_A_TO_B_RATES,
            'site_type_react_B_to_A': SITE_TYPE_REACT_B_TO_A_RATES,
        }
    }
    
    output_filename = f"simulation_results_{LATTICE_MODE}_numba.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)

    end_time = time.time()
    execution_time = end_time - start_time
        
    print(f"\n✅ Simulation complete...")
    print(f"   - Saved results to {output_filename}")
    print(f"   - Total execution time: {execution_time:.2f} seconds.")
