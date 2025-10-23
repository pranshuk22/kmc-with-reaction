import numpy as np
import pickle
import time
from itertools import product # We'll use this for iterating x,y

class Lattice:
    def __init__(self, size):
        self.size = size  # n x n

class MovementMatrixRandom:
    """
    Creates a lattice with randomly distributed site types ('blue', 'red'),
    each with its own user-defined rates for all event types.
    
    Instead of stripes, it generates an n x n 'site_map' (e.g., [[0, 1, 0...], ...])
    based on 'site_type_fractions' and a seed.
    """
    def __init__(self, n, site_type_fractions, seed,
                 site_type_hop_rates, 
                 site_type_ads_A_rates, site_type_des_A_rates,
                 site_type_ads_B_rates, site_type_des_B_rates,
                 site_type_react_A_to_B_rates, site_type_react_B_to_A_rates):
        
        self.n = n
        self.directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.rng = np.random.default_rng(seed)

        # --- 1. Create the random site map ---
        total_sites = n * n
        num_types = len(site_type_fractions)
        
        # Determine number of sites for each type
        site_counts = [int(f * total_sites) for f in site_type_fractions]
        # Ensure exact total by assigning remainder to the last type
        site_counts[-1] = total_sites - sum(site_counts[:-1]) 
        
        # Create a 1D array with the correct number of 0s (blue) and 1s (red)
        flat_map = []
        for site_type, count in enumerate(site_counts):
            flat_map.extend([site_type] * count)
        
        # Shuffle the 1D array randomly
        self.rng.shuffle(flat_map)
        
        # Reshape into the n x n lattice map
        self.site_map = np.array(flat_map).reshape((n, n))
        print("Generated random site map.")

        # --- 2. Create Rate Matrices based on the random map ---
        hop_matrix = np.zeros((n, n, 4))
        ads_A_matrix = np.zeros((n, n))
        des_A_matrix = np.zeros((n, n))
        ads_B_matrix = np.zeros((n, n))
        des_B_matrix = np.zeros((n, n))
        react_A_to_B_matrix = np.zeros((n, n))
        react_B_to_A_matrix = np.zeros((n, n))

        # Populate rate matrices by looking up the type of each site
        for x, y in product(range(n), range(n)):
            site_type = self.site_map[x, y] # Get site type (0 or 1)
            
            # Get rates for this specific site type
            rate_hop = site_type_hop_rates[site_type]
            rate_ads_A = site_type_ads_A_rates[site_type]
            rate_des_A = site_type_des_A_rates[site_type]
            rate_ads_B = site_type_ads_B_rates[site_type]
            rate_des_B = site_type_des_B_rates[site_type]
            rate_react_A = site_type_react_A_to_B_rates[site_type]
            rate_react_B = site_type_react_B_to_A_rates[site_type]
            
            # Assign rates to this specific (x, y) coordinate
            hop_matrix[x, y, :] = rate_hop
            ads_A_matrix[x, y] = rate_ads_A
            des_A_matrix[x, y] = rate_des_A
            ads_B_matrix[x, y] = rate_ads_B
            des_B_matrix[x, y] = rate_des_B
            react_A_to_B_matrix[x, y] = rate_react_A
            react_B_to_A_matrix[x, y] = rate_react_B
        
        # --- 3. Build the fast event catalog (This part is unchanged) ---
        self.events = []
        event_rates = []

        for x, y in product(range(n), range(n)):
            site = (x, y)
            
            # Hopping events
            for d, (dx, dy) in enumerate(self.directions):
                rate = float(hop_matrix[x, y, d])
                if rate > 0:
                    self.events.append(("hop", site, (dx, dy), rate))
                    event_rates.append(rate)
            
            # Adsorption A
            rate = ads_A_matrix[x, y]
            if rate > 0:
                self.events.append(("in_A", site, rate))
                event_rates.append(rate)
            
            # Adsorption B
            rate = ads_B_matrix[x, y]
            if rate > 0:
                self.events.append(("in_B", site, rate))
                event_rates.append(rate)
            
            # Desorption A
            rate = des_A_matrix[x, y]
            if rate > 0:
                self.events.append(("out_A", site, rate))
                event_rates.append(rate)

            # Desorption B
            rate = des_B_matrix[x, y]
            if rate > 0:
                self.events.append(("out_B", site, rate))
                event_rates.append(rate)

            # Reaction A -> B
            rate = react_A_to_B_matrix[x, y]
            if rate > 0:
                self.events.append(("react_A_to_B", site, rate))
                event_rates.append(rate)

            # Reaction B -> A
            rate = react_B_to_A_matrix[x, y]
            if rate > 0:
                self.events.append(("react_B_to_A", site, rate))
                event_rates.append(rate)
        
        self.cumulative_rates = np.cumsum(event_rates)
        if len(self.cumulative_rates) > 0:
            self.total_rate_val = self.cumulative_rates[-1]
        else:
            self.total_rate_val = 0.0

    @property
    def total_rate(self):
        return self.total_rate_val

    def pick_event(self):
        """
        Picks an event using O(log M) binary search.
        """
        if self.total_rate_val == 0:
            return None, 0.0

        T = self.total_rate_val
        r = np.random.rand() * T
        
        index = np.searchsorted(self.cumulative_rates, r, side='right')
        
        return self.events[index], T

class KMCSimulation:
    """
    --- NO CHANGES NEEDED HERE ---
    This class is generic. It just runs the events
    given to it by the MovementMatrix object.
    """
    def __init__(self, lattice, movement_matrix, max_unique_particles, total_time, seed):
        self.lattice = lattice
        self.mm = movement_matrix
        self.total_time = total_time
        self.max_particles = max_unique_particles
        self.rng = np.random.default_rng(seed)
        
        self.particles = {}
        self.occupation = {}
        
        self.available_pids = set(range(self.max_particles))
        self.next_new_pid = self.max_particles
        self.time_stamps = [0.0]
        self.snapshots = [{}] 
        self.trajectories = {}

    def _execute(self, ev):
        kind, data = ev[0], ev[1:]
        
        if kind == "in_A":
            site, _ = data
            if site not in self.occupation and len(self.particles) < self.max_particles:
                pid = self.available_pids.pop() if self.available_pids else self.next_new_pid
                if pid == self.next_new_pid: self.next_new_pid += 1
                
                self.particles[pid] = {'site': site, 'type': 'A'}
                self.occupation[site] = pid
                self.trajectories.setdefault(pid, []).append(site)
                return True
        
        elif kind == "in_B":
            site, _ = data
            if site not in self.occupation and len(self.particles) < self.max_particles:
                pid = self.available_pids.pop() if self.available_pids else self.next_new_pid
                if pid == self.next_new_pid: self.next_new_pid += 1
                
                self.particles[pid] = {'site': site, 'type': 'B'}
                self.occupation[site] = pid
                self.trajectories.setdefault(pid, []).append(site)
                return True
                
        elif kind == "out_A":
            site, _ = data
            if site in self.occupation:
                pid = self.occupation[site]
                if self.particles[pid]['type'] == 'A':
                    del self.particles[pid]
                    del self.occupation[site]
                    if pid < self.max_particles: self.available_pids.add(pid)
                    return True

        elif kind == "out_B":
            site, _ = data
            if site in self.occupation:
                pid = self.occupation[site]
                if self.particles[pid]['type'] == 'B':
                    del self.particles[pid]
                    del self.occupation[site]
                    if pid < self.max_particles: self.available_pids.add(pid)
                    return True
                
        elif kind == "hop":
            src, (dx, dy), _ = data
            if src in self.occupation:
                pid_at_src = self.occupation[src]
                dest = ((src[0] + dx) % self.lattice.size, (src[1] + dy) % self.lattice.size)
                
                if dest not in self.occupation:
                    del self.occupation[src]
                    self.occupation[dest] = pid_at_src
                    self.particles[pid_at_src]['site'] = dest 
                    self.trajectories[pid_at_src].append(dest)
                    return True
        
        elif kind == "react_A_to_B":
            site, _ = data
            if site in self.occupation:
                pid = self.occupation[site]
                if self.particles[pid]['type'] == 'A':
                    self.particles[pid]['type'] = 'B'
                    return True
        
        elif kind == "react_B_to_A":
            site, _ = data
            if site in self.occupation:
                pid = self.occupation[site]
                if self.particles[pid]['type'] == 'B':
                    self.particles[pid]['type'] = 'A'
                    return True

        return False
    
    def run(self, max_snapshots=10000):
        t = 0.0
        last_recorded_t = 0.0
        min_interval = self.total_time / max_snapshots
        
        while t < self.total_time:
            ev, total_rate = self.mm.pick_event()
            
            if ev is None or total_rate == 0:
                print("No events possible. Stopping simulation.")
                break
                
            self._execute(ev)
            
            dt = -np.log(self.rng.random()) / total_rate
            t += dt
            
            if t - last_recorded_t >= min_interval:
                snapshot_data = {p_data['site']: p_data['type'] 
                                 for pid, p_data in self.particles.items()}
                self.snapshots.append(snapshot_data)
                self.time_stamps.append(t)
                last_recorded_t = t
        
        snapshot_data = {p_data['site']: p_data['type'] 
                         for pid, p_data in self.particles.items()}
        self.snapshots.append(snapshot_data)
        self.time_stamps.append(t)

        return self.trajectories, self.snapshots, self.time_stamps


# Main execution block
if __name__ == "__main__":
    # Simulation Parameters
    LATTICE_SIZE = 20
    TOTAL_TIME = 10000
    RANDOM_SEED = 42

    # --- Site Type Configuration ---
    
    # NEW: Define the fraction of each site type
    # [0.5, 0.5] means 50% Type 0 (Blue) and 50% Type 1 (Red)
    # You could change this to [0.7, 0.3] for 70% Blue, 30% Red
    SITE_TYPE_FRACTIONS = [0.5, 0.5]
    
    NUM_SITE_TYPES = len(SITE_TYPE_FRACTIONS)
    
    # Site Type 0: Blue
    # Site Type 1: Red
    
    # We rename the lists from 'STRIPE_' to 'SITE_TYPE_' for clarity
    SITE_TYPE_HOP_RATES = [1.0, 1.0] 
    
    SITE_TYPE_ADS_A_RATES = [0.1, 0.0]
    SITE_TYPE_DES_A_RATES = [0.001, 0.01]
    SITE_TYPE_ADS_B_RATES = [0.0, 0.0]
    SITE_TYPE_DES_B_RATES = [0.01, 0.1]
    
    SITE_TYPE_REACT_A_TO_B_RATES = [0.0, 0.01]
    SITE_TYPE_REACT_B_TO_A_RATES = [0.0, 0.0]

    # --- Sanity Checks ---
    assert len(SITE_TYPE_HOP_RATES) == NUM_SITE_TYPES
    assert len(SITE_TYPE_ADS_A_RATES) == NUM_SITE_TYPES
    assert len(SITE_TYPE_DES_A_RATES) == NUM_SITE_TYPES
    assert len(SITE_TYPE_ADS_B_RATES) == NUM_SITE_TYPES
    assert len(SITE_TYPE_DES_B_RATES) == NUM_SITE_TYPES
    assert len(SITE_TYPE_REACT_A_TO_B_RATES) == NUM_SITE_TYPES
    assert len(SITE_TYPE_REACT_B_TO_A_RATES) == NUM_SITE_TYPES
    
    # ------------------------------------------------------------------
    
    start_time = time.time()
    print("Starting KMC simulation (Random Site Distribution)...")
    
    lattice = Lattice(LATTICE_SIZE)
    
    # --- Instantiate the NEW MovementMatrixRandom class ---
    movement = MovementMatrixRandom(
        n=LATTICE_SIZE,
        site_type_fractions=SITE_TYPE_FRACTIONS,
        seed=RANDOM_SEED,
        site_type_hop_rates=SITE_TYPE_HOP_RATES,
        site_type_ads_A_rates=SITE_TYPE_ADS_A_RATES,
        site_type_des_A_rates=SITE_TYPE_DES_A_RATES,
        site_type_ads_B_rates=SITE_TYPE_ADS_B_RATES,
        site_type_des_B_rates=SITE_TYPE_DES_B_RATES,
        site_type_react_A_to_B_rates=SITE_TYPE_REACT_A_TO_B_RATES,
        site_type_react_B_to_A_rates=SITE_TYPE_REACT_B_TO_A_RATES
    )
    
    sim = KMCSimulation(lattice, movement, LATTICE_SIZE * LATTICE_SIZE, TOTAL_TIME, seed=RANDOM_SEED)
    
    trajectories, snapshots, time_stamps = sim.run()
    
    # --- Update results dictionary to save the new site_map ---
    results = {
        'lattice_size': LATTICE_SIZE,
        'site_map': movement.site_map, # <-- ADDED THIS
        'site_type_fractions': SITE_TYPE_FRACTIONS, # <-- ADDED THIS
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
    
    output_filename = "simulation_results_AB_random.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)

    end_time = time.time()
    execution_time = end_time - start_time
        
    print(f"\n Simulation complete...")
    print(f"   - Saved results to {output_filename}")
    print(f"   - Total execution time: {execution_time:.2f} seconds.")