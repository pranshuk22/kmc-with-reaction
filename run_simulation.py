import numpy as np
import pickle
import time
from itertools import product # Used for iterating (x,y)

class Lattice:
    def __init__(self, size):
        self.size = size  # n x n

class MovementMatrix:
    """
    --- UNIFIED MOVEMENT MATRIX ---
    Creates a site map based on the specified 'mode' ('stripes' or 'random')
    and then builds the event catalog from that map.
    
    The 'site_map' is an n x n array where each cell contains the site type
    (e.g., 0 for 'blue', 1 for 'red').
    """
    def __init__(self, n, mode, seed, 
                 site_type_hop_rates, 
                 site_type_ads_A_rates, site_type_des_A_rates,
                 site_type_ads_B_rates, site_type_des_B_rates,
                 site_type_react_A_to_B_rates, site_type_react_B_to_A_rates,
                 num_site_types=None, site_type_fractions=None):
        
        self.n = n
        self.directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.rng = np.random.default_rng(seed)

        # --- 1. Generate the site_map based on the chosen mode ---
        
        if mode == 'stripes':
            print("Generating 'stripes' site map...")
            if num_site_types is None:
                num_site_types = len(site_type_hop_rates)
            
            stripe_width = n // num_site_types
            self.site_map = np.zeros((n, n), dtype=int)
            for stripe_index in range(num_site_types):
                x_start = stripe_index * stripe_width
                # Ensure the last stripe goes all the way to the edge
                x_end = (x_start + stripe_width) if stripe_index < num_site_types - 1 else n
                
                # --- THIS IS THE FIX ---
                # Assign site type to VERTICAL columns, not horizontal rows.
                # [:, x_start:x_end] = "all rows, columns from x_start to x_end"
                self.site_map[:, x_start:x_end] = stripe_index
            
        elif mode == 'random':
            print("Generating 'random' site map...")
            if site_type_fractions is None:
                raise ValueError("site_type_fractions must be provided for 'random' mode")
            
            total_sites = n * n
            
            # Determine number of sites for each type
            site_counts = [int(f * total_sites) for f in site_type_fractions]
            site_counts[-1] = total_sites - sum(site_counts[:-1]) 
            
            flat_map = []
            for site_type, count in enumerate(site_counts):
                flat_map.extend([site_type] * count)
            
            self.rng.shuffle(flat_map)
            self.site_map = np.array(flat_map).reshape((n, n))
        
        else:
            raise ValueError(f"Unknown LATTICE_MODE: '{mode}'. Must be 'stripes' or 'random'.")

        # --- 2. Create Rate Matrices based on the generated site_map ---
        
        hop_matrix = np.zeros((n, n, 4))
        ads_A_matrix = np.zeros((n, n))
        des_A_matrix = np.zeros((n, n))
        ads_B_matrix = np.zeros((n, n))
        des_B_matrix = np.zeros((n, n))
        react_A_to_B_matrix = np.zeros((n, n))
        react_B_to_A_matrix = np.zeros((n, n))

        # Populate rate matrices by looking up the type of each site
        for x, y in product(range(n), range(n)):
            # NOTE: site_map is indexed [row, col] which is [y, x]
            # But since our map is built relative to x/y, we index [x,y]
            # To fix the *previous* bug, we must index the map as [y, x]
            # OR, we fix the map generation. I've fixed the map generation.
            # So now site_map[x, y] is correct...
            # ... NO, wait.
            # plt.imshow plots [row, col] -> (y, x)
            # Our simulation logic uses (x, y)
            # The site_map[x_start:x_end, :] was creating (x=0..9, y=all) = 0
            # imshow plots this as (y=0..9, x=all) = 0, which is horizontal.
            #
            # The FIX: self.site_map[:, x_start:x_end] = stripe_index
            # This creates (y=all, x=0..9) = 0
            # imshow plots this as (x=0..9, y=all) = 0, which is vertical.
            #
            # Now, when we query rates, we must get the right (x,y)
            # site_type = self.site_map[x, y] <- This is [row, col]
            #
            # Let's test this:
            # map = np.zeros((3,3))
            # map[:, 0:2] = 1
            # map = [[1, 1, 0],
            #        [1, 1, 0],
            #        [1, 1, 0]]
            #
            # site_map[x=0, y=0] should be 1. map[0, 0] is 1.
            # site_map[x=1, y=0] should be 1. map[0, 1] is 1.
            # site_map[x=2, y=0] should be 0. map[0, 2] is 0.
            #
            # This is indexed (row, col) which is (y, x)
            # So it should be:
            site_type = self.site_map[y, x] # Get site type (0 or 1)
            
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
        
        # --- 3. Build the fast event catalog (Common to both modes) ---
        
        self.events = []
        event_rates = []

        # This loop iterates (x, y) from (0,0), (0,1)...
        # This matches the rate matrices, so this is correct.
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
    --- NO CHANGES NEEDED ---
    This class is fully generic and just executes the events
    passed to it by the MovementMatrix.
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
    
    # --- 1. CHOOSE LATTICE MODE ---
    # Options: 'stripes' or 'random'
    LATTICE_MODE = 'random' 
    
    # --- 2. Simulation Parameters ---
    LATTICE_SIZE = 20
    TOTAL_TIME = 10000
    RANDOM_SEED = 42

    # --- 3. Site Type Configuration (Common) ---
    # Site Type 0: Blue
    # Site Type 1: Red
    SITE_TYPE_HOP_RATES = [1.0, 1.0] 
    
    SITE_TYPE_ADS_A_RATES = [0.1, 0.0]
    SITE_TYPE_DES_A_RATES = [0.001, 0.01]
    SITE_TYPE_ADS_B_RATES = [0.0, 0.0]
    SITE_TYPE_DES_B_RATES = [0.01, 0.1]
    
    SITE_TYPE_REACT_A_TO_B_RATES = [0.0, 0.01]
    SITE_TYPE_REACT_B_TO_A_RATES = [0.0, 0.0]

    NUM_SITE_TYPES = len(SITE_TYPE_HOP_RATES)

    # --- 4. Mode-Specific Configuration ---
    
    # For 'random' mode: Specify fractions for [Type 0, Type 1, ...]
    SITE_TYPE_FRACTIONS = [0.5, 0.5] # 50% Blue, 50% Red
    
    # --- 5. Sanity Checks ---
    assert LATTICE_MODE in ['stripes', 'random'], "LATTICE_MODE must be 'stripes' or 'random'"
    assert len(SITE_TYPE_HOP_RATES) == NUM_SITE_TYPES
    assert len(SITE_TYPE_ADS_A_RATES) == NUM_SITE_TYPES
    # ... (all other assert lines) ...
    if LATTICE_MODE == 'random':
        assert len(SITE_TYPE_FRACTIONS) == NUM_SITE_TYPES, "Fractions list must match number of site types"
        assert np.isclose(sum(SITE_TYPE_FRACTIONS), 1.0), "SITE_TYPE_FRACTIONS must sum to 1.0"
    
    # ------------------------------------------------------------------
    
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
    
    output_filename = f"simulation_results_{LATTICE_MODE}.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)

    end_time = time.time()
    execution_time = end_time - start_time
        
    print(f"\n Simulation complete...")
    print(f"   - Saved results to {output_filename}")
    print(f"   - Total execution time: {execution_time:.2f} seconds.")