# run_simulation_fast.py

import numpy as np
import pickle
import time

class Lattice:
    def __init__(self, size):
        self.size = size  # n x n

class MovementMatrixStripes:
    """
    Creates vertical stripes with user-defined rates for all event types:
    - Hopping
    - Adsorption (A, B)
    - Desorption (A, B)
    - Reaction (A->B, B->A)
    
    --- OPTIMIZATION 1 ---
    This class pre-calculates a cumulative rate array for O(log M) event picking.
    """
    def __init__(self, n, num_stripes, 
                 stripe_hop_rates, 
                 stripe_ads_A_rates, stripe_des_A_rates,
                 stripe_ads_B_rates, stripe_des_B_rates,
                 stripe_react_A_to_B_rates, stripe_react_B_to_A_rates):
        
        self.n = n
        self.directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        # --- Create Rate Matrices for ALL event types ---
        hop_matrix = np.zeros((n, n, 4))
        ads_A_matrix = np.zeros((n, n))
        des_A_matrix = np.zeros((n, n))
        ads_B_matrix = np.zeros((n, n))
        des_B_matrix = np.zeros((n, n))
        react_A_to_B_matrix = np.zeros((n, n))
        react_B_to_A_matrix = np.zeros((n, n))
        
        stripe_width = n // num_stripes

        # Populate all rate matrices based on stripe index
        for stripe_index in range(num_stripes):
            x_start = stripe_index * stripe_width
            x_end = x_start + stripe_width
            
            # Get rates for this stripe
            rate_hop = stripe_hop_rates[stripe_index]
            rate_ads_A = stripe_ads_A_rates[stripe_index]
            rate_des_A = stripe_des_A_rates[stripe_index]
            rate_ads_B = stripe_ads_B_rates[stripe_index]
            rate_des_B = stripe_des_B_rates[stripe_index]
            rate_react_A = stripe_react_A_to_B_rates[stripe_index]
            rate_react_B = stripe_react_B_to_A_rates[stripe_index]
            
            for x in range(x_start, min(x_end, n)):
                for y in range(n):
                    hop_matrix[x, y, :] = rate_hop
                    ads_A_matrix[x, y] = rate_ads_A
                    des_A_matrix[x, y] = rate_des_A
                    ads_B_matrix[x, y] = rate_ads_B
                    des_B_matrix[x, y] = rate_des_B
                    react_A_to_B_matrix[x, y] = rate_react_A
                    react_B_to_A_matrix[x, y] = rate_react_B
        
        # --- Build the fast event catalog ---
        self.events = []
        event_rates = []

        for x in range(n):
            for y in range(n):
                site = (x, y)
                
                # Hopping events
                for d, (dx, dy) in enumerate(self.directions):
                    rate = float(hop_matrix[x, y, d])
                    if rate > 0:
                        # Hop event is particle-agnostic; _execute will check occupation
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
        
        # Create the cumulative rate array and store the total rate
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
        
        # Use binary search to find the index
        index = np.searchsorted(self.cumulative_rates, r, side='right')
        
        return self.events[index], T

class KMCSimulation:
    """
    --- OPTIMIZATION 2 ---
    Uses two dictionaries for O(1) lookups:
    1. self.particles:  pid -> {'site': (x,y), 'type': 'A'/'B'}
    2. self.occupation: site -> pid
    """
    def __init__(self, lattice, movement_matrix, max_unique_particles, total_time, seed):
        self.lattice = lattice
        self.mm = movement_matrix
        self.total_time = total_time
        self.max_particles = max_unique_particles
        self.rng = np.random.default_rng(seed)
        
        # self.particles: {pid: {'site': (x, y), 'type': 'A' or 'B'}}
        self.particles = {}
        # self.occupation: {(x, y): pid} - for O(1) lookups
        self.occupation = {}
        
        self.available_pids = set(range(self.max_particles))
        self.next_new_pid = self.max_particles
        self.time_stamps = [0.0]
        # Snapshots will store {(x, y): 'A', ...}
        self.snapshots = [{}] 
        self.trajectories = {}

    def _execute(self, ev):
        kind, data = ev[0], ev[1:]
        
        if kind == "in_A":
            site, _ = data
            # O(1) check
            if site not in self.occupation and len(self.particles) < self.max_particles:
                pid = self.available_pids.pop() if self.available_pids else self.next_new_pid
                if pid == self.next_new_pid: self.next_new_pid += 1
                
                # Update both maps
                self.particles[pid] = {'site': site, 'type': 'A'}
                self.occupation[site] = pid
                
                self.trajectories.setdefault(pid, []).append(site)
                return True
        
        elif kind == "in_B":
            site, _ = data
            # O(1) check
            if site not in self.occupation and len(self.particles) < self.max_particles:
                pid = self.available_pids.pop() if self.available_pids else self.next_new_pid
                if pid == self.next_new_pid: self.next_new_pid += 1
                
                # Update both maps
                self.particles[pid] = {'site': site, 'type': 'B'}
                self.occupation[site] = pid
                
                self.trajectories.setdefault(pid, []).append(site)
                return True
                
        elif kind == "out_A":
            site, _ = data
            # O(1) check and lookup
            if site in self.occupation:
                pid = self.occupation[site]
                # Check particle type before desorbing
                if self.particles[pid]['type'] == 'A':
                    del self.particles[pid]
                    del self.occupation[site]
                    if pid < self.max_particles: self.available_pids.add(pid)
                    return True

        elif kind == "out_B":
            site, _ = data
            # O(1) check and lookup
            if site in self.occupation:
                pid = self.occupation[site]
                # Check particle type before desorbing
                if self.particles[pid]['type'] == 'B':
                    del self.particles[pid]
                    del self.occupation[site]
                    if pid < self.max_particles: self.available_pids.add(pid)
                    return True
                
        elif kind == "hop":
            src, (dx, dy), _ = data
            # O(1) check and lookup
            if src in self.occupation:
                pid_at_src = self.occupation[src]
                dest = ((src[0] + dx) % self.lattice.size, (src[1] + dy) % self.lattice.size)
                
                # O(1) check
                if dest not in self.occupation:
                    # Update both maps
                    del self.occupation[src]
                    self.occupation[dest] = pid_at_src
                    # Update the particle's site, type remains unchanged
                    self.particles[pid_at_src]['site'] = dest 
                    
                    self.trajectories[pid_at_src].append(dest)
                    return True
        
        elif kind == "react_A_to_B":
            site, _ = data
            if site in self.occupation:
                pid = self.occupation[site]
                # Check type before reacting
                if self.particles[pid]['type'] == 'A':
                    self.particles[pid]['type'] = 'B' # Just change type
                    return True
        
        elif kind == "react_B_to_A":
            site, _ = data
            if site in self.occupation:
                pid = self.occupation[site]
                # Check type before reacting
                if self.particles[pid]['type'] == 'B':
                    self.particles[pid]['type'] = 'A' # Just change type
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
                # Create snapshot in {(x, y): type} format
                snapshot_data = {p_data['site']: p_data['type'] 
                                 for pid, p_data in self.particles.items()}
                self.snapshots.append(snapshot_data)
                self.time_stamps.append(t)
                last_recorded_t = t
        
        # Add final state
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

    # --- Stripe Configuration ---
    # Stripe 0: Blue
    # Stripe 1: Red
    NUM_STRIPES = 2
    
    # Per user request:
    # Hopping: 1.0 for both
    STRIPE_HOP_RATES = [1.0, 1.0] 
    
    # Blue (0): Ads A: 0.1, Des A: 0.001, Ads B: 0.0, Des B: 0.01
    # Red (1):  Ads A: 0.0, Des A: 0.01,  Ads B: 0.0, Des B: 0.1
    STRIPE_ADS_A_RATES = [0.1, 0.0]
    STRIPE_DES_A_RATES = [0.001, 0.01]
    STRIPE_ADS_B_RATES = [0.0, 0.0]
    STRIPE_DES_B_RATES = [0.01, 0.1]
    
    # Blue (0): A->B: 0, B->A: 0
    # Red (1):  A->B: 0.01, B->A: 0
    STRIPE_REACT_A_TO_B_RATES = [0.0, 0.01]
    STRIPE_REACT_B_TO_A_RATES = [0.0, 0.0]

    # --- Sanity Checks ---
    assert len(STRIPE_HOP_RATES) == NUM_STRIPES
    assert len(STRIPE_ADS_A_RATES) == NUM_STRIPES
    assert len(STRIPE_DES_A_RATES) == NUM_STRIPES
    assert len(STRIPE_ADS_B_RATES) == NUM_STRIPES
    assert len(STRIPE_DES_B_RATES) == NUM_STRIPES
    assert len(STRIPE_REACT_A_TO_B_RATES) == NUM_STRIPES
    assert len(STRIPE_REACT_B_TO_A_RATES) == NUM_STRIPES
    
    # ------------------------------------------------------------------
    
    start_time = time.time()
    print("Starting KMC simulation...")
    
    lattice = Lattice(LATTICE_SIZE)
    movement = MovementMatrixStripes(
        n=LATTICE_SIZE,
        num_stripes=NUM_STRIPES,
        stripe_hop_rates=STRIPE_HOP_RATES,
        stripe_ads_A_rates=STRIPE_ADS_A_RATES,
        stripe_des_A_rates=STRIPE_DES_A_RATES,
        stripe_ads_B_rates=STRIPE_ADS_B_RATES,
        stripe_des_B_rates=STRIPE_DES_B_RATES,
        stripe_react_A_to_B_rates=STRIPE_REACT_A_TO_B_RATES,
        stripe_react_B_to_A_rates=STRIPE_REACT_B_TO_A_RATES
    )
    
    # Max particles can still be total lattice size
    sim = KMCSimulation(lattice, movement, LATTICE_SIZE * LATTICE_SIZE, TOTAL_TIME, seed=RANDOM_SEED)
    
    trajectories, snapshots, time_stamps = sim.run()
    
    results = {
        'lattice_size': LATTICE_SIZE,
        'num_stripes': NUM_STRIPES,
        'total_time': TOTAL_TIME,
        'trajectories': trajectories,
        'snapshots': snapshots, # These are now {(x,y): 'A'/'B'}
        'time_stamps': time_stamps,
        'params': {
            'hop_rates': STRIPE_HOP_RATES,
            'ads_A_rates': STRIPE_ADS_A_RATES,
            'des_A_rates': STRIPE_DES_A_RATES,
            'ads_B_rates': STRIPE_ADS_B_RATES,
            'des_B_rates': STRIPE_DES_B_RATES,
            'react_A_to_B': STRIPE_REACT_A_TO_B_RATES,
            'react_B_to_A': STRIPE_REACT_B_TO_A_RATES,
        }
    }
    
    output_filename = "simulation_results_AB.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)

    end_time = time.time()
    execution_time = end_time - start_time
        
    print(f"\n Simulation complete...")
    print(f"   - Saved results to {output_filename}")
    print(f"   - Total execution time: {execution_time:.2f} seconds.")