import pickle
import numpy as np
import os
import sys

def analyze_results(filename, output_filename):
    """
    Loads A/B simulation results (from 'stripes' or 'random' mode)
    and performs a detailed analysis of particle counts (Total, A, and B),
    equilibrium, and fluctuations.
    Saves the report to a text file.
    """
    
    if not os.path.exists(filename):
        print(f"❌ Error: Results file not found at '{filename}'")
        print("➡️ Please run 'python run_simulation_fast.py' first to generate the results.")
        return

    # --- 1. Load Data ---
    print(f"Loading results from '{filename}'...")
    with open(filename, 'rb') as f:
        results = pickle.load(f)

    snapshots = results['snapshots']
    time_stamps = np.array(results['time_stamps'])
    n = results['lattice_size']
    trajectories = results['trajectories']
    
    # Load new unified data structure
    lattice_mode = results['lattice_mode']
    site_map = results['site_map'] # The n x n map
    num_site_types = results['num_site_types']
    
    if len(snapshots) == 0 or len(time_stamps) == 0:
        print("Error: No snapshots or timestamps found. Cannot analyze.")
        return

    # --- 2. Process Snapshots (A/B Aware) ---
    print("Processing snapshots to get Total, A, and B particle counts over time...")
    
    # --- THIS SECTION IS RE-WRITTEN ---
    # We no longer calculate boundaries; we use the site_map
    
    # Create history arrays
    total_particles_history = []
    a_particles_history = []
    b_particles_history = []
    
    # Create (num_snapshots x num_site_types) arrays
    site_type_total_history = np.zeros((len(snapshots), num_site_types))
    site_type_A_history = np.zeros((len(snapshots), num_site_types))
    site_type_B_history = np.zeros((len(snapshots), num_site_types))

    for snap_idx, snapshot in enumerate(snapshots):
        # snapshot is a dict {(x, y): 'A' or 'B'}
        total_particles_history.append(len(snapshot))
        
        a_count_snap = 0
        b_count_snap = 0
        
        # Count particles in each site type for this snapshot
        for (x, y), p_type in snapshot.items():
            # Tally total A/B
            if p_type == 'A':
                a_count_snap += 1
            elif p_type == 'B':
                b_count_snap += 1

            # --- KEY CHANGE ---
            # Get the site type (0 or 1) by looking up its (x,y) coord
            # in the site_map. Map is (row, col) so we use (y, x).
            site_type = site_map[y, x] 
            
            site_type_total_history[snap_idx, site_type] += 1
            if p_type == 'A':
                site_type_A_history[snap_idx, site_type] += 1
            elif p_type == 'B':
                site_type_B_history[snap_idx, site_type] += 1
        
        a_particles_history.append(a_count_snap)
        b_particles_history.append(b_count_snap)
                    
    total_particles_history = np.array(total_particles_history)
    a_particles_history = np.array(a_particles_history)
    b_particles_history = np.array(b_particles_history)

    # --- 3. Calculate Averages ---
    
    # Averages over the *entire* simulation
    avg_total_particles = np.mean(total_particles_history)
    avg_a_particles = np.mean(a_particles_history)
    avg_b_particles = np.mean(b_particles_history)
    
    avg_site_type_total = np.mean(site_type_total_history, axis=0)
    avg_site_type_A = np.mean(site_type_A_history, axis=0)
    avg_site_type_B = np.mean(site_type_B_history, axis=0)
    
    # Averages for the *last 25%* of the time
    quarter_index = len(time_stamps) // 4
    if quarter_index > 0:
        avg_total_last_q = np.mean(total_particles_history[-quarter_index:])
        avg_a_last_q = np.mean(a_particles_history[-quarter_index:])
        avg_b_last_q = np.mean(b_particles_history[-quarter_index:])
        
        avg_site_type_total_last_q = np.mean(site_type_total_history[-quarter_index:, :], axis=0)
        avg_site_type_A_last_q = np.mean(site_type_A_history[-quarter_index:, :], axis=0)
        avg_site_type_B_last_q = np.mean(site_type_B_history[-quarter_index:, :], axis=0)
    else: # Handle very short simulations
        avg_total_last_q = avg_total_particles
        avg_a_last_q = avg_a_particles
        avg_b_last_q = avg_b_particles
        avg_site_type_total_last_q = avg_site_type_total
        avg_site_type_A_last_q = avg_site_type_A
        avg_site_type_B_last_q = avg_site_type_B


    # --- 4. Find Time to Reach Average ---
    
    def find_time_to_avg(history, avg_val, times):
        rounded_avg = np.round(avg_val)
        if rounded_avg == 0 and avg_val > 0: # Handle low averages
             rounded_avg = 1
        
        indices_above_avg = np.where(history >= rounded_avg)[0]
        
        if len(indices_above_avg) > 0:
            first_index = indices_above_avg[0]
            time_to_avg = times[first_index]
            return time_to_avg, first_index
        else:
            return np.nan, -1 # Never reached average

    # Overall
    t_avg_total, idx_total = find_time_to_avg(total_particles_history, avg_total_particles, time_stamps)
    t_avg_A, idx_A = find_time_to_avg(a_particles_history, avg_a_particles, time_stamps)
    t_avg_B, idx_B = find_time_to_avg(b_particles_history, avg_b_particles, time_stamps)
    
    # Per Site Type
    t_avg_site_type_total, idx_site_type_total = [], []
    t_avg_site_type_A, idx_site_type_A = [], []
    t_avg_site_type_B, idx_site_type_B = [], []

    for i in range(num_site_types):
        t, idx = find_time_to_avg(site_type_total_history[:, i], avg_site_type_total[i], time_stamps)
        t_avg_site_type_total.append(t); idx_site_type_total.append(idx)
        
        t, idx = find_time_to_avg(site_type_A_history[:, i], avg_site_type_A[i], time_stamps)
        t_avg_site_type_A.append(t); idx_site_type_A.append(idx)
        
        t, idx = find_time_to_avg(site_type_B_history[:, i], avg_site_type_B[i], time_stamps)
        t_avg_site_type_B.append(t); idx_site_type_B.append(idx)

    # --- 5. Calculate Post-Equilibrium Variance ---
    
    def calc_post_avg_stats(history, first_index):
        if first_index != -1 and first_index < len(history) - 1:
            post_avg_history = history[first_index:]
            mean_val = np.mean(post_avg_history)
            variance = np.var(post_avg_history)
            std_dev = np.std(post_avg_history)
            fluctuation_coeff = std_dev / mean_val if mean_val > 0 else 0
            return variance, std_dev, fluctuation_coeff
        else:
            return np.nan, np.nan, np.nan

    # Overall
    var_total, std_total, fluc_total = calc_post_avg_stats(total_particles_history, idx_total)
    var_A, std_A, fluc_A = calc_post_avg_stats(a_particles_history, idx_A)
    var_B, std_B, fluc_B = calc_post_avg_stats(b_particles_history, idx_B)
    
    # Per Site Type
    var_s_total, std_s_total, fluc_s_total = [], [], []
    var_s_A, std_s_A, fluc_s_A = [], [], []
    var_s_B, std_s_B, fluc_s_B = [], [], []
    
    for i in range(num_site_types):
        v, s, f = calc_post_avg_stats(site_type_total_history[:, i], idx_site_type_total[i])
        var_s_total.append(v); std_s_total.append(s); fluc_s_total.append(f)
        
        v, s, f = calc_post_avg_stats(site_type_A_history[:, i], idx_site_type_A[i])
        var_s_A.append(v); std_s_A.append(s); fluc_s_A.append(f)

        v, s, f = calc_post_avg_stats(site_type_B_history[:, i], idx_site_type_B[i])
        var_s_B.append(v); std_s_B.append(s); fluc_s_B.append(f)

    # --- 6. Additional Statistics ---
    
    total_sites = n * n
    
    def calc_density(counts, num_sites):
        if num_sites == 0:
            return 0.0
        return counts / num_sites
        
    # --- KEY CHANGE: Calculate sites per type ---
    sites_per_type = []
    for i in range(num_site_types):
        count = np.count_nonzero(site_map == i)
        sites_per_type.append(count)

    # Density (Full)
    dens_total = calc_density(avg_total_particles, total_sites)
    dens_A = calc_density(avg_a_particles, total_sites)
    dens_B = calc_density(avg_b_particles, total_sites)
    
    dens_s_total, dens_s_A, dens_s_B = [], [], []
    for i in range(num_site_types):
        sites_in_type = sites_per_type[i]
        dens_s_total.append(calc_density(avg_site_type_total[i], sites_in_type))
        dens_s_A.append(calc_density(avg_site_type_A[i], sites_in_type))
        dens_s_B.append(calc_density(avg_site_type_B[i], sites_in_type))

    # Density (Last 25%)
    dens_total_last_q = calc_density(avg_total_last_q, total_sites)
    dens_A_last_q = calc_density(avg_a_last_q, total_sites)
    dens_B_last_q = calc_density(avg_b_last_q, total_sites)
    
    dens_s_total_last_q, dens_s_A_last_q, dens_s_B_last_q = [], [], []
    for i in range(num_site_types):
        sites_in_type = sites_per_type[i]
        dens_s_total_last_q.append(calc_density(avg_site_type_total_last_q[i], sites_in_type))
        dens_s_A_last_q.append(calc_density(avg_site_type_A_last_q[i], sites_in_type))
        dens_s_B_last_q.append(calc_density(avg_site_type_B_last_q[i], sites_in_type))

    # Trajectory Analysis (no change needed)
    total_unique_particles = len(trajectories)
    hops_per_particle = [len(traj) - 1 for traj in trajectories.values() if len(traj) > 0]
    avg_hops = np.mean(hops_per_particle) if hops_per_particle else 0
    max_hops = np.max(hops_per_particle) if hops_per_particle else 0
    
    # --- 7. Write Report to File ---
    
    print(f"\nWriting analysis report to '{output_filename}'...")
    
    with open(output_filename, 'w') as f:
        f.write("--- KMC Simulation Analysis Report (A/B) ---\n")
        f.write(f"Source File: {filename}\n")
        
        f.write("\n--- Simulation Parameters ---\n")
        f.write(f"Lattice Mode:                 {lattice_mode}\n")
        f.write(f"Total Simulation Time:        {results['total_time']:.2f}\n")
        f.write(f"Total Lattice Sites:          {total_sites} ({n}x{n})\n")
        
        if lattice_mode == 'random':
            f.write(f"Site Type Fractions:          {results['site_type_fractions']}\n")
        
        # --- Overall Stats ---
        f.write("\n" + "="*40 + "\n")
        f.write("--- Overall Lattice Statistics ---\n")
        f.write("="*40 + "\n")
        
        f.write("\n[Overall - ALL PARTICLES]\n")
        f.write(f"  Avg. Count (Full):          {avg_total_particles:.2f}\n")
        f.write(f"  Avg. Count (Last 25%):      {avg_total_last_q:.2f}\n")
        f.write(f"  Avg. Density (Full):          {dens_total:.4f} p/site\n")
        f.write(f"  Avg. Density (Last 25%):      {dens_total_last_q:.4f} p/site\n")
        f.write(f"  Time to Reach Avg. (First):   {t_avg_total:.2f}\n")
        f.write(f"  Variance (Post-First Avg.):   {var_total:.2f}\n")
        f.write(f"  Std. Dev. (Post-First Avg.):  {std_total:.2f}\n")
        f.write(f"  Fluctuation Coeff (Std/Mean): {fluc_total:.4f}\n")

        f.write("\n[Overall - TYPE A PARTICLES]\n")
        # ... (identical to before) ...
        f.write(f"  Avg. Count (Full):          {avg_a_particles:.2f}\n")
        f.write(f"  Avg. Count (Last 25%):      {avg_a_last_q:.2f}\n")
        f.write(f"  Avg. Density (Full):          {dens_A:.4f} p/site\n")
        f.write(f"  Avg. Density (Last 25%):      {dens_A_last_q:.4f} p/site\n")
        f.write(f"  Time to Reach Avg. (First):   {t_avg_A:.2f}\n")
        f.write(f"  Variance (Post-First Avg.):   {var_A:.2f}\n")
        f.write(f"  Std. Dev. (Post-First Avg.):  {std_A:.2f}\n")
        f.write(f"  Fluctuation Coeff (Std/Mean): {fluc_A:.4f}\n")
        
        f.write("\n[Overall - TYPE B PARTICLES]\n")
        # ... (identical to before) ...
        f.write(f"  Avg. Count (Full):          {avg_b_particles:.2f}\n")
        f.write(f"  Avg. Count (Last 25%):      {avg_b_last_q:.2f}\n")
        f.write(f"  Avg. Density (Full):          {dens_B:.4f} p/site\n")
        f.write(f"  Avg. Density (Last 25%):      {dens_B_last_q:.4f} p/site\n")
        f.write(f"  Time to Reach Avg. (First):   {t_avg_B:.2f}\n")
        f.write(f"  Variance (Post-First Avg.):   {var_B:.2f}\n")
        f.write(f"  Std. Dev. (Post-First Avg.):  {std_B:.2f}\n")
        f.write(f"  Fluctuation Coeff (Std/Mean): {fluc_B:.4f}\n")
        
        # --- Per-Site-Type Stats ---
        f.write("\n" + "="*40 + "\n")
        f.write("--- Per-Site-Type Statistics ---\n")
        f.write("="*40 + "\n")
        
        site_type_names = ['Blue', 'Red'] # From previous context
        
        for i in range(num_site_types):
            sites_in_type = sites_per_type[i]
            name = site_type_names[i % len(site_type_names)]
            
            f.write(f"\n--- [Site Type {i} ('{name}') | {sites_in_type} sites] ---\n")
            
            f.write(f"\n  [Site Type {i} - ALL PARTICLES]\n")
            f.write(f"    Avg. Count (Full):            {avg_site_type_total[i]:.2f}\n")
            f.write(f"    Avg. Count (Last 25%):        {avg_site_type_total_last_q[i]:.2f}\n")
            f.write(f"    Avg. Density (Full):          {dens_s_total[i]:.4f} p/site\n")
            f.write(f"    Avg. Density (Last 25%):      {dens_s_total_last_q[i]:.4f} p/site\n")
            f.write(f"    Time to Reach Avg. (First):   {t_avg_site_type_total[i]:.2f}\n")
            f.write(f"    Variance (Post-First Avg.):   {var_s_total[i]:.2f}\n")
            f.write(f"    Std. Dev. (Post-First Avg.):  {std_s_total[i]:.2f}\n")
            f.write(f"    Fluctuation Coeff (Std/Mean): {fluc_s_total[i]:.4f}\n")

            f.write(f"\n  [Site Type {i} - TYPE A PARTICLES]\n")
            f.write(f"    Avg. Count (Full):            {avg_site_type_A[i]:.2f}\n")
            f.write(f"    Avg. Count (Last 25%):        {avg_site_type_A_last_q[i]:.2f}\n")
            f.write(f"    Avg. Density (Full):          {dens_s_A[i]:.4f} p/site\n")
            f.write(f"    Avg. Density (Last 25%):      {dens_s_A_last_q[i]:.4f} p/site\n")
            f.write(f"    Time to Reach Avg. (First):   {t_avg_site_type_A[i]:.2f}\n")
            f.write(f"    Variance (Post-First Avg.):   {var_s_A[i]:.2f}\n")
            f.write(f"    Std. Dev. (Post-First Avg.):  {std_s_A[i]:.2f}\n")
            f.write(f"    Fluctuation Coeff (Std/Mean): {fluc_s_A[i]:.4f}\n")
            
            f.write(f"\n  [Site Type {i} - TYPE B PARTICLES]\n")
            f.write(f"    Avg. Count (Full):            {avg_site_type_B[i]:.2f}\n")
            f.write(f"    Avg. Count (Last 25%):        {avg_site_type_B_last_q[i]:.2f}\n")
            f.write(f"    Avg. Density (Full):          {dens_s_B[i]:.4f} p/site\n")
            f.write(f"    Avg. Density (Last 25%):      {dens_s_B_last_q[i]:.4f} p/site\n")
            f.write(f"    Time to Reach Avg. (First):   {t_avg_site_type_B[i]:.2f}\n")
            f.write(f"    Variance (Post-First Avg.):   {var_s_B[i]:.2f}\n")
            f.write(f"    Std. Dev. (Post-First Avg.):  {std_s_B[i]:.2f}\n")
            f.write(f"    Fluctuation Coeff (Std/Mean): {fluc_s_B[i]:.4f}\n")

        # --- Trajectory Stats ---
        f.write("\n" + "="*40 + "\n")
        f.write("--- Trajectory Statistics ---\n")
        f.write("="*40 + "\n")
        f.write(f"Total Unique Particles Served: {total_unique_particles}\n")
        f.write(f"Avg. Hops per Particle:        {avg_hops:.2f}\n")
        f.write(f"Max. Hops by a Particle:       {max_hops}\n")
    
    print("\n✅ Analysis complete.")
    print(f"Results saved to '{output_filename}'")


if __name__ == "__main__":
    
    # --- Smart File Loading ---
    INPUT_FILENAMES = {
        'stripes': "simulation_results_stripes.pkl",
        'random': "simulation_results_random.pkl"
    }
    
    OUTPUT_FILENAMES = {
        'stripes': "analysis_stripes.txt",
        'random': "analysis_random.txt"
    }
    
    loaded_mode = None
    for mode, fname in INPUT_FILENAMES.items():
        if os.path.exists(fname):
            loaded_mode = mode
            break

    if loaded_mode is None:
        print(f"❌ Error: No simulation results file found.")
        print("   Please run 'python run_simulation_fast.py' first.")
        print(f"   (Was looking for {list(INPUT_FILENAMES.values())})")
        sys.exit(1)
        
    input_file = INPUT_FILENAMES[loaded_mode]
    output_file = OUTPUT_FILENAMES[loaded_mode]
    
    analyze_results(filename=input_file, output_filename=output_file)