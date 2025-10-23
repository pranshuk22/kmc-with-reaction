import pickle
import numpy as np
import os
import sys

def analyze_results_ab(filename="simulation_results_AB.pkl", output_filename="analysis_AB.txt"):
    """
    Loads A/B simulation results from a pickle file and performs
    a detailed analysis of particle counts (Total, A, and B),
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
    num_stripes = results['num_stripes']
    trajectories = results['trajectories']
    
    if len(snapshots) == 0 or len(time_stamps) == 0:
        print("Error: No snapshots or timestamps found. Cannot analyze.")
        return

    # --- 2. Process Snapshots (A/B Aware) ---
    print("Processing snapshots to get Total, A, and B particle counts over time...")
    
    # Define stripe boundaries
    stripe_width = n // num_stripes
    stripe_boundaries = []
    for i in range(num_stripes):
        x_start = i * stripe_width
        x_end = x_start + stripe_width
        if i == num_stripes - 1:
            x_end = n # Ensure the last stripe goes to the edge
        stripe_boundaries.append((x_start, x_end))
    
    # Create history arrays
    total_particles_history = []
    a_particles_history = []
    b_particles_history = []
    
    # Create (num_snapshots x num_stripes) arrays
    stripe_total_history = np.zeros((len(snapshots), num_stripes))
    stripe_A_history = np.zeros((len(snapshots), num_stripes))
    stripe_B_history = np.zeros((len(snapshots), num_stripes))

    for snap_idx, snapshot in enumerate(snapshots):
        # snapshot is a dict {(x, y): 'A' or 'B'}
        total_particles_history.append(len(snapshot))
        
        a_count_snap = 0
        b_count_snap = 0
        
        # Count particles in each stripe for this snapshot
        for (x, y), p_type in snapshot.items():
            # Tally total A/B
            if p_type == 'A':
                a_count_snap += 1
            elif p_type == 'B':
                b_count_snap += 1

            # Tally by stripe
            for stripe_idx, (x_start, x_end) in enumerate(stripe_boundaries):
                if x_start <= x < x_end:
                    stripe_total_history[snap_idx, stripe_idx] += 1
                    if p_type == 'A':
                        stripe_A_history[snap_idx, stripe_idx] += 1
                    elif p_type == 'B':
                        stripe_B_history[snap_idx, stripe_idx] += 1
                    break # Particle found, move to next particle
        
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
    
    avg_stripe_total = np.mean(stripe_total_history, axis=0)
    avg_stripe_A = np.mean(stripe_A_history, axis=0)
    avg_stripe_B = np.mean(stripe_B_history, axis=0)
    
    # Averages for the *last 25%* of the time
    quarter_index = len(time_stamps) // 4
    if quarter_index > 0:
        avg_total_last_q = np.mean(total_particles_history[-quarter_index:])
        avg_a_last_q = np.mean(a_particles_history[-quarter_index:])
        avg_b_last_q = np.mean(b_particles_history[-quarter_index:])
        
        avg_stripe_total_last_q = np.mean(stripe_total_history[-quarter_index:, :], axis=0)
        avg_stripe_A_last_q = np.mean(stripe_A_history[-quarter_index:, :], axis=0)
        avg_stripe_B_last_q = np.mean(stripe_B_history[-quarter_index:, :], axis=0)
    else: # Handle very short simulations
        avg_total_last_q = avg_total_particles
        avg_a_last_q = avg_a_particles
        avg_b_last_q = avg_b_particles
        avg_stripe_total_last_q = avg_stripe_total
        avg_stripe_A_last_q = avg_stripe_A
        avg_stripe_B_last_q = avg_stripe_B


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
    
    # Per Stripe
    t_avg_stripe_total, idx_stripe_total = [], []
    t_avg_stripe_A, idx_stripe_A = [], []
    t_avg_stripe_B, idx_stripe_B = [], []

    for i in range(num_stripes):
        t, idx = find_time_to_avg(stripe_total_history[:, i], avg_stripe_total[i], time_stamps)
        t_avg_stripe_total.append(t); idx_stripe_total.append(idx)
        
        t, idx = find_time_to_avg(stripe_A_history[:, i], avg_stripe_A[i], time_stamps)
        t_avg_stripe_A.append(t); idx_stripe_A.append(idx)
        
        t, idx = find_time_to_avg(stripe_B_history[:, i], avg_stripe_B[i], time_stamps)
        t_avg_stripe_B.append(t); idx_stripe_B.append(idx)

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
    
    # Per Stripe
    var_s_total, std_s_total, fluc_s_total = [], [], []
    var_s_A, std_s_A, fluc_s_A = [], [], []
    var_s_B, std_s_B, fluc_s_B = [], [], []
    
    for i in range(num_stripes):
        v, s, f = calc_post_avg_stats(stripe_total_history[:, i], idx_stripe_total[i])
        var_s_total.append(v); std_s_total.append(s); fluc_s_total.append(f)
        
        v, s, f = calc_post_avg_stats(stripe_A_history[:, i], idx_stripe_A[i])
        var_s_A.append(v); std_s_A.append(s); fluc_s_A.append(f)

        v, s, f = calc_post_avg_stats(stripe_B_history[:, i], idx_stripe_B[i])
        var_s_B.append(v); std_s_B.append(s); fluc_s_B.append(f)

    # --- 6. Additional Statistics ---
    
    total_sites = n * n
    
    def calc_density(counts, num_sites):
        return counts / num_sites
        
    # Density (Full)
    dens_total = calc_density(avg_total_particles, total_sites)
    dens_A = calc_density(avg_a_particles, total_sites)
    dens_B = calc_density(avg_b_particles, total_sites)
    
    dens_s_total, dens_s_A, dens_s_B = [], [], []
    for i in range(num_stripes):
        sites_in_stripe = (stripe_boundaries[i][1] - stripe_boundaries[i][0]) * n
        dens_s_total.append(calc_density(avg_stripe_total[i], sites_in_stripe))
        dens_s_A.append(calc_density(avg_stripe_A[i], sites_in_stripe))
        dens_s_B.append(calc_density(avg_stripe_B[i], sites_in_stripe))

    # Density (Last 25%)
    dens_total_last_q = calc_density(avg_total_last_q, total_sites)
    dens_A_last_q = calc_density(avg_a_last_q, total_sites)
    dens_B_last_q = calc_density(avg_b_last_q, total_sites)
    
    dens_s_total_last_q, dens_s_A_last_q, dens_s_B_last_q = [], [], []
    for i in range(num_stripes):
        sites_in_stripe = (stripe_boundaries[i][1] - stripe_boundaries[i][0]) * n
        dens_s_total_last_q.append(calc_density(avg_stripe_total_last_q[i], sites_in_stripe))
        dens_s_A_last_q.append(calc_density(avg_stripe_A_last_q[i], sites_in_stripe))
        dens_s_B_last_q.append(calc_density(avg_stripe_B_last_q[i], sites_in_stripe))

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
        f.write(f"Total Simulation Time:        {results['total_time']:.2f}\n")
        f.write(f"Total Lattice Sites:          {total_sites} ({n}x{n})\n")
        f.write(f"Stripe Boundaries:            {stripe_boundaries}\n")
        
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
        f.write(f"  Avg. Count (Full):          {avg_a_particles:.2f}\n")
        f.write(f"  Avg. Count (Last 25%):      {avg_a_last_q:.2f}\n")
        f.write(f"  Avg. Density (Full):          {dens_A:.4f} p/site\n")
        f.write(f"  Avg. Density (Last 25%):      {dens_A_last_q:.4f} p/site\n")
        f.write(f"  Time to Reach Avg. (First):   {t_avg_A:.2f}\n")
        f.write(f"  Variance (Post-First Avg.):   {var_A:.2f}\n")
        f.write(f"  Std. Dev. (Post-First Avg.):  {std_A:.2f}\n")
        f.write(f"  Fluctuation Coeff (Std/Mean): {fluc_A:.4f}\n")
        
        f.write("\n[Overall - TYPE B PARTICLES]\n")
        f.write(f"  Avg. Count (Full):          {avg_b_particles:.2f}\n")
        f.write(f"  Avg. Count (Last 25%):      {avg_b_last_q:.2f}\n")
        f.write(f"  Avg. Density (Full):          {dens_B:.4f} p/site\n")
        f.write(f"  Avg. Density (Last 25%):      {dens_B_last_q:.4f} p/site\n")
        f.write(f"  Time to Reach Avg. (First):   {t_avg_B:.2f}\n")
        f.write(f"  Variance (Post-First Avg.):   {var_B:.2f}\n")
        f.write(f"  Std. Dev. (Post-First Avg.):  {std_B:.2f}\n")
        f.write(f"  Fluctuation Coeff (Std/Mean): {fluc_B:.4f}\n")
        
        # --- Per-Stripe Stats ---
        f.write("\n" + "="*40 + "\n")
        f.write("--- Per-Stripe Statistics ---\n")
        f.write("="*40 + "\n")
        
        stripe_names = ['Blue', 'Red'] # From previous context
        
        for i in range(num_stripes):
            x_start, x_end = stripe_boundaries[i]
            sites_in_stripe = (x_end - x_start) * n
            name = stripe_names[i % len(stripe_names)]
            
            f.write(f"\n--- [Stripe {i} ('{name}') | x={x_start}-{x_end-1} | {sites_in_stripe} sites] ---\n")
            
            f.write("\n  [Stripe {i} - ALL PARTICLES]\n")
            f.write(f"    Avg. Count (Full):            {avg_stripe_total[i]:.2f}\n")
            f.write(f"    Avg. Count (Last 25%):        {avg_stripe_total_last_q[i]:.2f}\n")
            f.write(f"    Avg. Density (Full):          {dens_s_total[i]:.4f} p/site\n")
            f.write(f"    Avg. Density (Last 25%):      {dens_s_total_last_q[i]:.4f} p/site\n")
            f.write(f"    Time to Reach Avg. (First):   {t_avg_stripe_total[i]:.2f}\n")
            f.write(f"    Variance (Post-First Avg.):   {var_s_total[i]:.2f}\n")
            f.write(f"    Std. Dev. (Post-First Avg.):  {std_s_total[i]:.2f}\n")
            f.write(f"    Fluctuation Coeff (Std/Mean): {fluc_s_total[i]:.4f}\n")

            f.write("\n  [Stripe {i} - TYPE A PARTICLES]\n")
            f.write(f"    Avg. Count (Full):            {avg_stripe_A[i]:.2f}\n")
            f.write(f"    Avg. Count (Last 25%):        {avg_stripe_A_last_q[i]:.2f}\n")
            f.write(f"    Avg. Density (Full):          {dens_s_A[i]:.4f} p/site\n")
            f.write(f"    Avg. Density (Last 25%):      {dens_s_A_last_q[i]:.4f} p/site\n")
            f.write(f"    Time to Reach Avg. (First):   {t_avg_stripe_A[i]:.2f}\n")
            f.write(f"    Variance (Post-First Avg.):   {var_s_A[i]:.2f}\n")
            f.write(f"    Std. Dev. (Post-First Avg.):  {std_s_A[i]:.2f}\n")
            f.write(f"    Fluctuation Coeff (Std/Mean): {fluc_s_A[i]:.4f}\n")
            
            f.write("\n  [Stripe {i} - TYPE B PARTICLES]\n")
            f.write(f"    Avg. Count (Full):            {avg_stripe_B[i]:.2f}\n")
            f.write(f"    Avg. Count (Last 25%):        {avg_stripe_B_last_q[i]:.2f}\n")
            f.write(f"    Avg. Density (Full):          {dens_s_B[i]:.4f} p/site\n")
            f.write(f"    Avg. Density (Last 25%):      {dens_s_B_last_q[i]:.4f} p/site\n")
            f.write(f"    Time to Reach Avg. (First):   {t_avg_stripe_B[i]:.2f}\n")
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
    # Call the new function with the new default filenames
    analyze_results_ab(filename="simulation_results_AB.pkl", output_filename="analysis_AB.txt")