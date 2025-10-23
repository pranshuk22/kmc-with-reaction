import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def generate_plots_ab(filename="simulation_results_AB.pkl", output_filename="plots_AB.png"):
    """
    Loads A/B simulation results and generates a multi-panel plot
    saving it to a single image file.
    
    The plot will contain:
    1. Total, A, and B particle counts vs. Time (Full System)
    2. A and B particle counts vs. Time (Per Stripe)
    3. Histogram of Total Particle Count (Full System)
    """
    
    if not os.path.exists(filename):
        print(f"❌ Error: Results file not found at '{filename}'")
        print("➡️ Please run 'python run_simulation_fast.py' first to generate the data.")
        return

    print(f"Loading results from '{filename}' for plotting...")
    with open(filename, 'rb') as f:
        results = pickle.load(f)

    snapshots = results['snapshots']
    time_stamps = np.array(results['time_stamps'])
    n = results['lattice_size']
    num_stripes = results['num_stripes']
    
    if len(snapshots) == 0 or len(time_stamps) == 0:
        print("Error: No snapshots or timestamps found. Cannot generate plots.")
        return

    # --- Process Snapshots (A/B Aware) ---
    print("Processing snapshots for A/B plot data...")
    
    stripe_width = n // num_stripes
    stripe_boundaries = []
    for i in range(num_stripes):
        x_start = i * stripe_width
        x_end = x_start + stripe_width
        if i == num_stripes - 1:
            x_end = n
        stripe_boundaries.append((x_start, x_end))
    
    # Create history arrays
    total_particles_history = []
    a_particles_history = []
    b_particles_history = []
    
    # Create (num_snapshots x num_stripes) arrays
    stripe_A_history = np.zeros((len(snapshots), num_stripes))
    stripe_B_history = np.zeros((len(snapshots), num_stripes))

    for snap_idx, snapshot in enumerate(snapshots):
        # snapshot is a dict {(x, y): 'A' or 'B'}
        total_particles_history.append(len(snapshot))
        
        a_count_snap = 0
        b_count_snap = 0
        
        counts_A_in_snapshot = [0] * num_stripes
        counts_B_in_snapshot = [0] * num_stripes

        # Count particles in each stripe for this snapshot
        for (x, y), p_type in snapshot.items():
            if p_type == 'A':
                a_count_snap += 1
            elif p_type == 'B':
                b_count_snap += 1

            for stripe_idx, (x_start, x_end) in enumerate(stripe_boundaries):
                if x_start <= x < x_end:
                    if p_type == 'A':
                        counts_A_in_snapshot[stripe_idx] += 1
                    elif p_type == 'B':
                        counts_B_in_snapshot[stripe_idx] += 1
                    break # Particle found, move to next particle
        
        a_particles_history.append(a_count_snap)
        b_particles_history.append(b_count_snap)
        
        for i in range(num_stripes):
            stripe_A_history[snap_idx, i] = counts_A_in_snapshot[i]
            stripe_B_history[snap_idx, i] = counts_B_in_snapshot[i]
                    
    total_particles_history = np.array(total_particles_history)
    a_particles_history = np.array(a_particles_history)
    b_particles_history = np.array(b_particles_history)

    # --- Data for Histogram ---
    
    def find_time_to_avg(history, avg_val, times):
        rounded_avg = np.round(avg_val)
        indices_above_avg = np.where(history >= rounded_avg)[0]
        if len(indices_above_avg) > 0:
            return times[indices_above_avg[0]]
        return np.nan

    avg_total_particles = np.mean(total_particles_history)
    time_to_avg_overall = find_time_to_avg(total_particles_history, avg_total_particles, time_stamps)
    
    first_index_overall_for_hist = -1
    if not np.isnan(time_to_avg_overall):
        first_index_overall_for_hist = np.searchsorted(time_stamps, time_to_avg_overall)

    data_to_hist = total_particles_history
    hist_title_suffix = '(Full Simulation)'
    if first_index_overall_for_hist != -1 and first_index_overall_for_hist < len(total_particles_history) - 1:
        data_to_hist = total_particles_history[first_index_overall_for_hist:]
        hist_title_suffix = '(Post-Equilibrium)'

    # --- Generate Plots ---
    print(f"Generating multi-panel plot and saving to '{output_filename}'...")

    # We need 2 plots + 1 plot per stripe
    num_plots = 2 + num_stripes
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), sharex=True)
    fig.suptitle('KMC Simulation Analysis (A/B Particles)', fontsize=18, weight='bold')

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Colors
    color_A = '#0072B2' # Blue
    color_B = '#D55E00' # Red
    color_Total = '#333333' # Dark Grey

    # --- Plot 1: Total, A, and B Particle Count vs. Time ---
    ax0 = axes[0]
    ax0.plot(time_stamps, total_particles_history, label='Total Count', alpha=0.9, linewidth=2.5, color=color_Total)
    ax0.plot(time_stamps, a_particles_history, label='Count A', alpha=0.8, linewidth=1.5, color=color_A)
    ax0.plot(time_stamps, b_particles_history, label='Count B', alpha=0.8, linewidth=1.5, color=color_B)
    
    ax0.set_title('1. Total System Occupancy vs. Time', fontsize=14)
    ax0.set_ylabel('Total Particle Count', fontsize=12)
    ax0.legend(fontsize=10)
    ax0.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Plot 2...N: Stripe Particle Counts vs. Time ---
    stripe_names = ['Blue', 'Red'] # From user context
    
    for i in range(num_stripes):
        ax = axes[i + 1]
        name = stripe_names[i % len(stripe_names)]
        x_start, x_end = stripe_boundaries[i]
        title = f'2.{i}. Occupancy for Stripe {i} ("{name}") (x={x_start}-{x_end-1})'

        ax.plot(time_stamps, stripe_A_history[:, i], label='Count A', alpha=0.8, color=color_A, linewidth=1.5)
        ax.plot(time_stamps, stripe_B_history[:, i], label='Count B', alpha=0.8, color=color_B, linewidth=1.5)

        # Calculate and plot averages for this stripe
        avg_A = np.mean(stripe_A_history[:, i])
        avg_B = np.mean(stripe_B_history[:, i])
        ax.axhline(avg_A, color=color_A, linestyle=':', alpha=0.7, linewidth=1.5, label=f'Avg A ({avg_A:.2f})')
        ax.axhline(avg_B, color=color_B, linestyle=':', alpha=0.7, linewidth=1.5, label=f'Avg B ({avg_B:.2f})')
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Particle Count', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Last Plot: Particle Count Distribution (Histogram) ---
    ax_hist = axes[-1]
    
    # Ensure bins are appropriate
    min_bin = int(np.min(data_to_hist))
    max_bin = int(np.max(data_to_hist))
    num_bins = max(1, max_bin - min_bin) # One bin for each integer value
    
    ax_hist.hist(data_to_hist, bins=num_bins, density=True, alpha=0.75, color='skyblue', edgecolor='black', label='Count Distribution')
    
    if len(data_to_hist) > 0:
        mean_hist_data = np.mean(data_to_hist)
        std_hist_data = np.std(data_to_hist)
        ax_hist.axvline(mean_hist_data, color='r', linestyle='--', label=f'Mean ({mean_hist_data:.2f})', linewidth=2)
        ax_hist.axvline(mean_hist_data + std_hist_data, color='orange', linestyle=':', label=f'Mean +/- Std ({std_hist_data:.2f})', linewidth=1.5)
        ax_hist.axvline(mean_hist_data - std_hist_data, color='orange', linestyle=':', linewidth=1.5)

    ax_hist.set_title(f'3. Distribution of Total Particle Count {hist_title_suffix}', fontsize=14)
    ax_hist.set_xlabel('Total Particle Count', fontsize=12)
    ax_hist.set_ylabel('Probability Density', fontsize=12)
    ax_hist.legend(fontsize=10)
    ax_hist.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set shared X-axis label
    axes[-1].set_xlabel('Time (s)', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent suptitle overlap
    plt.savefig(output_filename, dpi=300)
    print(f"✅ All plots saved to '{output_filename}'")


if __name__ == "__main__":
    generate_plots_ab(filename="simulation_results_AB.pkl", output_filename="plots_AB.png")