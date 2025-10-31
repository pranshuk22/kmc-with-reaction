import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_plots_ab(filename="simulation_results_AB.pkl", output_filename="plots_AB.png"):
    """
    Loads A/B simulation results and generates a multi-panel plot.
    Supports both 'stripes' and 'random' lattice modes.

    The plot will contain:
    1. Total, A, and B particle counts vs. Time (Full System)
    2. Per-region (stripe or site-type) A/B counts vs. Time
    3. Histogram of Total Particle Count
    """

    if not os.path.exists(filename):
        print(f"❌ Error: Results file not found at '{filename}'")
        print("➡️ Please run the simulation first to generate the data.")
        return

    print(f"Loading results from '{filename}' for plotting...")
    with open(filename, 'rb') as f:
        results = pickle.load(f)

    snapshots = results['snapshots']
    time_stamps = np.array(results['time_stamps'])
    n = results['lattice_size']
    lattice_mode = results.get('lattice_mode', 'stripes')
    num_site_types = results.get('num_site_types', 2)
    site_map = results.get('site_map', None)

    if len(snapshots) == 0 or len(time_stamps) == 0:
        print("Error: No snapshots or timestamps found. Cannot generate plots.")
        return

    print(f"Processing snapshots for lattice mode: '{lattice_mode}'...")

    total_particles_history = []
    a_particles_history = []
    b_particles_history = []

    if lattice_mode == 'stripes':
        # Stripe-based segmentation
        num_regions = num_site_types
        stripe_width = n // num_regions
        region_boundaries = [(i * stripe_width, n if i == num_regions - 1 else (i + 1) * stripe_width)
                             for i in range(num_regions)]
    else:
        # Random mode uses site types (0, 1, etc.)
        if site_map is None:
            print("❌ Error: site_map not found in results file for 'random' mode.")
            return
        num_regions = num_site_types
        region_masks = [(site_map == i) for i in range(num_regions)]

    region_A_history = np.zeros((len(snapshots), num_regions))
    region_B_history = np.zeros((len(snapshots), num_regions))

    for snap_idx, snapshot in enumerate(snapshots):
        total_particles_history.append(len(snapshot))

        a_count_snap = 0
        b_count_snap = 0
        counts_A = np.zeros(num_regions)
        counts_B = np.zeros(num_regions)

        for (x, y), p_type in snapshot.items():
            if p_type == 'A':
                a_count_snap += 1
            elif p_type == 'B':
                b_count_snap += 1

            if lattice_mode == 'stripes':
                for r, (x_start, x_end) in enumerate(region_boundaries):
                    if x_start <= x < x_end:
                        if p_type == 'A':
                            counts_A[r] += 1
                        else:
                            counts_B[r] += 1
                        break
            else:
                # Random mode: use site_map to determine region type
                if site_map[y, x] < num_regions:
                    site_type = site_map[y, x]
                    if p_type == 'A':
                        counts_A[site_type] += 1
                    else:
                        counts_B[site_type] += 1

        a_particles_history.append(a_count_snap)
        b_particles_history.append(b_count_snap)
        region_A_history[snap_idx, :] = counts_A
        region_B_history[snap_idx, :] = counts_B

    total_particles_history = np.array(total_particles_history)
    a_particles_history = np.array(a_particles_history)
    b_particles_history = np.array(b_particles_history)

    # --- Histogram ---
    avg_total = np.mean(total_particles_history)
    hist_data = total_particles_history

    # --- Plot ---
    print(f"Generating multi-panel plot → '{output_filename}'")

    num_plots = 2 + num_regions
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots), sharex=False)
    fig.suptitle(f"KMC Simulation Analysis ({lattice_mode.upper()} mode)", fontsize=18, weight='bold')

    plt.style.use('seaborn-v0_8-whitegrid')
    color_A = '#0072B2'
    color_B = '#D55E00'
    color_Total = '#333333'

    # --- Plot 1: Total A/B ---
    ax0 = axes[0]
    ax0.plot(time_stamps, total_particles_history, color=color_Total, label='Total', lw=2)
    ax0.plot(time_stamps, a_particles_history, color=color_A, label='A', lw=1.5)
    ax0.plot(time_stamps, b_particles_history, color=color_B, label='B', lw=1.5)
    ax0.set_title("1. Total Particle Counts vs. Time", fontsize=14)
    ax0.set_ylabel("Count")
    ax0.legend()

    # --- Region plots ---
    region_labels = [f"Type {i}" for i in range(num_regions)]
    for i in range(num_regions):
        ax = axes[i + 1]
        ax.plot(time_stamps, region_A_history[:, i], color=color_A, label='A')
        ax.plot(time_stamps, region_B_history[:, i], color=color_B, label='B')
        avg_A = np.mean(region_A_history[:, i])
        avg_B = np.mean(region_B_history[:, i])
        ax.axhline(avg_A, color=color_A, linestyle=':', alpha=0.6)
        ax.axhline(avg_B, color=color_B, linestyle=':', alpha=0.6)
        ax.set_title(f"2.{i} Region {region_labels[i]} A/B Occupancy")
        ax.set_ylabel("Count")
        ax.legend()

    # --- Histogram ---
    ax_hist = axes[-1]
    ax_hist.hist(hist_data, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    mean_val = np.mean(hist_data)
    std_val = np.std(hist_data)
    ax_hist.axvline(mean_val, color='red', linestyle='--', lw=2, label=f"Mean = {mean_val:.1f}")
    ax_hist.axvline(mean_val + std_val, color='orange', linestyle=':', lw=1.5)
    ax_hist.axvline(mean_val - std_val, color='orange', linestyle=':', lw=1.5)
    ax_hist.set_title("3. Distribution of Total Particle Count")
    ax_hist.set_xlabel("Total Particle Count")
    ax_hist.set_ylabel("Probability Density")
    ax_hist.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Plot saved to '{output_filename}'")

if __name__ == "__main__":
    generate_plots_ab("simulation_results_stripes.pkl", "plots_stripes.png")
