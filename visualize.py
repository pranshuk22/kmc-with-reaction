# visualize.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
import sys
import os

class LatticePlot:
    """
    Visualizer for the KMC simulation.
    Displays particles 'A' and 'B' at their current positions.
    """
    def __init__(self, data):
        # Load all data from the results dictionary
        self.lattice_size = data['lattice_size']
        self.num_stripes = data['num_stripes']
        self.snapshots = data['snapshots']
        self.time_stamps = data['time_stamps']
        # Get hop rates from the new 'params' sub-dictionary
        self.stripe_rates = data.get('params', {}).get('hop_rates', [])
        
        # Particle colors
        self.particle_colors = {
            'A': '#0072B2',  # Blue
            'B': '#D55E00'   # Vermillion/Red
        }

        # Setup the plot
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle("Interactive KMC Simulation (A/B Particles)", fontsize=16, weight='bold')
        plt.subplots_adjust(bottom=0.2, top=0.85)
        
        self._configure_axes()
        self._draw_background_stripes()
        
        # This list will hold the text artists
        self.particle_texts = []
        
        # Info box for time and particle count
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      fontsize=12, family='monospace', va='top', ha='left',
                                      bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8))
        
        self._create_slider()

    def _configure_axes(self):
        """Set up titles, limits, and grid for the main plot axis."""
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        self.ax.set_xlim(-0.5, self.lattice_size - 0.5)
        self.ax.set_ylim(-0.5, self.lattice_size - 0.5)
        self.ax.set_aspect('equal')

    def _draw_background_stripes(self):
        """Draw colored vertical rectangles to indicate stripe regions."""
        stripe_width = self.lattice_size / self.num_stripes
        # Colors match user's "blue" and "red" concept
        stripe_colors = ['#d6e8ee', '#f5dcdc']  # Light blue and light red
        stripe_names = ['Blue', 'Red']

        for i in range(self.num_stripes):
            rate = self.stripe_rates[i] if i < len(self.stripe_rates) else 'N/A'
            name = stripe_names[i % len(stripe_names)]
            color = stripe_colors[i % len(stripe_colors)]
            
            text = f"Stripe {i} ({name})\nHop Rate: {rate}"
            x_start = i * stripe_width
            
            self.ax.add_patch(
                plt.Rectangle((x_start - 0.5, -0.5), stripe_width, self.lattice_size,
                              color=color, alpha=0.5, zorder=0)
            )
            self.ax.text(x_start + stripe_width/2 - 0.5, self.lattice_size + 0.5,
                         text, ha='center', va='bottom', fontsize=9)
            if i > 0:
                self.ax.axvline(x=x_start - 0.5, color='gray', linestyle='--')

    def _create_slider(self):
        """Create and configure the time slider widget."""
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(
            ax=ax_slider,
            label='Time (s)',
            valmin=0.0,
            valmax=self.time_stamps[-1],
            valinit=0.0,
            valfmt='%.2f s'
        )
        self.slider.on_changed(self.update)

    def update(self, val):
        """Function called when the slider value changes."""
        time = self.slider.val
        step_idx = np.searchsorted(self.time_stamps, time, side="right") - 1
        current_snapshot = self.snapshots[step_idx]

        # 1. Clear the old particles from the plot
        for txt in self.particle_texts:
            txt.remove()
        self.particle_texts.clear()

        # 2. Draw new particles for the current frame
        # The snapshot format is now {(x, y): 'A', (x2, y2): 'B', ...}
        for (x, y), p_type in current_snapshot.items():
            p_color = self.particle_colors.get(p_type, 'black')
            
            # Display 'A' or 'B'
            txt = self.ax.text(x, y, str(p_type), ha='center', va='center',
                               color='white', fontsize=9, weight='bold', zorder=3,
                               bbox=dict(boxstyle='circle,pad=0.2', 
                                         fc=p_color, ec='none', alpha=0.9))
            self.particle_texts.append(txt)
        
        # 3. Update info text box
        self.info_text.set_text(f"Time: {time:.2f} s\nParticles: {len(current_snapshot)}")
        
        self.fig.canvas.draw_idle()

    def show(self):
        self.update(0)  # Initialize plot at t=0
        plt.show()

def plot_particle_count(data):
    """
    Generates two separate plots:
    1. A plot of the total, A, and B particle counts vs. time.
    2. If num_stripes > 1, a second plot with A/B counts for each stripe.
    """
    # --- Extract data from the dictionary ---
    snapshots = data['snapshots']
    time_stamps = data['time_stamps']
    num_stripes = data['num_stripes']
    lattice_size = data['lattice_size']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Pre-calculate A/B counts for all plots ---
    total_counts = []
    a_counts = []
    b_counts = []
    
    # Stripe counts: list of lists, one for each stripe
    stripe_counts_A = [[] for _ in range(num_stripes)]
    stripe_counts_B = [[] for _ in range(num_stripes)]
    stripe_width = lattice_size // num_stripes

    for snap in snapshots:
        # For Plot 1
        total_counts.append(len(snap))
        a_count_snap = 0
        b_count_snap = 0
        
        # For Plot 2
        counts_A_in_snapshot = [0] * num_stripes
        counts_B_in_snapshot = [0] * num_stripes

        # Loop over particles in the snapshot
        # Format is {(x, y): 'A', ...}
        for (x, y), p_type in snap.items():
            # Tally for Plot 1
            if p_type == 'A':
                a_count_snap += 1
            elif p_type == 'B':
                b_count_snap += 1
            
            # Tally for Plot 2
            if num_stripes > 1:
                stripe_index = min(x // stripe_width, num_stripes - 1)
                if p_type == 'A':
                    counts_A_in_snapshot[stripe_index] += 1
                elif p_type == 'B':
                    counts_B_in_snapshot[stripe_index] += 1
        
        a_counts.append(a_count_snap)
        b_counts.append(b_count_snap)
        
        if num_stripes > 1:
            for i in range(num_stripes):
                stripe_counts_A[i].append(counts_A_in_snapshot[i])
                stripe_counts_B[i].append(counts_B_in_snapshot[i])


    # --- Plot 1: Total System Occupancy (A vs B) ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(time_stamps, total_counts, lw=2.5, color="black", label="Total Count", alpha=0.8)
    ax1.plot(time_stamps, a_counts, lw=2, color="#0072B2", label="Count A")
    ax1.plot(time_stamps, b_counts, lw=2, color="#D55E00", label="Count B")
    
    ax1.set_title("Total System Occupancy (A vs B) Over Time", fontsize=16)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Number of Particles")
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Occupancy per Stripe (A vs B) ---
    if num_stripes > 1:
        fig2, axes2 = plt.subplots(
            nrows=num_stripes, 
            ncols=1, 
            figsize=(10, 3 * num_stripes), 
            sharex=True
        )
        fig2.suptitle("Particle Occupancy per Stripe (A vs B)", fontsize=16, y=0.96)
        
        stripe_names = ['Blue', 'Red'] # As defined by user

        for i in range(num_stripes):
            ax = axes2 if num_stripes == 1 else axes2[i]
            name = stripe_names[i % len(stripe_names)]
            
            counts_A = stripe_counts_A[i]
            counts_B = stripe_counts_B[i]
            
            ax.plot(time_stamps, counts_A, lw=2, label=f"Count A", color="#0072B2")
            ax.plot(time_stamps, counts_B, lw=2, label=f"Count B", color="#D55E00")
            
            ax.set_ylabel("Particle Count")
            ax.set_title(f"Stripe {i} ('{name}') Occupancy", fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        (axes2 if num_stripes == 1 else axes2[-1]).set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


if __name__ == "__main__":
    # Updated filename
    INPUT_FILENAME = "simulation_results_AB.pkl"
    
    if not os.path.exists(INPUT_FILENAME):
        print(f"❌ Error: Results file '{INPUT_FILENAME}' not found.")
        # Updated script name
        print("➡️ Please run 'python run_simulation_fast.py' first to generate the data.")
        sys.exit(1)
        
    with open(INPUT_FILENAME, 'rb') as f:
        simulation_data = pickle.load(f)
    print("✅ Data loaded successfully. Launching visualizer...")

    # Launch the interactive plot
    # (Uncommented to run)
    plot = LatticePlot(simulation_data)
    
    # Launch the particle count plots
    # plot_particle_count(simulation_data)

    # Show the interactive plot last
    plot.show()
