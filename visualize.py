import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
import pickle
import sys
import os
from itertools import product

class LatticePlot:
    """
    Visualizer for the KMC simulation.
    Displays particles 'A' and 'B' at their current positions.
    Now reads the 'site_map' to draw any background (stripes or random).
    """
    def __init__(self, data):
        # Load all data from the results dictionary
        self.lattice_size = data['lattice_size']
        self.lattice_mode = data['lattice_mode']
        self.site_map = data['site_map'] # The n x n map of site types
        self.num_site_types = data['num_site_types']
        self.snapshots = data['snapshots']
        self.time_stamps = data['time_stamps']
        self.site_type_rates = data.get('params', {}).get('site_type_hop_rates', [])
        
        # Particle colors
        self.particle_colors = {
            'A': '#0072B2',  # Blue
            'B': '#D55E00'   # Vermillion/Red
        }
        # Background colors
        self.site_type_colors = ['#d6e8ee', '#f5dcdc'] # Light blue, light red
        self.site_type_names = ['Blue', 'Red']

        # Setup the plot
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle(f"Interactive KMC Simulation (Mode: {self.lattice_mode})", fontsize=16, weight='bold')
        plt.subplots_adjust(bottom=0.2, top=0.85)
        
        self._configure_axes()
        self._draw_background() # Use the new unified background drawer
        
        self.particle_texts = []
        
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

    def _draw_background(self):
        """
        Draws the background for EITHER 'stripes' or 'random' mode
        by reading the self.site_map.
        """
        # 1. Draw the background colors using imshow (fast and works for both modes)
        cmap = ListedColormap(self.site_type_colors[:self.num_site_types])
        
        # Use origin='lower' to match plt.plot's coordinate system (0,0 at bottom-left)
        self.ax.imshow(self.site_map, cmap=cmap, interpolation='nearest', alpha=0.5, 
                       zorder=0, origin='lower',
                       extent=(-0.5, self.lattice_size - 0.5, -0.5, self.lattice_size - 0.5))

        # 2. If we are in 'stripes' mode, add the text labels
        if self.lattice_mode == 'stripes':
            stripe_width = self.lattice_size / self.num_site_types
            for i in range(self.num_site_types):
                rate = self.site_type_rates[i] if i < len(self.site_type_rates) else 'N/A'
                name = self.site_type_names[i % len(self.site_type_names)]
                color = self.site_type_colors[i % len(self.site_type_colors)]
                
                text = f"Stripe {i} ({name})\nHop Rate: {rate}"
                x_start = i * stripe_width
                
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

        # 1. Clear the old particles
        for txt in self.particle_texts:
            txt.remove()
        self.particle_texts.clear()

        # 2. Draw new particles
        for (x, y), p_type in current_snapshot.items():
            p_color = self.particle_colors.get(p_type, 'black')
            
            txt = self.ax.text(x, y, str(p_type), ha='center', va='center',
                               color='white', fontsize=9, weight='bold', zorder=3,
                               bbox=dict(boxstyle='circle,pad=0.2', 
                                         fc=p_color, ec='none', alpha=0.9))
            self.particle_texts.append(txt)
        
        # 3. Update info text
        self.info_text.set_text(f"Time: {time:.2f} s\nParticles: {len(current_snapshot)}")
        self.fig.canvas.draw_idle()

    def show(self):
        self.update(0)  # Initialize plot at t=0
        plt.show()

def plot_particle_count(data):
    """
    Generates two separate plots:
    1. A plot of the total, A, and B particle counts vs. time.
    2. A second plot with A/B counts for each *site type* (works for stripes and random).
    """
    # --- Extract data from the dictionary ---
    snapshots = data['snapshots']
    time_stamps = data['time_stamps']
    site_map = data['site_map'] # The n x n map
    lattice_mode = data['lattice_mode']
    lattice_size = data['lattice_size']
    num_site_types = data['num_site_types']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Pre-calculate A/B counts for all plots ---
    total_counts = []
    a_counts = []
    b_counts = []
    
    # Counts per site type: list of lists, one for each type (0, 1, ...)
    site_type_counts_A = [[] for _ in range(num_site_types)]
    site_type_counts_B = [[] for _ in range(num_site_types)]

    for snap in snapshots:
        total_counts.append(len(snap))
        a_count_snap, b_count_snap = 0, 0
        
        # Temp storage for this snapshot's counts
        counts_A_in_snapshot = [0] * num_site_types
        counts_B_in_snapshot = [0] * num_site_types

        # Loop over particles in the snapshot
        for (x, y), p_type in snap.items():
            # Tally for Plot 1 (Total)
            if p_type == 'A':
                a_count_snap += 1
            elif p_type == 'B':
                b_count_snap += 1
            
            # --- THIS IS THE KEY CHANGE ---
            # Get the site type (0 or 1) by looking it up in the map
            site_type = site_map[x, y]
            
            # Tally for Plot 2 (Per Site Type)
            if p_type == 'A':
                counts_A_in_snapshot[site_type] += 1
            elif p_type == 'B':
                counts_B_in_snapshot[site_type] += 1
        
        a_counts.append(a_count_snap)
        b_counts.append(b_count_snap)
        
        # Add this snapshot's counts to the time-series lists
        for i in range(num_site_types):
            site_type_counts_A[i].append(counts_A_in_snapshot[i])
            site_type_counts_B[i].append(counts_B_in_snapshot[i])


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

    # --- Plot 2: Occupancy per Site Type (A vs B) ---
    if num_site_types > 1:
        fig2, axes2 = plt.subplots(
            nrows=num_site_types, 
            ncols=1, 
            figsize=(10, 3 * num_site_types), 
            sharex=True
        )
        if num_site_types == 1: # Handle case of 1 type
             axes2 = [axes2] 
             
        fig2.suptitle(f"Particle Occupancy per Site Type (Mode: {lattice_mode})", fontsize=16, y=0.96)
        
        site_type_names = ['Blue', 'Red'] # As defined by user

        for i in range(num_site_types):
            ax = axes2[i]
            name = site_type_names[i % len(site_type_names)]
            
            counts_A = site_type_counts_A[i]
            counts_B = site_type_counts_B[i]
            
            ax.plot(time_stamps, counts_A, lw=2, label=f"Count A", color="#0072B2")
            ax.plot(time_stamps, counts_B, lw=2, label=f"Count B", color="#D55E00")
            
            ax.set_ylabel("Particle Count")
            ax.set_title(f"Site Type {i} ('{name}') Occupancy", fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        axes2[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


if __name__ == "__main__":
    
    # --- Smart File Loading ---
    # List of possible filenames
    INPUT_FILENAMES = ["simulation_results_stripes.pkl", "simulation_results_random.pkl"]
    
    loaded_file = None
    for fname in INPUT_FILENAMES:
        if os.path.exists(fname):
            loaded_file = fname
            break # Found a file, stop searching

    if loaded_file is None:
        print(f"❌ Error: No simulation results file found.")
        print("   Please run 'python run_simulation_fast.py' first.")
        print(f"   (Was looking for {INPUT_FILENAMES[0]} or {INPUT_FILENAMES[1]})")
        sys.exit(1)
        
    print(f"✅ Data loaded from '{loaded_file}'. Launching visualizer...")
    with open(loaded_file, 'rb') as f:
        simulation_data = pickle.load(f)

    # Launch the interactive plot
    plot = LatticePlot(simulation_data)
    
    # Launch the particle count plots
    plot_particle_count(simulation_data)

    # Show the interactive plot last (it blocks execution)
    plot.show()