"""
webgl_visualize.py
WebGL-backed lattice visualizer using Plotly (Scattergl).
Produces an interactive HTML file with slider + play controls.

Usage:
    python webgl_visualize.py <input_pickle.pkl> [output.html]

Dependencies:
    pip install plotly pillow numpy
"""

import sys
import pickle
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import base64

# Color maps (site types and particle types)
SITE_TYPE_COLORS = [
    (214, 232, 238),  # light blue-ish (type 0)
    (245, 220, 220),  # light red-ish  (type 1)
    # add more if you have >2 site types
]
PARTICLE_COLOR_MAP = {'A': 'rgb(0,114,178)', 'B': 'rgb(213,94,0)'}  # A-blue, B-red

def site_map_to_data_uri(site_map, site_type_colors=SITE_TYPE_COLORS, scale=20):
    """
    Convert integer site_map (n x n) into a PNG data URI with colored tiles.
    scale: upscale factor for nicer display (each lattice cell becomes scale x scale px)
    """
    n_y, n_x = site_map.shape
    img = Image.new('RGB', (n_x * scale, n_y * scale), color=(255,255,255))
    px = img.load()
    for yy in range(n_y):
        for xx in range(n_x):
            t = int(site_map[yy, xx])
            color = site_type_colors[t % len(site_type_colors)]
            # fill scale x scale block
            for sy in range(scale):
                for sx in range(scale):
                    px[xx*scale + sx, (n_y-1-yy)*scale + sy] = color
                    # note we flip y so that origin=(0,0) is bottom-left in plot
    buf = BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return "data:image/png;base64," + b64

def build_frames_from_snapshots(snapshots, time_stamps, lattice_size):
    """
    Build list of frame data dictionaries for Plotly animation.
    Each frame holds x,y, marker colors, and hover text.
    """
    frames = []
    max_particles = max(len(s) for s in snapshots) if snapshots else 0

    for snap, t in zip(snapshots, time_stamps):
        if len(snap) == 0:
            xs = []
            ys = []
            colors = []
            texts = []
        else:
            xs = [coord[0] + 0.0 for coord in snap.keys()]  # x
            ys = [coord[1] + 0.0 for coord in snap.keys()]  # y
            colors = [PARTICLE_COLOR_MAP.get(p, 'black') for p in snap.values()]
            texts = [f"({x},{y})<br>type: {p}" for (x,y), p in snap.items()]

        # Create marker dict with lengths padded to max_particles for consistent shape
        frames.append(go.Frame(
            data=[
                # trace 0: particle scatter (will be scattergl in figure)
                go.Scattergl(x=xs, y=ys, mode='markers',
                             marker=dict(color=colors, size=12, line=dict(width=0)),
                             hoverinfo='text', text=texts)
            ],
            name=f"{t:.6f}",
            layout=go.Layout(annotations=[dict(
                text=f"Time: {t:.3f} s — Particles: {len(snap)}",
                x=0.01, y=0.99, showarrow=False, xanchor='left', yanchor='top',
                font=dict(size=12, family='monospace')
            )])
        ))
    return frames

def create_figure(site_map, snapshots, time_stamps, output_html='lattice_visualization.html'):
    n_y, n_x = site_map.shape
    # make static background image data URI
    data_uri = site_map_to_data_uri(site_map, scale=20)

    # Build initial particle trace from snapshot 0 (or empty)
    init_snap = snapshots[0] if snapshots else {}
    xs0 = [coord[0] for coord in init_snap.keys()]
    ys0 = [coord[1] for coord in init_snap.keys()]
    colors0 = [PARTICLE_COLOR_MAP.get(p, 'black') for p in init_snap.values()]
    texts0 = [f"({x},{y})<br>type: {p}" for (x,y), p in init_snap.items()]

    particle_trace = go.Scattergl(
        x=xs0, y=ys0,
        mode='markers',
        marker=dict(color=colors0, size=12, line=dict(width=0)),
        hoverinfo='text',
        text=texts0,
        name='Particles'
    )

    # Build frames (only contain updated particle trace)
    frames = build_frames_from_snapshots(snapshots, time_stamps, lattice_size=max(n_x, n_y))

    # Layout with static background image
    layout = go.Layout(
        title=f"WebGL Lattice Viewer (size={n_x}x{n_y})",
        xaxis=dict(range=[-0.5, n_x - 0.5], scaleratio=1, constrain='domain', showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.5, n_y - 0.5], scaleanchor='x', showgrid=False, zeroline=False),
        width=800, height=800,
        images=[dict(
            source=data_uri,
            xref="x", yref="y",
            x=-0.5, y=n_y - 0.5,  # place top-left at ( -0.5, n_y-0.5 )
            sizex=n_x, sizey=n_y,
            sizing="stretch",
            opacity=0.8,
            layer="below"
        )],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.05,
            x=1.02,
            xanchor="right",
            yanchor="top",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, mode='immediate')]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
            ]
        )],
        sliders=[dict(
            active=0,
            pad={"t": 50},
            steps=[dict(method='animate',
                        args=[[f.name], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                        label=f"{i}: {time_stamps[i]:.2f}s")
                   for i, f in enumerate(frames)]
        )]
    )

    fig = go.Figure(data=[particle_trace], layout=layout, frames=frames)

    # Save to HTML
    fig.write_html(output_html, include_plotlyjs='cdn', auto_open=False)
    print(f"Saved interactive HTML visualization to '{output_html}'")

def load_simulation_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python webgl_visualize.py <input.pkl> [output.html]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'lattice_visualization.html'
    data = load_simulation_pickle(inp)

    # Expect site_map shape to be (n,n), snapshots list of dicts mapping (x,y)->'A'|'B'
    site_map = np.array(data['site_map'])
    snapshots = data['snapshots']
    time_stamps = np.array(data['time_stamps'])

    create_figure(site_map, snapshots, time_stamps, output_html=out)
