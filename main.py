# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import random

def draw_catan_terrain_map():
    # --- Parameters ---
    radius = 1.0  # hex side length
    hex_radius = radius

    # axial coordinates for the 7-hex (2–3–2) layout
    axial_coords = [(0, 0),
                    (1, 0), (1, -1), (0, -1),
                    (-1, 0), (-1, 1), (0, 1)]

    # convert axial to cartesian
    def axial_to_cart(q, r):
        x = hex_radius * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
        y = hex_radius * (1.5 * r)
        return (x, y)

    hex_centers = [axial_to_cart(q, r) for q, r in axial_coords]

    # --- Random terrain + number assignment ---
    terrain_types = {
        "Forest": "#2E8B57",
        "Field": "#F4E04D",
        "Pasture": "#9ACD32",
        "Hill": "#D2691E",
        "Mountain": "#A9A9A9",
        # "Desert": "#EEDD82"
    }

    terrain_list = random.choices(list(terrain_types.keys()), k=len(hex_centers))
    dice_numbers = random.sample([2, 3, 4, 5, 6, 8, 9, 10, 11, 12], len(hex_centers))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    for (hx, hy), terrain, number in zip(hex_centers, terrain_list, dice_numbers):
        color = terrain_types[terrain]
        hex_patch = RegularPolygon(
            (hx, hy),
            numVertices=6,
            radius=hex_radius,
            orientation=np.radians(0),
            facecolor=color,
            alpha=1,
            edgecolor='k'
        )
        ax.add_patch(hex_patch)
        # Center text: dice number
        ax.text(hx, hy, str(number), ha='center', va='center',
                fontsize=16, fontweight='bold', color='black')
        # Terrain label
        ax.text(hx, hy - 0.6, terrain, ha='center', va='center',
                fontsize=9, color='black', alpha=0.7)

    ax.scatter(
        [hx for hx, hy in hex_centers],
        [hy for hx, hy in hex_centers],
        c=[terrain_types[t] for t in terrain_list],
        s=40,
        alpha=0
    )
    plt.title("Quantum Catan Challenge — Random Terrain Map", fontsize=14)
    plt.show()

    return terrain_list, dice_numbers

# Run the generator
terrains, numbers = draw_catan_terrain_map()