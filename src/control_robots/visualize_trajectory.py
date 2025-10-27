#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trajectory Visualizer

Reads the 'trajectory.json' file (which contains 6D robot poses in mm/rad)
and generates a 2D plot of the (X, Y) path using Matplotlib.
The resulting image is saved as 'trajectory_visualization.png'.
"""

import json
import logging
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    logging.critical(f"Import Error: {e}. Please ensure 'matplotlib' and 'numpy' are installed.")
    logging.critical("Run: pip install matplotlib numpy")
    sys.exit(1)

LOG = logging.getLogger(__name__)


def visualize_trajectory():
    """
    Reads the trajectory and saves a 2D plot.
    """
    input_file = "trajectory.json"
    output_file = "trajectory_visualization.png"

    LOG.info(f"Starting trajectory visualization... Reading '{input_file}'")

    try:
        # 1. Load trajectory.json
        with open(input_file, 'r') as f:
            trajectory = json.load(f)

        if not trajectory:
            LOG.error(f"'{input_file}' is empty. Cannot visualize.")
            return

        # 2. Extract X and Y coordinates
        # Each 'pose' in the trajectory is [X, Y, Z, Rx, Ry, Rz]
        x_coords = [pose[0] for pose in trajectory]
        y_coords = [pose[1] for pose in trajectory]

        LOG.info(f"Extracted {len(x_coords)} (X, Y) points.")

        # 3. Create the plot
        plt.figure(figsize=(8, 11))  # Approximate A4 ratio

        # Plot the path (lines) and the points (markers)
        plt.plot(x_coords, y_coords, linestyle='-', marker='o', markersize=3, color='blue')

        # Plot the start point in green
        plt.plot(x_coords[0], y_coords[0], marker='o', markersize=8, color='green', label='Start')

        # Plot the end point in red
        plt.plot(x_coords[-1], y_coords[-1], marker='x', markersize=8, color='red', label='End')

        plt.title("Trajectory 2D Visualization (Robot Base Frame [mm])")
        plt.xlabel("X coordinate (mm)")
        plt.ylabel("Y coordinate (mm)")

        # --- Critical Steps for correct visualization ---

        # 1. Set equal aspect ratio, so 1mm on X axis = 1mm on Y axis
        plt.axis('equal')

        # 2. Invert Y-axis. In many robot/image systems, Y increases
        #    downwards. Matplotlib's default is upwards.
        plt.gca().invert_yaxis()

        # --- End Critical Steps ---

        plt.grid(True)
        plt.legend()

        # 4. Save the file
        plt.savefig(output_file)
        LOG.info(f"Visualization saved successfully to '{output_file}'")

    except FileNotFoundError:
        LOG.error(f"'{input_file}' not found. Run 'convert_to_trajectory.py' first!")
    except Exception as e:
        LOG.error(f"An error occurred during visualization: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    visualize_trajectory()