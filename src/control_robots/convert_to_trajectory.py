#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import numpy as np

LOG = logging.getLogger(__name__)

def interpolate(p1, p2, factor):
    """Linear interpolation between two points."""
    return p1 * (1 - factor) + p2 * factor


def bilinear_interpolate(p_tl, p_tr, p_bl, p_br, u, v):
    """
    Bilinear interpolation between 4 points [X, Y, Z]
    u: horizontal ratio (0-1)
    v: vertical ratio (0-1)
    """
    p_top = interpolate(p_tl, p_tr, u)
    p_bottom = interpolate(p_bl, p_br, u)
    return interpolate(p_top, p_bottom, v)


def convert_coordinates_to_trajectory():
    """Main conversion function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        LOG.info("Loading calibration (calibration.json)...")
        with open("calibration.json", 'r') as f:
            config = json.load(f)

        pen_orient = np.array(config["pen_orientation_rad"])
        img_w = config["image_processing"]["warped_image_width_px"]
        img_h = config["image_processing"]["warped_image_height_px"]

        # Get the calibrated drawing height
        drawing_z = config["drawing_height_mm"]
        LOG.info(f"Calibrated drawing height (Z) set to: {drawing_z} mm")

        # Load corner points as numpy arrays
        p_tl = np.array(config["paper_corners_robot_mm"]["top_left"])
        p_tr = np.array(config["paper_corners_robot_mm"]["top_right"])
        p_br = np.array(config["paper_corners_robot_mm"]["bottom_right"])
        p_bl = np.array(config["paper_corners_robot_mm"]["bottom_left"])

        LOG.info(f"Image size: {img_w}x{img_h} px")
        LOG.info("Corner points (TL, TR, BR, BL) loaded successfully.")

        LOG.info("Loading point pairs (pairing.json)...")
        with open("../drawing_robots/Output_pictures/config/pairing.json", 'r') as f:
            pairs = json.load(f)

        try:
            pairs.sort(key=lambda item: item["num_value"])
        except KeyError:
            LOG.error("Error during sorting! Are you sure 'num_value' is the key in 'pairing.json'?")
            return

        LOG.info(f"Successfully sorted {len(pairs)} points.")

        #Assemble Trajectory
        trajectory_6d = []
        for item in pairs:
            # Get pixel coordinates
            px_coord = item["dot_coord"]
            x_px = px_coord["x"]
            y_px = px_coord["y"]

            # Calculate ratios (u, v)
            u = x_px / img_w
            v = y_px / img_h

            # Bilinear interpolation for 3D robot coordinates
            robot_xyz_interpolated = bilinear_interpolate(p_tl, p_tr, p_bl, p_br, u, v)

            # Use X and Y from interpolation
            robot_x = robot_xyz_interpolated[0]
            robot_y = robot_xyz_interpolated[1]

            # 6D Pose Assembly: [X_interp, Y_interp, Z_CALIBRATED] + [Rx, Ry, Rz]
            # Override Z with the calibrated 'drawing_height_mm'
            pose_6d = [robot_x, robot_y, drawing_z] + list(pen_orient)
            trajectory_6d.append(pose_6d)

        # 5. Save Trajectory
        output_file = "trajectory.json"
        with open(output_file, 'w') as f:
            json.dump(trajectory_6d, f, indent=2)

        LOG.info(f"Conversion successful! Trajectory with {len(trajectory_6d)} points saved to: {output_file}")

    except FileNotFoundError as e:
        LOG.error(f"Error: Required file not found! {e}")
    except KeyError as e:
        LOG.error(f"Error: Key not found! Missing key from 'calibration.json' or 'pairing.json': {e}")
    except Exception as e:
        LOG.error(f"An unknown error occurred: {e}")


if __name__ == "__main__":
    convert_coordinates_to_trajectory()