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
        with open("src\control_robots/calibration.json", 'r') as f:
            config = json.load(f)

        pen_orient = np.array(config["pen_orientation_rad"])
        img_w = config["image_processing"]["warped_image_width_px"]
        img_h = config["image_processing"]["warped_image_height_px"]

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
        with open("src/new_drawing_robots/Output_pictures/config/pairing.json", 'r') as f:
            data = json.load(f)

        # Ellenőrzés: a "pairings" kulcs legyen jelen
        if "pairings" not in data:
            LOG.error("pairing.json does not contain 'pairings' key!")
            return

        pairs = data["pairings"]

        # Sort by number_info.number
        try:
            pairs.sort(key=lambda item: item["number_info"]["number"])
        except KeyError:
            LOG.error("Error during sorting! Are you sure 'number_info.number' exists in all pairings?")
            return

        LOG.info(f"Successfully sorted {len(pairs)} points.")

        # Assemble Trajectory
        trajectory_6d = []
        for item in pairs:
            # Get pixel coordinates
            try:
                px_coord = item["dot_coordinates"]
                x_px = px_coord["x"]
                y_px = px_coord["y"]
            except KeyError as e:
                LOG.warning(f"Skipping point due to missing coordinates: {e}")
                continue

            # Calculate ratios (u, v)
            u = x_px / img_w
            v = y_px / img_h

            # Bilinear interpolation for 3D robot coordinates
            robot_xyz_interpolated = bilinear_interpolate(p_tl, p_tr, p_bl, p_br, u, v)

            robot_x = robot_xyz_interpolated[0]
            robot_y = robot_xyz_interpolated[1]

            # 6D Pose: [X_interp, Y_interp, Z_CALIBRATED] + [Rx, Ry, Rz]
            pose_6d = [robot_x, robot_y, drawing_z] + list(pen_orient)
            trajectory_6d.append(pose_6d)

        # Save Trajectory
        output_file = "trajectory.json"
        with open(output_file, 'w') as f:
            json.dump(trajectory_6d, f, indent=2)

        LOG.info(f"Conversion successful! Trajectory with {len(trajectory_6d)} points saved to: {output_file}")

    except FileNotFoundError as e:
        LOG.error(f"Error: Required file not found! {e}")
    except KeyError as e:
        LOG.error(f"Error: Missing key in 'calibration.json' or 'pairing.json': {e}")
    except Exception as e:
        LOG.error(f"An unknown error occurred: {e}")


if __name__ == "__main__":
    convert_coordinates_to_trajectory()
