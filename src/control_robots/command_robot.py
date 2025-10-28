#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import time
import sys
import json
import logging
from Dashboard import Dashboard
from rtdeState import RtdeState
import numpy as np

CONFIG = {}
LOG = logging.getLogger(__name__)


def load_configuration(config_file="calibration.json"):
    """Loads configuration from JSON file into global CONFIG dict."""
    global CONFIG
    try:
        with open(config_file, 'r') as f:
            CONFIG = json.load(f)
        LOG.info(f"Configuration loaded successfully from: {config_file}")
        return True
    except FileNotFoundError:
        LOG.error(f"Configuration file not found: {config_file}")
        return False
    except Exception as e:
        LOG.error(f"Error loading configuration: {e}")
        return False


class URScriptClient:
    """Client for Secondary Interface (30002) to send URScript commands."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        """Connects to the Secondary Interface (30002)."""
        LOG.info(f"Connecting (Secondary): {self.host}:{self.port}...")
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.host, self.port))
            self.connected = True
            LOG.info("Connected (Secondary: 30002)!")
            return True
        except Exception as e:
            LOG.error(f"Connection error (Secondary): {e}")
            return False

    def send_script(self, script):
        """Sends a URScript command"""
        if not self.connected:
            LOG.error("Not connected (Secondary).")
            return False
        try:
            if not script.endswith('\n'):
                script += '\n'
            self.socket.sendall(script.encode('utf-8'))
            LOG.debug(f"Script sent: {script.strip()}")
            return True
        except Exception as e:
            LOG.error(f"Script send error: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        """Disconnects from the Secondary Interface."""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.connected = False
            LOG.info("Disconnected (Secondary).")


def mm_to_m(position_mm):
    """Converts [X_mm, Y_mm, Z_mm, Rx, Ry, Rz] -> [X_m, Y_m, Z_m, Rx, Ry, Rz]"""
    position_m = list(position_mm)
    for i in range(3):
        position_m[i] = position_mm[i] / 1000.0
    return position_m


def move_to_position_mm(client, rtde_state, pose_mm_rad, speed=None, accel=None, interpolate_steps=0,
                        wait_for_enter=False):
    """
    Sends a 'movel' command with a pose in mm/rad.
    If interpolate_steps > 1, it gets the current pose from RTDE
    and sends 'interpolate_steps' blended movel commands.
    If wait_for_enter=True, it waits for user input before each step.
    """
    speed = speed if speed is not None else CONFIG.get("default_speed_ms", 0.1)
    accel = accel if accel is not None else CONFIG.get("default_accel_mss", 0.5)

    if interpolate_steps <= 1:
        # Original behavior: one single move
        pose_m_rad = mm_to_m(pose_mm_rad)
        pos_str = "p[" + ",".join([f"{pos:.6f}" for pos in pose_m_rad]) + "]"
        script = f"movel({pos_str}, a={accel}, v={speed})\n"
        LOG.info(f"Sending single move command: {script.strip()}")
        return client.send_script(script)

    # --- Interpolated move logic ---
    try:
        # 1. Get Target Pose (in meters)
        target_pose_m = mm_to_m(pose_mm_rad)

        # 2. Get Current Pose (in meters) from RTDE
        state = rtde_state.receive()
        if state is None or not hasattr(state, 'actual_TCP_pose'):
            LOG.error("Cannot interpolate: Failed to get current TCP from RTDE.")
            return False

        start_pose_m = list(state.actual_TCP_pose)
        LOG.info(f"Interpolating move from {start_pose_m} to {target_pose_m} in {interpolate_steps} steps.")

        # 3. Generate intermediate poses using numpy.linspace
        num_points = interpolate_steps + 1
        waypoints_m = []
        for i in range(6):  # X, Y, Z, Rx, Ry, Rz tengelyekre
            interp_axis = np.linspace(start_pose_m[i], target_pose_m[i], num=num_points)
            waypoints_m.append(interp_axis)

        # 4. Send all intermediate 'movel' commands
        all_ok = True
        for i in range(1, num_points):  # Start from 1 (first interpolated point)
            pose_step_m = [waypoints_m[j][i] for j in range(6)]

            # --- ÚJ RÉSZ: VÁRAKOZÁS AZ ENTERRE ---
            LOG.info(f"Preparing interpolated step {i}/{interpolate_steps}...")
            if wait_for_enter:
                try:
                    input(f"  ==> Press ENTER to execute step {i}/{interpolate_steps}...")
                except KeyboardInterrupt:
                    LOG.warning("Move sequence cancelled by user.")
                    return False
            # --- ÚJ RÉSZ VÉGE ---

            pos_str = "p[" + ",".join([f"{pos:.6f}" for pos in pose_step_m]) + "]"
            script = f"movel({pos_str}, a={accel}, v={speed})\n"

            LOG.debug(f"Sending interpolated move ({i}/{interpolate_steps}): {script.strip()}")
            if not client.send_script(script):
                LOG.error(f"Failed at interpolated step {i}.")
                all_ok = False
                break

        return all_ok

    except Exception as e:
        LOG.error(f"Error during move interpolation: {e}")
        return False


def check_dashboard_status():
    """Queries basic status from the Dashboard server (Dashboard.py)."""
    LOG.info("Dashboard Status (29999) Check")
    try:
        dash = Dashboard(CONFIG["robot_ip"])
        dash.connect()
        LOG.info("Dashboard Status")
        LOG.info(f"  Robot Mode: {dash.sendAndReceive('robotmode')}")
        LOG.info(f"  Program State: {dash.sendAndReceive('programstate')}")
        LOG.info(f"  Safety Status: {dash.sendAndReceive('safetystatus')}")
        dash.close()
    except Exception as e:
        LOG.error(f"Dashboard error: {e}")


def check_rtde_status():
    """Queries detailed data (TCP, Joints) via RTDE (rtdeState.py)."""
    LOG.info("RTDE Status (30004) Check")
    state_monitor = None
    try:
        state_monitor = RtdeState(
            CONFIG["robot_ip"],
            CONFIG["rtde_config_file"]
        )
        state_monitor.initialize()
        state = state_monitor.receive()

        if state is not None:
            LOG.info("RTDE Status")
            if hasattr(state, 'actual_TCP_pose'):
                tcp_m_rad = [f"{p:.3f}" for p in state.actual_TCP_pose]
                LOG.info(f"  Actual TCP (m/rad): {tcp_m_rad}")
            if hasattr(state, 'actual_q'):
                joints_rad = [f"{q:.3f}" for q in state.actual_q]
                LOG.info(f"  Actual Joints (rad): {joints_rad}")
        else:
            LOG.warning("No data received from RTDE.")
    except Exception as e:
        LOG.error(f"RTDE error: {e}")
    finally:
        if state_monitor:
            state_monitor.con.send_pause()
            state_monitor.con.disconnect()


def execute_drawing(client):
    """Loads 'trajectory.json' and executes the drawing sequence."""
    LOG.info("Starting Drawing Sequence")

    try:
        with open("trajectory.json", 'r') as f:
            trajectory = json.load(f)
        if not trajectory:
            LOG.error("'trajectory.json' is empty. Run 'convert_to_trajectory.py' first!")
            return
    except FileNotFoundError:
        LOG.error("'trajectory.json' not found. Run 'convert_to_trajectory.py' first!")
        return

    # Load calibration data
    safe_z_offset = CONFIG["safe_travel_height_offset_mm"]
    home_pos = CONFIG["home_position_mm"]

    wait_long = CONFIG.get("move_wait_time_s", 3)
    wait_short = CONFIG.get("drawing_move_wait_time_s", 0.5)

    try:

        # 1. Move ABOVE start point (Safe Z)
        # 'trajectory.json' already contains 6D poses [X,Y,Z,Rx,Ry,Rz]
        first_pose_draw = trajectory[0]

        # Calculate safe pose (with Z offset)
        start_pose_safe = list(first_pose_draw)
        start_pose_safe[2] += safe_z_offset  # Increase Z (less negative)

        LOG.info("Moving above start point (Pen Up)...")
        input("Press ENTER to move above start point...")
        if not move_to_position_mm(client, start_pose_safe):
            raise Exception("Failed to move above start point.")
        time.sleep(wait_long)

        # 2. Lower pen to drawing Z
        LOG.info("Lowering pen to drawing height...")
        input("ATTENTION: Lowering pen! Press ENTER to continue...")
        if not move_to_position_mm(client, first_pose_draw):
            raise Exception("Failed to lower pen.")
        time.sleep(wait_long)

        # 3. Execute drawing (the rest of the trajectory)
        LOG.info(f"Starting draw... {len(trajectory) - 1} points remaining.")
        input("Press ENTER to begin drawing...")

        # We are already at the first point, so start from index 1
        for i, pose in enumerate(trajectory[1:]):
            LOG.info(f"Drawing point: {i + 1} / {len(trajectory) - 1}...")
            if not move_to_position_mm(client, pose):
                raise Exception(f"Failed at point {i + 1}.")
            time.sleep(wait_short)  # Short wait between drawing points

        # 4. Lift pen (Safe Z)
        last_pose_draw = trajectory[-1]

        # Calculate safe end pose (with Z offset)
        end_pose_safe = list(last_pose_draw)
        end_pose_safe[2] += safe_z_offset  # Increase Z

        LOG.info("Drawing complete. Lifting pen...")
        if not move_to_position_mm(client, end_pose_safe):
            raise Exception("Failed to lift pen.")
        time.sleep(wait_long)

        # 5. Return to Home
        LOG.info("Returning to Home position...")
        if not move_to_position_mm(client, home_pos):
            raise Exception("Failed to return home.")
        time.sleep(wait_long)

        LOG.info("--- Drawing Sequence Finished Successfully ---")

    except Exception as e:
        LOG.error(f"Error during drawing sequence: {e}")
        LOG.warning("Attempting emergency return to Home position...")
        try:
            # Emergency fallback: try to lift and go home
            safe_home = list(home_pos)
            safe_home[2] += safe_z_offset  # Go to home, but higher
            move_to_position_mm(client, safe_home)
        except:
            LOG.error("Emergency home return FAILED.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("robot_controller.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    if not load_configuration("calibration.json"):
        LOG.critical("Critical Error: Failed to load configuration.")
        sys.exit(1)

    # 1. Connect to Secondary port (30002)
    client = URScriptClient(CONFIG["robot_ip"], CONFIG["secondary_port"])
    if not client.connect():
        LOG.critical("Critical Error: Failed to connect to Secondary port (30002).")
        sys.exit(1)

    # 2. Connect to RTDE (30004) and keep it alive
    state_monitor = None
    try:
        state_monitor = RtdeState(
            CONFIG["robot_ip"],
            CONFIG["rtde_config_file"]
        )
        state_monitor.initialize()
        LOG.info("RTDE Monitor connected and initialized (30004).")
    except Exception as e:
        LOG.critical(f"Critical Error: Failed to connect RTDE: {e}")
        client.disconnect()
        sys.exit(1)

    # --- Main Loop ---
    try:
        while True:
            print("\n" + "=" * 30)
            print("      UR Robot Controller CLI")
            print("=" * 30)
            print("1. Get Full Status (Dashboard + RTDE)")
            print("2. Start Drawing (from trajectory.json)")
            print("3. Go Home (Fast Interpolated, 10 steps)")
            print("4. Safe Home Move (10 steps, manual confirm)")  # ÚJ MENÜPONT
            print("0. Exit")
            print("=" * 30)
            choice = input("Select (0-4): ")  # Frissítve 0-4 -re

            if choice == '1':
                # ... (Ez a rész változatlan) ...
                check_dashboard_status()
                LOG.info("RTDE Status (30004) Check [Live]")
                try:
                    state = state_monitor.receive()
                    if state is not None:
                        LOG.info("RTDE Status")
                        if hasattr(state, 'actual_TCP_pose'):
                            tcp_m_rad = [f"{p:.3f}" for p in state.actual_TCP_pose]
                            LOG.info(f"  Actual TCP (m/rad): {tcp_m_rad}")
                        if hasattr(state, 'actual_q'):
                            joints_rad = [f"{q:.3f}" for q in state.actual_q]
                            LOG.info(f"  Actual Joints (rad): {joints_rad}")
                    else:
                        LOG.warning("No data received from RTDE.")
                except Exception as e:
                    LOG.error(f"RTDE error: {e}")

            elif choice == '2':
                # Pass both client and rtde_state
                execute_drawing(client, state_monitor)

            elif choice == '3':
                LOG.info("Moving to Home position (fast, interpolated)")
                # A 'wait_for_enter' itt alapértelmezetten False,
                # így ez a mozgás gyors lesz, megállás nélkül.
                move_to_position_mm(
                    client,
                    state_monitor,
                    CONFIG["home_position_mm"],
                    interpolate_steps=10
                    # wait_for_enter=False (ez az alapértelmezett)
                )
                time.sleep(CONFIG.get("move_wait_time_s", 3))

            # --- ÚJ MENÜPONT KEZELÉSE ---
            elif choice == '4':
                LOG.info("Starting SAFE Home move (10 steps, manual confirm)")
                # Itt a 'wait_for_enter'-t True-ra állítjuk!
                move_to_position_mm(
                    client,
                    state_monitor,
                    CONFIG["home_position_mm"],
                    interpolate_steps=10,
                    wait_for_enter=True  # A lényegi különbség!
                )
                LOG.info("Safe Home move complete.")
                time.sleep(1)  # Rövid várakozás a végén
            # --- ÚJ MENÜPONT VÉGE ---

            elif choice == '0':
                LOG.info("Exiting")
                break

            else:
                LOG.warning("Invalid choice")

    except KeyboardInterrupt:
        LOG.warning("Program interrupted by user")
    finally:
        # Disconnect *both* clients
        if state_monitor:
            state_monitor.con.send_pause()
            state_monitor.con.disconnect()
            LOG.info("RTDE Monitor disconnected.")

        client.disconnect()
        LOG.info("Program finished.")


if __name__ == "__main__":
    main()