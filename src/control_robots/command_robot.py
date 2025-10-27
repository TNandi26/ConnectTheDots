#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import time
import sys
import json
import logging
from Dashboard import Dashboard
from rtdeState import RtdeState

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


def move_to_position_mm(client, pose_mm_rad, speed=None, accel=None):
    """
    Sends a 'movel' command with a pose in mm/rad.
    Automatically converts mm to meters for the command.
    """
    pose_m_rad = mm_to_m(pose_mm_rad)
    pos_str = "p[" + ",".join([f"{pos:.6f}" for pos in pose_m_rad]) + "]"

    speed = speed if speed is not None else CONFIG.get("default_speed_ms", 0.1)
    accel = accel if accel is not None else CONFIG.get("default_accel_mss", 0.5)

    script = f"movel({pos_str}, a={accel}, v={speed})\n"

    LOG.info(f"Sending move command: {script.strip()}")
    return client.send_script(script)


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

    # Connect to the movement port (keep connection open)
    client = URScriptClient(CONFIG["robot_ip"], CONFIG["secondary_port"])
    if not client.connect():
        LOG.critical("Critical Error: Failed to connect to Secondary port (30002).")
        sys.exit(1)

    try:
        while True:
            print("UR Robot Controller CLI")
            print("1. Get Full Status (Dashboard + RTDE)")
            print("2. Start Drawing (from trajectory.json)")
            print("3. Return to Home Position")
            print("0. Exit")
            choice = input("Select (0-3): ")

            if choice == '1':
                check_dashboard_status()
                check_rtde_status()

            elif choice == '2':
                execute_drawing(client)

            elif choice == '3':
                LOG.info("Moving to Home position")
                move_to_position_mm(client, CONFIG["home_position_mm"])
                time.sleep(CONFIG.get("move_wait_time_s", 5))

            elif choice == '0':
                LOG.info("Exiting")
                break

            else:
                LOG.warning("Invalid choice")

    except KeyboardInterrupt:
        LOG.warning("Program interrupted by user")
    finally:
        client.disconnect()
        LOG.info("Drawing is finished.")


if __name__ == "__main__":
    main()