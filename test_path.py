#!/usr/bin/env python3
"""Simple test - check WaypointTracker directly"""
import sys
import math
import numpy as np

sys.path.insert(0, '.')

from apf import APF
from apf_orca import FleetManager, DiffDriveConfig, GlobalPath, WaypointTracker

cfg = DiffDriveConfig(robot_radius=1.0, wheel_base=0.8, max_linear_speed=2.0,
    max_angular_speed=10.0, max_wheel_speed=3.0, tracking_error=0.40,
    orientation_time=0.5, time_horizon=3.0, neighbor_dist=6.0, sim_dt=0.1)

# Create path manually
path = GlobalPath()
path.set_straight_line_path((2.0, 2.0), (18.0, 18.0), n_steps=50)

tracker = WaypointTracker(max_speed=2.0, lookahead_window=15, carrot_steps=3, goal_tolerance=0.3)

# Test tracker
for step in range(5):
    pos = np.array([2.0, 2.0]) + np.array([step*0.5, step*0.5])  # simulated movement
    v_path, reached = tracker.compute(pos, path)
    print(f"Step {step}: pos=({pos[0]:.1f},{pos[1]:.1f}) V_path={v_path}, reached={reached}")