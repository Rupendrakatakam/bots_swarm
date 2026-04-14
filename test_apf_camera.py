#!/usr/bin/env python3
"""
Clean test script without matplotlib to verify camera blob APF interaction.
"""
import sys
import math
import numpy as np

sys.path.insert(0, '.')

from apf import APF, CircleObstacle, RectObstacle, DynamicObstacle
from apf_orca import FleetManager, DiffDriveConfig, CameraBlob
from nh_orca import NHORCAPlanner

# Config
cfg = DiffDriveConfig(
    robot_radius=1.0,
    wheel_base=0.8,
    max_linear_speed=2.0,
    max_angular_speed=10.0,
    max_wheel_speed=3.0,
    tracking_error=0.40,
    orientation_time=0.5,
    time_horizon=3.0,
    neighbor_dist=6.0,
    sim_dt=0.1,
)

# APF with enable_static_repulsion=False (key setting!)
apf = APF(
    k_att=1.9,
    k_rep=10.0,
    rho_0=1.5,
    max_force=7.0,
    vortex_gain=0.35,
    static_circles=[],
    static_rects=[],
    enable_static_repulsion=False,
)

print("=== APF CONFIG ===")
print(f"enable_static_repulsion = {apf.enable_static_repulsion}")
print(f"static_circles count = {len(apf.static_circles)}")
print(f"static_rects count = {len(apf.static_rects)}")

# Create two robots
fleet = FleetManager(cfg, apf)

# Robot A: diagonal path (2,2) -> (18,18)
robot_a = fleet.add_robot('A', goal_tolerance=0.3, lookahead_window=20, carrot_steps=8)
fleet.update_state('A', 2.0, 2.0, math.pi/4, 0.0, 0.0)
fleet.set_goal('A', 18.0, 18.0)
robot_a.path.set_straight_line_path((2.0, 2.0), (18.0, 18.0), n_steps=50)

# Robot B: horizontal path (2,12) -> (18,2)
robot_b = fleet.add_robot('B', goal_tolerance=0.3, lookahead_window=20, carrot_steps=8)
fleet.update_state('B', 2.0, 12.0, 0.0, 0.0, 0.0)
fleet.set_goal('B', 18.0, 2.0)
robot_b.path.set_straight_line_path((2.0, 12.0), (18.0, 2.0), n_steps=50)

# Camera blob at (7,7) - this should trigger APF when Robot A gets close
camera_blobs = [CameraBlob(x=7.0, y=7.0, radius=1.2, priority=1.5)]

print(f"\n=== CAMERA BLOB ===")
print(f"Blob at: ({camera_blobs[0].x}, {camera_blobs[0].y})")
print(f"Blob radius: {camera_blobs[0].radius}")
print(f"Blob priority: {camera_blobs[0].priority}")

print(f"\n=== RUNNING 40 STEPS (to reach blob at 7,7) ===")
dt = 0.1
for step in range(40):
    commands = fleet.tick_all(camera_blobs=camera_blobs)
    
    for rid in ['A', 'B']:
        robot = fleet.robots[rid]
        pos = (robot.state.pose.x, robot.state.pose.y)
        
        # Check actual APF output stored
        f_unknown = robot._last_f_unknown
        apf_mag = np.linalg.norm(f_unknown)
        
        # Get command and integrate position
        cmd = commands[rid]
        new_yaw = robot.state.pose.yaw + cmd.omega * dt
        new_x = robot.state.pose.x + cmd.v * math.cos(new_yaw) * dt
        new_y = robot.state.pose.y + cmd.v * math.sin(new_yaw) * dt
        fleet.update_state(rid, new_x, new_y, new_yaw, cmd.v * math.cos(new_yaw), cmd.v * math.sin(new_yaw))
        
        dist_to_blob = np.linalg.norm(np.array(pos) - np.array([7.0, 7.0]))
        
        print(f"Step {step:2d} {rid}: pos=({pos[0]:5.1f},{pos[1]:5.1f}) blob_dist={dist_to_blob:4.1f} APF_force={apf_mag:.2f} cmd_v={cmd.v:.2f}")

print("\n=== DONE ===")
print("If APF_force shows > 0 when blob_dist < rho_0, then APF is working correctly!")