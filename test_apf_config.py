#!/usr/bin/env python3
"""
Test script to verify APF configuration works correctly:
- With enable_static_repulsion=True (default): generates forces from static obstacles
- With enable_static_repulsion=False: no forces from static obstacles, only dynamic
"""

import numpy as np
from apf import APF, CircleObstacle, RectObstacle, DynamicObstacle

def test_apf_static_repulsion():
    print("Testing APF static repulsion configuration...")
    
    # Create APF with static obstacles
    static_circle = CircleObstacle(5.0, 5.0, 1.0)
    static_rect = RectObstacle(2.0, 2.0, 4.0, 3.0)
    
    # Test 1: Enable static repulsion (default behavior)
    print("\n=== Test 1: Static repulsion ENABLED ===")
    apf_enabled = APF(
        k_att=1.0,
        k_rep=5.0,
        rho_0=2.0,
        max_force=10.0,
        vortex_gain=0.4,
        static_circles=[static_circle],
        static_rects=[static_rect],
        enable_static_repulsion=True
    )
    
    # Position robot near static obstacle
    robot_pos = (5.5, 5.0)  # Near the circle
    goal_pos = (0.0, 0.0)
    
    force_enabled = apf_enabled.get_force(robot_pos, goal_pos, [])
    print(f"Robot position: {robot_pos}")
    print(f"Force with static repulsion enabled: {force_enabled}")
    print(f"Force magnitude: {np.linalg.norm(force_enabled):.4f}")
    
    # Test 2: Disable static repulsion (new behavior)
    print("\n=== Test 2: Static repulsion DISABLED ===")
    apf_disabled = APF(
        k_att=1.0,
        k_rep=5.0,
        rho_0=2.0,
        max_force=10.0,
        vortex_gain=0.4,
        static_circles=[static_circle],
        static_rects=[static_rect],
        enable_static_repulsion=False  # <-- This is the key change
    )
    
    force_disabled = apf_disabled.get_force(robot_pos, goal_pos, [])
    print(f"Robot position: {robot_pos}")
    print(f"Force with static repulsion disabled: {force_disabled}")
    print(f"Force magnitude: {np.linalg.norm(force_disabled):.4f}")
    
    # Test 3: Get repulsive only (what the intention blender uses)
    print("\n=== Test 3: Repulsive only (what IntentionBlender uses) ===")
    repulsive_only_enabled = apf_enabled.get_repulsive_only(robot_pos, [])
    repulsive_only_disabled = apf_disabled.get_repulsive_only(robot_pos, [])
    
    print(f"Repulsive only with static ENABLED: {repulsive_only_enabled} (mag: {np.linalg.norm(repulsive_only_enabled):.4f})")
    print(f"Repulsive only with static DISABLED: {repulsive_only_disabled} (mag: {np.linalg.norm(repulsive_only_disabled):.4f})")
    
    # Verification
    print("\n=== VERIFICATION ===")
    static_only_enabled = np.linalg.norm(repulsive_only_enabled) > 0.1
    static_only_disabled = np.linalg.norm(repulsive_only_disabled) < 0.1  # Should be near zero
    # Check that attractive force is still present in total force (goal-seeking behavior)
    attractive_force_enabled = np.linalg.norm(apf_enabled.get_force(robot_pos, goal_pos, []) - apf_enabled.get_repulsive_only(robot_pos, []))
    attractive_force_disabled = np.linalg.norm(apf_disabled.get_force(robot_pos, goal_pos, []) - apf_disabled.get_repulsive_only(robot_pos, []))
    
    print(f"Static repulsion enabled generates force: {static_only_enabled}")
    print(f"Static repulsion disabled generates near-zero force: {static_only_disabled}")
    print(f"Attractive force still works (enabled): {attractive_force_enabled > 0.1}")
    print(f"Attractive force still works (disabled): {attractive_force_disabled > 0.1}")
    
    if static_only_enabled and static_only_disabled:
        print("\n✅ STATIC REPULSION TOGGLE WORKS CORRECTLY!")
        return True
    else:
        print("\n❌ STATIC REPULSION TOGGLE FAILED")
        return False

if __name__ == "__main__":
    test_apf_static_repulsion()