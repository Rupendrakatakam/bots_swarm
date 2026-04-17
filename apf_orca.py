"""
integration.py — Core Navigation Pipeline
==========================================

Implements the full DAG pipeline described in the architecture report:

    Hybrid A* → Waypoint Tracker → APF (Shield) → Intention Blender
              → NH-ORCA (Scalpel) → Kinematic Mapper → Wheels

PIPELINE AT A GLANCE
---------------------
Phase 1  GLOBAL PLANNER   (once at startup)
         Hybrid A* → list of (x, y, θ) waypoints around static obstacles.

Phase 2  WAYPOINT TRACKER  (60 Hz)
         Floating carrot: finds closest waypoint, targets K steps ahead.
         → V_path

Phase 3  THE SHIELD  (60 Hz)
         APF repulsion from UNKNOWN camera blobs only (not swarm agents).
         → F_unknown

Phase 4  INTENTION BLENDER  (60 Hz)
         V_pref = V_path + F_unknown

Phase 5  THE SCALPEL  (60 Hz)
         NH-ORCA against swarm telemetry only (not camera blobs).
         → V_safe (vx, vy)

Phase 6  KINEMATIC MAPPER + LOW-PASS FILTER  (60 Hz)
         V_safe → (v, ω) → (vl, vr) → hardware

SEPARATION GUARANTEE
---------------------
APF  never sees swarm agents.
ORCA never sees camera blobs.
They communicate ONLY through V_pref. This is the architectural firewall.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrow
from geometry import get_closest_point_on_circle, get_closest_point_on_rect

from hybrid_astar import HybridAStar
from apf import APF, DynamicObstacle, CircleObstacle, RectObstacle
from nh_orca import (
    NHORCAPlanner, DiffDriveConfig, SwarmAgent,
    unicycle_to_wheels, wheels_to_unicycle
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Pose2D:
    """Robot pose in 2-D space."""
    x:   float = 0.0
    y:   float = 0.0
    yaw: float = 0.0   # [rad]

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class RobotState:
    """Complete state of one robot, updated every tick from odometry."""
    pose:     Pose2D       = field(default_factory=Pose2D)
    vel:      np.ndarray   = field(default_factory=lambda: np.zeros(2))  # [vx, vy]
    # Last command sent to wheels (used for low-pass filtering)
    prev_v:   float        = 0.0
    prev_w:   float        = 0.0


@dataclass
class CameraBlob:
    """
    One unknown obstacle detection from the overhead camera.
    Only position and estimated size — no velocity.
    """
    x:      float
    y:      float
    radius: float   = 0.30   # estimated bounding radius [m]
    priority: float = 1.0    # repulsion multiplier (1.0 = normal)


@dataclass
class SwarmTelemetry:
    """
    Perfect telemetry for one OTHER swarm robot.
    Source: central tracker or broadcast from each robot.
    """
    robot_id: str
    x:        float
    y:        float
    vx:       float
    vy:       float
    radius:   float    # inflated radius (r + ε) of that robot


@dataclass
class WheelCommand:
    """Final hardware command."""
    v:     float = 0.0   # linear speed  [m/s]
    omega: float = 0.0   # angular speed [rad/s]
    vl:    float = 0.0   # left wheel    [m/s]
    vr:    float = 0.0   # right wheel   [m/s]


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE LOGGER — per-tick diagnostic for every phase
# ─────────────────────────────────────────────────────────────────────────────

class PipelineLogger:
    """
    Attaches to a RobotController and records one LogEntry per tick.

    WHAT EACH FIELD ANSWERS
    -----------------------
    apf_active        : Was APF actually doing anything? (F_unknown magnitude > 0.01)
                        If True WITH camera_blobs=[] → APF is firing on static obstacles
                        only. Check if static rects/circles are too close to the path.

    apf_force_mag     : How hard APF pushed this tick [m/s].
                        Spikes here = robot near a static obstacle.
                        Sustained non-zero with no blobs = static repulsion fighting path.

    orca_active       : Did ORCA compute any half-planes? (n_halfplanes > 0)
                        If False the entire time → robots never come close enough
                        for ORCA to trigger. Check neighbor_dist.

    orca_n_halfplanes : How many constraints the LP had to solve.
                        0 = ORCA idle. 1-2 = normal. 3+ = crowded space.

    orca_adjustment   : ||V_safe - V_pref|| [m/s] — how much ORCA shifted the velocity.
                        High values = ORCA is doing heavy dodging.
                        If orca_active=True but orca_adjustment≈0 → V_pref was already safe.

    kinematic_region  : Which NH region was selected this tick.
                        R_A1 = normal driving.
                        R_A2 = tight turn while moving.
                        R_B  = stop-and-spin ← primary jitter source.
                        Repeated R_B → orientation_time T is too small, or
                        V_safe direction keeps flipping (caused by APF/ORCA fighting).

    v_path_mag        : Speed from waypoint tracker [m/s].
                        If near zero → robot is stalling on waypoint (increase carrot_steps).

    v_pref_mag        : Speed after intention blend [m/s].
                        v_pref_mag < v_path_mag → APF is fighting path (pushing backwards).

    v_safe_mag        : Speed after ORCA [m/s].
                        v_safe_mag << v_pref_mag → ORCA is heavily braking.

    cmd_v / cmd_omega : Final wheel commands sent.
    """

    from dataclasses import dataclass as _dc

    @_dc
    class LogEntry:
        step:               int
        robot_id:           str
        pos:                tuple
        # Phase 2
        v_path_mag:         float
        # Phase 3 (APF)
        apf_active:         bool
        apf_force_mag:      float
        n_camera_blobs:     int
        # Phase 4 (blend)
        v_pref_mag:         float
        # Phase 5 (ORCA)
        orca_active:        bool
        orca_n_halfplanes:  int
        orca_adjustment:    float
        # Phase 6 (kinematics)
        kinematic_region:   str
        heading_error_deg:  float
        v_safe_mag:         float
        cmd_v:              float
        cmd_omega:          float

    def __init__(self, robot_id: str, print_every: int = 50):
        self.robot_id    = robot_id
        self.print_every = print_every
        self.entries: List['PipelineLogger.LogEntry'] = []
        self._step       = 0

    def record(
        self,
        pos:            tuple,
        v_path:         np.ndarray,
        f_unknown:      np.ndarray,
        v_pref:         np.ndarray,
        n_camera_blobs: int,
        orca_planner,                  # NHORCAPlanner instance
        mapper,                        # MotorMapper instance
        cmd:            'WheelCommand',
        yaw:            float,
    ):
        import math
        apf_mag  = float(np.linalg.norm(f_unknown))
        pref_mag = float(np.linalg.norm(v_pref))
        safe_mag = float(np.linalg.norm(
            getattr(orca_planner, '_last_safe_vel', np.zeros(2))
        ))
        n_hp     = getattr(orca_planner, '_last_n_halfplanes', 0)
        pref_clipped = getattr(orca_planner, '_last_pref_clipped', v_pref)
        adjustment = float(np.linalg.norm(
            getattr(orca_planner, '_last_safe_vel', v_pref) - pref_clipped
        ))

        # Heading error: between V_pref direction and current yaw
        if pref_mag > 1e-3:
            theta_H = math.atan2(v_pref[1], v_pref[0])
            h_err   = math.degrees(abs(((theta_H - yaw + math.pi) % (2*math.pi)) - math.pi))
        else:
            h_err = 0.0

        region = getattr(orca_planner.mapper, '_last_region', '?')

        entry = PipelineLogger.LogEntry(
            step              = self._step,
            robot_id          = self.robot_id,
            pos               = (round(pos[0], 2), round(pos[1], 2)),
            v_path_mag        = round(float(np.linalg.norm(v_path)), 3),
            apf_active        = apf_mag > 0.01,
            apf_force_mag     = round(apf_mag, 3),
            n_camera_blobs    = n_camera_blobs,
            v_pref_mag        = round(pref_mag, 3),
            orca_active       = n_hp > 0,
            orca_n_halfplanes = n_hp,
            orca_adjustment   = round(adjustment, 3),
            kinematic_region  = region,
            heading_error_deg = round(h_err, 1),
            v_safe_mag        = round(safe_mag, 3),
            cmd_v             = round(cmd.v, 3),
            cmd_omega         = round(cmd.omega, 3),
        )
        self.entries.append(entry)

        if self._step % self.print_every == 0:
            # --- ADD THESE to record() ---
            v_path_actual_mag = round(float(np.linalg.norm(v_path)), 3)
            v_pref_actual_mag = round(pref_mag, 3)
            safe_vel = getattr(orca_planner, '_last_safe_vel', np.zeros(2))
            v_safe_actual_mag = round(float(np.linalg.norm(safe_vel)), 3)
            self._print(entry)

        self._step += 1

    def _print(self, e: 'PipelineLogger.LogEntry'):
        apf_flag  = f"APF={'ON ' if e.apf_active else 'off'} ({e.apf_force_mag:.2f})"
        orca_flag = f"ORCA={'ON ' if e.orca_active else 'off'} ({e.orca_n_halfplanes}hp)"
        kin_flag  = f"KIN={e.kinematic_region}(herr={e.heading_error_deg:.0f}°)"

        # Show the full velocity chain so we can see WHERE speed is lost
        chain = (f"V_path={e.v_path_mag:.2f} → "
                 f"V_pref={e.v_pref_mag:.2f} → "
                 f"V_safe={e.v_safe_mag:.2f} → "
                 f"cmd_v={e.cmd_v:.2f}")

        print(f"[{e.robot_id}] step={e.step:4d} | {apf_flag} | {orca_flag} | "
              f"{kin_flag} | {chain}")

    def jitter_report(self):
        """
        Print a summary of where jitter-causing events occurred.
        Call this after run_simulation() finishes.
        """
        rb_steps    = [e for e in self.entries if e.kinematic_region == 'R_B']
        apf_on_static = [e for e in self.entries
                         if e.apf_active and e.n_camera_blobs == 0]
        big_orca    = [e for e in self.entries if e.orca_adjustment > 0.5]
        big_apf     = [e for e in self.entries if e.apf_force_mag > 1.0]

        print(f"\n{'='*60}")
        print(f"JITTER REPORT — {self.robot_id}  ({len(self.entries)} steps total)")
        print(f"{'='*60}")
        print(f"  R_B (stop-and-spin) events   : {len(rb_steps):4d}  "
              f"← main jitter source if high")
        print(f"  APF active, no camera blobs  : {len(apf_on_static):4d}  "
              f"← static obstacles fighting path")
        print(f"  ORCA big adjustments (>0.5)  : {len(big_orca):4d}  "
              f"← heavy swarm dodging")
        print(f"  APF big forces (>1.0 m/s)    : {len(big_apf):4d}  "
              f"← near static obstacle")

        if rb_steps:
            positions = [e.pos for e in rb_steps[:5]]
            print(f"  First R_B positions          : {positions}")
            print(f"  → Fix: increase orientation_time T or tracking_error ε")

        if apf_on_static:
            positions = [e.pos for e in apf_on_static[:5]]
            print(f"  First no-blob APF positions  : {positions}")
            print(f"  → Fix: reduce rho_0, or remove boundary walls from APF")

        if big_orca:
            positions = [e.pos for e in big_orca[:5]]
            print(f"  First big ORCA positions     : {positions}")
            print(f"  → Expected near robot crossings")
        print(f"{'='*60}\n")
# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — GLOBAL PLANNER INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class GlobalPath:
    """
    Stores the A*-generated waypoint list and exposes it to the tracker.

    In a real system, call set_path() with the output of your Hybrid A*
    planner. Here we also provide a straight-line stub for testing.

    IMPORTANT DESIGN RULE
    ----------------------
    The path is computed BLIND to other robots and camera blobs.
    It only knows about the static map (walls, pillars, U-shapes).
    This keeps the global planning complexity at O(1) per robot.
    """

    def __init__(self):
        self.waypoints: List[np.ndarray] = []   # list of [x, y] arrays

    def set_path(self, waypoints: List[Tuple[float, float]]):
        """
        Load a new path.  Each waypoint is (x, y, theta).
        Call this once at startup (or whenever the goal changes).
        """
        # FIX: Just grab wp[0] (x) and wp[1] (y), ignoring wp[2] (theta)
        self.waypoints = [np.array([wp[0], wp[1]], dtype=float) for wp in waypoints]

    def set_straight_line_path(
        self,
        start: Tuple[float, float],
        goal:  Tuple[float, float],
        n_steps: int = 50
    ):
        """Stub: straight-line path for testing without a full A* planner."""
        xs = np.linspace(start[0], goal[0], n_steps)
        ys = np.linspace(start[1], goal[1], n_steps)
        self.waypoints = [np.array([x, y]) for x, y in zip(xs, ys)]

    @property
    def is_loaded(self) -> bool:
        return len(self.waypoints) > 0


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — WAYPOINT TRACKER (floating carrot)
# ─────────────────────────────────────────────────────────────────────────────

class WaypointTracker:
    """
    Computes V_path: the velocity vector pulling the robot toward the
    immediate A* path segment.

    FLOATING CARROT LOGIC (plain English)
    --------------------------------------
    Problem: if the robot misses waypoint N by 1 cm, it circles forever.
    Fix: instead of requiring the robot to HIT waypoint N to unlock N+1,
         we scan a window of indices [current, current+lookahead_window]
         and find the CLOSEST waypoint in that window.
         Then target the waypoint CARROT_STEPS ahead of that.
         The window slides forward as the robot progresses.

    This means the robot always chases a moving carrot, never stalls.
    """

    def __init__(
        self,
        max_speed: float = 0.26, # [m/s]
        lookahead_window: int = 15, # how many waypoints to scan ahead
        carrot_steps: int = 3, # how many steps ahead to target
        goal_tolerance: float = 0.15, # [m] distance to declare "reached"
        curvature_steps: int = 10, # look ahead N steps for curvature
        max_curvature: float = 1.5, # rad/m threshold to trigger braking
    ):
        self.max_speed = max_speed
        self.lookahead_window = lookahead_window
        self.carrot_steps = carrot_steps
        self.goal_tolerance = goal_tolerance
        self.curvature_steps = curvature_steps
        self.max_curvature = max_curvature
        self._current_idx = 0
        self._prev_v_path = np.zeros(2)
        self.path_alpha = 0.4 # Path velocity smoothing factor
        self._last_carrot: Optional[np.ndarray] = None # exposed for ImprovedAPF

    def reset(self):
        """Call this when a new path is loaded."""
        self._current_idx = 0
        self._prev_v_path = np.zeros(2)  # Reset path smoothing

    def compute(
        self,
        robot_pos: np.ndarray,    # [x, y]
        path:      GlobalPath
    ) -> Tuple[np.ndarray, bool]: # (V_path [vx,vy], reached_goal)
        """
        Compute V_path toward the carrot waypoint.

        Returns
        -------
        V_path       : np.ndarray [vx, vy] — velocity toward carrot [m/s]
        reached_goal : bool — True when robot is within goal_tolerance of last wp
        """
        if not path.is_loaded:
            return np.zeros(2), False

        wps  = path.waypoints
        n_wp = len(wps)

        if np.linalg.norm(robot_pos - wps[-1]) < self.goal_tolerance:
            return np.zeros(2), True

        lo = self._current_idx
        hi = min(lo + self.lookahead_window, n_wp)
        dists = [np.linalg.norm(robot_pos - wps[i]) for i in range(lo, hi)]
        closest_local = int(np.argmin(dists))
        self._current_idx = lo + closest_local

        # ── Global reset: if robot drifted far from window, resync ───
        # This happens when ORCA deflects the robot far off the A* path.
        # Without this, the scan window falls behind and V_path becomes stale.
        DRIFT_THRESHOLD = self.max_speed * 4.0 # 4 s of travel = serious drift
        min_window_dist = dists[closest_local]
        if min_window_dist > DRIFT_THRESHOLD and self._current_idx > 0:
            # Robot is very far from any waypoint in the current window.
            # Do a global search across the full path to resync.
            all_dists = [np.linalg.norm(robot_pos - wps[i]) for i in range(n_wp)]
            global_best = int(np.argmin(all_dists))
            # Only go forward, never backward (prevents looping)
            if global_best > self._current_idx:
                self._current_idx = global_best
                # Recompute window from new position
                hi = min(self._current_idx + self.lookahead_window, n_wp)
                dists = [np.linalg.norm(robot_pos - wps[i]) for i in range(self._current_idx, hi)]
                closest_local = int(np.argmin(dists))
                self._current_idx = self._current_idx + closest_local

        carrot_idx = min(self._current_idx + self.carrot_steps, n_wp - 1)
        carrot = wps[carrot_idx]
        self._last_carrot = carrot.copy()
        diff = carrot - robot_pos
        dist       = np.linalg.norm(diff)

        if dist < 1e-4:
            return np.zeros(2), False

        curvature_factor = 1.0
        if n_wp > self.curvature_steps:
            future_idx = min(self._current_idx + self.curvature_steps, n_wp - 1)
            if future_idx > self._current_idx:
                current_dir = wps[future_idx] - wps[self._current_idx]
                current_dir_norm = np.linalg.norm(current_dir)
                if current_dir_norm > 1e-4:
                    current_dir = current_dir / current_dir_norm
                    robot_to_carrot = diff / dist
                    cross = abs(current_dir[0] * robot_to_carrot[1] - current_dir[1] * robot_to_carrot[0])
                    angle_diff = math.asin(np.clip(cross, -1.0, 1.0))
                    curvature = abs(angle_diff) / max(dist, 0.1)
                    if curvature > self.max_curvature:
                        curvature_factor = max(0.3, 1.0 - (curvature - self.max_curvature) / 2.0)

        speed  = min(self.max_speed * curvature_factor, dist)
        V_path = speed * diff / dist
        
        # Smooth path velocity to reduce sudden direction changes
        V_path = self.path_alpha * V_path + (1 - self.path_alpha) * self._prev_v_path
        self._prev_v_path = V_path.copy()
        
        return V_path, False


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 + 4 — THE SHIELD + INTENTION BLENDER
# ─────────────────────────────────────────────────────────────────────────────

class IntentionBlender:
    """
    Combines the waypoint pull (V_path) and the obstacle push (F_unknown)
    into V_pref, the input to NH-ORCA.

    V_pref = V_path + F_unknown

    This is the ARCHITECTURAL FIREWALL:
        APF  never knows ORCA exists.
        ORCA never sees camera blobs.
        They meet ONLY here, in V_pref.

    CAMERA BLOBS RULE
    -----------------
    Pass only UNKNOWN obstacles (camera detections of non-swarm entities).
    NEVER pass swarm agent positions here.  Use nh_orca.py for those.
    """

    def __init__(
        self,
        apf: APF,
        max_pref_speed: float = 0.26, # clip V_pref to this speed [m/s]
        robot_id: str = 'default',
    ):
        self.apf = apf
        self.max_pref_speed = max_pref_speed
        self.robot_id = robot_id

    def compute(
        self,
        robot_pos: Tuple[float, float],
        v_path: np.ndarray, # from waypoint tracker
        camera_blobs: List[CameraBlob], # UNKNOWN obstacles ONLY
        goal_pos: Optional[Tuple[float, float]] = None, # carrot waypoint for ImprovedAPF rho_g
    ) -> np.ndarray: # V_pref [vx, vy]
        """
        Blend the path velocity with APF repulsion from camera blobs.

        EMERGENCY: if repulsion magnitude > 2x max_pref_speed, ignore path
        and output pure repulsion direction to prevent collision.
        """
        if not camera_blobs:
            spd = np.linalg.norm(v_path)
            if spd > self.max_pref_speed:
                return v_path * (self.max_pref_speed / spd)
            return v_path.copy()

        # Build DynamicObstacle list from camera blobs
        dyn_obs = [
            DynamicObstacle(b.x, b.y, b.radius, b.priority)
            for b in camera_blobs
        ]

        # APF returns ONLY the repulsive component
        # (the attractive pull is handled by the waypoint tracker above)
        # goal_pos enables ImprovedAPF's GNRO fix (rho_g calculation). If None,
        # ImprovedAPF silently falls back to classical behaviour.
        # robot_id enables per-robot LocalMinimaState tracking.
        F_unknown = self.apf.get_repulsive_only(robot_pos, dyn_obs, goal_pos=goal_pos, robot_id=self.robot_id)

        f_mag = np.linalg.norm(F_unknown)

        # ── Emergency: repulsion very strong — ignore path ────────────────
        # Do NOT let path attraction cancel emergency repulsion.
        # If repulsion magnitude > emergency threshold, ignore path and
        # output pure repulsion at max speed.
        EMERGENCY_THRESHOLD = self.max_pref_speed * 2.0  # > 1x max_speed = emergency
        if f_mag > EMERGENCY_THRESHOLD:
            repulsion_dir = F_unknown / f_mag
            return repulsion_dir * self.max_pref_speed

        # ── Normal blend: path + repulsion ────────────────────────────────
        V_pref = v_path + F_unknown

        # Clip to max preferred speed
        speed = np.linalg.norm(V_pref)
        if speed > self.max_pref_speed:
            V_pref = V_pref * (self.max_pref_speed / speed)

        return V_pref


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — LOW-PASS FILTER + KINEMATIC EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

class MotorMapper:
    """
    Phase 6: Translate V_safe → physical wheel commands.

    LOW-PASS FILTER (The Shock Absorber)
    --------------------------------------
    Raw (v, ω) from NH-ORCA can jump sharply between ticks.
    Sending such abrupt commands to the motors causes:
        • Physical jitter and vibration in the chassis
        • Motor overcurrent spikes
        • Noisy odometry feedback

    We apply an exponential moving average (EMA):
        cmd_new = α * cmd_target + (1 - α) * cmd_prev

    α = 0.2 → very smooth (slow to react, good for steady cruising)
    α = 0.5 → balanced (recommended default)
    α = 0.9 → very responsive (good for high-speed obstacle avoidance)

    Tune α based on your motor controller's bandwidth and the environment.
    A cluttered environment with fast obstacles → higher α.
    A warehouse with wide lanes → lower α.
    """

    def __init__(
        self,
        cfg:          DiffDriveConfig,
        filter_alpha: float = 0.25,
    ):
        self.cfg = cfg
        self.alpha = filter_alpha
        self._prev_heading_error = 0.0
        self.heading_alpha = 0.3 # EMA smoothing for heading

    def compute(
        self,
        vx: float,
        vy: float,
        yaw: float,
        state: RobotState,
    ) -> WheelCommand:
        """
        Convert V_safe = (vx, vy) to a smoothed WheelCommand.

        Uses:
        - Soft-stop deceleration (not abrupt snap to zero)
        - Heading error EMA smoothing
        - Acceleration limits for smooth speed transitions
        - Coupled speed/steering: slow down when turning
        """
        cfg = self.cfg
        dt = cfg.sim_dt

        V_H = math.hypot(vx, vy)

        # ── Stop command: decelerate smoothly, don't snap ────────────────
        if V_H < 1e-4:
            v_stop = max(0.0, state.prev_v - cfg.max_linear_decel * dt)
            w_stop = 0.0
            if abs(state.prev_w) > 1e-4:
                w_stop = math.copysign(
                    max(0.0, abs(state.prev_w) - cfg.max_angular_accel * dt),
                    state.prev_w
                )
            state.prev_v = v_stop
            state.prev_w = w_stop
            vl, vr = unicycle_to_wheels(v_stop, w_stop, cfg.wheel_base, cfg.max_wheel_speed)
            v_act, w_act = wheels_to_unicycle(vl, vr, cfg.wheel_base)
            return WheelCommand(v=v_act, omega=w_act, vl=vl, vr=vr)

        theta_H = math.atan2(vy, vx)
        theta_err_raw = _wrap_angle(theta_H - yaw)

        # Smooth heading error with EMA
        theta_err = (self.heading_alpha * theta_err_raw +
                     (1.0 - self.heading_alpha) * self._prev_heading_error)
        self._prev_heading_error = theta_err

        omega_target = float(np.clip(
            theta_err / cfg.orientation_time,
            -cfg.max_angular_speed,
            cfg.max_angular_speed
        ))

        # ── Paper Eq.(8): optimal forward speed for this heading error ────
        v_target = float(np.clip(
            _optimal_v(V_H, theta_err),
            0.0,
            cfg.max_linear_speed
        ))

        # ── Coupling: large turn rate forces reduced forward speed ────────
        # Physical constraint: vr = v + ω*L/2 ≤ max_wheel_speed
        v_coupled_limit = max(
            0.0,
            cfg.max_wheel_speed - abs(omega_target) * cfg.wheel_base / 2.0
        )
        v_target = min(v_target, v_coupled_limit)

        # ── Turn-aware speed: sharp turns → slow down (human behavior) ───
        # When the steering wheel is turned sharply, humans brake before the
        # turn, then accelerate through it. We detect this via omega_target
        # magnitude and apply a proportional speed reduction.
        # omega_target > 0.5 * max_angular_speed → entering sharp turn
        TURN_SLOW_THRESHOLD = 0.5 * cfg.max_angular_speed
        if abs(omega_target) > TURN_SLOW_THRESHOLD:
            # Map [0.5*w_max, w_max] → [0.60, 1.0] speed factor
            # At threshold (0.5*w_max) → 60% speed retained
            # At max (1.0*w_max) → 60% speed retained
            # (Linear from threshold to max, constant beyond)
            turn_sharpness = (abs(omega_target) - TURN_SLOW_THRESHOLD) / (0.5 * cfg.max_angular_speed)
            turn_slow_factor = max(0.60, 1.0 - turn_sharpness * 0.40)
            v_target *= turn_slow_factor

        # ── Acceleration limits (smooth transitions) ──────────────────────
        dv = v_target - state.prev_v
        if dv >= 0:
            dv_limited = min(dv, cfg.max_linear_accel * dt)
        else:
            dv_limited = max(dv, -cfg.max_linear_decel * dt)
        v_filt = float(np.clip(state.prev_v + dv_limited, 0.0, cfg.max_linear_speed))

        dw = omega_target - state.prev_w
        dw_limited = float(np.clip(
            dw,
            -cfg.max_angular_accel * dt,
            cfg.max_angular_accel * dt
        ))
        omega_filt = float(np.clip(
            state.prev_w + dw_limited,
            -cfg.max_angular_speed,
            cfg.max_angular_speed
        ))

        state.prev_v = v_filt
        state.prev_w = omega_filt

        # ── Wheel speeds → reconstruct actual v, ω ────────────────────────
        vl, vr = unicycle_to_wheels(
            v_filt, omega_filt, cfg.wheel_base, cfg.max_wheel_speed
        )
        v_act, w_act = wheels_to_unicycle(vl, vr, cfg.wheel_base)

        return WheelCommand(v=v_act, omega=w_act, vl=vl, vr=vr)


# EMERGENCY BRAKE — hard geometric safety net
# ─────────────────────────────────────────────────────────────────────────────

class EmergencyBrake:
    """
    Monitors the robot's immediate neighbourhood every tick and triggers
    three escalating responses when an obstacle is too close.

    LEVEL 1 — SLOW DOWN (warning zone): distance < warn_dist → scale speed
    LEVEL 2 — HARD BRAKE (danger zone): distance < brake_dist → cmd_v = 0
    LEVEL 3 — REVERSE (collision imminent): distance < reverse_dist → reverse

    Also checks boundary walls (proximity levels same as blob levels).

    APF is "soft" physics — it pushes but never guarantees non-overlap.
    This is the hard geometric check that catches what APF misses when
    the blob moves faster than the robot can react.
    """

    def __init__(
        self,
        robot_radius: float = 1.0,
        warn_dist: float = 0.8, # gap: slow down
        brake_dist: float = 0.3, # gap: hard brake
        reverse_dist: float = 0.0, # gap: reverse (≤0 = overlapping)
        slow_factor: float = 0.4, # fraction of max speed in warning
        reverse_speed: float = 0.3, # [m/s] reverse speed
        world_x_min: float = 0.0,
        world_x_max: float = 20.0,
        world_y_min: float = 0.0,
        world_y_max: float = 20.0,
    ):
        self.robot_radius = robot_radius
        self.warn_dist = warn_dist
        self.brake_dist = brake_dist
        self.reverse_dist = reverse_dist
        self.slow_factor = slow_factor
        self.reverse_speed = reverse_speed
        self.world_x_min = world_x_min
        self.world_x_max = world_x_max
        self.world_y_min = world_y_min
        self.world_y_max = world_y_max

    def check(
        self,
        cmd: WheelCommand,
        robot_pos: np.ndarray,
        robot_yaw: float,
        camera_blobs: List[CameraBlob],
    ) -> WheelCommand:
        """Inspect cmd against immediate geometry and override if unsafe."""
        # ── Check blob obstacles ──────────────────────────────────────────
        min_blob_gap = float('inf')
        closest_blob_dir = None

        for blob in camera_blobs:
            blob_pos = np.array([blob.x, blob.y])
            centre_dist = np.linalg.norm(robot_pos - blob_pos)
            gap = centre_dist - blob.radius - self.robot_radius

            if gap < min_blob_gap:
                min_blob_gap = gap
                if centre_dist > 1e-9:
                    closest_blob_dir = (robot_pos - blob_pos) / centre_dist

        # ── Check boundary walls ──────────────────────────────────────────
        # Find smallest gap to any wall
        min_wall_gap = float('inf')
        push_dir = np.zeros(2)  # direction to push away from wall

        x, y = robot_pos[0], robot_pos[1]
        r = self.robot_radius

        # Left wall
        gap_left = x - r - self.world_x_min
        if gap_left < min_wall_gap:
            min_wall_gap = gap_left
            push_dir = np.array([1.0, 0.0])
        # Right wall
        gap_right = self.world_x_max - r - x
        if gap_right < min_wall_gap:
            min_wall_gap = gap_right
            push_dir = np.array([-1.0, 0.0])
        # Bottom wall
        gap_bottom = y - r - self.world_y_min
        if gap_bottom < min_wall_gap:
            min_wall_gap = gap_bottom
            push_dir = np.array([0.0, 1.0])
        # Top wall
        gap_top = self.world_y_max - r - y
        if gap_top < min_wall_gap:
            min_wall_gap = gap_top
            push_dir = np.array([0.0, -1.0])

        # Use whichever is worse (smaller gap)
        min_gap = min(min_blob_gap, min_wall_gap)

        # If only blob triggered, save its direction for Level 3
        if min_blob_gap <= min_wall_gap and closest_blob_dir is not None:
            closest_dir = closest_blob_dir
        else:
            closest_dir = push_dir

        if min_gap == float('inf') or closest_dir is None:
            return cmd  # no obstacles — passthrough

        # ── Level 3: Reverse ──────────────────────────────────────────────
        if min_gap <= self.reverse_dist:
            heading_vec = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
            into_obs = float(np.dot(heading_vec, -closest_dir))  # > 0 = toward obs
            if into_obs > 0.1:
                vl = -self.reverse_speed
                vr = -self.reverse_speed
                return WheelCommand(
                    v=-self.reverse_speed,
                    omega=cmd.omega * 0.5,
                    vl=vl,
                    vr=vr,
                )

        # ── Level 2: Hard brake ───────────────────────────────────────────
        if min_gap <= self.brake_dist:
            return WheelCommand(
                v=0.0,
                omega=cmd.omega * 0.8,
                vl=-cmd.omega * 0.4,
                vr=cmd.omega * 0.4,
            )

        # ── Level 1: Slow down ────────────────────────────────────────────
        if min_gap <= self.warn_dist:
            scale = max(
                self.slow_factor,
                (min_gap - self.brake_dist) / (self.warn_dist - self.brake_dist),
            )
            return WheelCommand(
                v=cmd.v * scale,
                omega=cmd.omega,
                vl=cmd.vl * scale,
                vr=cmd.vr * scale,
            )

        return cmd  # all clear


# ─────────────────────────────────────────────────────────────────────────────
# CORE ROBOT CONTROLLER — puts all phases together for one robot
# ─────────────────────────────────────────────────────────────────────────────

class RobotController:
    """
    Full navigation stack for ONE differential-drive robot.

    Instantiate one controller per robot. Each controller is independent —
    no communication between controllers (ORCA handles coordination internally
    through shared telemetry).

    USAGE EXAMPLE
    -------------
        cfg = DiffDriveConfig(robot_radius=0.22, wheel_base=0.287, ...)

        apf = APF(
            k_att=1.0, k_rep=2.0, rho_0=1.5, max_force=5.0,
            static_circles=[CircleObstacle(5,5,0.3)],
            static_rects=[RectObstacle(0,0,10,0.2)],
        )

        ctrl = RobotController(robot_id='robot_0', cfg=cfg, apf=apf)

        # At startup:
        ctrl.set_goal(3.0, 0.0)
        ctrl.path.set_straight_line_path((0,0), (3,0))  # or real A* path

        # Every control tick (~60 Hz):
        ctrl.update_state(x, y, yaw, vx, vy)
        cmd = ctrl.tick(
            camera_blobs  = [CameraBlob(2.0, 0.5, 0.3)],   # unknown blobs
            swarm_telemetry = [                               # other robots
                SwarmTelemetry('robot_1', 1.5, 0.0, 0.1, 0.0, cfg.inflated_radius)
            ]
        )
        # cmd.v, cmd.omega → publish to /cmd_vel
        # cmd.vl, cmd.vr   → send directly to wheel controllers
    """

    def __init__(
        self,
        robot_id: str,
        cfg:      DiffDriveConfig,
        apf:      APF,
        filter_alpha:     float = 0.25,
        lookahead_window: int   = 15,
        carrot_steps:     int   = 3,
        goal_tolerance:   float = 0.15,
    ):
        self.robot_id = robot_id
        self.cfg      = cfg

        # Subsystems
        self.path     = GlobalPath()
        self.tracker  = WaypointTracker(
            max_speed        = cfg.max_linear_speed,
            lookahead_window = lookahead_window,
            carrot_steps     = carrot_steps,
            goal_tolerance   = goal_tolerance,
        )
        self.blender = IntentionBlender(apf, max_pref_speed=cfg.max_linear_speed, robot_id=robot_id)
        self.orca = NHORCAPlanner(cfg)
        self.mapper = MotorMapper(cfg, filter_alpha=filter_alpha)
        self.emergency = EmergencyBrake(
            robot_radius=cfg.robot_radius,
            warn_dist=cfg.robot_radius * 0.8, # 0.8m gap → slow down
            brake_dist=cfg.robot_radius * 0.3, # 0.3m gap → hard brake
            reverse_dist=0.0, # overlapping → reverse
            slow_factor=0.35,
            reverse_speed=cfg.max_linear_speed * 0.2,
            world_x_min=cfg.world_x_min,
            world_x_max=cfg.world_x_max,
            world_y_min=cfg.world_y_min,
            world_y_max=cfg.world_y_max,
        )

        # State
        self.state        = RobotState()
        self.goal:        Optional[Tuple[float, float]] = None
        self.reached_goal = False

# Debug/log (last tick values)
        self._last_v_path = np.zeros(2)
        self._last_f_unknown = np.zeros(2)
        self._last_v_pref = np.zeros(2)
        self._last_v_safe = np.zeros(2)

# Yield coordination state (set by FleetManager._detect_and_assign_yields)
        # KEY DESIGN: soft yield only — we scale V_pref before ORCA, never hard-stop.
        # _forced_stop was removed because it caused permanent deadlock: the robot
        # stops, ORCA's next tick sees vel=(0,0), but path_progress doesn't advance,
        # so _forced_stop stays True forever.
        self.yield_priority: int = 0 # 0=normal, higher=has right-of-way
        self.is_yielding: bool = False # True = commanded to slow by fleet
        self._yield_ticks_left: int = 0 # countdown — yield auto-releases after N ticks
        self._yield_scale: float = 1.0 # 1.0 = full speed, 0.30-0.60 = soft yield
        self.path_progress: float = 0.0 # fraction 0..1 of waypoints already passed

        self.logger = PipelineLogger(robot_id, print_every=100)

    # ── State update (call with fresh odometry) ───────────────────────────

    def update_state(
        self,
        x:   float,
        y:   float,
        yaw: float,
        vx:  float = 0.0,
        vy:  float = 0.0,
    ):
        """Update robot state from odometry. Call before tick()."""
        self.state.pose = Pose2D(x, y, yaw)
        self.state.vel  = np.array([vx, vy])

    # ── Goal setter ───────────────────────────────────────────────────────

    def set_goal(self, gx: float, gy: float):
        """
        Set a new navigation goal.
        If you have a Hybrid A* planner, also call:
            self.path.set_path(astar_result)
        Otherwise, a straight-line path is generated as a stub.
        """
        self.goal         = (gx, gy)
        self.reached_goal = False
        self.tracker.reset()
        # Stub straight-line path if no A* planner connected
        if not self.path.is_loaded:
            pos = (self.state.pose.x, self.state.pose.y)
            self.path.set_straight_line_path(pos, (gx, gy))

    # ── Main tick — runs all 6 phases ─────────────────────────────────────

    def tick(
        self,
        camera_blobs:     List[CameraBlob]     = None,
        swarm_telemetry:  List[SwarmTelemetry] = None,
    ) -> WheelCommand:
        """
        Run one full pipeline tick. Returns wheel commands.

        Parameters
        ----------
        camera_blobs     : unknown dynamic obstacles from overhead camera.
                           ONLY non-swarm entities. Can be empty list.
        swarm_telemetry  : perfect telemetry of all OTHER swarm robots.
                           ONLY swarm agents. Can be empty list.

        Returns
        -------
        WheelCommand : v, omega, vl, vr ready to send to hardware.
        """
        camera_blobs    = camera_blobs    or []
        swarm_telemetry = swarm_telemetry or []

        pos = self.state.pose.position
        yaw = self.state.pose.yaw
        vel = self.state.vel

        # ── Stop if goal reached ──────────────────────────────────────────
        if self.reached_goal:
            return WheelCommand()

        # ── Phase 2: Waypoint Tracker → V_path ───────────────────────────
        v_path, reached = self.tracker.compute(pos, self.path)
        if reached:
            self.reached_goal = True
            return WheelCommand()

        # ── Track path progress (0.0 = start, 1.0 = end) ───────────────────
        if self.path.is_loaded:
            wp_idx = self.tracker._current_idx
            total_wp = len(self.path.waypoints)
            self.path_progress = float(wp_idx) / max(total_wp - 1, 1)
        else:
            self.path_progress = 0.0

        # ── Phase 3+4: Shield + Blender → V_pref ─────────────────────────
        goal_pos = tuple(self.tracker._last_carrot) if hasattr(self.tracker, '_last_carrot') and self.tracker._last_carrot is not None else None
        v_pref = self.blender.compute(
            robot_pos = tuple(pos),
            v_path = v_path,
            camera_blobs = camera_blobs, # ONLY unknown obstacles
            goal_pos = goal_pos, # carrot → ImprovedAPF rho_g (Eq.1-3)
        )

# ── Fleet yield: reduce V_pref BEFORE ORCA (critical ordering) ───
    # By reducing V_pref here, ORCA sees the correct intended velocity.
    # This means other robots' ORCA correctly predicts "A is slowing",
    # preventing the velocity mismatch that caused ghost collisions.
    # The _yield_ticks_left countdown handles auto-release.
        if self.is_yielding and self._yield_scale < 1.0:
            v_pref = v_pref * self._yield_scale
            self._yield_ticks_left -= 1
            if self._yield_ticks_left <= 0:
                self.is_yielding = False
                self._yield_scale = 1.0

    # ── Phase 5: NH-ORCA → V_safe ─────────────────────────────────────
        neighbors = [
            SwarmAgent(
                pos = np.array([t.x, t.y]),
                vel = np.array([t.vx, t.vy]),
                radius = t.radius,
                max_speed = self.cfg.max_linear_speed,
            )
            for t in swarm_telemetry # ONLY swarm agents
        ]
        v_safe_vx, v_safe_vy = self.orca.update(
            my_pos = tuple(pos),
            my_vel = tuple(vel),
            my_yaw = yaw,
            pref_vel = tuple(v_pref), # ORCA receives the already-scaled V_pref
            neighbors = neighbors,
        )
        
        # Reconstruct 2D V_safe from (v, ω)
        # For debug/log — actual command is already in (v, ω)
        v_safe_2d = np.array([v_safe_vx, v_safe_vy])

        # ── Phase 6: Kinematic mapper + low-pass filter ───────────────────
        # MotorMapper now receives the true safe holonomic direction and
        # computes heading error → (v, ω) correctly.
        cmd = self.mapper.compute(
            vx    = v_safe_vx,
            vy    = v_safe_vy,
            yaw   = yaw,
            state = self.state,
        )
        # ── Save debug values ─────────────────────────────────────────────
        self._last_v_path    = v_path
        self._last_f_unknown = v_pref - v_path
        self._last_v_pref    = v_pref
        self._last_v_safe    = v_safe_2d

        # ── Log this tick ─────────────────────────────────────────────────
        self.logger.record(
            pos            = tuple(pos),
            v_path         = v_path,
            f_unknown      = v_pref - v_path,
            v_pref         = v_pref,
            n_camera_blobs = len(camera_blobs),
            orca_planner   = self.orca,
            mapper         = self.mapper,
             cmd            = cmd,
             yaw=yaw,
        )

        # ── Emergency brake override (last safety net before hardware) ────
        # Runs AFTER the full pipeline. Catches what APF and ORCA missed.
        # This is the geometric hard-check that Gazebo physics requires.
        cmd = self.emergency.check(
            cmd=cmd,
            robot_pos=pos,
            robot_yaw=yaw,
            camera_blobs=camera_blobs,
        )

# ── Fleet-level soft yield override ──────────────────────────────
    # IMPORTANT: yield scaling was ALREADY applied to V_pref before ORCA
    # in the Phase 4/5 section above. _yield_ticks_left was decremented there.
    # is_yielding was set to False when _yield_ticks_left hit 0.
    # Here we just apply one final soft-scale to the wheel command as a
    # safety net, ensuring the command truly reflects the yield intent.
    # We keep omega (steering direction) — freezing omega caused oscillations.
        if self.is_yielding:
            cmd = WheelCommand(
                v = cmd.v * self._yield_scale,
                omega = cmd.omega, # keep steering — don't freeze direction
                vl = cmd.vl * self._yield_scale,
                vr = cmd.vr * self._yield_scale,
        )

        return cmd

    def get_debug_info(self) -> Dict:
        """Return intermediate pipeline values for logging / visualisation."""
        return {
            'robot_id':   self.robot_id,
            'pos':        (self.state.pose.x, self.state.pose.y),
            'yaw_deg':    math.degrees(self.state.pose.yaw),
            'v_path':     tuple(self._last_v_path),
            'f_unknown':  tuple(self._last_f_unknown),
            'v_pref':     tuple(self._last_v_pref),
            'v_safe':     tuple(self._last_v_safe),
            'goal':       self.goal,
            'reached':    self.reached_goal,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FLEET MANAGER — orchestrates N robots
# ─────────────────────────────────────────────────────────────────────────────

class FleetManager:
    """
    Manages a full swarm of RobotControllers.

    WHAT IT DOES
    ------------
    • Holds one RobotController per robot.
    • Each tick, collects all robots' states and passes the correct
      swarm_telemetry to each robot (its OWN state is excluded).
    • Applies the APF firewall: camera_blobs go to APF, swarm telemetry
      goes to NH-ORCA. They never cross.

    HOW SWARM TELEMETRY IS BUILT
    ----------------------------
    Each robot broadcasts its (x, y, vx, vy) — from odometry.
    The fleet manager assembles SwarmTelemetry for all OTHER robots
    and passes it to each robot's NH-ORCA.

    NOTE: This manager is framework-agnostic.
    In a ROS 2 system, replace tick_all() with a ROS timer callback,
    replace update_state() with an odometry subscriber, and replace
    WheelCommand with a Twist publisher.
    """

    def __init__(self, cfg: DiffDriveConfig, apf: APF):
        self.cfg        = cfg
        self.apf        = apf
        self.robots:    Dict[str, RobotController] = {}

    def add_robot(self, robot_id: str, **kwargs) -> RobotController:
        """
        Add a robot to the fleet.

        Optional kwargs forwarded to RobotController:
            filter_alpha, lookahead_window, carrot_steps, goal_tolerance
        """
        ctrl = RobotController(robot_id, self.cfg, self.apf, **kwargs)
        self.robots[robot_id] = ctrl
        return ctrl
    
    def update_state(
        self,
        robot_id: str,
        x: float, y: float, yaw: float,
        vx: float = 0.0, vy: float = 0.0,
    ):
        """Update one robot's odometry. Call this from your sensor callback."""
        if robot_id in self.robots:
            self.robots[robot_id].update_state(x, y, yaw, vx, vy)

    def set_goal(self, robot_id: str, gx: float, gy: float):
        """Set a goal for one robot."""
        if robot_id in self.robots:
            self.robots[robot_id].set_goal(gx, gy)
    
    def tick_all(
        self,
        camera_blobs: List[CameraBlob] = None, # shared across all robots
    ) -> Dict[str, WheelCommand]:
        """
        Run one full tick for ALL robots.

        Each robot gets:
        camera_blobs → the shared list of unknown obstacles (APF)
        swarm_telemetry → telemetry of all OTHER robots (NH-ORCA)

        Returns
        -------
        Dict mapping robot_id → WheelCommand
        """
        camera_blobs = camera_blobs or []

        # ── PHASE 0: Conflict detection + yield assignment ──────────────
        self._detect_and_assign_yields()

        # Build telemetry snapshot for this tick
        # (all robots broadcast their current state)
        all_telemetry: Dict[str, SwarmTelemetry] = {
            rid: SwarmTelemetry(
                robot_id = rid,
                x        = ctrl.state.pose.x,
                y        = ctrl.state.pose.y,
                vx       = ctrl.state.vel[0],
                vy       = ctrl.state.vel[1],
radius = self.cfg.social_radius, # uses 20% virtual bubble — other robots plan around the 20% clearance
            )
            for rid, ctrl in self.robots.items()
        }

        commands: Dict[str, WheelCommand] = {}

        for rid, ctrl in self.robots.items():
            # Each robot sees ALL OTHER robots' telemetry — not its own
            swarm_tel = [
                tel for other_id, tel in all_telemetry.items()
                if other_id != rid
            ]

            # APF gets camera_blobs (unknown)
            # NH-ORCA gets swarm_tel (cooperative)
            # They NEVER swap inputs — this is the firewall
            commands[rid] = ctrl.tick(
                camera_blobs    = camera_blobs,   # unknown → APF only
                swarm_telemetry = swarm_tel,       # cooperative → ORCA only
            )

        return commands
    
    def _detect_and_assign_yields(self):
        """
        Predict trajectory conflicts and assign SOFT yield (speed reduction).
    
        KEY DIFFERENCES from old version:
        ----------------------------------
        OLD: used _forced_stop → hard zero velocity → ORCA mismatch → deadlock
        NEW: uses _yield_scale → proportional speed reduction → ORCA stays valid
    
        Why soft yield works:
        - We reduce the yielding robot's V_pref magnitude before ORCA runs
        - ORCA sees the correct reduced velocity and plans half-planes accordingly
        - The other robot's ORCA correctly predicts "A is slowing" (not "A is stopped")
        - No mismatch → no ghost collisions
    
        Yield assignment rule:
        - Lower path_progress (less committed) yields
        - Yield is soft: speed scales to _yield_scale (default 30%)
        - Yield auto-releases after _yield_ticks_left countdown expires
        - BOTH robots slow at crossings (60%/40%) — like real human negotiation
        - If robots are clearly separated (gap > CONFLICT_CLEAR_DIST), skip yield
        """
        YIELD_DURATION_TICKS = 25 # auto-release after 2.5s at 10Hz
        CONFLICT_TIME_WINDOW = 2.5 # [s] — only react to conflicts within 2.5s
        CONFLICT_CLEAR_DIST = 2.5 # [m] — surface gap to deactivate yield
        YIELD_FAST_SCALE = 0.60 # higher path_progress → 60% speed
        YIELD_SLOW_SCALE = 0.40 # lower path_progress → 40% speed (more conservative)
    
        # Combine inflated radii for conflict zone (use same radius as ORCA)
        combined_r = self.cfg.inflated_radius * 2.0
    
        for ctrl in self.robots.values():
            # NOTE: We do NOT clear is_yielding here.
            # If is_yielding is already True (from previous tick), we keep it.
            # The _yield_ticks_left countdown in tick() handles auto-release.
            # Only clear if the robot has reached its goal.
            if ctrl.reached_goal:
                ctrl.is_yielding = False
                ctrl._yield_scale = 1.0
                ctrl._yield_ticks_left = 0
    
        robot_ids = list(self.robots.keys())
        tau = self.cfg.time_horizon
    
        for i, rid_a in enumerate(robot_ids):
            for rid_b in robot_ids[i + 1:]:
                ctrl_a = self.robots[rid_a]
                ctrl_b = self.robots[rid_b]
    
                # Skip if either robot already at goal (clear path)
                if ctrl_a.reached_goal or ctrl_b.reached_goal:
                    continue
    
                pos_a = np.array([ctrl_a.state.pose.x, ctrl_a.state.pose.y])
                pos_b = np.array([ctrl_b.state.pose.x, ctrl_b.state.pose.y])
                vel_a = ctrl_a.state.vel
                vel_b = ctrl_b.state.vel
    
                # ── Check current surface gap first ──────────────────────
                current_dist = float(np.linalg.norm(pos_a - pos_b))
                surface_gap = current_dist - combined_r
                if surface_gap > CONFLICT_CLEAR_DIST:
                    continue # far apart — no yield needed
    
                # ── Trajectory intersection test ─────────────────────────
                rel_pos = pos_b - pos_a
                rel_vel = vel_a - vel_b
    
                a_c = float(np.dot(rel_vel, rel_vel))
                b_c = 2.0 * float(np.dot(rel_pos, rel_vel))
                c_c = float(np.dot(rel_pos, rel_pos)) - combined_r ** 2
    
                will_conflict = False
                if a_c < 1e-9:
                    will_conflict = (c_c < 0) # already overlapping
                else:
                    disc = b_c ** 2 - 4.0 * a_c * c_c
                    if disc >= 0:
                        sq = math.sqrt(disc)
                        t1 = (-b_c - sq) / (2.0 * a_c)
                        t2 = (-b_c + sq) / (2.0 * a_c)
                        for t in (t1, t2):
                            if 0.0 < t <= CONFLICT_TIME_WINDOW:
                                will_conflict = True
                                break
    
                if not will_conflict:
                    continue
    
                # ── Both robots slow proportionally at crossings ──────────
                # Higher path_progress = more "right of way" = slightly faster (60%)
                # Lower path_progress = less committed = slower (40%)
                # This mirrors human behavior: both cars at a crossing both slow
                # down, and the one that arrived first / is more committed goes first.
                progress_a = ctrl_a.path_progress
                progress_b = ctrl_b.path_progress
    
                if progress_a >= progress_b:
                    ctrl_a._yield_scale = YIELD_FAST_SCALE  # 0.60
                    ctrl_a._yield_ticks_left = YIELD_DURATION_TICKS
                    if not ctrl_a.is_yielding:
                        ctrl_a.is_yielding = True
    
                    ctrl_b._yield_scale = YIELD_SLOW_SCALE  # 0.40
                    ctrl_b._yield_ticks_left = YIELD_DURATION_TICKS
                    if not ctrl_b.is_yielding:
                        ctrl_b.is_yielding = True
                else:
                    ctrl_a._yield_scale = YIELD_SLOW_SCALE
                    ctrl_a._yield_ticks_left = YIELD_DURATION_TICKS
                    if not ctrl_a.is_yielding:
                        ctrl_a.is_yielding = True
    
                    ctrl_b._yield_scale = YIELD_FAST_SCALE
                    ctrl_b._yield_ticks_left = YIELD_DURATION_TICKS
                    if not ctrl_b.is_yielding:
                        ctrl_b.is_yielding = True
    
    def get_fleet_debug(self) -> List[Dict]:
        """Return debug info for all robots (for logging / visualisation)."""
        return [ctrl.get_debug_info() for ctrl in self.robots.values()]
    




# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _optimal_v(V_H: float, theta_err: float) -> float:
    """Paper Eq. (8): v* = V_H * θ * sin(θ) / (2*(1-cos(θ)))"""
    if abs(theta_err) < 1e-4:
        return V_H
    c = 1.0 - math.cos(theta_err)
    return V_H if abs(c) < 1e-10 else V_H * theta_err * math.sin(theta_err) / (2.0 * c)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB SIMULATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    fleet:         FleetManager,
    agent_data:    Dict,                  # {'robot_id': {'color': str, 'goal': (x,y)}}
    camera_blobs:  List[CameraBlob] = None,
    max_steps:     int   = 1500,
    dt:            float = 0.1,
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Run the full FleetManager pipeline and collect (x, y, yaw) histories.

    Parameters
    ----------
    fleet        : configured FleetManager with robots already added + goals set
    agent_data   : dict keyed by robot_id with 'color' and 'goal' fields
    camera_blobs : static list of unknown obstacles (or None)
    max_steps    : safety cutoff
    dt           : must match fleet.cfg.sim_dt

    Returns
    -------
    histories : Dict[robot_id → list of (x, y, yaw)]
    """
    camera_blobs = camera_blobs or []
    histories    = {rid: [] for rid in fleet.robots}

    # Record initial positions
    for rid, ctrl in fleet.robots.items():
        p = ctrl.state.pose
        histories[rid].append((p.x, p.y, p.yaw))

    for step in range(max_steps):
        # ── Run one pipeline tick ─────────────────────────────────────────
        commands = fleet.tick_all(camera_blobs=camera_blobs)

        all_reached = all(fleet.robots[rid].reached_goal for rid in fleet.robots)

        # ── Euler integration: propagate each robot's state ───────────────
        for rid, cmd in commands.items():
            ctrl = fleet.robots[rid]
            p    = ctrl.state.pose

            new_yaw = p.yaw + cmd.omega * dt
            new_x   = p.x   + cmd.v * math.cos(new_yaw) * dt
            new_y   = p.y   + cmd.v * math.sin(new_yaw) * dt

            # --- END OF run_simulation() LOOP ---
            vx = cmd.v * math.cos(new_yaw)
            vy = cmd.v * math.sin(new_yaw)

            fleet.update_state(rid, new_x, new_y, new_yaw, vx, vy)
            histories[rid].append((new_x, new_y, new_yaw))

        if all_reached:
            print(f"All goals reached at step {step}.")
            break

    return histories



def build_figure(
    fleet:        FleetManager,
    agent_data:   Dict,
    global_paths: Dict,                   # robot_id → list of (x, y) or (x, y, θ)
    histories:    Dict,
    world_size:   float = 20.0,
    static_circles: List = None,          # list of CircleObstacle
    static_rects:   List = None,          # list of RectObstacle
):
    """
    Build and return the (fig, ax, artists, update_fn) needed for FuncAnimation.
    """
    static_circles = static_circles or []
    static_rects   = static_rects   or []

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect('equal')
    ax.set_title('Multi-Agent Navigation: APF + NH-ORCA Pipeline', fontsize=14)
    ax.set_xticks(np.arange(0, world_size + 1, 1))
    ax.set_yticks(np.arange(0, world_size + 1, 1))
    ax.grid(True, linestyle=':', alpha=0.5)

    INFLATE = fleet.cfg.inflated_radius - fleet.cfg.robot_radius  # = ε

    # ── Static obstacles ──────────────────────────────────────────────────
    for obs in static_circles:
        ax.add_patch(Circle((obs.cx, obs.cy), obs.radius,
                            fill=True, color='red', alpha=0.35, zorder=2))
        ax.add_patch(Circle((obs.cx, obs.cy), obs.radius + INFLATE,
                            fill=False, color='orange', linestyle='--',
                            linewidth=1.2, zorder=2))

    for obs in static_rects:
        x0, y0, x1, y1 = obs.bounds
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                     fill=True, color='red', alpha=0.35, zorder=2))
        ax.add_patch(Rectangle((x0 - INFLATE, y0 - INFLATE),
                     (x1 - x0) + 2 * INFLATE, (y1 - y0) + 2 * INFLATE,
                     fill=False, color='orange', linestyle='--',
                     linewidth=1.2, zorder=2))

    # ── Boundary walls ─────────────────────────────────────────────────
    # Four walls around the world perimeter — dark gray, solid
    from matplotlib.patches import Rectangle as Rect2D
    wall_color = 'dimgray'
    wall_alpha = 0.6
    W = 0.15  # wall thickness [m]
    ax.add_patch(Rect2D((0, 0), W, world_size,
                 fill=True, color=wall_color, alpha=wall_alpha, zorder=1))
    ax.add_patch(Rect2D((world_size - W, 0), W, world_size,
                 fill=True, color=wall_color, alpha=wall_alpha, zorder=1))
    ax.add_patch(Rect2D((0, 0), world_size, W,
                 fill=True, color=wall_color, alpha=wall_alpha, zorder=1))
    ax.add_patch(Rect2D((0, world_size - W), world_size, W,
                 fill=True, color=wall_color, alpha=wall_alpha, zorder=1))

    # ── Global path lines (dashed) ────────────────────────────────────────
    for rid, path in global_paths.items():
        color = agent_data[rid]['color']
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, color=color, linestyle='--', linewidth=1.5,
                alpha=0.4, zorder=1)

    # ── Goal markers ──────────────────────────────────────────────────────
    for rid, data in agent_data.items():
        gx, gy = data['goal']
        color  = data['color']
        ax.scatter(gx, gy, s=220, c=color, marker='*', zorder=5)
    ax.annotate(f"Goal {rid}", (gx, gy),
                 textcoords='offset points', xytext=(6, 6), fontsize=8)

    # ── Trajectory prediction lines ─────────────────────────────────────
    # Show where each robot expects to go based on current velocity
    traj_lines = {}
    for rid in agent_data:
        color = agent_data[rid]['color']
        traj_lines[rid], = ax.plot([], [], color=color, linestyle=':',
                                   linewidth=1.5, alpha=0.5, zorder=1)

    # ── Yield indicator rings ────────────────────────────────────────────
    # Yellow dashed ring around yielding robots
    yield_rings = {}
    for rid in agent_data:
        x0, y0, _ = histories[rid][0]
        yield_rings[rid] = Circle((x0, y0), r * 1.8, fill=False,
                                  color='gold', linewidth=2.5,
                                  linestyle='--', zorder=6, alpha=0.0)
        ax.add_patch(yield_rings[rid])

    # ── Per-robot dynamic artists ─────────────────────────────────────────
    r = fleet.cfg.robot_radius
    L = fleet.cfg.wheel_base

    body_patches  = {}
    heading_lines = {}
    trail_lines   = {}
    left_wheels   = {}
    right_wheels  = {}

    for rid, data in agent_data.items():
        color = data['color']
        x0, y0, _ = histories[rid][0]

        body_patches[rid] = Circle((x0, y0), r, fill=True,
                                   color=color, alpha=0.6, zorder=4)
        ax.add_patch(body_patches[rid])

        heading_lines[rid], = ax.plot([], [], color='black',
                                      linewidth=2, zorder=5)
        trail_lines[rid],   = ax.plot([], [], color=color,
                                      linewidth=1.8, alpha=0.65, zorder=3)
        left_wheels[rid],   = ax.plot([], [], color='black',
                                      linewidth=4, solid_capstyle='round', zorder=5)
        right_wheels[rid],  = ax.plot([], [], color='black',
                                      linewidth=4, solid_capstyle='round', zorder=5)

    # ── Social bubble rings ───────────────────────────────────────────
    # Dashed circle showing the 20% virtual clearance zone
    social_r = fleet.cfg.robot_radius * (1.0 + fleet.cfg.social_bubble_factor)
    bubble_patches = {}
    for rid, data in agent_data.items():
        color = data['color']
        x0, y0, _ = histories[rid][0]
        bubble_patches[rid] = Circle(
            (x0, y0), social_r, fill=False, color=color, linewidth=1.5,
            linestyle='--', alpha=0.45, zorder=3
        )
        ax.add_patch(bubble_patches[rid])

    # ── Velocity arrows (V_path white, V_pref blue, V_safe green) ─────
    # Arrows scaled so max_speed → arrow length of 0.2 world units
    # scale_units='width' means U is in data units of x-axis width
    _ARROW_SCALE = 10.0 # max_speed / _ARROW_SCALE = visual arrow length
    v_path_quivers = {}
    v_pref_quivers = {}
    v_safe_quivers = {}
    for rid, data in agent_data.items():
        color = data['color']
        x0, y0, _ = histories[rid][0]
        # V_path arrow (white, shortest — raw intent)
        v_path_quivers[rid] = ax.quiver(
            x0, y0, 0.01, 0.01, color='white', scale=_ARROW_SCALE,
            width=0.004, alpha=0.9, zorder=7, pivot='mid'
        )
        # V_pref arrow (dodgerblue, medium — APF-blended intent)
        v_pref_quivers[rid] = ax.quiver(
            x0, y0, 0.01, 0.01, color='dodgerblue', scale=_ARROW_SCALE,
            width=0.005, alpha=0.9, zorder=7, pivot='mid'
        )
        # V_safe arrow (limegreen, shortest — ORCA-corrected intent)
        v_safe_quivers[rid] = ax.quiver(
            x0, y0, 0.01, 0.01, color='limegreen', scale=_ARROW_SCALE,
            width=0.006, alpha=0.9, zorder=7, pivot='mid'
        )

    # ── Step counter text ─────────────────────────────────────────────────
    step_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        fontsize=10, verticalalignment='top')

    # ── Update function ───────────────────────────────────────────────────
    wl = L * 0.6          # visual wheel length

    def update(frame):
        artists = [step_text]
        step_text.set_text(f'Step: {frame}')

        for rid in agent_data:
            hist = histories[rid]
            f    = min(frame, len(hist) - 1)
            x, y, theta = hist[f]

            # Trail
            trail_lines[rid].set_data(
                [p[0] for p in hist[:f + 1]],
                [p[1] for p in hist[:f + 1]]
            )

            # Body
            body_patches[rid].center = (x, y)

            # Heading arrow (centre → nose)
            heading_lines[rid].set_data(
                [x, x + r * math.cos(theta)],
                [y, y + r * math.sin(theta)]
            )

        # Wheel positions (perpendicular to heading)
        # Left wheel centre
        lx = x - (L / 2) * math.sin(theta)
        ly = y + (L / 2) * math.cos(theta)
        # Right wheel centre
        rx = x + (L / 2) * math.sin(theta)
        ry = y - (L / 2) * math.cos(theta)

        # Draw each wheel as a short line along heading direction
        left_wheels[rid].set_data(
            [lx - wl * math.cos(theta), lx + wl * math.cos(theta)],
            [ly - wl * math.sin(theta), ly + wl * math.sin(theta)]
        )
        right_wheels[rid].set_data(
            [rx - wl * math.cos(theta), rx + wl * math.cos(theta)],
            [ry - wl * math.sin(theta), ry + wl * math.sin(theta)]
        )

        # ── Trajectory prediction ─────────────────────────────────────────
        ctrl = fleet.robots[rid]
        vel = ctrl.state.vel
        tau_viz = 3.0  # predict 3 seconds ahead
        traj_pts = [(x, y)]
        for t in np.linspace(0, tau_viz, 20):
            traj_pts.append((x + vel[0] * t, y + vel[1] * t))
        traj_lines[rid].set_data([p[0] for p in traj_pts], [p[1] for p in traj_pts])

        # ── Yield indicator ring ───────────────────────────────────────────
        if ctrl.is_yielding:
            yield_rings[rid].center = (x, y)
            yield_rings[rid].set_alpha(0.9)
        else:
            yield_rings[rid].set_alpha(0.0)

        # ── Social bubble patch ─────────────────────────────────────────
        bubble_patches[rid].center = (x, y)

        # ── Velocity arrows (V_path white, V_pref blue, V_safe green) ──
        debug = ctrl.get_debug_info()
        v_path = debug['v_path']
        v_pref = debug['v_pref']
        v_safe = debug['v_safe']
        v_path_quivers[rid].set_UVC(v_path[0], v_path[1])
        v_path_quivers[rid].set_offsets(np.array([[x, y]]))
        v_pref_quivers[rid].set_UVC(v_pref[0], v_pref[1])
        v_pref_quivers[rid].set_offsets(np.array([[x, y]]))
        v_safe_quivers[rid].set_UVC(v_safe[0], v_safe[1])
        v_safe_quivers[rid].set_offsets(np.array([[x, y]]))

        artists.extend([
            trail_lines[rid], body_patches[rid],
            heading_lines[rid], left_wheels[rid], right_wheels[rid],
            traj_lines[rid], yield_rings[rid],
            bubble_patches[rid],
            v_path_quivers[rid], v_pref_quivers[rid], v_safe_quivers[rid],
        ])

        return artists

    return fig, update, max(len(h) for h in histories.values())
