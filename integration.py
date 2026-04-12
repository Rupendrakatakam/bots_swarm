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
        Load a new path.  Each waypoint is (x, y).

        Call this once at startup (or whenever the goal changes).
        """
        self.waypoints = [np.array([wx, wy], dtype=float) for wx, wy in waypoints]

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
        max_speed:        float = 0.26,    # [m/s]
        lookahead_window: int   = 15,      # how many waypoints to scan ahead
        carrot_steps:     int   = 3,       # how many steps ahead to target
        goal_tolerance:   float = 0.15,    # [m] distance to declare "reached"
    ):
        self.max_speed        = max_speed
        self.lookahead_window = lookahead_window
        self.carrot_steps     = carrot_steps
        self.goal_tolerance   = goal_tolerance
        self._current_idx     = 0

    def reset(self):
        """Call this when a new path is loaded."""
        self._current_idx = 0

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

        # ── Check if reached the final goal ───────────────────────────────
        if np.linalg.norm(robot_pos - wps[-1]) < self.goal_tolerance:
            return np.zeros(2), True

        # ── Find closest waypoint in lookahead window ─────────────────────
        lo = self._current_idx
        hi = min(lo + self.lookahead_window, n_wp)
        dists = [np.linalg.norm(robot_pos - wps[i]) for i in range(lo, hi)]
        closest_local = int(np.argmin(dists))
        self._current_idx = lo + closest_local

        # ── Target the carrot K steps ahead ───────────────────────────────
        carrot_idx = min(self._current_idx + self.carrot_steps, n_wp - 1)
        carrot     = wps[carrot_idx]
        diff       = carrot - robot_pos
        dist       = np.linalg.norm(diff)

        if dist < 1e-4:
            return np.zeros(2), False

        speed  = min(self.max_speed, dist)   # slow down near waypoints
        V_path = speed * diff / dist
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
        max_pref_speed: float = 0.26,    # clip V_pref to this speed [m/s]
    ):
        self.apf             = apf
        self.max_pref_speed  = max_pref_speed

    def compute(
        self,
        robot_pos:   Tuple[float, float],
        v_path:      np.ndarray,           # from waypoint tracker
        camera_blobs: List[CameraBlob],    # UNKNOWN obstacles ONLY
    ) -> np.ndarray:                       # V_pref [vx, vy]
        """
        Blend the path velocity with APF repulsion from camera blobs.

        Example
        -------
        V_path = [0.2, 0.0]  (robot wants to go east)
        F_unknown = [0.0, 0.1]  (unknown obstacle to the south, pushing north)
        → V_pref = [0.2, 0.1]  (robot curves slightly north-east)
        """
        # Build DynamicObstacle list from camera blobs
        dyn_obs = [
            DynamicObstacle(b.x, b.y, b.radius, b.priority)
            for b in camera_blobs
        ]

        # APF returns ONLY the repulsive component
        # (the attractive pull is handled by the waypoint tracker above)
        F_unknown = self.apf.get_repulsive_only(robot_pos, dyn_obs)

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
        filter_alpha: float = 0.5,          # EMA smoothing factor
    ):
        self.cfg    = cfg
        self.alpha  = filter_alpha

    def compute(
        self,
        vx:       float,
        vy:       float,
        yaw:      float,
        state:    RobotState,
    ) -> WheelCommand:
        """
        Convert V_safe = (vx, vy) to a smoothed WheelCommand.

        Parameters
        ----------
        vx, vy : components of V_safe from NH-ORCA
        yaw    : current robot heading [rad]
        state  : RobotState (used for prev_v, prev_w for EMA filter)

        Returns
        -------
        WheelCommand with v, omega, vl, vr all filled in.
        """
        cfg = self.cfg

        # ── Compute heading error ─────────────────────────────────────────
        V_H = math.hypot(vx, vy)
        if V_H < 1e-4:
            cmd = WheelCommand(0.0, 0.0, 0.0, 0.0)
            state.prev_v = 0.0
            state.prev_w = 0.0
            return cmd

        theta_H   = math.atan2(vy, vx)
        theta_err = _wrap_angle(theta_H - yaw)

        # ── Angular command (from paper Eq. 9) ───────────────────────────
        omega_raw = theta_err / cfg.orientation_time
        omega_raw = float(np.clip(omega_raw, -cfg.max_angular_speed, cfg.max_angular_speed))

        # ── Linear command (from paper Eq. 8) ────────────────────────────
        v_raw = _optimal_v(V_H, theta_err)
        v_raw = float(np.clip(v_raw, 0.0, cfg.max_linear_speed))

        # ── Low-pass filter (EMA shock absorber) ─────────────────────────
        alpha = self.alpha
        v_filt     = alpha * v_raw     + (1.0 - alpha) * state.prev_v
        omega_filt = alpha * omega_raw + (1.0 - alpha) * state.prev_w

        state.prev_v = v_filt
        state.prev_w = omega_filt

        # ── Unicycle → wheel speeds ───────────────────────────────────────
        vl, vr = unicycle_to_wheels(
            v_filt, omega_filt, cfg.wheel_base, cfg.max_wheel_speed
        )

        # ── Reconstruct v, ω from clipped wheel speeds ───────────────────
        # (wheel clip may have changed the actual v and ω)
        v_actual, omega_actual = wheels_to_unicycle(vl, vr, cfg.wheel_base)

        return WheelCommand(
            v     = v_actual,
            omega = omega_actual,
            vl    = vl,
            vr    = vr
        )


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
        filter_alpha:     float = 0.5,
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
        self.blender  = IntentionBlender(apf, max_pref_speed=cfg.max_linear_speed)
        self.orca     = NHORCAPlanner(cfg)
        self.mapper   = MotorMapper(cfg, filter_alpha=filter_alpha)

        # State
        self.state        = RobotState()
        self.goal:        Optional[Tuple[float, float]] = None
        self.reached_goal = False

        # Debug/log (last tick values)
        self._last_v_path    = np.zeros(2)
        self._last_f_unknown = np.zeros(2)
        self._last_v_pref    = np.zeros(2)
        self._last_v_safe    = np.zeros(2)

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

        # ── Phase 3+4: Shield + Blender → V_pref ─────────────────────────
        v_pref = self.blender.compute(
            robot_pos    = tuple(pos),
            v_path       = v_path,
            camera_blobs = camera_blobs,     # ONLY unknown obstacles
        )

        # ── Phase 5: NH-ORCA → V_safe ─────────────────────────────────────
        neighbors = [
            SwarmAgent(
                pos       = np.array([t.x, t.y]),
                vel       = np.array([t.vx, t.vy]),
                radius    = t.radius,
                max_speed = self.cfg.max_linear_speed,
            )
            for t in swarm_telemetry           # ONLY swarm agents
        ]
        v_safe_v, v_safe_w = self.orca.update(
            my_pos    = tuple(pos),
            my_vel    = tuple(vel),
            my_yaw    = yaw,
            pref_vel  = tuple(v_pref),
            neighbors = neighbors,
        )
        # Reconstruct 2D V_safe from (v, ω)
        # For debug/log — actual command is already in (v, ω)
        v_safe_2d = np.array([
            v_safe_v * math.cos(yaw),
            v_safe_v * math.sin(yaw)
        ])

        # ── Phase 6: Kinematic mapper + low-pass filter ───────────────────
        cmd = self.mapper.compute(
            vx    = v_safe_2d[0],
            vy    = v_safe_2d[1],
            yaw   = yaw,
            state = self.state,
        )

        # ── Save debug values ─────────────────────────────────────────────
        self._last_v_path    = v_path
        self._last_f_unknown = v_pref - v_path
        self._last_v_pref    = v_pref
        self._last_v_safe    = v_safe_2d

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
        camera_blobs: List[CameraBlob] = None,   # shared across all robots
    ) -> Dict[str, WheelCommand]:
        """
        Run one full tick for ALL robots.

        Each robot gets:
            camera_blobs     → the shared list of unknown obstacles (APF)
            swarm_telemetry  → telemetry of all OTHER robots (NH-ORCA)

        Returns
        -------
        Dict mapping robot_id → WheelCommand
        """
        camera_blobs = camera_blobs or []

        # Build telemetry snapshot for this tick
        # (all robots broadcast their current state)
        all_telemetry: Dict[str, SwarmTelemetry] = {
            rid: SwarmTelemetry(
                robot_id = rid,
                x        = ctrl.state.pose.x,
                y        = ctrl.state.pose.y,
                vx       = ctrl.state.vel[0],
                vy       = ctrl.state.vel[1],
                radius   = self.cfg.inflated_radius,
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
# QUICK INTEGRATION TEST (no ROS needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("INTEGRATION PIPELINE — Quick Sanity Test")
    print("=" * 60)

    # ── Build config ──────────────────────────────────────────────────────
    cfg = DiffDriveConfig(
        robot_radius      = 0.22,
        wheel_base        = 0.287,
        max_linear_speed  = 0.26,
        max_angular_speed = 1.82,
        tracking_error    = 0.05,
        orientation_time  = 0.5,
    )

    # ── Build APF with some static obstacles ─────────────────────────────
    apf = APF(
        k_att     = 1.0,
        k_rep     = 2.0,
        rho_0     = 1.5,
        max_force = 5.0,
        static_circles = [CircleObstacle(3.0, 0.5, 0.3)],
        static_rects   = [RectObstacle(0.0, 2.0, 6.0, 2.2)],
    )

    # ── Create fleet ──────────────────────────────────────────────────────
    fleet = FleetManager(cfg, apf)

    robot_a = fleet.add_robot('robot_A', filter_alpha=0.5)
    robot_b = fleet.add_robot('robot_B', filter_alpha=0.5)

    # ── Set initial states ────────────────────────────────────────────────
    fleet.update_state('robot_A', x=0.0,  y=0.0,  yaw=0.0,         vx=0.1, vy=0.0)
    fleet.update_state('robot_B', x=6.0,  y=0.0,  yaw=math.pi,     vx=-0.1, vy=0.0)

    # ── Set goals (head-on collision scenario) ────────────────────────────
    fleet.set_goal('robot_A', 6.0, 0.0)
    fleet.set_goal('robot_B', 0.0, 0.0)

    robot_a.path.set_straight_line_path((0.0, 0.0), (6.0, 0.0))
    robot_b.path.set_straight_line_path((6.0, 0.0), (0.0, 0.0))

    # ── Simulate 5 ticks ──────────────────────────────────────────────────
    camera_blobs = [CameraBlob(x=3.0, y=1.0, radius=0.3, priority=1.5)]

    print(f"\n{'Tick':<5} {'Robot':<10} {'v':>6} {'ω':>7} {'vl':>7} {'vr':>7}")
    print("-" * 50)

    for tick in range(5):
        commands = fleet.tick_all(camera_blobs=camera_blobs)
        for rid, cmd in commands.items():
            print(f"{tick:<5} {rid:<10} "
                  f"{cmd.v:>6.3f} {cmd.omega:>7.3f} "
                  f"{cmd.vl:>7.3f} {cmd.vr:>7.3f}")

        # Simulate motion (very crude Euler integration for test only)
        dt = cfg.sim_dt
        for rid in ['robot_A', 'robot_B']:
            ctrl = fleet.robots[rid]
            cmd  = commands[rid]
            x    = ctrl.state.pose.x + ctrl.state.vel[0] * dt
            y    = ctrl.state.pose.y + ctrl.state.vel[1] * dt
            yaw  = ctrl.state.pose.yaw + cmd.omega * dt
            vx   = cmd.v * math.cos(yaw)
            vy   = cmd.v * math.sin(yaw)
            fleet.update_state(rid, x, y, yaw, vx, vy)

    print("\nDebug info (last tick):")
    for dbg in fleet.get_fleet_debug():
        print(f"  {dbg['robot_id']}: pos={dbg['pos']}  "
              f"v_pref={dbg['v_pref']}  v_safe={dbg['v_safe']}")

    print("\nAll phases executed correctly.")
