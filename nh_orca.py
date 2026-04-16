"""
nh_orca.py — Non-Holonomic Optimal Reciprocal Collision Avoidance
=================================================================
Role in pipeline: Phase 5 "The Scalpel"

PURPOSE
-------
Handles COOPERATIVE swarm agents with PERFECT TELEMETRY.
Because all swarm robots are tracked centrally, we know exact (x, y, vx, vy).
This is where ORCA's mathematical guarantee is valid and correct.

ARCHITECTURAL RULE
------------------
NH-ORCA must NEVER see unknown camera blobs (humans, objects).
Those are handled exclusively by APF upstream.

NH vs STANDARD ORCA
--------------------
Standard ORCA: robots can slide in any direction instantly (holonomic).
NH-ORCA: inflates each robot's radius by ε (tracking error), then maps
         the holonomic output to real (v, ω) via paper Eq. 8-9.

Three key additions:
    1. r_inflated = r + ε
    2. P_AHV polygon (reachable velocity polygon)
    3. Kinematic mapping (v, ω) from v_H* in three regions

Reference: Alonso-Mora et al., "Optimal Reciprocal Collision Avoidance
           for Multiple Non-Holonomic Robots", DARS 2013.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ROBOT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiffDriveConfig:
    """
    All physical parameters for ONE differential-drive robot.

    HOW TO FILL IN FOR YOUR ROBOT
    ------------------------------
    Measure with a ruler:
        robot_radius  → centre to outermost edge [m]
        wheel_base    → left to right wheel contact patch [m]
        wheel_radius  → one wheel's radius [m]

    From datasheet / empirical testing:
        max_linear_speed   [m/s]
        max_angular_speed  [rad/s]
        max_wheel_speed    [m/s] per individual wheel

    Tune with experiments:
        tracking_error (ε) → how far NH robot drifts from holonomic path.
                             Start 0.05 m. Decrease for tighter packing.
        orientation_time (T) → time budget to rotate to new heading.
                               Must be >= sim_dt. Start 0.5 s.

    Example — TurtleBot3 Burger:
        robot_radius=0.105, wheel_base=0.160,
        max_linear_speed=0.22, max_angular_speed=2.84
    """
    robot_radius:        float = 0.220
    wheel_base:          float = 0.287
    wheel_radius:        float = 0.033
    max_linear_speed:    float = 0.26
    max_angular_speed:   float = 1.82
    max_wheel_speed:     float = 0.50
    max_linear_accel:    float = 2.0   # [m/s²] acceleration limit
    max_linear_decel:    float = 4.0   # [m/s²] deceleration limit
    max_angular_accel:   float = 6.0   # [rad/s²] angular accel limit
    tracking_error:      float = 0.05   # ε [m]
    orientation_time:    float = 0.50   # T [s]
    time_horizon: float = 5.0 # τ [s] collision lookahead
    neighbor_dist: float = 5.0 # [m] sensing radius
    sim_dt: float = 0.10 # [s] control period
    # Boundary world dimensions [m] — used by ORCA for wall constraints
    world_x_min: float = 0.0
    world_x_max: float = 20.0
    world_y_min: float = 0.0
    world_y_max: float = 20.0
    boundary_buffer: float = 0.5 # minimum gap from wall [m]
    # ── Social bubble (virtual safety margin) ──────────────────────────
    # The "virtual bubble" used in ORCA planning between swarm robots.
    # robot_radius * (1 + social_bubble_factor) = the clearance ORCA enforces.
    # 0.20 = 20% extra — keeps robots 20% further apart than physical edge.
    # This is SEPARATE from tracking_error (which is for NH math correctness).
    # Total ORCA planning radius = robot_radius × (1 + factor) + tracking_error
    social_bubble_factor: float = 0.20

    @property
    def inflated_radius(self) -> float:
        """r + ε — radius used inside NHORCAPlanner's own LP (kinematic correctness)."""
        return self.robot_radius + self.tracking_error

    @property
    def social_radius(self) -> float:
        """
        Radius passed in SwarmTelemetry to other robots' ORCA planners.
        = physical radius × (1 + social_bubble_factor) + tracking_error

        Example (robot_radius=1.0, factor=0.20, tracking_error=0.5):
        social_radius = 1.0 × 1.20 + 0.5 = 1.70m

        Combined radius between two robots = 2 × 1.70 = 3.40m.
        ORCA will keep robot centres at least 3.40m apart.

        Note: inflated_radius (r + ε = 1.5m) is still used in NH kinematic math.
        social_radius (1.7m) is used in ORCA planning for clearance.
        """
        return self.robot_radius * (1.0 + self.social_bubble_factor) + self.tracking_error


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NH KINEMATIC MAPPER  (Paper Eq. 8, 9, 13)
# ─────────────────────────────────────────────────────────────────────────────

class NHKinematicMapper:
    """
    Maps safe holonomic velocity v_H* = (Vx, Vy) → (v [m/s], ω [rad/s]).

    THREE REGIONS
    -------------
    R_A1 — Normal: small heading error, within angular speed limit.
            ω = θ_err / T,  v = optimal from Eq. (8).

    R_A2 — Tight turn while moving: ω must be clamped to ω_max.
            ω = ±ω_max,  v = reduced to keep wheel speeds feasible.

    R_B  — Stop and spin: error so large that driving forward would
            exceed ε tracking budget.
            v = 0,  ω = ±ω_max.  Robot spins until facing right direction.
    """

    def __init__(self, cfg: DiffDriveConfig):
        self.cfg = cfg
        self._last_region = 'R_A1'   # diagnostic

    def holonomic_to_controls(
        self,
        vx: float,
        vy: float,
        theta: float          # current robot heading [rad]
    ) -> Tuple[float, float]: # (v, ω)
        """Convert safe holonomic velocity (vx, vy) to (v, ω)."""
        V_H = math.hypot(vx, vy)
        if V_H < 1e-4:
            return 0.0, 0.0
        theta_H   = math.atan2(vy, vx)
        theta_err = _wrap_angle(theta_H - theta)
        return self._region_rules(V_H, theta_err)

    def max_holonomic_speed(self, theta_err: float) -> float:
        """
        Theorem 2: max holonomic speed for heading error theta_err
        such that tracking error stays ≤ ε.  Used to build P_AHV.
        """
        cfg   = self.cfg
        eps   = cfg.tracking_error
        T     = cfg.orientation_time
        w_max = cfg.max_angular_speed
        v_max = cfg.max_linear_speed

        if abs(theta_err) < 1e-6:
            return v_max

        w_needed = theta_err / T

        if abs(w_needed) <= w_max:
            # Region R_A1 — Eq. (13) first case
            costh  = math.cos(theta_err)
            sinth  = math.sin(theta_err)
            factor = 2.0 * (1.0 - costh) - sinth ** 2
            V_Hmax = v_max if factor <= 1e-9 else \
                     (eps / T) * math.sqrt(2.0 * (1.0 - costh) / factor)
            v_lim  = max(0.0, cfg.max_wheel_speed - abs(w_needed) * cfg.wheel_base / 2.0)
            return min(V_Hmax, v_lim, v_max)
        else:
            # Region R_B — ε = V_H * θ / ω_max → V_H^max = ε * ω_max / |θ|
            return min(eps * w_max / abs(theta_err), v_max)

    def _region_rules(self, V_H: float, theta_err: float) -> Tuple[float, float]:
        cfg   = self.cfg
        T     = cfg.orientation_time
        w_max = cfg.max_angular_speed
        v_max = cfg.max_linear_speed

        def v_lim_for_omega(w: float) -> float:
            return max(0.0, cfg.max_wheel_speed - abs(w) * cfg.wheel_base / 2.0)

        w_needed = theta_err / T

        if abs(w_needed) <= w_max:
            omega = w_needed
            v     = min(_optimal_v(V_H, theta_err), v_lim_for_omega(omega), v_max)
            v     = max(v, 0.0)
            self._last_region = 'R_A1'
        else:
            omega = math.copysign(w_max, theta_err)
            v_opt = _optimal_v(V_H, theta_err)
            if v_opt <= v_lim_for_omega(omega):
                v = v_opt
                self._last_region = 'R_A2'
            else:
                # Never stop completely - maintain minimum forward speed while turning
                v = v_lim_for_omega(omega) * 0.3  # 30% of max possible speed
                self._last_region = 'R_A2'  # Always stay in tight-turn region

        return (float(np.clip(v, 0.0, v_max)),
                float(np.clip(omega, -w_max, w_max)))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  P_AHV — SET OF ALLOWED HOLONOMIC VELOCITIES  (Paper Section 4.2)
# ─────────────────────────────────────────────────────────────────────────────

class AllowedHolonomicVelocities:
    """
    Convex polygon of holonomic velocities the robot can actually track
    with tracking error ≤ ε. Replaces the simple circular speed disc in ORCA.

    Built by sampling V_H^max in N directions, rotated at runtime to align
    with the robot's current heading.
    """

    def __init__(self, cfg: DiffDriveConfig, n_samples: int = 36):
        self.cfg    = cfg
        self.mapper = NHKinematicMapper(cfg)
        self.n      = n_samples
        self._base  = self._build()

    def get_polygon(self, theta: float) -> np.ndarray:
        """Return P_AHV polygon (n×2) rotated to current heading."""
        c, s = math.cos(theta), math.sin(theta)
        R    = np.array([[c, -s], [s, c]])
        return (R @ self._base.T).T

    def clip_velocity(
        self, vx: float, vy: float, theta: float
    ) -> Tuple[float, float]:
        """Scale velocity toward origin until it is inside P_AHV."""
        poly = self.get_polygon(theta)
        pt   = np.array([vx, vy])
        if _point_in_convex_polygon(pt, poly):
            return vx, vy
        # Binary search: find largest scale s.t. s*pt is inside polygon
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = (lo + hi) / 2.0
            if _point_in_convex_polygon(mid * pt, poly):
                lo = mid
            else:
                hi = mid
        result = lo * pt
        return float(result[0]), float(result[1])

    def _build(self) -> np.ndarray:
        pts = []
        for i in range(self.n):
            theta_H = 2.0 * math.pi * i / self.n
            V_max   = self.mapper.max_holonomic_speed(theta_H)
            pts.append([V_max * math.cos(theta_H), V_max * math.sin(theta_H)])
        return np.array(pts)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VELOCITY OBSTACLE + ORCA HALF-PLANE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SwarmAgent:
    """Telemetry for one swarm robot (all values perfectly known)."""
    pos:       np.ndarray   # [x, y]
    vel:       np.ndarray   # [vx, vy]
    radius:    float        # inflated radius (r + ε)
    max_speed: float


@dataclass
class HalfPlane:
    """
    Safe constraint in velocity space.
    Robot must pick v with: (v - point) · normal ≥ 0
    Normal points toward the safe side.
    """
    point:  np.ndarray
    normal: np.ndarray


def compute_orca_halfplane(
    ego:   SwarmAgent,
    other: SwarmAgent,
    tau:   float,
    c:     float = 0.5      # 0.5=equal responsibility, 1.0=ego avoids all
) -> Optional[HalfPlane]:
    """
    Build one ORCA half-plane for ego to avoid other.

    Steps:
    1. Compute relative velocity  v_rel = v_ego - v_other
    2. Check if v_rel is inside the VO cone (collision predicted within τ)
    3. Find u: smallest push to escape the VO
    4. Ego takes c*u → half-plane reference point = v_ego + c*u
    5. Return the safe half-plane

    c = 0.5 for swarm-to-swarm (both share responsibility equally)
    c = 1.0 for ego vs static obstacle (ego avoids entirely)
    """
    rel_pos  = other.pos - ego.pos
    rel_vel  = ego.vel   - other.vel
    dist     = np.linalg.norm(rel_pos)
    comb_rad = ego.radius + other.radius

    # ── Already overlapping: emergency push-out ───────────────────────────
    if dist < comb_rad:
        n = (-rel_pos / dist) if dist > 1e-6 else np.array([1.0, 0.0])
        u = (comb_rad / max(dist, 1e-6) + 1.0 / tau) * n - rel_vel
        return HalfPlane(point=ego.vel + c * u, normal=n)

    leg_len = math.sqrt(max(dist ** 2 - comb_rad ** 2, 0.0)) or 1e-9
    sin_a   = comb_rad / dist
    cos_a   = leg_len  / dist
    apex    = rel_pos  / dist

    left_leg  = np.array([ cos_a * apex[0] - sin_a * apex[1],
                            sin_a * apex[0] + cos_a * apex[1]])
    right_leg = np.array([ cos_a * apex[0] + sin_a * apex[1],
                           -sin_a * apex[0] + cos_a * apex[1]])

    # Is rel_vel inside the VO cone?
    in_left  = np.cross(left_leg,  rel_vel - apex * dist / tau) <= 0
    in_right = np.cross(right_leg, rel_vel - apex * dist / tau) >= 0
    if not (in_left and in_right):
        return None   # no collision predicted — no constraint needed

    # Find nearest escape
    trunc_centre = rel_pos / tau
    trunc_w      = rel_vel - trunc_centre
    trunc_dist   = np.linalg.norm(trunc_w)
    trunc_rad    = comb_rad / tau

    if trunc_dist < trunc_rad:
        n = (trunc_w / trunc_dist) if trunc_dist > 1e-9 else np.array([1.0, 0.0])
        u = (trunc_rad - trunc_dist) * n
    else:
        t_l    = np.dot(rel_vel, left_leg)
        dist_l = np.linalg.norm(rel_vel - t_l * left_leg)
        t_r    = np.dot(rel_vel, right_leg)
        dist_r = np.linalg.norm(rel_vel - t_r * right_leg)

        if dist_l <= dist_r:
            n = np.array([-left_leg[1],  left_leg[0]])
            u = dist_l * n
        else:
            n = np.array([ right_leg[1], -right_leg[0]])
            u = dist_r * n

    n_mag = np.linalg.norm(n)
    if n_mag > 1e-9:
        n = n / n_mag

    return HalfPlane(point=ego.vel + c * u, normal=n)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LINEAR PROGRAMMING SOLVER
# ─────────────────────────────────────────────────────────────────────────────

def solve_lp(
    halfplanes:   List[HalfPlane],
    pref_vel:     np.ndarray,
    max_speed:    float,
    pahv_polygon: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Find velocity v* closest to pref_vel satisfying all ORCA half-planes,
    speed disc, and P_AHV polygon.

    Uses incremental 2-D LP (O(n) expected time).
    Falls back to 3-D LP if no feasible solution exists (very crowded).
    """
    if _feasible(pref_vel, halfplanes, max_speed, pahv_polygon):
        return pref_vel.copy()

    current = pref_vel.copy()

    for i, hp in enumerate(halfplanes):
        if np.dot(current - hp.point, hp.normal) >= -1e-8:
            continue
        line_dir   = np.array([-hp.normal[1], hp.normal[0]])
        candidates = _intersect_line_disc(hp.point, line_dir, max_speed)
        best_v, best_d = None, math.inf
        for cand in candidates:
            if _feasible(cand, halfplanes[:i], max_speed, pahv_polygon):
                d = float(np.linalg.norm(cand - pref_vel))
                if d < best_d:
                    best_d, best_v = d, cand
        if best_v is not None:
            current = best_v

    if _feasible(current, halfplanes, max_speed, pahv_polygon):
        return current

    return _fallback_lp(halfplanes, max_speed)


def _fallback_lp(halfplanes: List[HalfPlane], max_speed: float) -> np.ndarray:
    """Crowded fallback: minimise maximum constraint violation (grid search)."""
    best_v, best_d = np.zeros(2), math.inf
    N = 24
    for i in range(N + 1):
        for j in range(N + 1):
            vx = max_speed * (2 * i / N - 1)
            vy = max_speed * (2 * j / N - 1)
            if vx ** 2 + vy ** 2 > max_speed ** 2:
                continue
            v     = np.array([vx, vy])
            worst = max((float(np.dot(hp.point - v, hp.normal)) for hp in halfplanes),
                        default=0.0)
            if worst < best_d:
                best_d, best_v = worst, v.copy()
    return best_v


# ─────────────────────────────────────────────────────────────────────────────
# 6.  NH-ORCA PLANNER — main per-robot algorithm
# ─────────────────────────────────────────────────────────────────────────────

class NHORCAPlanner:
    """
    Per-robot NH-ORCA planner.

    USAGE
    -----
        cfg     = DiffDriveConfig(robot_radius=0.22, ...)
        planner = NHORCAPlanner(cfg)

        # Every control tick:
        v, omega = planner.update(
            my_pos    = (x, y),
            my_vel    = (vx, vy),
            my_yaw    = yaw,
            pref_vel  = (px, py),    # V_pref from intention blender
            neighbors = [SwarmAgent(...), ...]
        )
        # → Twist(linear.x=v, angular.z=omega)

    SWARM AGENTS ONLY — no camera blobs here.
    """

    def __init__(self, cfg: DiffDriveConfig):
        self.cfg    = cfg
        self.mapper = NHKinematicMapper(cfg)
        self.pahv   = AllowedHolonomicVelocities(cfg)
        # Diagnostic slots (populated each update() call)
        self._last_n_halfplanes = 0
        self._last_pref_vel     = np.zeros(2)
        self._last_safe_vel     = np.zeros(2)
        self._last_pref_clipped = np.zeros(2)

    def update(
        self,
        my_pos:    Tuple[float, float],
        my_vel:    Tuple[float, float],
        my_yaw:    float,
        pref_vel:  Tuple[float, float],
        neighbors: List[SwarmAgent]
    ) -> Tuple[float, float]:
        cfg = self.cfg
        ego = SwarmAgent(
            pos       = np.array(my_pos,  dtype=float),
            vel       = np.array(my_vel,  dtype=float),
            radius    = cfg.inflated_radius,
            max_speed = cfg.max_linear_speed
        )
        pref = np.array(pref_vel, dtype=float)

        # Clip magnitude to max speed only (don't crush direction!)
        spd = np.linalg.norm(pref)
        if spd > cfg.max_linear_speed:
            pref *= cfg.max_linear_speed / spd

        # NOTE: Removed pre-clip to P_AHV - it was incorrectly crushing velocity to near-zero
        # The ORCA LP will find a valid velocity; P_AHV is just for post-clip safety check

# ── Step 2: Build ORCA half-planes (swarm only) ───────────────────────
        halfplanes: List[HalfPlane] = []
        for nb in neighbors:
            if np.linalg.norm(nb.pos - ego.pos) > cfg.neighbor_dist:
                continue
            hp = compute_orca_halfplane(ego, nb, cfg.time_horizon, c=0.5)
            if hp is not None:
                halfplanes.append(hp)

        # ── Boundary wall constraints ─────────────────────────────────────
        # Each wall adds a half-plane when robot is within PROX metres of it.
        # The half-plane pushes velocity toward the safe side (inside the map).
        x_min = cfg.world_x_min + cfg.boundary_buffer
        x_max = cfg.world_x_max - cfg.boundary_buffer
        y_min = cfg.world_y_min + cfg.boundary_buffer
        y_max = cfg.world_y_max - cfg.boundary_buffer
        PROX = 0.5  # [m] — only activate constraint when this close to wall

        # Left wall: normal points RIGHT (safe side is to the right / inside)
        if ego.pos[0] < x_min + PROX:
            n = np.array([1.0, 0.0])
            # point = velocity needed to maintain x >= x_min within time_horizon
            point = np.array([(x_min - ego.pos[0]) / cfg.time_horizon, 0.0])
            halfplanes.append(HalfPlane(point=point, normal=n))

        # Right wall: normal points LEFT (safe side is to the left / inside)
        if ego.pos[0] > x_max - PROX:
            n = np.array([-1.0, 0.0])
            point = np.array([(x_max - ego.pos[0]) / cfg.time_horizon, 0.0])
            halfplanes.append(HalfPlane(point=point, normal=n))

        # Bottom wall: normal points UP (safe side is upward / inside)
        if ego.pos[1] < y_min + PROX:
            n = np.array([0.0, 1.0])
            point = np.array([0.0, (y_min - ego.pos[1]) / cfg.time_horizon])
            halfplanes.append(HalfPlane(point=point, normal=n))

        # Top wall: normal points DOWN (safe side is downward / inside)
        if ego.pos[1] > y_max - PROX:
            n = np.array([0.0, -1.0])
            point = np.array([0.0, (y_max - ego.pos[1]) / cfg.time_horizon])
            halfplanes.append(HalfPlane(point=point, normal=n))

        # ── Step 3: LP — half-planes + speed disc only (NO P_AHV) ────────────
        # P_AHV is NOT a hard LP constraint.
        # The old pre-clip was incorrectly crushing velocity, removed for debugging
        v_safe = solve_lp(halfplanes, pref, cfg.max_linear_speed, pahv_polygon=None)

        # NOTE: Removed post-clip to P_AHV - it was also incorrectly crushing velocity

        # ── Diagnostics (read by PipelineLogger) ─────────────────────────
        self._last_n_halfplanes   = len(halfplanes)
        self._last_pref_vel       = pref.copy()
        self._last_safe_vel       = v_safe.copy()
        self._last_pref_clipped   = pref.copy()   # after pre-clip

        return float(v_safe[0]), float(v_safe[1])


# ─────────────────────────────────────────────────────────────────────────────
# 7.  WHEEL SPEED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def unicycle_to_wheels(
    v: float, omega: float,
    wheel_base: float, max_wheel_speed: float
) -> Tuple[float, float]:
    """
    Unicycle → diff-drive inverse kinematics.
        vl = v - ω * L/2
        vr = v + ω * L/2
    """
    L  = wheel_base
    vl = float(np.clip(v - omega * L / 2.0, -max_wheel_speed, max_wheel_speed))
    vr = float(np.clip(v + omega * L / 2.0, -max_wheel_speed, max_wheel_speed))
    return vl, vr


def wheels_to_unicycle(vl: float, vr: float, wheel_base: float) -> Tuple[float, float]:
    """Diff-drive forward kinematics → (v, ω)."""
    return (vl + vr) / 2.0, (vr - vl) / wheel_base


# ─────────────────────────────────────────────────────────────────────────────
# 8.  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _optimal_v(V_H: float, theta_err: float) -> float:
    """Paper Eq. (8): v* = V_H * θ * sin(θ) / (2*(1-cos(θ)))"""
    if abs(theta_err) < 1e-4:
        return V_H
    c = 1.0 - math.cos(theta_err)
    return V_H if abs(c) < 1e-10 else V_H * theta_err * math.sin(theta_err) / (2.0 * c)


def _feasible(
    v: np.ndarray,
    halfplanes: List[HalfPlane],
    max_speed: float,
    polygon: Optional[np.ndarray]
) -> bool:
    if np.dot(v, v) > max_speed ** 2 + 1e-8:
        return False
    if polygon is not None and not _point_in_convex_polygon(v, polygon):
        return False
    return all(np.dot(v - hp.point, hp.normal) >= -1e-8 for hp in halfplanes)


def _intersect_line_disc(
    point: np.ndarray, direction: np.ndarray, radius: float
) -> List[np.ndarray]:
    d    = direction / (np.linalg.norm(direction) + 1e-12)
    b    = 2.0 * float(np.dot(point, d))
    cc   = float(np.dot(point, point)) - radius ** 2
    disc = b * b - 4.0 * cc
    if disc < 0:
        return []
    sq = math.sqrt(disc)
    return [point + ((-b - sq) / 2.0) * d,
            point + ((-b + sq) / 2.0) * d]


def _point_in_convex_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e    = b - a
        if e[0] * (pt[1] - a[1]) - e[1] * (pt[0] - a[0]) < -1e-8:
            return False
    return True
