"""
apf.py — Artificial Potential Field with Strategy Pattern
==========================================================

ARCHITECTURE
------------
         ┌─────────────────────┐
         │     APFBase (ABC)   │  ← interface contract
         │  get_force()        │  IntentionBlender calls this
         │  get_repulsive_only()│
         └──────────┬──────────┘
                    │
    ┌───────────────┴────────────────┐
    │                                 │
    │  ClassicalAPF                   │  ImprovedAPF
    │  (original logic)               │  (GNRO fix + local minima)
    │  k_rep / rho^2 only             │  Eq.1-3 (GNRO) + Eq.4-6 (minima)
    └─────────────────────────────────┘
                    │
              ┌─────┴─────┐
              │ APF alias │ ← drop-in backwards-compatible constructor
              └───────────┘

SWAPPING IN simulation.py (zero changes elsewhere):
  Classical: apf = APF(k_att=1.9, k_rep=12.0, ...)
  Improved:  p = ImprovedAPFParams(eta=12.0, n_reg=0.5, ...)
             apf = ImprovedAPF(p, static_circles=[...])

PAPER MATH REFERENCE (Alonso-Mora et al., GNRO formulation)
----------------
Repulsive potential (GNRO fix, Eq.1–3):
  When ρ(X,Xi) ≤ ρ₀:
    F_rep1(X) = η × (1/ρ − 1/ρ₀) × ρ_g^n / ρ²     [Eq.1]
    F_rep2(X) = (n/2) × η × (1/ρ − 1/ρ₀)² × ρ_g^(n-1)  [Eq.2]
    Total = −(F_rep1 + F_rep2) in radial direction  [Eq.3]

  ρ  = surface-to-surface gap (robot edge to obstacle edge)
  ρ_g = distance from robot to GOAL (carrot waypoint)
  η  = repulsion gain (paper uses eta)
  n  = regulation constant (0 < n < 1, typically 0.5)

  Key insight: when far from goal (ρ_g large), repulsion is STRONGER,
  helping robot find path around obstacles. Near goal (ρ_g small),
  repulsion softens so robot can arrive.

Local minima + virtual target (Eq.4–6):
  Stuck if: step_length < β × l                        [Eq.4]
  x_t = x_b + β₁ × sin(n_vt × (x_g − x))               [Eq.5]
  y_t = y_b + β₂ × sin(n_vt × (y_g − y))               [Eq.6]

  (x_b, y_b) = closest obstacle position
  (x_g, y_g) = goal position
  State machine: normal → slow-accumulating → stuck →
                 virtual-target-active → expired → normal
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def closest_point_on_circle(
    robot_pos: np.ndarray,
    circle: Tuple[float, float, float],  # (cx, cy, radius)
) -> np.ndarray:
    cx, cy, r = circle
    centre = np.array([cx, cy])
    direction = robot_pos - centre
    dist = np.linalg.norm(direction)
    if dist < 1e-9:
        return centre + np.array([r, 0.0])
    return centre + (direction / dist) * r


def closest_point_on_rect(
    robot_pos: np.ndarray,
    rect: Tuple[float, float, float, float],  # (x_min, y_min, x_max, y_max)
) -> np.ndarray:
    x_min, y_min, x_max, y_max = rect
    cx = float(np.clip(robot_pos[0], x_min, x_max))
    cy = float(np.clip(robot_pos[1], y_min, y_max))
    return np.array([cx, cy])


# ─────────────────────────────────────────────────────────────────────────────
# OBSTACLE DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class CircleObstacle:
    def __init__(self, cx: float, cy: float, radius: float):
        self.cx, self.cy, self.radius = cx, cy, radius

    def closest_point(self, robot_pos: np.ndarray) -> np.ndarray:
        return closest_point_on_circle(robot_pos, (self.cx, self.cy, self.radius))


class RectObstacle:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.bounds = (x_min, y_min, x_max, y_max)

    def closest_point(self, robot_pos: np.ndarray) -> np.ndarray:
        return closest_point_on_rect(robot_pos, self.bounds)


class DynamicObstacle:
    def __init__(self, x: float, y: float, radius: float = 0.3, priority: float = 1.0):
        self.pos = np.array([x, y], dtype=float)
        self.radius = radius
        self.priority = priority


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassicalAPFParams:
    """Parameters for ClassicalAPF. All existing code uses these fields."""
    k_att: float = 1.0
    k_rep: float = 2.0
    rho_0: float = 1.5
    max_force: float = 5.0
    vortex_gain: float = 0.4
    robot_radius: float = 1.0
    enable_static_repulsion: bool = True


@dataclass
class ImprovedAPFParams:
    """Parameters for ImprovedAPF. Contains all classical params plus paper-specific."""
    # Classical params (for backwards compatibility)
    k_att: float = 1.9
    k_rep: float = 12.0
    rho_0: float = 2.5
    max_force: float = 12.0
    vortex_gain: float = 0.3
    robot_radius: float = 1.0
    enable_static_repulsion: bool = True
    # Paper-specific params (GNRO + local minima)
    eta: float = 12.0        # η — repulsion gain in Eq.1-2 (replaces k_rep for improved formula)
    n_reg: float = 0.5       # n — regulation constant in Eq.1-2 (0 < n < 1)
    step_length: float = 0.20  # l — normal step size [m] for stuck detection (Eq.4)
    beta_stuck: float = 3.0  # β — stuck threshold multiplier in Eq.4
    stuck_window: int = 8    # consecutive slow steps = "stuck"
    beta1: float = 2.0       # β₁ — x-amplitude in virtual target (Eq.5)
    beta2: float = 2.0       # β₂ — y-amplitude in virtual target (Eq.6)
    n_vt: float = 1.0        # oscillation frequency for virtual target (Eq.5-6)
    virtual_target_duration: int = 20  # ticks to chase virtual target before reverting


# ─────────────────────────────────────────────────────────────────────────────
# APF BASE — interface contract
# ─────────────────────────────────────────────────────────────────────────────

class APFBase(ABC):
    """Abstract base that both implementations honour."""

    @abstractmethod
    def get_force(
        self,
        robot_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        dynamic_obstacles: Optional[List[DynamicObstacle]] = None,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def get_repulsive_only(
        self,
        robot_pos: Tuple[float, float],
        dynamic_obstacles: Optional[List[DynamicObstacle]] = None,
        goal_pos: Optional[Tuple[float, float]] = None,
        robot_id: str = 'default',
    ) -> np.ndarray:
        ...

    # ── Shared geometry utilities ─────────────────────────────────────────

    @staticmethod
    def _surface_gap_static(
        robot_pos: np.ndarray,
        surface_pt: np.ndarray,
        robot_radius: float,
    ) -> float:
        """ρ = centre_dist − robot_radius (robot-surface to obstacle-surface gap)."""
        return max(float(np.linalg.norm(robot_pos - surface_pt)) - robot_radius, 0.01)

    @staticmethod
    def _dynamic_gap(
        robot_pos: np.ndarray,
        obs: DynamicObstacle,
        robot_radius: float,
    ) -> Tuple[float, float, np.ndarray]:
        """Returns (centre_dist, surface_gap, radial_dir from obs toward robot)."""
        diff = robot_pos - obs.pos
        centre_dist = float(np.linalg.norm(diff))
        surface_gap = centre_dist - obs.radius - robot_radius
        radial_dir = diff / centre_dist if centre_dist > 1e-9 else np.array([1.0, 0.0])
        return centre_dist, surface_gap, radial_dir

    @staticmethod
    def _emergency_push(
        penetration: float,
        push_dir: np.ndarray,
        max_force: float,
    ) -> np.ndarray:
        """Strong push-out when robot is already inside an obstacle."""
        tangent = np.array([-push_dir[1], push_dir[0]])
        mag = max_force * (1.0 + penetration * 2.0)
        return mag * push_dir + (mag * 0.5) * tangent

    def _static_surface_points(
        self,
        q: np.ndarray,
        static_circles: list,
        static_rects: list,
    ) -> List[np.ndarray]:
        pts = [obs.closest_point(q) for obs in static_circles]
        pts += [obs.closest_point(q) for obs in static_rects]
        return pts

    # ── Runtime mutation ───────────────────────────────────────────────────

    def add_static_circle(self, cx, cy, radius):
        self.static_circles.append(CircleObstacle(cx, cy, radius))

    def add_static_rect(self, x_min, y_min, x_max, y_max):
        self.static_rects.append(RectObstacle(x_min, y_min, x_max, y_max))

    def clear_static_obstacles(self):
        self.static_circles.clear()
        self.static_rects.clear()


# ─────────────────────────────────────────────────────────────────────────────
# CLASSICAL APF — original logic (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalAPF(APFBase):
    """Original APF: repulsion k_rep × (1/ρ − 1/ρ₀) / ρ² — no goal-distance scaling."""

    def __init__(
        self,
        params: ClassicalAPFParams,
        static_circles: Optional[List[CircleObstacle]] = None,
        static_rects: Optional[List[RectObstacle]] = None,
    ):
        self.p = params
        self.static_circles = static_circles or []
        self.static_rects = static_rects or []

    def get_force(
        self,
        robot_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        dynamic_obstacles: Optional[List[DynamicObstacle]] = None,
    ) -> np.ndarray:
        q = np.array(robot_pos, dtype=float)
        goal = np.array(goal_pos, dtype=float)
        F_att = -self.p.k_att * (q - goal)
        F_rep = self._rep_static(q) + self._rep_dynamic(q, dynamic_obstacles or [])
        mag = np.linalg.norm(F_rep)
        if mag > self.p.max_force:
            F_rep *= self.p.max_force / mag
        return F_att + F_rep

    def get_repulsive_only(
        self,
        robot_pos: Tuple[float, float],
        dynamic_obstacles: Optional[List[DynamicObstacle]] = None,
        goal_pos: Optional[Tuple[float, float]] = None,
        robot_id: str = 'default',
    ) -> np.ndarray:
        q = np.array(robot_pos, dtype=float)
        F_rep = np.zeros(2)
        if self.p.enable_static_repulsion:
            F_rep += self._rep_static(q)
        F_rep += self._rep_dynamic(q, dynamic_obstacles or [])
        mag = np.linalg.norm(F_rep)
        if mag > self.p.max_force:
            F_rep *= self.p.max_force / mag
        return F_rep

    def _rep_static(self, q: np.ndarray) -> np.ndarray:
        F = np.zeros(2)
        for pt in self._static_surface_points(q, self.static_circles, self.static_rects):
            rho = self._surface_gap_static(q, pt, self.p.robot_radius)
            if rho < self.p.rho_0:
                mag = self.p.k_rep * (1.0 / rho - 1.0 / self.p.rho_0) / (rho ** 2)
                dirn = (q - pt) / max(np.linalg.norm(q - pt), 1e-9)
                F += mag * dirn
        return F

    def _rep_dynamic(self, q: np.ndarray, obs_list: List[DynamicObstacle]) -> np.ndarray:
        F = np.zeros(2)
        for obs in obs_list:
            _, rho, rd = self._dynamic_gap(q, obs, self.p.robot_radius)
            if rho <= 0.0:
                F += self._emergency_push(abs(rho) + 0.1, rd, self.p.max_force)
                continue
            rho = max(rho, 0.02)
            if rho >= self.p.rho_0:
                continue
            k = self.p.k_rep * obs.priority
            m = k * (1.0 / rho - 1.0 / self.p.rho_0) / (rho ** 2)
            tg = np.array([-rd[1], rd[0]])
            F += m * rd + (m * self.p.vortex_gain) * tg
        return F


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL MINIMA STATE — isolated state machine (Eq.4–6)
# ─────────────────────────────────────────────────────────────────────────────

class LocalMinimaState:
    """
    Per-robot state for stuck detection and virtual target management.

    Isolated here so ImprovedAPF's calculation methods stay stateless and clean.

    State machine transitions:
        normal → (slow steps accumulate) → stuck/virtual-target-active → expired → normal
    """

    def __init__(self):
        self.prev_pos: Optional[np.ndarray] = None
        self.slow_step_count: int = 0
        self.virtual_target: Optional[np.ndarray] = None
        self.virtual_ticks_left: int = 0

    def record_step(self, current_pos: np.ndarray, threshold: float):
        """
        Update step tracker.

        threshold = beta_stuck × step_length (Eq.4).
        If step < threshold: increment slow counter (robot is barely moving).
        """
        if self.prev_pos is None:
            self.prev_pos = current_pos.copy()
            return

        step = float(np.linalg.norm(current_pos - self.prev_pos))
        self.prev_pos = current_pos.copy()

        # Eq.4: step < β × l → increment slow counter
        if step < threshold:
            self.slow_step_count += 1
        else:
            self.slow_step_count = 0  # Moving normally

        # Decay virtual timer if robot is moving
        if self.virtual_ticks_left > 0:
            self.virtual_ticks_left = max(0, self.virtual_ticks_left - 1)
            if self.virtual_ticks_left <= 0:
                self.virtual_target = None

    def is_stuck(self, stuck_window: int) -> bool:
        return self.slow_step_count >= stuck_window

    def activate(
        self,
        robot_pos: np.ndarray,
        real_goal: np.ndarray,
        obs_pos: np.ndarray,
        p: ImprovedAPFParams,
    ):
        """
        Compute virtual target using Eq.5 and Eq.6 and activate it.

        x_t = x_b + β₁ × sin(n_vt × (x_g − x))   [Eq.5]
        y_t = y_b + β₂ × sin(n_vt × (y_g − y))   [Eq.6]
        """
        dx = real_goal[0] - robot_pos[0]
        dy = real_goal[1] - robot_pos[1]
        x_t = obs_pos[0] + p.beta1 * math.sin(p.n_vt * dx)  # Eq.5
        y_t = obs_pos[1] + p.beta2 * math.sin(p.n_vt * dy)  # Eq.6
        self.virtual_target = np.array([x_t, y_t])
        self.virtual_ticks_left = p.virtual_target_duration
        self.slow_step_count = 0  # Reset to prevent immediate re-trigger

    def tick(self) -> Optional[np.ndarray]:
        """Decrement timer. Returns active target, or None if expired."""
        if self.virtual_ticks_left <= 0:
            self.virtual_target = None
            return None
        self.virtual_ticks_left -= 1
        if self.virtual_ticks_left <= 0:
            self.virtual_target = None
            return None
        return self.virtual_target

    @property
    def active(self) -> bool:
        return self.virtual_target is not None and self.virtual_ticks_left > 0


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED APF — GNRO fix + local minima escape (paper upgrades)
# ─────────────────────────────────────────────────────────────────────────────

class ImprovedAPF(APFBase):
    """
    Two upgrades over ClassicalAPF:

    1. GNRO Fix (Eq.1–3)
       Repulsion magnitude is scaled by ρ_g^n (distance to goal).
       Effect: when robot is far from goal, repulsion is STRONGER, helping
       it find a path around obstacles. Near goal, repulsion softens.

    2. Local Minima Escape (Eq.4–6)
       If consecutive step lengths fall below β×l (Eq.4), robot is stuck.
       A virtual target is computed (Eq.5–6) offset from nearest obstacle,
       luring robot out of the trap.

    INTEGRATION NOTE
    ----------------
    IntentionBlender.compute() must pass goal_pos and robot_id:
        F_unknown = self.apf.get_repulsive_only(
            robot_pos, dyn_obs,
            goal_pos=tuple(carrot),   # ← enables GNRO fix
            robot_id=self.robot_id,   # ← per-robot stuck state
        )
    Without goal_pos, ImprovedAPF silently falls back to classical behaviour.
    """

    def __init__(
        self,
        params: ImprovedAPFParams,
        static_circles: Optional[List[CircleObstacle]] = None,
        static_rects: Optional[List[RectObstacle]] = None,
    ):
        self.p = params
        self.static_circles = static_circles or []
        self.static_rects = static_rects or []
        self._states: dict = {}  # robot_id → LocalMinimaState

    # ── Public interface ──────────────────────────────────────────────────

    def get_force(
        self,
        robot_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        dynamic_obstacles: Optional[List[DynamicObstacle]] = None,
    ) -> np.ndarray:
        q = np.array(robot_pos, dtype=float)
        goal = np.array(goal_pos, dtype=float)
        F_att = -self.p.k_att * (q - goal)
        rho_g = max(float(np.linalg.norm(q - goal)), 0.01)
        F_rep = self._rep_static_improved(q, rho_g)
        F_rep += self._rep_dynamic_improved(q, rho_g, dynamic_obstacles or [])
        mag = np.linalg.norm(F_rep)
        if mag > self.p.max_force:
            F_rep *= self.p.max_force / mag
        return F_att + F_rep

    def get_repulsive_only(
        self,
        robot_pos: Tuple[float, float],
        dynamic_obstacles: Optional[List[DynamicObstacle]] = None,
        goal_pos: Optional[Tuple[float, float]] = None,
        robot_id: str = 'default',
    ) -> np.ndarray:
        q = np.array(robot_pos, dtype=float)
        dyn_obs = dynamic_obstacles or []

        # ── Resolve effective goal (handles virtual target logic) ──────────
        active_goal = self._resolve_goal(q, goal_pos, dyn_obs, robot_id)

        # ── Compute repulsion ─────────────────────────────────────────────
        F_rep = np.zeros(2)
        if active_goal is not None:
            # GNRO fix active — rho_g is used in equations
            rho_g = max(float(np.linalg.norm(q - active_goal)), 0.01)
            if self.p.enable_static_repulsion:
                F_rep += self._rep_static_improved(q, rho_g)
            F_rep += self._rep_dynamic_improved(q, rho_g, dyn_obs)
        else:
            # No goal available — classical fallback (no GNRO fix)
            if self.p.enable_static_repulsion:
                F_rep += self._rep_static_classical(q)
            F_rep += self._rep_dynamic_classical(q, dyn_obs)

        mag = np.linalg.norm(F_rep)
        if mag > self.p.max_force:
            F_rep *= self.p.max_force / mag
        return F_rep

    # ── Goal resolution — Eq.4–6 ──────────────────────────────────────────

    def _resolve_goal(
        self,
        q: np.ndarray,
        goal_pos: Optional[Tuple[float, float]],
        dyn_obs: List[DynamicObstacle],
        robot_id: str,
    ) -> Optional[np.ndarray]:
        if goal_pos is None:
            return None

        real_goal = np.array(goal_pos, dtype=float)
        state = self._state(robot_id)

        # Eq.4: record step length and check stuck
        threshold = self.p.beta_stuck * self.p.step_length
        state.record_step(q, threshold)

        # If virtual target still active (from previous stuck event)
        if state.active:
            result = state.tick()
            if result is not None:
                return result  # Keep chasing virtual target

        # If newly stuck, compute and activate virtual target
        if state.is_stuck(self.p.stuck_window):
            obs_pos = self._nearest_obstacle_pos(q, dyn_obs)
            if obs_pos is not None:
                state.activate(q, real_goal, obs_pos, self.p)  # Eq.5, Eq.6
                return state.virtual_target

        return real_goal

    def _state(self, robot_id: str) -> LocalMinimaState:
        if robot_id not in self._states:
            self._states[robot_id] = LocalMinimaState()
        return self._states[robot_id]

    def _nearest_obstacle_pos(
        self,
        q: np.ndarray,
        dyn_obs: List[DynamicObstacle],
    ) -> Optional[np.ndarray]:
        """Find (x_b, y_b): position of nearest obstacle for Eq.5/6."""
        best_d, best_p = float('inf'), None

        for obs in dyn_obs:
            d = float(np.linalg.norm(q - obs.pos)) - obs.radius
            if d < best_d:
                best_d, best_p = d, obs.pos.copy()

        for obs in self.static_circles:
            c = np.array([obs.cx, obs.cy])
            d = float(np.linalg.norm(q - c)) - obs.radius
            if d < best_d:
                best_d, best_p = d, c.copy()

        for obs in self.static_rects:
            pt = obs.closest_point(q)
            d = float(np.linalg.norm(q - pt))
            if d < best_d:
                best_d, best_p = d, pt.copy()

        return best_p

    # ── Core improved repulsion formula — Eq.1–3 ─────────────────────────

    def _improved_force_vector(
        self,
        rho: float,
        rho_g: float,
        radial_dir: np.ndarray,
        k_eff: float,
    ) -> np.ndarray:
        """
        Computes the improved repulsion vector for a single obstacle.

        Line-by-line paper mapping:

        n = self.p.n_reg
        rho_0 = self.p.rho_0
        base = (1/rho - 1/rho_0)

        Eq.1: F_rep1 = η × (1/ρ − 1/ρ₀) × ρ_g^n / ρ²
        Eq.2: F_rep2 = (n/2) × η × (1/ρ − 1/ρ₀)² × ρ_g^(n-1)
        Eq.3: total = -(F_rep1 + F_rep2) — repulsion outward
        """
        n = self.p.n_reg
        rho_0 = self.p.rho_0
        base = (1.0 / rho) - (1.0 / rho_0)

        # Eq.1: F_rep1 = η × (1/ρ − 1/ρ₀) × ρ_g^n / ρ²
        f1 = k_eff * base * (rho_g ** n) / (rho ** 2)

        # Eq.2: F_rep2 = (n/2) × η × (1/ρ − 1/ρ₀)² × ρ_g^(n-1)
        f2 = (n / 2.0) * k_eff * (base ** 2) * (rho_g ** (n - 1.0))

        # Eq.3: scalar = f1 + f2; radial_dir already points FROM obstacle TOWARD robot
        scalar = f1 + f2
        tangent = np.array([-radial_dir[1], radial_dir[0]])
        return scalar * radial_dir + (scalar * self.p.vortex_gain) * tangent

    def _rep_static_improved(self, q: np.ndarray, rho_g: float) -> np.ndarray:
        F = np.zeros(2)
        for pt in self._static_surface_points(q, self.static_circles, self.static_rects):
            rho = self._surface_gap_static(q, pt, self.p.robot_radius)
            if rho >= self.p.rho_0:
                continue
            dirn = (q - pt) / max(float(np.linalg.norm(q - pt)), 1e-9)
            F += self._improved_force_vector(rho, rho_g, dirn, self.p.eta)
        return F

    def _rep_dynamic_improved(
        self,
        q: np.ndarray,
        rho_g: float,
        obs_list: List[DynamicObstacle],
    ) -> np.ndarray:
        F = np.zeros(2)
        for obs in obs_list:
            _, rho, rd = self._dynamic_gap(q, obs, self.p.robot_radius)
            if rho <= 0.0:
                F += self._emergency_push(abs(rho) + 0.1, rd, self.p.max_force)
                continue
            rho = max(rho, 0.02)
            if rho >= self.p.rho_0:
                continue
            F += self._improved_force_vector(rho, rho_g, rd, self.p.eta * obs.priority)
        return F

    # ── Classical fallback (when goal_pos is None) ─────────────────────────

    def _rep_static_classical(self, q: np.ndarray) -> np.ndarray:
        F = np.zeros(2)
        for pt in self._static_surface_points(q, self.static_circles, self.static_rects):
            rho = self._surface_gap_static(q, pt, self.p.robot_radius)
            if rho < self.p.rho_0:
                mag = self.p.k_rep * (1.0 / rho - 1.0 / self.p.rho_0) / (rho ** 2)
                dirn = (q - pt) / max(float(np.linalg.norm(q - pt)), 1e-9)
                F += mag * dirn
        return F

    def _rep_dynamic_classical(
        self,
        q: np.ndarray,
        obs_list: List[DynamicObstacle],
    ) -> np.ndarray:
        F = np.zeros(2)
        for obs in obs_list:
            _, rho, rd = self._dynamic_gap(q, obs, self.p.robot_radius)
            if rho <= 0.0:
                F += self._emergency_push(abs(rho) + 0.1, rd, self.p.max_force)
                continue
            rho = max(rho, 0.02)
            if rho >= self.p.rho_0:
                continue
            k = self.p.k_rep * obs.priority
            m = k * (1.0 / rho - 1.0 / self.p.rho_0) / (rho ** 2)
            tg = np.array([-rd[1], rd[0]])
            F += m * rd + (m * self.p.vortex_gain) * tg
        return F

    # ── Debug queries ──────────────────────────────────────────────────────

    def is_stuck(self, robot_id: str = 'default') -> bool:
        return self._state(robot_id).is_stuck(self.p.stuck_window)

    def get_virtual_target(self, robot_id: str = 'default') -> Optional[np.ndarray]:
        return self._state(robot_id).virtual_target


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARDS-COMPATIBLE ALIAS (existing code needs zero changes)
# ─────────────────────────────────────────────────────────────────────────────

class APF(ClassicalAPF):
    """
    Drop-in alias for ClassicalAPF with the old keyword-argument constructor.

    Existing call (still works):
        apf = APF(k_att=1.9, k_rep=12.0, rho_0=2.5, max_force=12.0,
                  vortex_gain=0.3, robot_radius=cfg.robot_radius,
                  static_circles=INTERNAL_CIRCLES,
                  enable_static_repulsion=True)

    To switch to improved (in simulation.py only):
        from apf import ImprovedAPF, ImprovedAPFParams
        p = ImprovedAPFParams(eta=12.0, n_reg=0.5, step_length=0.2,
                              beta_stuck=3.0, stuck_window=8,
                              beta1=2.0, beta2=2.0, n_vt=1.0,
                              virtual_target_duration=20,
                              robot_radius=cfg.robot_radius, ...)
        apf = ImprovedAPF(p, static_circles=INTERNAL_CIRCLES)
    """

    def __init__(
        self,
        k_att: float = 1.0,
        k_rep: float = 2.0,
        rho_0: float = 1.5,
        max_force: float = 5.0,
        vortex_gain: float = 0.4,
        robot_radius: float = 1.0,
        static_circles: Optional[List[CircleObstacle]] = None,
        static_rects: Optional[List[RectObstacle]] = None,
        enable_static_repulsion: bool = True,
    ):
        super().__init__(
            ClassicalAPFParams(
                k_att=k_att,
                k_rep=k_rep,
                rho_0=rho_0,
                max_force=max_force,
                vortex_gain=vortex_gain,
                robot_radius=robot_radius,
                enable_static_repulsion=enable_static_repulsion,
            ),
            static_circles,
            static_rects,
        )