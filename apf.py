
"""
Artificial Potential Field (APF) - Local Path Planning
F_att = -k_att * (q - goal)
F_rep = k_rep * (1/rho - 1/rho_0) * (1/rho^2) * (grad(rho))
F_total = F_att + F_rep

adaptive repulsion :
U_rep = 1/2 * eta *((1/(rho(x,x_obs)) - 1/rho)) * rho^n (x,x_obs) , where rho(x,x_obs) <= rho_0
else 0 , rho(x,x_obs) > rho_0
where eta = repulsion gain
n = regulation const
rho(x,x_obs) = distance between robot and obstacle

F_rep = -grad(U_rep)

grad(U_rep) = -(F_rep1 + F_rep2) , where rho(x,x_obs) <= rho_0
else 0 , rho(x,x_obs) > rho_0

where F_rep1 = eta * (1/rho(x,x_obs) - 1/rho_0) * (rho^n(x,x_goal)/rho(x,x_i)^2) * (grad(rho(x,x_obs)))


Role in pipeline: Phase 3 "The Shield"
 
PURPOSE
-------
Handles UNKNOWN, UNCOOPERATIVE dynamic obstacles detected by the overhead
camera (humans, rolling objects, dropped boxes). These entities have no
telemetry — we cannot reliably estimate their velocity — so ORCA cannot
be used against them.
 
APF solves this by being VELOCITY-AGNOSTIC:
    It only cares about POSITION (x, y).
    As long as we know WHERE something is, we can push away from it.
    This makes it naturally robust to camera noise.
 
ARCHITECTURAL RULE (enforced here)
------------------------------------
APF must NEVER see the swarm agents.
Swarm agents are handled exclusively by NH-ORCA downstream.
This separation is what makes the whole pipeline work without deadlocks.
 
PIPELINE POSITION
-----------------
    Waypoint Tracker  →  V_path
    Camera blobs      →  APF  →  F_unknown
                                     ↓
                            Intention Blender  →  V_pref  →  NH-ORCA

"""

import math
import numpy as np
from typing import List, Tuple, Optional
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS  (self-contained — no external geometry module needed)
# ─────────────────────────────────────────────────────────────────────────────
 
def closest_point_on_circle(
    robot_pos: np.ndarray,
    circle: Tuple[float, float, float]          # (cx, cy, radius)
) -> np.ndarray:
    """
    Return the nearest point on the SURFACE of a circle to robot_pos.
 
    Why surface and not centre?
    Because we want the repulsion to start from the robot's physical edge,
    not from its geometric centre. This gives correct distance calculations.
 
    Example
    -------
    Robot at (3, 0), circular pillar centred at (0, 0) with radius 0.5.
    → closest surface point = (0.5, 0)   (the pillar's edge facing the robot)
    → distance to surface   = 2.5 m      (not 3 m to the centre)
    """
    cx, cy, r = circle
    centre = np.array([cx, cy])
    direction = robot_pos - centre
    dist = np.linalg.norm(direction)
    if dist < 1e-9:
        # Robot is inside the obstacle centre — push straight right as fallback
        return centre + np.array([r, 0.0])
    return centre + (direction / dist) * r
 
 
def closest_point_on_rect(
    robot_pos: np.ndarray,
    rect: Tuple[float, float, float, float]     # (x_min, y_min, x_max, y_max)
) -> np.ndarray:
    """
    Return the nearest point ON or INSIDE a filled axis-aligned rectangle.
 
    For collision avoidance we clamp the robot position to the rectangle,
    which gives us the nearest boundary point when outside, or the robot's
    own position when inside (meaning it is already overlapping — use the
    minimum-penetration push-out logic in the caller).
 
    Example
    -------
    Robot at (5, 1), wall rectangle from (0,0) to (3, 0.2).
    → clamped x = 3 (right edge), clamped y = 1 → clamped to (3, 0.2)
    → distance = sqrt(4 + 0.64) ≈ 2.15 m
    """
    x_min, y_min, x_max, y_max = rect
    cx = float(np.clip(robot_pos[0], x_min, x_max))
    cy = float(np.clip(robot_pos[1], y_min, y_max))
    return np.array([cx, cy])
 
 
def closest_point_on_segment(
    robot_pos: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray
) -> np.ndarray:
    """
    Return the nearest point on a LINE SEGMENT [p1, p2] to robot_pos.
    Useful for thin walls defined as line segments rather than thick rectangles.
    """
    seg   = p2 - p1
    t     = np.dot(robot_pos - p1, seg) / (np.dot(seg, seg) + 1e-12)
    t     = float(np.clip(t, 0.0, 1.0))
    return p1 + t * seg
 
 
# ─────────────────────────────────────────────────────────────────────────────
# OBSTACLE DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
 
class CircleObstacle:
    """A circular static obstacle (pillar, column, rounded wall)."""
    def __init__(self, cx: float, cy: float, radius: float):
        self.cx, self.cy, self.radius = cx, cy, radius
 
    def closest_point(self, robot_pos: np.ndarray) -> np.ndarray:
        return closest_point_on_circle(robot_pos, (self.cx, self.cy, self.radius))
 
 
class RectObstacle:
    """An axis-aligned rectangular static obstacle (wall, box, U-shape segment)."""
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.bounds = (x_min, y_min, x_max, y_max)
 
    def closest_point(self, robot_pos: np.ndarray) -> np.ndarray:
        return closest_point_on_rect(robot_pos, self.bounds)
 
 
class DynamicObstacle:
    """
    An UNKNOWN dynamic obstacle detected by the overhead camera.
 
    Fields
    ------
    x, y     : current position (camera centroid)
    radius   : estimated bounding radius (from bounding box diagonal / 2)
    priority : repulsion multiplier.
               1.0 = normal push.
               >1.0 = treat as higher priority (e.g., fast-moving human).
               Set this based on obstacle size or proximity, not velocity.
    """
    def __init__(self, x: float, y: float, radius: float = 0.3, priority: float = 1.0):
        self.pos      = np.array([x, y], dtype=float)
        self.radius   = radius
        self.priority = priority
 
 
# ─────────────────────────────────────────────────────────────────────────────
# APF CORE
# ─────────────────────────────────────────────────────────────────────────────
 
class APF:
    """
    Artificial Potential Fields — Phase 3 "The Shield".

    WHAT IT DOES (in plain English)
    --------------------------------
    Imagine the robot lives in a hilly landscape:
      • The GOAL is a deep valley — the robot rolls downhill toward it.
      • Every OBSTACLE is a sharp mountain — the robot is pushed away.
      • The OUTPUT is the slope (gradient) at the robot's current position.

    The robot follows this slope at every control tick.

    PARAMETERS
    ----------
    k_att       : Attractive gain.  Larger = robot chases goal more aggressively.
                  Typical range: 0.5 – 2.0
    k_rep       : Repulsive gain.  Larger = obstacles push harder.
                  Typical range: 1.0 – 5.0
    rho_0       : Influence radius [m].  Obstacles outside this distance are ignored.
                  Typical range: 0.5 – 2.0 m
    max_force   : Hard cap on the repulsive force magnitude.
                  CRITICAL: prevents the 1/d² singularity from blowing up
                  when a robot is very close to an obstacle.
                  Typical value: 3.0 – 8.0
    vortex_gain : Fraction of repulsive force added as a tangential "spin".
                  Breaks head-on deadlocks where two forces cancel perfectly.
                  Typical range: 0.3 – 0.7
    """

    def __init__(
        self,
        k_att: float = 1.0,
        k_rep: float = 2.0,
        rho_0: float = 1.5,
        max_force: float = 5.0,
        vortex_gain: float = 0.4,
        robot_radius: float = 1.0,  # NEW: robot's own physical radius
        static_circles: Optional[List[CircleObstacle]] = None,
        static_rects: Optional[List[RectObstacle]] = None,
        enable_static_repulsion: bool = True,
    ):
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
        self.max_force = max_force
        self.vortex_gain = vortex_gain
        self.robot_radius = robot_radius  # NEW

        # Static obstacles (set once at startup from the map)
        self.static_circles = static_circles or []
        self.static_rects = static_rects or []
        self.enable_static_repulsion = enable_static_repulsion

    # ── Main entry point ──────────────────────────────────────

    def get_force(
        self,
        robot_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        dynamic_obstacles: Optional[List[DynamicObstacle]] = None,
    ) -> np.ndarray:
        """
        Compute the total APF force on the robot.

        Parameters
        ----------
        robot_pos : (x, y) robot's current position
        goal_pos : (x, y) current carrot / waypoint from A*
        dynamic_obstacles : list of DynamicObstacle objects from the camera.
        Pass [] or None if no unknowns are detected.
        NEVER pass swarm agents here — use NH-ORCA for those.

        Returns
        -------
        force : np.ndarray shape (2,)
        A 2D velocity-space vector. Units: [m/s] (when gains are tuned).
        This becomes F_unknown in the intention blender.

        HOW TO READ THE OUTPUT
        ----------------------
        force = [1.2, -0.3]
        → push 1.2 m/s to the right (east)
        → push -0.3 m/s downward (south)
        force = [0, 0]
        → no obstacles nearby; attractive pull exactly cancelled; at goal
        """
        q = np.array(robot_pos, dtype=float)
        goal = np.array(goal_pos, dtype=float)

        F_att = self._attractive(q, goal)
        F_rep = np.zeros(2)

        if self.enable_static_repulsion:
            F_rep += self._repulsive_static(q)

        F_rep += self._repulsive_dynamic(q, dynamic_obstacles or [])

        rep_mag = np.linalg.norm(F_rep)
        if rep_mag > self.max_force:
            F_rep = F_rep * (self.max_force / rep_mag)

        return F_att + F_rep

    def get_repulsive_only(
        self,
        robot_pos: Tuple[float, float],
        dynamic_obstacles: Optional[List[DynamicObstacle]] = None,
    ) -> np.ndarray:
        """
        Return ONLY the repulsive component F_unknown (no attraction).

        This is the correct output to use in the integration pipeline.
        The attractive component is handled separately by the waypoint tracker.

        See architecture doc:
        V_pref = V_path (from waypoint tracker) + F_unknown (from here)
        """
        q = np.array(robot_pos, dtype=float)
        F_rep = np.zeros(2)

        if self.enable_static_repulsion:
            F_rep += self._repulsive_static(q)

        F_rep += self._repulsive_dynamic(q, dynamic_obstacles or [])

        rep_mag = np.linalg.norm(F_rep)
        if rep_mag > self.max_force:
            F_rep = F_rep * (self.max_force / rep_mag)

        return F_rep

    # ── Internal force components ─────────────────────────

    def _attractive(self, q: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Linear attractive force: F_att = -k_att * (robot - goal)

        WHY LINEAR (not quadratic)?
        Linear gives constant "pull velocity" regardless of distance.
        Quadratic would pull harder far away and weaker near the goal,
        which can cause the robot to overshoot and oscillate.
        Linear is simpler and works well as a velocity reference.
        """
        return -self.k_att * (q - goal)

    def _repulsive_static(self, q: np.ndarray) -> np.ndarray:
        """
        Repulsion from static map obstacles (walls, pillars).

        ρ = distance from robot SURFACE to obstacle SURFACE.
        Both robot_radius and obstacle geometry are accounted for.
        """
        F_rep = np.zeros(2)

        # Collect nearest surface points from all static obstacles
        surface_points = []
        for obs in self.static_circles:
            surface_points.append(obs.closest_point(q))
        for obs in self.static_rects:
            surface_points.append(obs.closest_point(q))

        for pt in surface_points:
            centre_dist = np.linalg.norm(q - pt)
            # Gap between robot surface and obstacle surface
            rho = centre_dist - self.robot_radius
            rho = max(rho, 0.01)  # floor to avoid division by zero
            if rho < self.rho_0:
                mag = self.k_rep * (1.0 / rho - 1.0 / self.rho_0) / (rho ** 2)
                dirn = (q - pt) / max(centre_dist, 1e-9)
                F_rep += mag * dirn

        return F_rep

    def _repulsive_dynamic(
        self,
        q: np.ndarray,
        obstacles: List[DynamicObstacle]
        ) -> np.ndarray:
        """
        Repulsion from unknown camera blobs.

        CRITICAL FIX: rho = dist_between_centres - blob_radius - robot_radius
        This is the TRUE gap between physical surfaces.
        Previously it was dist - blob_radius only, ignoring the robot's own body.

        EMERGENCY ZONE: when rho ≤ 0, robots are physically overlapping.
        A strong constant push-out force is applied, bypassing the normal
        formula (which would divide by zero or produce garbage at rho ≤ 0).
        """
        F_rep = np.zeros(2)

        for obs in obstacles:
            diff = q - obs.pos
            centre_dist = np.linalg.norm(diff)

            # Physical gap between robot edge and obstacle edge
            rho = centre_dist - obs.radius - self.robot_radius

            # ── EMERGENCY: already overlapping ───────────────────────────
            if rho <= 0.0:
                if centre_dist < 1e-9:
                    push_dir = np.array([1.0, 0.0])
                else:
                    push_dir = diff / centre_dist
                # Emergency force: proportional to penetration depth
                penetration = abs(rho) + 0.1  # how deep inside [m]
                emergency_mag = self.max_force * (1.0 + penetration * 2.0)
                # Add tangential component to slide out, not just push back
                tangent = np.array([-push_dir[1], push_dir[0]])
                F_rep += emergency_mag * push_dir + (emergency_mag * 0.5) * tangent
                continue

            # ── Floor and influence check ─────────────────────────────────
            rho = max(rho, 0.02)
            if rho >= self.rho_0:
                continue

            # ── Normal repulsion ──────────────────────────────────────────
            k_eff = self.k_rep * obs.priority
            mag = k_eff * (1.0 / rho - 1.0 / self.rho_0) / (rho ** 2)

            radial_dir = diff / centre_dist
            F_radial = mag * radial_dir

            tangent = np.array([-radial_dir[1], radial_dir[0]])
            F_vortex = (mag * self.vortex_gain) * tangent

            F_rep += F_radial + F_vortex

        return F_rep

    # ── Utility: update static obstacles at runtime ──────────────────────

    def add_static_circle(self, cx: float, cy: float, radius: float):
        """Add a circular static obstacle (e.g., discovered pillar)."""
        self.static_circles.append(CircleObstacle(cx, cy, radius))

    def add_static_rect(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float
    ):
        """Add a rectangular static obstacle (e.g., wall segment)."""
        self.static_rects.append(RectObstacle(x_min, y_min, x_max, y_max))

    def clear_static_obstacles(self):
        """Reset static obstacle list (call when map changes)."""
        self.static_circles.clear()
        self.static_rects.clear()