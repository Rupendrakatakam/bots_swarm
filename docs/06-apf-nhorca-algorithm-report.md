# APF + NH-ORCA Navigation Algorithm Report

## Overview

This report analyzes the multi-robot navigation pipeline implemented in the bots_swarm project. The system combines two complementary collision avoidance algorithms—Artificial Potential Fields (APF) and Non-Holonomic Optimal Reciprocal Collision Avoidance (NH-ORCA)—in a phased architecture that handles different types of obstacles and enables safe swarm navigation.

---

## 1. System Architecture

The navigation pipeline follows a six-phase architecture that separates concerns and creates an "architectural firewall" between different obstacle types:

```
Phase 1: Global Planner (Hybrid A*)     → Waypoints around static obstacles
Phase 2: Waypoint Tracker               → V_path (velocity toward carrot)
Phase 3: APF (The Shield)               → F_unknown (repulsion from unknown obstacles)
Phase 4: Intention Blender              → V_pref = V_path + F_unknown
Phase 5: NH-ORCA (The Scalpel)          → V_safe (collision-free velocity)
Phase 6: Kinematic Mapper + Low-Pass    → (v, ω) → (vl, vr) → wheels
```

### Key Design Principle

**Separation of Obstacle Types:**
- **APF** handles UNKNOWN, UNCOOPERATIVE obstacles (humans, rolling objects, dropped boxes) detected by the overhead camera. These have no telemetry—velocity cannot be reliably estimated.
- **NH-ORCA** handles COOPERATIVE swarm agents with PERFECT TELEMETRY (exact x, y, vx, vy from central tracking).
- **They never see each other's inputs.** Communication happens only through V_pref in the intention blender.

---

## 2. Artificial Potential Fields (APF) — Phase 3

### Purpose

APF provides velocity-agnostic collision avoidance. It only cares about POSITION, not velocity. As long as the system knows WHERE something is, it can push the robot away. This makes it naturally robust to camera noise.

### Physics Model

APF models the robot's environment as a hilly landscape:

- **Goal** = deep valley (robot rolls downhill toward it)
- **Obstacles** = sharp mountains (robot is pushed away)
- **Output** = slope (gradient) at the robot's current position

### Force Equations

**Attractive Force (toward goal):**
```
F_att = -k_att × (q - goal)
```
Where:
- `k_att` = attractive gain (typical: 0.5–2.0)
- `q` = robot position
- `goal` = target waypoint

**Repulsive Force (from obstacles):**
```
F_rep = k_rep × (1/ρ - 1/ρ₀) × (1/ρ²) × ∇ρ
```
Where:
- `ρ` = distance from robot to obstacle surface
- `ρ₀` = influence radius (obstacles beyond this are ignored)
- `k_rep` = repulsive gain (typical: 1.0–5.0)

**Total Force:**
```
F_total = F_att + F_rep
```

### Implementation Details

#### Geometry Helpers

The code provides three geometry functions for finding closest points on obstacles:

1. **closest_point_on_circle()** — Returns nearest point on circle surface (not center), giving correct distance calculations.

2. **closest_point_on_rect()** — Clamps robot position to rectangle bounds, finding nearest boundary point.

3. **closest_point_on_segment()** — Finds nearest point on a line segment (for thin walls).

#### Obstacle Types

1. **CircleObstacle** — Static circular obstacles (pillars, columns)
2. **RectObstacle** — Static rectangular obstacles (walls, boxes)
3. **DynamicObstacle** — Unknown dynamic obstacles from camera (position, radius, priority multiplier)

#### Repulsion Calculation

For **static obstacles**, the code iterates through all circles and rectangles, computes surface points, and applies the standard APF formula.

For **dynamic obstacles** (camera blobs), three key differences apply:

1. **Distance to surface**: `ρ = dist - obstacle_radius` (accounts for object size)
2. **Vortex component**: Adds tangential force to break head-on deadlocks
3. **Priority multiplier**: Scales force per obstacle based on size/proximity

```python
# Radial repulsion (same as static)
mag = k_eff × (1/ρ - 1/ρ₀) / ρ²

# Vortex (tangential) force - breaks symmetry
tangent = [-radial_dir[1], radial_dir[0]]  # 90° CCW rotation
F_vortex = mag × vortex_gain × tangent
```

### Parameters

| Parameter | Typical Range | Purpose |
|-----------|--------------|---------|
| k_att | 0.5–2.0 | How aggressively robot chases goal |
| k_rep | 1.0–5.0 | How hard obstacles push |
| rho_0 | 0.5–2.0 m | Influence radius |
| max_force | 3.0–8.0 | Prevents singularity blow-up when very close |
| vortex_gain | 0.3–0.7 | Tangential "spin" to break deadlocks |

---

## 3. NH-ORCA (Non-Holonomic ORCA) — Phase 5

### Purpose

NH-ORCA handles COOPERATIVE swarm agents with perfect telemetry. Since all swarm robots are tracked centrally with exact (x, y, vx, vy), ORCA's mathematical guarantee is valid and correct.

### Key Difference: Holonomic vs Non-Holonomic

- **Standard ORCA**: Robots can slide in any direction instantly (holonomic)
- **NH-ORCA**: Accounts for differential-drive kinematics (robots can only move forward/backward and rotate)

### Three Key Additions

1. **Inflated radius**: `r_inflated = r + ε` where ε is tracking error
2. **P_AHV polygon**: Reachable velocity polygon for non-holonomic constraints
3. **Kinematic mapping**: Converts holonomic (vx, vy) to (v, ω) via paper equations

### Velocity Obstacle (VO) Theory

ORCA builds on the Velocity Obstacle concept:

1. For each neighbor robot, compute the **VO cone** — region of velocities that lead to collision within time τ
2. Find the **minimal avoidance push** (vector u) to escape the VO
3. Create a **half-plane constraint**: robot must pick velocity v where (v - point) · normal ≥ 0

### P_AHV — Allowed Holonomic Velocities Polygon

The reachable velocity set for a differential-drive robot is NOT a circle—it's a convex polygon shaped by the robot's orientation and heading error.

**Theorem 2**: Maximum holonomic speed for heading error θ_err such that tracking error stays ≤ ε:
```
V_Hmax = (ε/T) × √(2×(1-cos(θ_err)) / factor)
```

The polygon is built by sampling V_H^max in N directions (default: 36 samples), then rotated at runtime to align with the robot's current heading.

### Three Kinematic Regions

The mapper converts V_safe (holonomic) to (v, ω) based on heading error:

| Region | Condition | Behavior |
|--------|-----------|----------|
| R_A1 | Small heading error, within ω_max | Normal driving: ω = θ_err / T, v = optimal |
| R_A2 | Tight turn needed | Clamp ω to ω_max, reduce v to keep wheels feasible |
| R_B | Error so large that driving forward would exceed ε | Stop and spin: v = 0, ω = ±ω_max |

**R_B is the primary jitter source** — occurs when the robot needs to turn almost 180° while moving forward.

### Half-Plane Computation

```python
def compute_orca_halfplane(ego, other, tau, c=0.5):
    # 1. Relative position and velocity
    rel_pos = other.pos - ego.pos
    rel_vel = ego.vel - other.vel
    
    # 2. Check if collision predicted within τ
    # Build VO cone from relative geometry
    
    # 3. Find minimal escape vector u
    # 4. Ego takes c*u (c=0.5 for swarm, c=1.0 for static obstacle)
    # 5. Return HalfPlane(point=ego.vel + c*u, normal=n)
```

### Linear Programming Solver

The LP finds velocity v* closest to pref_vel satisfying:
- All ORCA half-planes
- Speed disc (max speed)
- P_AHV polygon (optional—used as pre-clip for efficiency)

Uses incremental 2-D LP with O(n) expected time. Falls back to grid search if no solution exists.

---

## 4. Integration Pipeline — apf_orca.py

### Phase 2: Waypoint Tracker (Floating Carrot)

**Problem**: If robot misses waypoint N by 1cm, it circles forever.

**Solution**: Floating carrot logic:
1. Scan window [current_idx, current_idx + lookahead_window]
2. Find CLOSEST waypoint in that window
3. Target waypoint CARROT_STEPS ahead of that
4. Window slides forward as robot progresses

```python
V_path = speed × (carrot_pos - robot_pos) / distance
```

### Phase 4: Intention Blender

Combines the two force components:

```python
V_pref = V_path + F_unknown
```

Where:
- `V_path` = velocity from waypoint tracker (goal-seeking)
- `F_unknown` = repulsive force from APF (obstacle avoidance)

**Important**: Only camera blobs go to APF. Swarm agents go to NH-ORCA. They meet only here in V_pref.

### Phase 6: Motor Mapper (Low-Pass Filter)

Raw (v, ω) commands can jump sharply between ticks, causing:
- Physical jitter and vibration
- Motor overcurrent spikes
- Noisy odometry

**Solution**: Exponential Moving Average (EMA) filter:

```python
v_filt = α × v_raw + (1 - α) × v_prev
omega_filt = α × omega_raw + (1 - α) × omega_prev
```

Where α = 0.5 (balanced). Higher α = more responsive, lower α = smoother.

---

## 5. Fleet Manager

Manages a swarm of robots:

1. **Holds** one RobotController per robot
2. **Each tick**, collects all robots' states
3. **Builds telemetry** for all OTHER robots (excludes self)
4. **Enforces firewall**: camera_blobs → APF, swarm_telemetry → ORCA

---

## 6. Simulation — simulation.py

### Configuration

```python
cfg = DiffDriveConfig(
    robot_radius=1.0,
    wheel_base=0.8,
    max_linear_speed=2.0,
    max_angular_speed=3.5,
    tracking_error=0.40,      # ε — how far robot drifts from holonomic path
    orientation_time=0.25,    # T — time budget to rotate
    time_horizon=3.0,         # τ — collision lookahead
    neighbor_dist=6.0,        # sensing radius
    sim_dt=0.1,               # control period
)
```

### Obstacles

1. **INTERNAL_CIRCLES/RECTS** — Used by both A* and APF
2. **BOUNDARY_RECTS** — Used only by A* (NOT by APF, to avoid fighting path-following)
3. **Camera blobs** — Unknown dynamic obstacles (passed to APF only)

### Simulation Loop

1. Call `fleet.tick_all(camera_blobs)` — runs full pipeline for all robots
2. Euler integration: update position/heading from commands
3. Repeat until all goals reached or max_steps

---

## 7. Data Flow Summary

```
                    ┌─────────────────┐
                    │  Hybrid A*      │ → waypoints (static map only)
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Waypoint Tracker│ → V_path
                    └────────┬────────┘
                             ↓
         ┌──────────────────────────────────────┐
         │         INTENTION BLENDER            │
         │  V_pref = V_path + F_unknown         │
         └─────────────┬────────────────────────┘
                       ↓
    ┌──────────────────┴──────────────────┐
    ↓                                         ↓
┌─────────────┐                    ┌─────────────────┐
│    APF      │                    │   NH-ORCA       │
│ (camera     │                    │   (swarm        │
│  blobs)     │                    │   telemetry)    │
│    ↓        │                    │       ↓         │
│  F_unknown  │                    │    V_safe       │
└──────┬──────┘                    └────────┬────────┘
       ↓                                    ↓
       └──────────────┬─────────────────────┘
                      ↓
             ┌─────────────────┐
             │ Motor Mapper    │
             │ (v, ω) → wheels │
             └─────────────────┘
```

---

## 8. Tuning Guidelines

### APF Parameters

| Issue | Fix |
|-------|-----|
| Robot overshoots goal | Reduce k_att |
| Robot gets too close to obstacles | Increase k_rep or rho_0 |
| Robot stuck in singularity | Increase max_force |
| Head-on deadlock with dynamic obstacle | Increase vortex_gain |

### NH-ORCA Parameters

| Issue | Fix |
|-------|-----|
| Robot freezes in crowded space | Increase tracking_error ε |
| Excessive stop-and-spin (R_B) | Increase orientation_time T |
| Robot doesn't avoid far neighbors | Increase neighbor_dist |
| Too few/late collisions | Decrease time_horizon τ |

### Motor Mapper

| Issue | Fix |
|-------|-----|
| Chassis vibrates/jitters | Decrease filter_alpha (more smoothing) |
| Slow reaction to obstacles | Increase filter_alpha (more responsive) |

---

## 9. Diagnostic Output

The PipelineLogger records per-tick diagnostics:

- **apf_active**: Was APF doing anything?
- **apf_force_mag**: How hard APF pushed [m/s]
- **orca_active**: Did ORCA compute any half-planes?
- **orca_n_halfplanes**: Number of constraints (0 = idle, 1-2 = normal, 3+ = crowded)
- **orca_adjustment**: How much ORCA shifted velocity from V_pref
- **kinematic_region**: R_A1 (normal), R_A2 (tight turn), R_B (stop-and-spin)
- **heading_error_deg**: Error between V_pref direction and current yaw

Run `ctrl.logger.jitter_report()` after simulation to see summary statistics.

---

## 10. Conclusion

The APF + NH-ORCA pipeline provides a robust multi-robot navigation system by:

1. **Separating obstacle types** — Unknown obstacles handled by APF (position-based), cooperative agents by ORCA (velocity-based)
2. **Hierarchical planning** — Global path from A*, local avoidance from APF/ORCA
3. **Non-holonomic awareness** — P_AHV polygon and kinematic mapping account for differential-drive constraints
4. **Diagnostic visibility** — Extensive logging enables debugging and tuning

The architectural firewall ensures APF never sees swarm agents and ORCA never sees camera blobs—they communicate only through V_pref, preventing deadlocks and conflicting commands.