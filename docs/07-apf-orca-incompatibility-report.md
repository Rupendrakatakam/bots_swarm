# APF + ORCA Incompatibility Analysis Report

## Executive Summary

This report documents the fundamental incompatibilities discovered between Artificial Potential Fields (APF) and Non-Holonomic Optimal Reciprocal Collision Avoidance (NH-ORCA) when used together in a unified navigation pipeline. Through extensive testing and iterative debugging, we determined that these two algorithms operate on fundamentally different paradigms that fight each other when combined, leading to 360° loops, squiggly trajectories, and collisions.

---

## 1. Introduction: The Problem Statement

### 1.1 Initial Design Intent

The original system design attempted to leverage both algorithms' strengths:

| Algorithm | Intended Role | Why It Was Chosen |
|-----------|--------------|------------------|
| **APF** | Handle unknown dynamic obstacles (camera blobs) | Position-based, velocity-agnostic, robust to camera noise |
| **ORCA** | Handle cooperative swarm agents | Velocity-based, mathematically guaranteed collision-free, predicts future collisions |

### 1.2 What Actually Happened

During testing, we encountered multiple failure modes:

| Failure Mode | Symptom | Frequency |
|-------------|---------|----------|
| **360° Loops** | Robot traces a complete circle when blob passes by | Frequent |
| **Squiggly Cusps** | Robot path has sharp direction changes | Frequent |
| **Dynamic Blob Collisions** | Robot hits moving camera blob | Occasional |
| **Static Circle Collisions** | Robot drives through static obstacles | Occasional |
| **Wall Collisions** | Robot hits boundary walls | Rare (after fixes) |

---

## 2. Theoretical Incompatibility Analysis

### 2.1 Fundamental Paradigm Conflict

APF and ORCA are based on completely different mathematical frameworks:

| Aspect | APF | ORCA |
|--------|-----|------|
| **Input** | Position of obstacles | Position + Velocity of obstacles |
| **Output** | Force (vector) | Velocity constraint (half-plane) |
| **Time Model** | Instantaneous (current position only) | Predictive (τ time horizon) |
| **Mathematical Basis** | Gradient descent | Linear programming |
| **Guarantee** | None (soft constraints) | Collision-free if solution exists |

### 2.2 APF: Force-Based Position Control

APF treats the environment as a potential energy landscape:

```
F_total = F_att + F_rep
```

- **Goal** = valley (minimum potential)
- **Obstacle** = hill (maximum potential)
- **Robot** = ball rolling downhill

The robot follows the **gradient** of the potential field — it moves in the direction of steepest descent. This is inherently **position-only** — APF doesn't care about where things are going, only where they are.

**Key limitation**: When a dynamic obstacle moves in a circle around the robot, the repulsive force vector rotates in lockstep with the obstacle, causing the robot to trace a circle in physical space.

### 2.3 ORCA: Velocity-Based Constraint Solving

ORCA treats collision avoidance as a constrained optimization problem:

```
Find v* closest to V_pref subject to:
  - All ORCA half-planes (avoid all neighbors)
  - Speed ≤ max_speed
  - P_AHV polygon (non-holonomic constraints)
```

The key insight is that ORCA uses **relative velocity** to predict future collisions. If the relative velocity points toward the obstacle, collision is predicted within time τ.

**Key strength**: ORCA knows WHERE something will be in τ seconds, not just where it is now.

### 2.4 Why They Fight

When combined in the pipeline:

```
V_path → APF adds F_rep → V_pref = V_path + F_rep
         ↓
     ORCA receives V_pref, applies half-planes
         ↓
     V_safe = ORCA_LP(V_pref, constraints)
```

The conflict arises because:

1. **APF suggests a direction** based on current obstacle positions
2. **ORCA constrains that direction** based on predicted future positions
3. **The LP picks the "closest valid"** to APF's suggestion — but APF's suggestion may be fundamentally wrong

APF pushes, ORCA constrains. The result is neither pure APF behavior nor pure ORCA behavior — it's a confused hybrid that exhibits both systems' worst traits.

---

## 3. Testing and Failure Analysis

### 3.1 Test Scenarios

We tested the APF+ORCA system with multiple scenarios:

| Scenario | Description | Challenge Level |
|----------|-------------|-----------------|
| **Circle Cross** | Two robots cross paths at 90° | Medium |
| **Head-On** | Two robots approach head-on | High |
| **Dynamic Blob** | Robot navigates around moving blob | Very High |
| **Static Obstacle** | Robot navigates through static circles | Medium |
| **Multi-Robot** | 4+ robots in shared space | Very High |

### 3.2 Failure 1: 360° Loops (Dynamic Blobs in APF)

**Observation**: When a camera blob moved in a circular path around the robot, the robot traced a circle in physical space.

**Root Cause**:
- APF computes radial repulsion from blob center
- As blob moves around robot, force vector rotates with it
- Robot smoothly tracks the rotating force vector
- Result: continuous circular motion

**Code Location**: `apf_orca.py:1096` — camera_blobs passed to APF
```python
camera_blobs = [], # Dynamic blobs removed from APF; ORCA handles them completely
```

**Fix**: Move dynamic blobs from APF to ORCA. ORCA uses velocity prediction, not pure radial repulsion.

### 3.3 Failure 2: Squiggly Cusps (is_reversing Hysteresis)

**Observation**: Robot path had sharp direction changes ("cusps") when approaching obstacles.

**Root Cause**: MotorMapper had `is_reversing` boolean with hysteresis — when switching from forward to reverse, the frame reference flipped, causing sudden heading changes.

**Code Location**: `nh_orca.py` (earlier version) — is_reversing logic

**Fix**: Remove is_reversing entirely. Let the kinematic regions handle all motion naturally.

### 3.4 Failure 3: Dynamic Blob Collisions (Wrong Responsibility Factor)

**Observation**: Robot collided with moving camera blobs even when ORCA was active.

**Root Cause**:
- ORCA assumed c=0.5 (equal responsibility) for all neighbors
- Camera blobs are UNCOOPERATIVE — they don't dodge the robot
- With c=0.5, ORCA only avoids half the collision — robot moves 50% of what's needed

**Code Location**: `nh_orca.py:498`
```python
hp = compute_orca_halfplane(ego, nb, cfg.time_horizon, c=0.5)  # WRONG for blobs
```

**Evidence from log**:
```
[C] step=100 | ORCA=ON (1hp) | KIN=R_A1(herr=148°) | V_pref=0.80 → V_safe=2.00
```
ORCA increased magnitude but direction was wrong because it only shifted by c*u = 0.5*u.

**Fix**: Add `c` field to SwarmAgent, set c=1.0 for camera blobs.

### 3.5 Failure 4: Static Circle Collisions (Missing from ORCA)

**Observation**: Robot drove through static circular obstacles.

**Root Cause**:
- APF knew about static_circles (added to V_pref)
- ORCA did NOT receive static_circles
- ORCA LP could override APF's direction entirely
- Result: robot curved into static obstacle

**Code Location**: `apf_orca.py:1132-1141` — static circles not passed to ORCA
```python
# Before: only camera_blobs and swarm_telemetry passed to ORCA
```

**Fix**: Inject static circles as zero-velocity SwarmAgent with c=1.0.

---

## 4. Root Cause Summary

### 4.1 The Architectural Gap

```
APF knows about:  static_circles ✓    camera_blobs ✗    swarm ✗
ORCA knows about: static_circles ✗    camera_blobs ✓    swarm ✓

NEITHER has the complete picture!
```

### 4.2 The Parameter Hell

APF has many parameters that interact with ORCA's parameters:

| APF Parameter | ORCA Parameter | Interaction |
|--------------|---------------|-------------|
| k_rep | time_horizon | High repulsion + short horizon = fighting |
| rho_0 | neighbor_dist | Large influence + small neighbor = late reactions |
| max_force | tracking_error | Strong push + small buffer = collisions |
| vortex_gain | — | Adds asymmetry that ORCA can't predict |

The tuning space is N-dimensional and non-linear. Finding stable parameters is nearly impossible.

### 4.3 The Prediction Gap

| Algorithm | Prediction |
|-----------|-----------|
| APF | NONE — reacts to current position only |
| ORCA | τ seconds — predicts future collision |

When APF pushes the robot toward a "safe" position, that position may not be safe in τ seconds because APF didn't account for where the obstacle will be.

---

## 5. The Fixes Applied

### 5.1 Fix A: Per-Neighbor Responsibility Factor

**File**: `nh_orca.py`

1. Added `c` field to SwarmAgent dataclass:
```python
@dataclass
class SwarmAgent:
    pos:       np.ndarray
    vel:       np.ndarray
    radius:    float
    max_speed: float
    c:         float = 0.5   # 0.5=cooperative, 1.0=uncooperative
```

2. Changed half-plane computation to use per-neighbor c:
```python
hp = compute_orca_halfplane(ego, nb, cfg.time_horizon, c=nb.c)
```

### 5.2 Fix B: Static Circles in ORCA

**File**: `apf_orca.py`

Inject static circles as uncooperative obstacles:
```python
for obs in self.blender.apf.static_circles:
    neighbors.append(
        SwarmAgent(
            pos = np.array([obs.cx, obs.cy]),
            vel = np.zeros(2),
            radius = obs.radius,
            max_speed = 0.0,
            c = 1.0,  # uncooperative
        )
    )
```

### 5.3 Fix C: Remove Blobs from APF

**File**: `apf_orca.py:1096`

```python
v_pref = self.blender.compute(
    robot_pos = tuple(pos),
    v_path = v_path,
    camera_blobs = [],  # Now handled by ORCA only
    goal_pos = goal_pos,
)
```

---

## 6. Post-Fix Pipeline State

| System | Static Obstacles | Dynamic Blobs | Swarm Agents |
|--------|------------------|---------------|---------------|
| **APF** | Soft bias (optional) | None | None |
| **ORCA** | Hard constraint (c=1.0) | Hard constraint (c=1.0) | Hard constraint (c=0.5) |

---

## 7. Conclusion: Why APF and ORCA Are Incompatible

### 7.1 Fundamental Issues

1. **Different time models**: APF is instantaneous, ORCA is predictive
2. **Different mathematical frameworks**: Gradient descent vs linear programming
3. **Conflicting outputs**: Force vs velocity constraint
4. **Parameter explosion**: N-dimensional tuning space

### 7.2 Why ORCA-Only Works

| Aspect | APF+ORCA | ORCA-Only |
|--------|----------|-----------|
| Paradigm | Two conflicting | One consistent |
| Prediction | APF has none | τ time horizon |
| Parameters | 10+ interacting | 3 (c, τ, neighbor_dist) |
| Guarantees | None | Collision-free if solution exists |
| Predictability | Low (forces fight) | High (constraints enforced) |

### 7.3 Recommendation

**Remove APF entirely** from the navigation pipeline. Let ORCA handle:
- Static obstacles (as zero-velocity, c=1.0 agents)
- Dynamic obstacles (as uncooperative, c=1.0 agents)
- Swarm agents (as cooperative, c=0.5 agents)

APF's only remaining value was soft bias for static obstacles — but ORCA now handles those as hard constraints, making APF redundant.

---

## 8. Appendix: Change History

| # | Change | Result |
|---|--------|--------|
| 1 | Added `is_reversing` hysteresis to MotorMapper | Fixed 360° spin loops, introduced squiggly cusps |
| 2 | Fixed EMA frame flip when `is_reversing` toggles | Reduced squiggle severity |
| 3 | Removed `swarm_telemetry` from EmergencyBrake | Stopped brake↔ORCA feedback oscillation |
| 4 | Fixed EmergencyBrake wheel kinematics (preserve ω) | Better dodge arcs |
| 5 | Reverted `tracking_error` 0.45→0.25 | Stopped wall collisions |
| 6 | Removed `is_reversing` entirely | Eliminated all cusps permanently |
| 7 | Removed `camera_blobs` from APF, added to ORCA | Eliminated 360° loops |
| 8 | Added per-neighbor `c` factor to ORCA | Fixed blob collisions |
| 9 | Injected static circles into ORCA | Fixed static obstacle collisions |

---

*Report generated: April 2026*
*System: bots_swarm navigation pipeline*