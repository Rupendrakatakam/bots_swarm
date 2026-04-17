# Codebase Documentation — Part 2: `apf_orca.py` + `simulation.py`

*Continuation of [Part 1](file:///home/rupendra/.gemini/antigravity/brain/6f9f1fe9-34ba-4b49-84a2-3534c5be719a/artifacts/codebase_documentation.md)*

---

## 5. `apf_orca.py` — Pipeline Integration (1759 lines)

This is the central file. It wires all phases together and contains all runtime coordination logic.

### 5.1 Data Types

#### `Pose2D` (line 62)
```python
x: float = 0.0      # [m] world x
y: float = 0.0      # [m] world y
yaw: float = 0.0    # [rad] heading angle (0 = +x axis, π/2 = +y axis)
```
Property `position` returns `np.array([x, y])`.

#### `RobotState` (line 74)
```python
pose: Pose2D                 # current (x, y, yaw)
vel: np.ndarray = [0, 0]     # current velocity [vx, vy] from odometry
prev_v: float = 0.0          # last linear command (for low-pass filter)
prev_w: float = 0.0          # last angular command (for low-pass filter)
```

#### `CameraBlob` (line 84)
One unknown obstacle detected by the overhead camera.
```python
x, y: float           # world position [m]
radius: float = 0.30  # estimated bounding radius [m]
priority: float = 1.0 # repulsion multiplier (1.5 = 50% stronger push)
vx, vy: float = 0.0   # velocity estimate [m/s] — for predictive EmergencyBrake
```

#### `SwarmTelemetry` (line 100)
Perfect telemetry for one other swarm robot.
```python
robot_id: str        # unique identifier
x, y: float          # position [m]
vx, vy: float        # velocity [m/s]
radius: float        # social_radius (r × 1.20 + ε) — what ORCA uses for clearance
```

#### `WheelCommand` (line 113)
Final hardware command output.
```python
v: float = 0.0       # linear speed [m/s]
omega: float = 0.0   # angular speed [rad/s]
vl: float = 0.0      # left wheel speed [m/s]
vr: float = 0.0      # right wheel speed [m/s]
```

### 5.2 `PipelineLogger` (line 126)

Records one `LogEntry` per tick per robot. Prints summary every 100 ticks.

#### LogEntry Fields

| Field | Type | What It Answers |
|-------|------|-----------------|
| `apf_active` | bool | Is APF doing anything? (force > 0.01) |
| `apf_force_mag` | float | How hard APF pushed [m/s] |
| `n_camera_blobs` | int | How many blobs visible this tick |
| `v_path_mag` | float | Speed from waypoint tracker |
| `v_pref_mag` | float | Speed after intention blend |
| `v_safe_mag` | float | Speed after ORCA |
| `orca_active` | bool | Did ORCA compute half-planes? |
| `orca_n_halfplanes` | int | Number of LP constraints |
| `orca_adjustment` | float | ‖V_safe − V_pref‖ — ORCA correction magnitude |
| `kinematic_region` | str | R_A1, R_A2, or R_B |
| `heading_error_deg` | float | Angle between V_pref direction and robot yaw |
| `cmd_v`, `cmd_omega` | float | Final wheel commands |

#### `jitter_report()` — Post-Simulation Diagnostic

Prints counts of:
- **R_B events**: Stop-and-spin ticks (main jitter source)
- **APF with no blobs**: Static obstacles fighting the path
- **Big ORCA adjustments (>0.5 m/s)**: Heavy swarm dodging
- **Big APF forces (>1.0 m/s)**: Robot near static obstacle

### 5.3 Phase 1 — `GlobalPath` (line 321)

Stores A* waypoints as `List[np.ndarray]` of `[x, y]` points.

| Method | Purpose |
|--------|---------|
| `set_path(waypoints)` | Load waypoints from Hybrid A* (each is `(x, y, θ)`, only x,y used) |
| `set_straight_line_path(start, goal, n=50)` | Testing stub: N evenly-spaced points |
| `is_loaded` | True when waypoints exist |

**Design rule**: Path is computed BLIND to other robots and camera blobs. Only static map. This keeps planning O(1) per robot.

### 5.4 Phase 2 — `WaypointTracker` (line 366)

Computes `V_path`: velocity vector pulling robot toward the A* path.

#### Floating Carrot Algorithm

**Problem**: If robot misses waypoint N by 1cm, it circles forever trying to hit it exactly.

**Solution**: Instead of requiring exact waypoint hits:
1. Scan a window of `lookahead_window=15` waypoints ahead of current index
2. Find the CLOSEST waypoint in that window
3. Target the waypoint `carrot_steps=3` steps ahead of that
4. The "carrot" always moves forward — robot never stalls

#### Constructor Parameters

| Parameter | Value | Unit | Purpose |
|-----------|-------|------|---------|
| `max_speed` | 2.0 | m/s | Maximum path-following speed (= `cfg.max_linear_speed`) |
| `lookahead_window` | 20 | waypoints | How far ahead to scan for closest waypoint |
| `carrot_steps` | 10 | waypoints | How far ahead of closest to place the carrot |
| `goal_tolerance` | 0.5 | m | Distance to declare "goal reached" |
| `curvature_steps` | 10 | waypoints | Look-ahead for curvature estimation |
| `max_curvature` | 1.5 | rad/m | Curvature threshold for speed reduction |
| `path_alpha` | 0.4 | — | EMA smoothing factor for V_path |

#### Drift Recovery (line 436)
If robot drifts more than `max_speed × 4 = 8m` from the scan window (ORCA deflected it far), performs a GLOBAL search across all waypoints to resync. Only moves forward, never backward.

#### Curvature Braking (line 464)
Estimates path curvature by comparing current direction to future direction. If curvature > 1.5 rad/m, reduces speed to 30-100% (proportional to curvature excess). Mimics human behavior of slowing before sharp turns.

#### Output
```
V_path = speed × (carrot − robot_pos) / ||carrot − robot_pos||
V_path = 0.4 × V_path_new + 0.6 × V_path_prev  ← EMA smoothing
```

### 5.5 Phase 3+4 — `IntentionBlender` (line 493)

Combines path velocity with APF repulsion. This is the architectural firewall.

#### Three Response Zones

**Zone 1 — Normal blend** (line 586): Robot far from blobs.
```
V_pref = V_path + F_unknown
if ||V_pref|| > max_pref_speed: scale down
```

**Zone 2 — Repulsion dominant** (line 581): APF force > EMERGENCY_THRESHOLD (`1.0 × max_speed = 2.0 m/s`).
```
V_pref = (F_unknown / ||F_unknown||) × max_pref_speed
```
Path completely ignored. Pure escape at max speed.

**Zone 3 — Emergency overlap** (line 566): Surface gap ≤ 0 (robot INSIDE blob).
```
V_pref = (F_unknown / ||F_unknown||) × max_pref_speed
```
If APF returns zero (degenerate), pushes directly away from nearest blob.

**Why threshold was changed from 2.0× to 1.0×**: At 2.0×, APF force of 3.14 m/s was BELOW the 4.0 threshold, so the path kept pulling the robot INTO the obstacle. At 1.0×, threshold=2.0, and any APF force above 2.0 immediately overrides path.

### 5.6 Phase 6 — `MotorMapper` (line 598)

Translates `V_safe = (vx, vy)` from ORCA into smoothed `(v, ω)` wheel commands.

#### Constructor Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `filter_alpha` | 0.25→0.50 | EMA filter strength for commands |
| `heading_alpha` | 0.50 | EMA smoothing for heading error (was 0.3) |

#### Processing Pipeline

1. **Stop command** (V_H < 0.0001): Decelerate smoothly at `max_linear_decel × dt`, don't snap to zero
2. **Heading error**: `θ_err = atan2(vy, vx) − yaw`, smoothed with EMA: `θ_err = 0.5×raw + 0.5×prev`
3. **Angular velocity**: `ω = clip(θ_err / orientation_time, −ω_max, ω_max)`
4. **Forward speed** (Eq.8): `v = optimal_v(V_H, θ_err)` — see nh_orca.py §4.4
5. **Coupling limit**: `v ≤ max_wheel_speed − |ω| × wheel_base/2` — physical wheel constraint
6. **Turn-aware braking**: If `|ω| > 0.5 × ω_max`, reduce v by 0-40% (sharp turn → slow down)
7. **Acceleration limits**: `Δv ≤ accel × dt` (ramp up), `Δv ≥ −decel × dt` (ramp down)
8. **Wheel conversion**: `vl = v − ω×L/2`, `vr = v + ω×L/2` → reconstruct actual `(v, ω)`

### 5.7 `EmergencyBrake` (line 747)

Last safety net. Runs AFTER the full pipeline. Hard geometric check.

#### Three Escalating Levels

| Level | Trigger | Action | Current Threshold |
|-------|---------|--------|-------------------|
| **Level 1: Slow** | gap < `warn_dist` | Scale speed by `slow_factor` to `1.0` (proportional) | 1.84m (combined_r × 0.8) |
| **Level 2: Brake** | gap < `brake_dist` | `v = 0`, escape rotation toward away-direction | 0.92m (combined_r × 0.4) |
| **Level 3: Reverse** | gap < `reverse_dist` | `v = −reverse_speed`, no steering | 0.1m (before overlap) |

#### Constructor Parameters

| Parameter | Value | Unit | Purpose |
|-----------|-------|------|---------|
| `robot_radius` | 1.0 | m | For gap calculation |
| `warn_dist` | 1.84 | m | Level 1 trigger (combined_r × 0.8) |
| `brake_dist` | 0.92 | m | Level 2 trigger (combined_r × 0.4) |
| `reverse_dist` | 0.1 | m | Level 3 trigger (0.1m BEFORE overlap) |
| `slow_factor` | 0.20 | — | Minimum speed fraction in warning zone |
| `reverse_speed` | 0.8 | m/s | Reverse speed (max_speed × 0.4) |
| `max_angular_speed` | 5.0 | rad/s | For escape rotation during Level 2 |

#### Predictive Lookahead (line 810)
For each blob, checks BOTH current AND predicted positions up to `lookahead=3` ticks ahead:
```python
for step in range(lookahead + 1):
    bx_pred = blob.x + blob.vx × step × dt
    by_pred = blob.y + blob.vy × step × dt
    gap = ||robot − predicted_blob|| − blob.radius − robot_radius
```
Uses the worst (smallest) gap across all predictions.

### 5.8 `RobotController` (line 889)

Full navigation stack for ONE robot. Combines all phases.

#### `tick(camera_blobs, swarm_telemetry)` — The Main Loop (line 1030)

Execution order per tick:
```
1. Check goal reached → return zero command
2. WaypointTracker.compute() → V_path
3. IntentionBlender.compute() → V_pref (with 3-zone logic)
4. Yield bypass check: if APF active (>0.5 m/s), SKIP yield scaling
5. If not APF-active and yielding: V_pref × yield_scale
6. NHORCAPlanner.update() → V_safe
7. MotorMapper.compute() → WheelCommand
8. EmergencyBrake.check() → final WheelCommand override
9. If still yielding: scale command speed (safety net)
```

**Critical Fix — Yield Bypass** (line 1073):
```python
apf_component = ||V_pref − V_path||
apf_is_firing = apf_component > 0.5

if is_yielding and yield_scale < 1.0 and NOT apf_is_firing:
    V_pref *= yield_scale  # normal yield
elif is_yielding and apf_is_firing:
    # consume yield tick but DON'T reduce velocity — robot needs escape speed
```

### 5.9 `FleetManager` (line 1196)

Orchestrates N robots.

#### `tick_all(camera_blobs)` (line 1251)
1. Run `_detect_and_assign_yields()` — conflict prediction
2. Build `SwarmTelemetry` for all robots (using `social_radius`)
3. For each robot: call `tick()` with camera_blobs (→APF) and other robots' telemetry (→ORCA)

#### `_detect_and_assign_yields()` (line 1304) — Conflict Prediction

**Constants**:
| Constant | Value | Purpose |
|----------|-------|---------|
| `YIELD_DURATION_TICKS` | 25 | Auto-release after 2.5s |
| `CONFLICT_TIME_WINDOW` | 2.5s | Only react to conflicts within 2.5s |
| `CONFLICT_CLEAR_DIST` | 2.5m | Surface gap to deactivate yield |
| `YIELD_FAST_SCALE` | 0.60 | Higher path_progress → 60% speed |
| `YIELD_SLOW_SCALE` | 0.40 | Lower path_progress → 40% speed |

**Algorithm**:
1. For each robot pair (A, B):
2. Skip if surface gap > 2.5m (clearly separated)
3. Trajectory intersection test using quadratic: `at² + bt + c = 0` where:
   - `a = ||v_rel||²`
   - `b = 2 × rel_pos · rel_vel`
   - `c = ||rel_pos||² − combined_radius²`
4. If discriminant ≥ 0 and any root `0 < t ≤ 2.5s` → conflict predicted
5. Robot with higher `path_progress` gets right-of-way (60% speed)
6. Robot with lower `path_progress` yields more (40% speed)

---

## 6. `simulation.py` — Orchestration & Visualization (382 lines)

### 6.1 Purpose
Configures the scenario, runs the simulation loop, and renders matplotlib animation.

### 6.2 Configuration Block (line 242)

All tuned values in one place:

#### `DiffDriveConfig` — Current Tuned Values
| Parameter | Value | Why This Value |
|-----------|-------|----------------|
| `robot_radius` | 1.0m | Physical constraint |
| `wheel_base` | 0.8m | Physical constraint |
| `max_linear_speed` | 2.0 m/s | Maximum safe forward speed |
| `max_angular_speed` | 5.0 rad/s | Maximum rotation rate |
| `max_wheel_speed` | 3.0 m/s | Individual wheel limit |
| `max_linear_accel` | 2.0 m/s² | Prevents motor overcurrent |
| `max_linear_decel` | 3.0 m/s² | Allows faster braking than acceleration |
| `max_angular_accel` | 4.0 rad/s² | Smooth turn transitions |
| `tracking_error` | 0.30m | Was 0.50 — reduced to shrink social_radius (1.7→1.5m), fewer premature yields |
| `orientation_time` | 0.8s | Was 1.2 — faster rotation for escape maneuvers |
| `time_horizon` | 3.0s | ORCA looks 3s ahead for collisions |
| `neighbor_dist` | 6.0m | Only consider nearby robots for ORCA |
| `sim_dt` | 0.1s | 10 Hz control rate |

#### `ImprovedAPFParams` — Current Tuned Values
| Parameter | Value | Why This Value |
|-----------|-------|----------------|
| `eta` | 18.0 | Was 12 — stronger repulsion for fast-moving blobs |
| `rho_0` | 4.0m | Was 2.5 — earlier detection. With combined_radius=2.3m, APF starts at 4+2.3=6.3m centre distance |
| `max_force` | 18.0 | Was 12 — higher ceiling before clamping |
| `vortex_gain` | 0.4 | Was 0.3 — stronger tangential sliding |

### 6.3 Obstacle Setup (line 261)

| Obstacle | Type | Position | Radius | Purpose |
|----------|------|----------|--------|---------|
| Static pillar | `CircleObstacle` | (8, 8) | 1.5m | A* routes around it |
| 4 boundary walls | `RectObstacle` | Around perimeter | — | Containment |

### 6.4 Dynamic Blob Orbit (line 29)

The camera blob orbits in a circle:
```
centre = (8.0, 8.0), radius = 4.0m, period = 400 ticks = 40s
```

**Position at tick `step`**:
```
angle = 2π × (step mod 400) / 400
blob_x = 8.0 + 4.0 × cos(angle)
blob_y = 8.0 + 4.0 × sin(angle)
```

**Velocity** (tangential to orbit):
```
ω_blob = 2π / (400 × 0.1) = 0.157 rad/s
vx = −4.0 × 0.157 × sin(angle) ≈ ±0.628 m/s
vy = +4.0 × 0.157 × cos(angle) ≈ ±0.628 m/s
tangential_speed = 0.628 m/s
```

**Blob properties**: `radius=1.3m, priority=1.5` (50% stronger repulsion than normal).

### 6.5 Agent Definitions (line 299)

| Robot | Start | Goal | Color |
|-------|-------|------|-------|
| A | (2, 5, π/4) | (19, 18) | royalblue |
| B | (2, 15, π/2) | (15, 5) | seagreen |
| C | (2, 10, 0) | (19, 14) | yellow |

### 6.6 Fleet Construction (line 307)

Per-robot overrides passed to `RobotController`:
| Parameter | Value | Why |
|-----------|-------|-----|
| `filter_alpha` | 0.5 | Balanced smoothing for velocity commands |
| `goal_tolerance` | 0.5m | A* waypoints are ~1m apart, need tolerance > precision |
| `lookahead_window` | 20 | Wider scan for sparse A* paths in 20m world |
| `carrot_steps` | 10 | Further carrot for smoother following |

### 6.7 `run_simulation()` (line 20)

Main loop. For each tick:
1. Compute blob position and velocity from orbit equations
2. Create `CameraBlob` with position + velocity
3. Call `fleet.tick_all(camera_blobs)` → `Dict[robot_id, WheelCommand]`
4. Euler integration for each robot:
```
new_yaw = yaw + ω × dt
new_x = x + v × cos(new_yaw) × dt
new_y = y + v × sin(new_yaw) × dt
```
5. Update fleet state with new odometry
6. Check if all goals reached → break

Returns `(histories, blob_history)` for animation.

### 6.8 `build_figure()` (line 82)

Builds matplotlib animation with:
- Gray boundary walls (0.3m thick)
- Red static obstacles
- Purple blob (updates position each frame)
- Dashed A* paths per robot
- Star goal markers, square start markers
- Per-robot: body circle, heading line, wheel markers, trail line, social bubble ring
- Speed text display (computed from position delta / dt)

---

## 7. Complete Parameter Tuning Guide

### 7.1 If Robot Collides With Blob

| Try This | Parameter | File | Effect |
|----------|-----------|------|--------|
| Earlier APF detection | `rho_0` ↑ | simulation.py | APF starts pushing sooner |
| Stronger repulsion | `eta` ↑ | simulation.py | APF pushes harder |
| Higher force ceiling | `max_force` ↑ | simulation.py | APF force not clamped too early |
| Larger brake zones | `warn_dist`, `brake_dist` ↑ | apf_orca.py | EmergencyBrake fires sooner |
| More prediction | `lookahead` ↑ | apf_orca.py | EmergencyBrake sees further ahead |

### 7.2 If Robot Oscillates/Jitters

| Try This | Parameter | File | Effect |
|----------|-----------|------|--------|
| Smoother heading | `heading_alpha` ↓ | apf_orca.py | Less heading overshoot |
| Smoother commands | `filter_alpha` ↓ | simulation.py | More command smoothing |
| Less vortex | `vortex_gain` ↓ | simulation.py | Less tangential push |
| Path smoothing | `path_alpha` ↓ | apf_orca.py | Smoother V_path transitions |

### 7.3 If Robots Deadlock (Both Stopped)

| Try This | Parameter | File | Effect |
|----------|-----------|------|--------|
| Shorter yield | `YIELD_DURATION_TICKS` ↓ | apf_orca.py | Yield releases faster |
| Wider gap threshold | `CONFLICT_CLEAR_DIST` ↑ | apf_orca.py | Yield deactivates sooner |
| Less yielding | `YIELD_SLOW_SCALE` ↑ | apf_orca.py | Yielding robot moves faster |

### 7.4 If Robot Gets Stuck in Local Minimum

| Try This | Parameter | File | Effect |
|----------|-----------|------|--------|
| Faster stuck detection | `stuck_window` ↓ | simulation.py | Detect stuck sooner |
| Larger virtual target | `beta1`, `beta2` ↑ | simulation.py | Bigger escape offset |
| Longer virtual chase | `virtual_target_duration` ↑ | simulation.py | Chase escape point longer |

### 7.5 If Robot Takes Too Wide a Detour

| Try This | Parameter | File | Effect |
|----------|-----------|------|--------|
| Smaller APF influence | `rho_0` ↓ | simulation.py | APF activates closer |
| Weaker repulsion | `eta` ↓ | simulation.py | Less push from obstacles |
| Lower vortex | `vortex_gain` ↓ | simulation.py | Less tangential deflection |
| Less social buffer | `social_bubble_factor` ↓ | nh_orca.py | Robots pass closer to each other |

---

## 8. Data Flow Diagram — One Complete Tick

```
┌──────────────────────────────────────────────────────────────────┐
│ FleetManager.tick_all(camera_blobs)                             │
│                                                                  │
│  1. _detect_and_assign_yields() → set is_yielding, yield_scale  │
│  2. Build SwarmTelemetry for all robots                         │
│  3. For each robot:                                              │
│     ┌──────────────────────────────────────────────────────────┐ │
│     │ RobotController.tick(camera_blobs, swarm_telemetry)      │ │
│     │                                                          │ │
│     │  Phase 2: WaypointTracker → V_path (toward carrot)       │ │
│     │     ↓                                                    │ │
│     │  Phase 3: APF.get_repulsive_only(blobs) → F_unknown      │ │
│     │     ↓                                                    │ │
│     │  Phase 4: IntentionBlender                               │ │
│     │     Zone 3 (overlap): pure escape                        │ │
│     │     Zone 2 (APF>2.0): pure repulsion                     │ │
│     │     Zone 1 (normal):  V_pref = V_path + F_unknown        │ │
│     │     ↓                                                    │ │
│     │  Yield check: if APF active → SKIP yield scaling         │ │
│     │               if not → V_pref × yield_scale              │ │
│     │     ↓                                                    │ │
│     │  Phase 5: NH-ORCA LP(V_pref, halfplanes) → V_safe        │ │
│     │     ↓                                                    │ │
│     │  Phase 6: MotorMapper(V_safe) → WheelCommand             │ │
│     │     ↓                                                    │ │
│     │  EmergencyBrake.check() → override if gap too small       │ │
│     │     ↓                                                    │ │
│     │  Final yield safety net → scale cmd.v if still yielding   │ │
│     └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Return Dict[robot_id → WheelCommand]                           │
└──────────────────────────────────────────────────────────────────┘
```

---

> [!NOTE]
> **Paper reference**: Alonso-Mora et al., "Optimal Reciprocal Collision Avoidance for Multiple Non-Holonomic Robots", DARS 2013. Equations 1-9 and 13 are from this paper.
