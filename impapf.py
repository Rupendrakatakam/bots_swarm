"""
apf.py — Artificial Potential Field (APF) Path Planning
         with Local Minima Solutions
=========================================================

WHAT IS APF?
  Imagine the world as a landscape with hills and valleys.
  The goal is a deep valley that attracts the robot like a magnet.
  Obstacles are hills that push the robot away.
  The robot always rolls "downhill" toward the goal.

THE PROBLEM — Local Minima:
  Sometimes two hills trap the robot in a "false valley" before
  the real goal. The forces balance and the robot freezes. This
  non-goal trapped position is called a LOCAL MINIMUM.

  Example: a robot trapped between two symmetric obstacles
  halfway to the goal. Both obstacles push equally, the goal
  pulls forward, and the forces cancel → robot is frozen.

TWO FIXES IMPLEMENTED
──────────────────────────────────────────────────────────────

FIX 1 — Modified Repulsive Field (Ge & Cui formulation)
  Problem solved: robot gets trapped NEAR the goal itself
                  (obstacle right next to goal).
  
  Traditional repulsion:
    F_rep = K * (1/d - 1/d0) * (1/d²) * [away from obstacle]
  
  Modified repulsion (multiply by d_goal^n):
    F_rep' = K * (1/d - 1/d0) * (1/d²) * d_goal^n * [away]
           + (n/2)*K * (1/d - 1/d0)² * d_goal^(n-1) * [toward goal]
  
  How it works:
    • d_goal^n → when robot is NEAR goal, this → 0
      so repulsion fades away → robot CAN arrive
    • Second term = "secondary attractor" — gentle push toward
      goal that activates only inside an obstacle's influence zone
    • When robot is FAR from goal, d_goal^n is large, so
      repulsion behaves normally

  Simple analogy:
    Traditional: every obstacle always repels at full strength
    Modified:    obstacles "step aside" as you approach the goal

──────────────────────────────────────────────────────────────

FIX 2 — Dynamic Virtual Target Points
  Problem solved: robot trapped in a U-shaped obstacle or
                  stuck between two obstacles mid-journey.
  
  How it works:
    Step 1: DETECT — monitor net displacement over 40 steps.
            If robot barely moved (< 0.8 m in 40 steps × 0.15 m/step),
            it's oscillating in a local minimum.
    
    Step 2: IDENTIFY — find the "effective obstacle" (closest one).
    
    Step 3: ESCAPE — place a virtual sub-goal to the SIDE of the trap:
              vt = robot_pos
                 + perpendicular_direction × 3.5 m
                 + goal_direction × 1.4 m (small forward nudge)
    
    Step 4: NAVIGATE to virtual target using normal APF.
    
    Step 5: RESUME — once virtual target is reached, discard it
            and continue toward the real goal.

  Simple analogy:
    Like a car stuck in traffic — instead of pushing forward,
    take the next side street, get ahead of the jam, rejoin.

──────────────────────────────────────────────────────────────

MAZE:
  Robot starts at (1.5, 1.5) → Goal at (18.5, 18.5)
  A U-shaped trap cluster + 4 scattered obstacles force the
  robot into two local minima during its journey.

OUTPUT:
  1. plt.show()              — live preview window
  2. apf_simulation.mp4     — saved to outputs folder
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")      # headless; change to "TkAgg" for live window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from collections import deque

# ══════════════════════════════════════════════════════════════
#  WORLD & MAZE
# ══════════════════════════════════════════════════════════════
WORLD = 20.0
START = np.array([ 1.5,  1.5])
GOAL  = np.array([18.5, 18.5])

# Circular obstacles: (centre_x, centre_y, radius)
# ─── U-trap along the diagonal path ──────────────────────────
# Verified: traps robot in two sequential local minima
OBSTACLES = [
    # U-shaped cluster (back wall + two arms)
    (11.0, 12.0, 1.3),   # back wall of U
    ( 9.3, 10.5, 1.1),   # left  arm
    (12.7, 10.5, 1.1),   # right arm
    # Scattered obstacles
    ( 6.0, 10.0, 1.0),   # left wall mid-path
    (16.0,  8.0, 1.0),   # right mid — creates 1st local min
    (15.5, 16.5, 1.0),   # near goal
    ( 5.0, 14.0, 0.9),   # upper left
]

# ══════════════════════════════════════════════════════════════
#  APF PARAMETERS
# ══════════════════════════════════════════════════════════════
K_ATT         = 2.0    # attractive gain
K_REP         = 120.0  # repulsive gain
D0            = 2.5    # obstacle influence radius [m]
N_POWER       = 2      # exponent for d_goal modifier
STEP_SIZE     = 0.15   # movement per step [m]
MAX_ITER      = 2000   # safety cap

# Stuck detection
DISP_WINDOW   = 40     # rolling window of positions to track
DISP_THRESH   = 0.80   # net displacement < this in window => stuck
MIN_FREE_STEP = 80     # don't check until robot has moved this many steps
VIRTUAL_DIST  = 3.5    # side-step distance for virtual target [m]
VT_ARRIVE     = 0.55   # arrival tolerance for virtual target [m]
VT_MAX_STEPS  = 150    # abort virtual mode after this many steps

# ══════════════════════════════════════════════════════════════
#  FORCE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def attractive_force(pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """
    Pull robot toward current navigation goal.
    Returns unit-vector scaled by K_ATT.
    """
    diff = goal - pos
    dist = np.linalg.norm(diff)
    return K_ATT * diff / dist if dist > 1e-6 else np.zeros(2)


def repulsive_force_modified(pos: np.ndarray,
                              obstacles: list,
                              real_goal: np.ndarray) -> np.ndarray:
    """
    FIX 1 — Modified repulsive field.

    Key difference from traditional APF:
      Traditional F_rep pushes robot away at full strength always.
      Modified F_rep fades to zero as robot approaches the real goal.

    This prevents the 'goal-near-obstacle' deadlock:
      • Traditional: robot frozen next to obstacle next to goal
      • Modified:    repulsion melts away → robot glides to goal

    Secondary term (toward goal) kicks in inside influence zone
    to prevent any residual attractive-repulsive balance.
    """
    d_goal   = np.linalg.norm(real_goal - pos)
    goal_dir = (real_goal - pos) / (d_goal + 1e-9)
    total    = np.zeros(2)

    for (cx, cy, r) in obstacles:
        diff   = pos - np.array([cx, cy])
        d_surf = max(np.linalg.norm(diff) - r, 1e-3)   # dist to surface

        if d_surf < D0:
            away = diff / (np.linalg.norm(diff) + 1e-9)

            # Primary repulsion — fades as robot nears goal
            F1 = (K_REP
                  * (1.0/d_surf - 1.0/D0)
                  * (1.0/d_surf**2)
                  * (d_goal**N_POWER))
            total += F1 * away

            # Secondary attractor — gentle push toward goal
            F2 = ((N_POWER / 2.0)
                  * K_REP
                  * (1.0/d_surf - 1.0/D0)**2
                  * (d_goal**(N_POWER - 1)))
            total += F2 * goal_dir

    return total


def is_stuck(pos_history: deque) -> bool:
    """
    FIX 2 — Stuck detector using net displacement.

    If robot barely moved in DISP_WINDOW steps → oscillating → stuck.

    Why displacement (not step length)?
      Step length is always = STEP_SIZE (we normalise every move).
      But a stuck robot takes full steps while circling the same spot.
      Net displacement over N steps = near-zero when truly stuck.

    Example:
      40 steps × 0.15 m/step = 6.0 m maximum possible travel.
      If net displacement < 0.8 m → robot only moved 13% of max.
      Clearly looping or oscillating → STUCK.
    """
    if len(pos_history) < DISP_WINDOW:
        return False
    hist = list(pos_history)
    return np.linalg.norm(hist[-1] - hist[-DISP_WINDOW]) < DISP_THRESH


def find_virtual_target(pos: np.ndarray,
                         goal: np.ndarray,
                         obstacles: list) -> np.ndarray:
    """
    FIX 2 — Compute virtual sub-goal to escape the trap.

    1. Find effective obstacle (closest one = likely cause of trap).
    2. Build two perpendiculars to the robot→goal axis.
    3. Choose perpendicular pointing AWAY from effective obstacle.
    4. Virtual target = side-step + small forward nudge.

    The forward nudge (0.4 × VD) ensures net progress toward goal
    so the robot doesn't just move sideways forever.
    """
    # Effective obstacle
    best_d, eff_obs = float("inf"), np.zeros(2)
    for (cx, cy, r) in obstacles:
        obs = np.array([cx, cy])
        d   = np.linalg.norm(pos - obs) - r
        if d < best_d:
            best_d, eff_obs = d, obs

    g_vec  = goal - pos
    g_dist = np.linalg.norm(g_vec)
    if g_dist < 1e-6:
        return goal
    g_norm = g_vec / g_dist

    # Two perpendicular candidates
    perp_L = np.array([-g_norm[1],  g_norm[0]])
    perp_R = np.array([ g_norm[1], -g_norm[0]])

    # Pick the one pointing away from trap obstacle
    away = pos - eff_obs
    perp = perp_L if np.dot(perp_L, away) >= np.dot(perp_R, away) else perp_R

    vt = pos + perp * VIRTUAL_DIST + g_norm * (VIRTUAL_DIST * 0.4)
    return np.clip(vt, 1.0, WORLD - 1.0)


# ══════════════════════════════════════════════════════════════
#  SIMULATION
# ══════════════════════════════════════════════════════════════

def run_simulation():
    """
    Runs the full APF simulation with both fixes active.
    Returns path array (N,2) and events list.
    """
    pos         = START.copy().astype(float)
    path        = [pos.copy()]
    pos_hist    = deque(maxlen=DISP_WINDOW + 5)
    pos_hist.append(pos.copy())

    vt        = None    # current virtual target (None = not active)
    vt_timer  = 0
    free_step = 0       # steps since last reset (avoid premature detection)
    events    = []      # (step, type, data)

    for i in range(MAX_ITER):
        nav_goal = vt if vt is not None else GOAL

        # Compute & apply forces
        f_net = (attractive_force(pos, nav_goal)
                 + repulsive_force_modified(pos, OBSTACLES, GOAL))
        mag = np.linalg.norm(f_net)
        if mag > 1e-9:
            pos = np.clip(pos + (f_net / mag) * STEP_SIZE, 0.5, WORLD - 0.5)
        path.append(pos.copy())
        pos_hist.append(pos.copy())
        free_step += 1

        # Reached virtual target?
        if vt is not None:
            vt_timer += 1
            if np.linalg.norm(pos - vt) < VT_ARRIVE or vt_timer > VT_MAX_STEPS:
                events.append((i, "vt_cleared", pos.copy()))
                vt = None; vt_timer = 0; free_step = 0
                pos_hist.clear(); pos_hist.append(pos.copy())

        # Reached real goal?
        if np.linalg.norm(pos - GOAL) < 0.5:
            events.append((i, "goal_reached", pos.copy()))
            break

        # Stuck? → insert virtual target (FIX 2)
        if (vt is None
                and free_step > MIN_FREE_STEP
                and is_stuck(pos_hist)):
            new_vt = find_virtual_target(pos, GOAL, OBSTACLES)
            events.append((i, "stuck",  pos.copy()))
            events.append((i, "vt_set", new_vt.copy()))
            vt = new_vt; vt_timer = 0; free_step = 0
            pos_hist.clear(); pos_hist.append(pos.copy())

    return np.array(path), events


# ── Run ────────────────────────────────────────────────────────
print("Running APF simulation …")
path, events = run_simulation()

n_stuck = sum(1 for _, e, _ in events if e == "stuck")
reached = any(e == "goal_reached" for _, e, _ in events)
total_dist = sum(np.linalg.norm(path[i+1]-path[i]) for i in range(len(path)-1))

print(f"  Steps        : {len(path)}")
print(f"  Distance (m) : {total_dist:.1f}")
print(f"  Goal reached : {reached}")
print(f"  Local minima escaped (Fix 2): {n_stuck}")
print("  Event log:")
for s, e, d in events:
    print(f"    step {s:4d}  [{e:12s}]  @ {np.round(d,2)}")

# Per-step event lookup for animator
event_map: dict[int, list] = {}
for step, etype, edata in events:
    event_map.setdefault(step, []).append((etype, edata))


# ══════════════════════════════════════════════════════════════
#  ANIMATION  (dark theme, annotated)
# ══════════════════════════════════════════════════════════════
C_BG    = "#0d1117"
C_PANEL = "#161b22"
C_OBS   = "#e05252"
C_AURA  = "#e05252"
C_PATH  = "#4dabf7"
C_ROBOT = "#69db7c"
C_GOAL  = "#ffd43b"
C_START = "#74c0fc"
C_VT    = "#ff922b"
C_STUCK = "#ff6b6b"
C_GRID  = "#21262d"

fig, ax = plt.subplots(figsize=(8, 8), facecolor=C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, WORLD); ax.set_ylim(0, WORLD); ax.set_aspect("equal")
ax.tick_params(colors="#555", labelsize=8)
for sp in ax.spines.values():
    sp.set_edgecolor("#2d333b")
ax.set_xlabel("x  (m)", color="#8b949e", fontsize=9)
ax.set_ylabel("y  (m)", color="#8b949e", fontsize=9)
ax.set_title(
    "Artificial Potential Field — Local Minima Solutions\n"
    "FIX 1: Modified Repulsion  ·  FIX 2: Virtual Sub-goals",
    color="white", fontsize=10, pad=10
)

# Grid
for v in np.arange(0, WORLD + 1, 2):
    ax.axvline(v, color=C_GRID, lw=0.4, zorder=0)
    ax.axhline(v, color=C_GRID, lw=0.4, zorder=0)

# Influence auras
for (cx, cy, r) in OBSTACLES:
    ax.add_patch(plt.Circle((cx, cy), D0, color=C_AURA,
                             alpha=0.07, linewidth=0, zorder=1))

# Obstacle bodies
for (cx, cy, r) in OBSTACLES:
    ax.add_patch(plt.Circle((cx, cy), r, color=C_OBS,
                             alpha=0.88, linewidth=1.2, zorder=3))

# U-trap annotation
ax.text(11.0, 13.7, "U-trap", color="#ff9090", fontsize=8,
        ha="center", va="bottom", zorder=4,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#2a1515",
                  edgecolor="#ff6b6b", alpha=0.8))

# Start / Goal
ax.plot(*START, "o", color=C_START, ms=12, zorder=6, mec="white", mew=1.2)
ax.text(START[0]+0.3, START[1]-0.4, "Start", color=C_START, fontsize=8, va="top")
ax.plot(*GOAL, "*", color=C_GOAL, ms=18, zorder=6, mec="white", mew=1.0)
ax.text(GOAL[0]+0.3, GOAL[1]-0.5, "Goal", color=C_GOAL, fontsize=8, va="top")

# Animated artists
trail_line, = ax.plot([], [], "-", color=C_PATH, lw=1.8, alpha=0.55, zorder=4)
robot_dot,  = ax.plot([], [], "o", color=C_ROBOT, ms=11, zorder=7,
                       mec="white", mew=1.2)
dyn: list = []   # accumulates stuck marks & virtual target dots

# Status banner (top-right)
status_box = ax.text(
    WORLD - 0.3, WORLD - 0.4,
    "Navigating to goal…",
    ha="right", va="top", color="white", fontsize=8.5, zorder=10,
    bbox=dict(boxstyle="round,pad=0.45", facecolor=C_PANEL,
              edgecolor="#30363d", alpha=0.93)
)
# Step counter (top-left)
step_text = ax.text(
    0.3, WORLD - 0.4, "Step: 0",
    ha="left", va="top", color="#6e7681", fontsize=8, zorder=10
)
# Distance counter
dist_text = ax.text(
    0.3, WORLD - 1.1, "Dist to goal: —",
    ha="left", va="top", color="#6e7681", fontsize=8, zorder=10
)

# Legend
ax.legend(handles=[
    mpatches.Patch(color=C_OBS,             label="Obstacle (solid)"),
    mpatches.Patch(color=C_AURA, alpha=0.3, label=f"Influence zone  d₀={D0} m"),
    mpatches.Patch(color=C_PATH,            label="Robot path trail"),
    mpatches.Patch(color=C_ROBOT,           label="Robot"),
    mpatches.Patch(color=C_STUCK,           label="Local min detected  (×)"),
    mpatches.Patch(color=C_VT,              label="Virtual sub-goal  (◆)"),
], loc="lower right", facecolor=C_PANEL, edgecolor="#30363d",
   labelcolor="white", fontsize=7.5, framealpha=0.92)

# Mutable animation state
state = {"status": "Navigating to goal…", "bg": C_PANEL}


def init_anim():
    trail_line.set_data([], [])
    robot_dot.set_data([], [])
    return [trail_line, robot_dot, status_box, step_text, dist_text]


def update(frame):
    i = min(frame, len(path) - 1)

    trail_line.set_data(path[:i+1, 0], path[:i+1, 1])
    robot_dot.set_data([path[i, 0]], [path[i, 1]])
    step_text.set_text(f"Step: {i}")
    dist_text.set_text(f"Dist to goal: {np.linalg.norm(path[i]-GOAL):.1f} m")

    if i in event_map:
        for etype, edata in event_map[i]:

            if etype == "stuck":
                mk, = ax.plot(*edata, "x", color=C_STUCK,
                               ms=15, mew=2.8, zorder=8)
                dyn.append(mk)
                state["status"] = "⚠  Local Minimum detected!  Inserting virtual target…"
                state["bg"]     = "#3d0e0e"

            elif etype == "vt_set":
                # Diamond at virtual target position
                md, = ax.plot(*edata, "D", color=C_VT,
                               ms=9, mec="white", mew=0.9, zorder=8)
                # Dashed line: robot → virtual target
                ml, = ax.plot([path[i,0], edata[0]],
                               [path[i,1], edata[1]],
                               "--", color=C_VT, lw=1.2, alpha=0.55, zorder=5)
                dyn.extend([md, ml])

            elif etype == "vt_cleared":
                state["status"] = "✓  Virtual target reached — resuming to goal…"
                state["bg"]     = "#0e3d1e"

            elif etype == "goal_reached":
                state["status"] = "🎯  Goal Reached!"
                state["bg"]     = "#2d2808"

    status_box.set_text(state["status"])
    status_box.get_bbox_patch().set_facecolor(state["bg"])

    return [trail_line, robot_dot, status_box, step_text, dist_text] + dyn


# Subsample to ~600 frames (≈ 20 s at 30 fps)
N      = len(path)
skip   = max(1, N // 600)
frames = list(range(0, N, skip))
if frames[-1] != N - 1:
    frames.append(N - 1)

anim = FuncAnimation(
    fig, update,
    frames=frames,
    init_func=init_anim,
    interval=33,
    blit=False,
)
plt.tight_layout(pad=1.5)


# ── 1. Show live window ────────────────────────────────────────
print("\nShowing animation (close window to continue to MP4 export) …")
try:
    plt.show()
except Exception as exc:
    print(f"  plt.show() skipped ({exc})")


# ── 2. Save MP4 ───────────────────────────────────────────────
out = "/mnt/user-data/outputs/apf_simulation.mp4"
print(f"Saving MP4 → {out} …")
writer = FFMpegWriter(fps=30, bitrate=2000,
                      metadata={"title": "APF Local Minima Solution"})
anim.save(out, writer=writer, dpi=120)
print("Done! ✓")