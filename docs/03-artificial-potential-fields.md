# Guide 3: Artificial Potential Fields

> **What this explains:** How invisible "magnet" forces let a robot dodge moving obstacles in real time — and how `apf.py` does this.

---

## The Big Idea

Imagine this:

- Your **goal** is a giant magnet that **pulls** you toward it.
- Every **obstacle** is surrounded by an invisible force field that **pushes** you away.

You're a tiny metal ball. You just roll wherever the combined pull and push takes you.

That's **Artificial Potential Fields (APF)**. The entire algorithm is just that — invisible forces.

> **"Potential field"** is a physics term. In everyday language, it just means "an area where forces act on things." Like a gravitational field around Earth — you feel pulled toward the ground without anything touching you.

---

## The Two Forces

### 1. Attractive Force (The Pull Toward the Goal)

The goal acts like a magnet. The farther away you are, the harder it pulls.

```
Attractive Force = k_att × (goal_position - current_position)
```

**What this means:**

- `goal_position - current_position` is a **vector** — an arrow pointing from where you are to where you want to go. It has both a direction and a length.
- `k_att` is a number that controls **how strong the pull is**.

> **What is a vector?** Think of it as an arrow. The arrow points in a direction and has a length. In our case, the arrow points toward the goal, and its length is proportional to the distance.

**Simple analogy:** Imagine the goal is a rubber band attached to the robot. The farther the robot is from the goal, the more stretched the rubber band is, and the harder it pulls.

### 2. Repulsive Force (The Push Away from Obstacles)

Each obstacle has a "danger zone" around it. If the robot enters this zone, the obstacle pushes it away.

```
Repulsive Force = k_rep × (1/distance - 1/rho_0) × (1/distance²) × direction_away
```

**Don't panic at this formula.** Let's break it down piece by piece.

| Part | What It Does |
|------|-------------|
| `1/distance - 1/rho_0` | Gets bigger as you get closer to the obstacle. At the edge of the danger zone (`rho_0`), this is zero. At the obstacle's surface, it's huge. |
| `1/distance²` | Makes the push **explode** as you get very close. This ensures the robot really really avoids collisions. |
| `direction_away` | An arrow pointing from the obstacle to the robot — "push this way" |
| `k_rep` | Controls the overall strength of the push |

**Simple analogy:** Imagine each obstacle is surrounded by a springy bubble. The closer you get to the obstacle, the harder the bubble pushes you back. Touch the obstacle and the spring is infinitely strong — you can't get through.

---

## The Danger Zone (rho_0)

```
rho_0 = 15.0
```

This is the **radius of influence** around each obstacle. If the robot is more than 15 units away from an obstacle, that obstacle exerts **zero** repulsive force. The robot doesn't even "feel" it.

But once the robot enters the 15-unit zone, the repulsive force kicks in and gets stronger the closer it gets.

> **Think of it like a personal space bubble.** If someone is far away, you don't care. If they step into your personal space, you start feeling uncomfortable and want to move away. The closer they get, the more uncomfortable you feel.

---

## Combining the Forces

The robot adds the two forces together:

```
Total Force = Attractive Force + Repulsive Force
```

Then it moves a small step in the direction of the total force.

```
step_vector = (Total Force / |Total Force|) × step_size
```

**What this means:**

1. `Total Force / |Total Force|` — This **normalizes** the force. It turns the force into a unit vector (an arrow of length 1) that just tells us the direction.
2. `× step_size` — This scales the arrow to our desired step length.

**Simple analogy:** Imagine someone is pulling you toward the goal while someone else is pushing you away from obstacles. You take one small step in whatever direction the combined tug-of-war sends you. Then you recalculate. Then take another step. Repeat until you reach the goal.

---

## The Parameters and What They Do

Here are all the knobs you can turn, and what happens when you turn them:

### k_att (Attractive Gain) = 2.0

**What it controls:** How aggressively the robot moves toward the goal.

| If you increase it | If you decrease it |
|---|---|
| Robot charges toward the goal | Robot meanders lazily |
| Might ignore obstacles and crash | More cautious, but slower |

> **Think of it like:** How hungry the robot is for the goal. High k_att = very hungry, will rush. Low k_att = not in a hurry.

### k_rep (Repulsive Gain) = 50,000.0

**What it controls:** How strongly obstacles push the robot away.

| If you increase it | If you decrease it |
|---|---|
| Robot gives obstacles a very wide berth | Robot cuts corners closer to obstacles |
| Very safe, but might take long detours | Risky — might clip obstacles |

> **Yes, 50,000 seems huge.** It is! But remember the formula has `1/distance²` in it, which makes the numbers very small at typical distances. The large k_rep compensates for that.

### rho_0 (Influence Radius) = 15.0

**What it controls:** How far away the robot starts "feeling" obstacles.

| If you increase it | If you decrease it |
|---|---|
| Robot detects obstacles from farther away | Robot only reacts when very close |
| Takes wider, smoother avoidance paths | Might react too late and get stuck |

> **Think of it like:** How far ahead a driver looks. Good drivers scan far ahead (large rho_0). Bad drivers only notice things right in front of them (small rho_0).

### step_size = 0.5

**What it controls:** How big each movement step is.

| If you increase it | If you decrease it |
|---|---|
| Robot moves faster but more jerkily | Robot moves slower but more smoothly |
| Might overshoot and miss narrow gaps | Very precise, but takes longer |

> **Think of it like:** The size of your footsteps. Big steps = fast but clumsy. Small steps = slow but precise.

### wp_threshold = 3.0

**What it controls:** How close the robot needs to get to a waypoint before it moves on to the next one.

| If you increase it | If you decrease it |
|---|---|
| Robot accepts "close enough" more easily | Robot tries to hit waypoints more precisely |
| Faster progress, but might cut corners | Slower, but follows the planned path more closely |

---

## Why It's "Real-Time"

The key insight: **the forces are recalculated every single frame.**

This means:
- If an obstacle moves, the repulsive force changes instantly
- The robot doesn't need to re-plan its entire path
- It just reacts to the new forces

> **Analogy:** Imagine you're walking through a crowded room. You don't plan your entire route before you start walking. You just keep walking toward the exit while adjusting your path around people as they move. That's real-time reactive navigation.

This is why APF is perfect for handling **moving obstacles** — the ones that change position every frame. The robot doesn't need to predict where they'll be. It just reacts to where they are right now.

---

## The Weakness: Local Minima

APF has a famous problem called **local minima**. Here's what happens:

Imagine the robot is between two obstacles. The obstacle on the left pushes it right. The obstacle on the right pushes it left. The forces **cancel out**.

The robot is stuck. The total force is zero. It doesn't move. But the goal is still far away.

This is like being in a valley between two hills. The goal is on the other side, but you're stuck at the bottom with no force pushing you out.

### How We Solve It

We don't use APF alone. We use it **with** Hybrid A* (the global planner from Guide 2).

- **Hybrid A*** plans the overall route in advance, avoiding all known static obstacles
- **APF** handles the real-time dodging of moving obstacles along that route

The waypoints from Hybrid A* act as a guide. Even if APF gets confused momentarily, the next waypoint pulls it back on track.

> **Analogy:** Hybrid A* is your GPS giving you the route. APF is you, the driver, swerving to avoid a pothole. The GPS doesn't need to recalculate — you just dodge and get back on the route.

---

## What the Code Does (Step by Step)

Here's the APF section of `apf.py`, explained in plain English:

### 1. Set up the visual elements

```python
fig, ax = plt.subplots(figsize=(8, 8))
```

**Translation:** "Create an 8×8 inch drawing window."

### 2. Draw the static elements

```python
ax.plot(full_path_arr[:, 0], full_path_arr[:, 1], 'k--', alpha=0.5, label='Hybrid A* Smooth Plan')
ax.scatter(waypoints[:-1, 0], waypoints[:-1, 1], c='orange', s=50, marker='s', label='Waypoints')
ax.scatter(start_state[0], start_state[1], c='green', s=150, zorder=5, label='Start')
ax.scatter(goal[0], goal[1], c='gold', s=200, marker='*', zorder=5, label='Goal')
```

**Translation:** "Draw the planned path as a dashed black line. Draw waypoints as orange squares. Mark the start (green dot) and goal (gold star)."

### 3. Draw the obstacles and their danger zones

```python
obs_scatter = ax.scatter(obstacles[:, 0], obstacles[:, 1], c='red', s=100, marker='X', label='Obstacles')
obs_circles = [plt.Circle((obs[0], obs[1]), safety_radius, color='r', fill=True, alpha=0.2) for obs in obstacles]
for circle in obs_circles:
    ax.add_patch(circle)
```

**Translation:** "Draw red X marks at each obstacle's position. Draw a semi-transparent red circle around each one showing the danger zone (safety_radius = 12 units)."

### 4. Set up the robot's trail

```python
trail_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.8, label='APF Executed Path')
robot_dot, = ax.plot([], [], 'bo', markersize=8)
```

**Translation:** "Create an empty blue line (the robot's trail) and an empty blue dot (the robot itself). They start empty and get filled in during the animation."

### 5. The animation loop

```python
def animate(frame):
```

This function runs **500 times** (once per frame). Each time, it:

**A. Moves the obstacles:**

```python
obstacles[1][0] = 50.0 + 20.0 * math.sin(frame * 0.05)
obstacles[2][1] = 20.0 + 15.0 * math.cos(frame * 0.05)
```

**Translation:** "Obstacle 1 moves left and right in a smooth wave pattern. Obstacle 2 moves up and down in a smooth wave pattern. The other two obstacles stay still."

> **What are sin and cos?** They're math functions that create smooth back-and-forth motion. Think of a pendulum swinging — that's a sine wave. `sin` and `cos` are the same wave, just shifted. We use them because they create natural-looking, smooth movement.

**B. Calculates the forces:**

```python
F_att = k_att * (target - current_pos)
F_rep = np.array([0.0, 0.0])

for obs in obstacles:
    d = math.dist(current_pos, obs)
    if 0 < d < rho_0:
        direction = (current_pos - obs) / d
        magnitude = k_rep * (1.0/d - 1.0/rho_0) * (1.0 / (d**2))
        F_rep += direction * magnitude

F_total = F_att + F_rep
```

**Translation:**
1. Calculate the attractive force (pull toward current waypoint)
2. Start with zero repulsive force
3. For each obstacle: if it's within the danger zone, calculate how hard it pushes and add that to the total repulsive force
4. Add attractive and repulsive forces together

**C. Move the robot:**

```python
force_magnitude = np.linalg.norm(F_total)
if force_magnitude > 0:
    step_vector = (F_total / force_magnitude) * step_size
    current_pos += step_vector
```

**Translation:** "If there's any net force, take a step in that direction. The step size is fixed at 0.5 units."

> **`np.linalg.norm`** just calculates the length of the force vector. It tells us "how strong is the total force?"

**D. Record the trail:**

```python
path_x.append(current_pos[0])
path_y.append(current_pos[1])
```

**Translation:** "Remember where the robot is so we can draw its path."

**E. Check if we reached the waypoint:**

```python
if math.dist(current_pos, target) < wp_threshold:
    current_wp_idx += 1
```

**Translation:** "If we're within 3 units of the current waypoint, move on to the next one."

---

## Real-World Examples

### Robot Vacuum Cleaners

A Roomba uses something very similar to APF. The room boundaries and furniture legs act as repulsive forces. The uncleaned areas act as attractive forces. The robot just follows the combined forces.

### Drones Dodging Trees

A drone flying through a forest uses APF to avoid trees in real time. The GPS gives it the general route (global plan), but APF handles the last-second dodging of individual trees and branches.

### Robotic Arms in Factories

Industrial robot arms use APF to avoid colliding with each other, with workers, and with equipment. Each obstacle creates a repulsive field, and the target position creates an attractive field.

### Video Game AI

Many video games use APF for non-player character (NPC) movement. Enemies are attracted to the player but repelled by walls and other enemies. It creates natural-looking movement without complex pathfinding.

### Self-Driving Cars

While self-driving cars use much more sophisticated systems, the core idea of APF is present: stay attracted to your lane and route while being repelled by other vehicles, pedestrians, and road boundaries.

---

## Key Terms Glossary

| Term | Simple Definition |
|------|-------------------|
| **Artificial Potential Field (APF)** | A navigation method using invisible attractive and repulsive forces |
| **Attractive force** | The pull toward the goal (like a magnet) |
| **Repulsive force** | The push away from obstacles (like a force field) |
| **k_att** | How strong the attractive pull is |
| **k_rep** | How strong the repulsive push is |
| **rho_0** | The radius of an obstacle's "danger zone" |
| **step_size** | How far the robot moves each frame |
| **wp_threshold** | How close to a waypoint counts as "reached" |
| **Vector** | An arrow with a direction and a length |
| **Normalize** | Scale a vector to length 1 (keep direction, remove magnitude) |
| **Local minima** | When forces cancel out and the robot gets stuck |
| **Real-time** | Recalculating every frame to react to changes instantly |
| **Reactive** | Responding to the current situation rather than following a pre-planned route |
| **Sine/Cosine wave** | A smooth back-and-forth motion pattern (like a pendulum) |

---

**Previous:** [Guide 2: Pathfinding & Hybrid A*](02-pathfinding-and-hybrid-a-star.md)
**Next:** [Guide 4: How It All Works Together](04-how-it-all-works-together.md) — Now let's see how both planners combine into one complete system.
