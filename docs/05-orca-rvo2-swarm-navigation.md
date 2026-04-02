# Guide 5: ORCA/RVO2 Swarm Navigation

> **What this explains:** How multiple robots navigate through each other without colliding — using velocity-based collision avoidance instead of forces.

---

## The Problem

Imagine you're walking down a crowded hallway. Someone is walking toward you from the opposite direction.

What happens?

You both adjust your paths slightly — maybe you step a bit to the right, they step a bit to their right — and you pass each other smoothly. Neither of you stops. Neither of you crashes. You both just... figure it out.

Now imagine **10 people** all trying to cross the same hallway at the same time, each going to a different spot on the other side.

That's **multi-agent collision avoidance**. And it's much harder than avoiding a single obstacle.

---

## Why APF Isn't Enough for Swarms

In Guide 3, we learned about Artificial Potential Fields (APF). APF works great for **one robot avoiding obstacles**. But it has serious problems when **multiple robots** are involved:

| Problem | What Happens |
|---------|-------------|
| **Symmetry** | Two robots heading toward each other both apply the same repulsive force. They both step the same direction. They still collide. |
| **No coordination** | Each robot acts independently. It doesn't know that the other robot is also trying to avoid it. |
| **Oscillation** | Robots can get stuck going back and forth, never making progress. |
| **Deadlock** | All forces cancel out and nobody moves. |

> **The "two people in a hallway" problem:** You've definitely experienced this. You're walking toward someone. You step right. They also step right (their left). You both step the other way. Same thing. You do this 2-3 times before one of you laughs and goes the other way. This is exactly what happens to robots using simple force-based avoidance.

---

## Enter ORCA: Optimal Reciprocal Collision Avoidance

ORCA takes a completely different approach from APF.

### The Key Insight

Instead of thinking about **forces** ("push me away from that obstacle"), ORCA thinks about **velocities** ("what speed and direction can I safely move at?").

Here's the big idea:

1. **Every robot has a preferred velocity** — the speed and direction it *wants* to go (usually straight toward its goal).
2. **Every other robot creates a "forbidden zone"** of velocities that would lead to a collision.
3. **The robot picks the safest velocity** that's as close as possible to its preferred velocity but outside all forbidden zones.

> **Analogy:** Imagine you're driving and there's a "no entry" sign on one road. You don't push against the sign (that's what APF does). You just pick a different road that's as close to your preferred route as possible (that's what ORCA does).

### The "Reciprocal" Part

This is what makes ORCA special. When two robots are on a collision course, **they share the responsibility** of avoiding each other.

- Robot A says: "I'll adjust my velocity by half of what's needed."
- Robot B says: "I'll adjust my velocity by the other half."

Both robots move a little, and together they avoid the collision. This is much more natural and efficient than one robot doing all the dodging.

> **Real-world analogy:** When two people walk toward each other in a hallway, both usually adjust slightly. It's a shared effort. ORCA formalizes this intuition mathematically.

---

## Velocity Obstacles: The Core Concept

A **velocity obstacle** is the set of all velocities that would cause a collision with another agent within a certain time window.

### How It Works (The Simple Version)

Imagine Robot A and Robot B. Robot A wants to know: "Which of my possible velocities would cause me to crash into Robot B in the next few seconds?"

For each possible velocity, Robot A asks: "If I move at this speed and direction, will I hit Robot B?"

- If yes → that velocity is **forbidden**
- If no → that velocity is **safe**

The forbidden velocities form a cone-shaped region in velocity space. This is the **velocity obstacle**.

Robot A then picks the safest velocity that's closest to its preferred velocity (straight toward its goal).

### The Time Horizon

ORCA doesn't worry about collisions that are very far in the future. It only considers collisions within a **time horizon** — say, the next 2 seconds.

This is practical: planning too far ahead is wasteful because things change. Planning just far enough ahead gives the robot time to react.

> **Analogy:** When you drive, you don't plan to avoid a car that's 10 miles away. You focus on the next few seconds. That's your time horizon.

---

## The RVO2 Library

We don't implement ORCA from scratch. We use a library called **RVO2** (Reciprocal Velocity Obstacles, version 2). It's a well-tested, optimized implementation of the ORCA algorithm.

### Setting Up the Simulator

```python
sim = rvo2.PyRVOSimulator(1/60.0, 2.5, 10, 2.0, 2.0, 0.3, 1.5)
```

This creates the simulation world. Let's break down every parameter:

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| **timeStep** | 1/60.0 | How much time passes each simulation step (1/60th of a second = 60 Hz) |
| **neighborDist** | 2.5 | How far a robot looks for other robots (only considers neighbors within this distance) |
| **maxNeighbors** | 10 | Maximum number of neighbors each robot considers at once |
| **timeHorizon** | 2.0 | How far ahead each robot plans to avoid collisions (in seconds) |
| **timeHorizonObst** | 2.0 | How far ahead each robot plans to avoid static obstacles |
| **radius** | 0.3 | The physical size of each robot (how big the collision circle is) |
| **maxSpeed** | 1.5 | The maximum speed any robot can move at |

### Adding Robots

```python
for i in range(num_agents):
    angle = i * (2 * math.pi / num_agents)
    start_x = circle_radius * math.cos(angle)
    start_y = circle_radius * math.sin(angle)
    agent_id = sim.addAgent((start_x, start_y))
```

Each robot is added to the simulator at a specific position. The simulator tracks each robot's position, velocity, and preferred velocity.

### Setting Preferred Velocities

```python
vector_x = goal[0] - pos[0]
vector_y = goal[1] - pos[1]
dist = math.hypot(vector_x, vector_y)

if dist > 0.1:
    pref_vx = (vector_x / dist) * 1.5
    pref_vy = (vector_y / dist) * 1.5
else:
    pref_vx, pref_vy = 0.0, 0.0

sim.setAgentPrefVelocity(agent_id, (pref_vx, pref_vy))
```

**Translation:** "Calculate the direction from where I am to where I want to go. Normalize it (make it length 1). Multiply by my desired speed (1.5). That's my preferred velocity."

### Taking a Step

```python
sim.doStep()
```

This is where the magic happens. When you call `doStep()`, the RVO2 library:

1. Looks at every robot's preferred velocity
2. Calculates velocity obstacles for every pair of nearby robots
3. Finds the safest velocity for each robot (closest to preferred but outside all forbidden zones)
4. Moves each robot to its new position

All of this happens in one call. The library handles all the math.

---

## The Circle Cross: The Ultimate Test

Our simulation uses what we call the **circle cross** — the hardest test for any collision avoidance algorithm.

### The Setup

- **10 robots** arranged in a circle (radius 5 units)
- Each robot's goal is the **exact opposite point** on the circle
- All 10 robots start moving at the same time

```
        Goal 5
          ●
    ●           ●
  Goal 4       Goal 6
    |           |
●---●-----------●---●
R4  |     R     |  R6
    |           |
  Goal 3       Goal 7
    ●           ●
          ●
        Goal 2
```

(Every robot must cross through the center, where all paths intersect.)

### Why This Is Hard

All 10 robots want to go through the center of the circle at the same time. Their paths all cross. If the algorithm isn't good, they'll all get stuck in the middle, pushing against each other forever.

A good algorithm will have them:
- Coordinate their movements
- Take turns passing through the center
- Avoid collisions smoothly
- All reach their goals efficiently

### Symmetry-Breaking Noise

```python
noise_angle = random.uniform(0, 2 * math.pi)
noise_dist = random.uniform(0, 0.05)
pref_vx += math.cos(noise_angle) * noise_dist
pref_vy += math.sin(noise_angle) * noise_dist
```

This adds a **tiny random nudge** to each robot's preferred velocity. It's very small (0 to 0.05 units) — barely noticeable.

**Why?** Because perfect symmetry causes deadlock. If two robots are perfectly mirrored, they'll make perfectly mirrored decisions, which means they'll still collide. The tiny random nudge breaks the symmetry and lets them make slightly different decisions.

> **The hallway analogy again:** When you and another person keep stepping the same way, eventually one of you does something slightly different — maybe you hesitate for a fraction of a second, or step at a slightly different angle. That tiny difference breaks the deadlock. The noise in our code does the same thing.

---

## Velocity Vectors: The Arrows

Our visualization shows **arrows** on each robot. These are called **quiver plots** in matplotlib.

Each arrow shows the robot's **current velocity** — which way it's moving and how fast.

- **Arrow direction** = which way the robot is moving
- **Arrow length** = how fast the robot is moving

When you watch the animation, you'll see:
- Arrows pointing toward goals when robots are moving freely
- Arrows curving away from other robots when they're avoiding collisions
- Arrows getting shorter when robots slow down to let others pass
- Arrows getting longer when robots speed up after the path is clear

---

## What the Code Does (Step by Step)

### Step 1: Create the Simulator

```python
sim = rvo2.PyRVOSimulator(1/60.0, 2.5, 10, 2.0, 2.0, 0.3, 1.5)
```

**Translation:** "Create a simulation world where time moves in 1/60-second steps, robots look 2.5 units for neighbors, consider up to 10 neighbors, plan 2 seconds ahead, are 0.3 units in radius, and can move at most 1.5 units per second."

### Step 2: Place the Robots

```python
num_agents = 10
circle_radius = 5.0

for i in range(num_agents):
    angle = i * (2 * math.pi / num_agents)
    start_x = circle_radius * math.cos(angle)
    start_y = circle_radius * math.sin(angle)
    goal_x = -start_x
    goal_y = -start_y
    agent_id = sim.addAgent((start_x, start_y))
    goals.append((goal_x, goal_y))
```

**Translation:** "Place 10 robots evenly around a circle of radius 5. Each robot's goal is the point directly opposite it on the circle (negate both x and y)."

### Step 3: Set Up the Visuals

```python
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
```

**Translation:** "Create an 8×8 inch drawing window showing the area from -7 to 7 on both axes (a bit bigger than the circle of radius 5)."

### Step 4: The Animation Loop

```python
def animate(frame):
```

This function runs **400 times** (once per frame). Each time:

**A. For each robot, calculate its preferred velocity:**

```python
for i, agent_id in enumerate(agents):
    pos = sim.getAgentPosition(agent_id)
    vel = sim.getAgentVelocity(agent_id)
    goal = goals[i]
    
    vector_x = goal[0] - pos[0]
    vector_y = goal[1] - pos[1]
    dist = math.hypot(vector_x, vector_y)
    
    if dist > 0.1:
        pref_vx = (vector_x / dist) * 1.5
        pref_vy = (vector_y / dist) * 1.5
        # Add tiny random noise
        noise_angle = random.uniform(0, 2 * math.pi)
        noise_dist = random.uniform(0, 0.05)
        pref_vx += math.cos(noise_angle) * noise_dist
        pref_vy += math.sin(noise_angle) * noise_dist
    else:
        pref_vx, pref_vy = 0.0, 0.0
    
    sim.setAgentPrefVelocity(agent_id, (pref_vx, pref_vy))
```

**Translation:** "For each robot: figure out which way its goal is, set that as the preferred velocity, add a tiny random nudge, and tell the simulator."

**B. Run the simulation step:**

```python
sim.doStep()
```

**Translation:** "RVO2, figure out safe velocities for everyone and move them."

**C. Update the visuals:**

```python
pos_arr = np.array(current_positions)
scat_agents.set_offsets(pos_arr)

vel_arr = np.array(current_velocities)
quiver.set_offsets(pos_arr)
quiver.set_UVC(vel_arr[:, 0], vel_arr[:, 1])
```

**Translation:** "Update the robot dots to their new positions. Update the arrows to show their new velocities."

---

## ORCA vs. APF: A Comparison

| Aspect | APF (Guide 3) | ORCA (This Guide) |
|--------|--------------|-------------------|
| **Approach** | Force-based (push/pull) | Velocity-based (safe speeds) |
| **Best for** | Single robot, dynamic obstacles | Multiple robots, mutual avoidance |
| **Collision model** | Repulsive force fields | Velocity obstacle cones |
| **Coordination** | None (each robot acts alone) | Reciprocal (robots share avoidance) |
| **Deadlock risk** | High (local minima) | Low (velocity space is richer) |
| **Computation** | Simple per-frame calculation | More complex but optimized in library |
| **Analogy** | Magnets pushing and pulling | Choosing safe lanes on a highway |

---

## Real-World Examples

### Swarm Drones

Light shows with hundreds of drones use ORCA-like algorithms. Each drone needs to fly to its position in the formation without colliding with any other drone. The reciprocal nature means all drones adjust smoothly.

### Crowd Simulation

Video games and movies use ORCA to simulate realistic crowds. NPCs (non-player characters) walk through crowds naturally, avoiding each other the way real people do.

### Warehouse Robots

In warehouses with dozens of robots moving simultaneously, ORCA ensures they don't collide at intersections. Each robot plans its velocity considering all nearby robots.

### Autonomous Forklifts

In factories, self-driving forklifts use velocity-based collision avoidance to navigate shared spaces. The reciprocal approach means both forklifts adjust, just like human drivers would.

### Robot Soccer

RoboCup (robot soccer competitions) uses ORCA for multi-robot coordination. Players need to move to positions on the field while avoiding teammates and opponents.

### Pedestrian Modeling

Urban planners use ORCA to simulate pedestrian flow in stadiums, train stations, and shopping malls. This helps design spaces that handle crowds efficiently.

---

## Key Terms Glossary

| Term | Simple Definition |
|------|-------------------|
| **ORCA** | Optimal Reciprocal Collision Avoidance — velocity-based multi-agent avoidance |
| **RVO2** | Reciprocal Velocity Obstacles, version 2 — the library implementing ORCA |
| **Velocity obstacle** | The set of velocities that would cause a collision |
| **Preferred velocity** | The speed and direction a robot *wants* to go (usually toward its goal) |
| **Reciprocal** | Both agents share the responsibility of avoiding each other |
| **Time horizon** | How far ahead the robot plans to avoid collisions |
| **Neighbor distance** | How far a robot looks for other robots |
| **Symmetry-breaking noise** | Tiny random nudges to prevent deadlock from perfect symmetry |
| **Quiver plot** | A visualization showing velocity vectors as arrows |
| **Deadlock** | When robots get stuck because their avoidance decisions cancel out |
| **Time step** | How much simulation time passes between each update |
| **Agent** | A robot or moving entity in the simulation |
| **Multi-agent** | Involving multiple robots operating simultaneously |
| **Swarm** | A group of robots working in the same space |

---

**Previous:** [Guide 4: How It All Works Together](04-how-it-all-works-together.md)
**Back to:** [Documentation Index](README.md)
