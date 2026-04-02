# Guide 2: Pathfinding & Hybrid A*

> **What this explains:** How a robot finds a path from point A to point B when it has to actually *drive* — not teleport — and how `apf.py` does this.

---

## The Problem

Imagine you're in a parking lot. You're at one end, and you need to get to the other end. There are parked cars everywhere.

Your job: find a smooth, drivable path from where you are to where you want to go, without hitting anything.

Sounds simple, right? But there's a catch:

**You're driving a car.** You can't:
- Teleport sideways
- Spin in place instantly
- Turn sharper than your steering wheel allows
- Go through walls (the parked cars)

This is the fundamental problem of **robot pathfinding**. And the solution we use is called **Hybrid A***.

---

## First, What Is A*?

A* (pronounced "A-star") is one of the most famous algorithms in computer science. It finds the shortest path from a starting point to a goal.

### How Regular A* Works (The Simple Version)

Imagine a chessboard. You're on one square and want to get to another square. You can move to adjacent squares. Some squares are blocked (obstacles).

A* works like this:

1. **Start at your current square.**
2. **Look at all the squares you can move to.**
3. **For each one, calculate a "cost":**
   - **g-cost** = How far you've already traveled to get here
   - **h-cost** = A guess of how far you still have to go (straight-line distance to the goal)
   - **Total cost (f-cost)** = g-cost + h-cost
4. **Pick the square with the lowest total cost.**
5. **Repeat steps 2–4 until you reach the goal.**

The "guess" (h-cost) is called a **heuristic**. It's what makes A* smart — it doesn't blindly explore in every direction. It has a sense of which way the goal is.

### The Problem With Regular A*

Regular A* works on a **grid** — like a chessboard. Each step, you jump from one square to the next.

**But real vehicles don't work like that.**

A car:
- Has a specific **direction** it's facing
- Can only **turn so sharply**
- Moves in **smooth curves**, not square-by-square jumps

If you used regular A* for a car, you'd get a path that looks like a jagged staircase. The car literally could not follow it.

---

## Enter Hybrid A*

Hybrid A* fixes this problem by planning in **continuous space** instead of a grid.

### What Does "Continuous Space" Mean?

Instead of jumping between squares, the robot can be at **any position** on the map. Any real number. Not just (10, 10) or (11, 10) — but (10.347, 10.891) if that's where it needs to be.

### The State: (x, y, θ)

Every position the robot can be in is described by **three numbers**:

| Symbol | What It Means | Example |
|--------|--------------|---------|
| **x** | How far right on the map | 10.0 |
| **y** | How far up on the map | 10.0 |
| **θ** (theta) | Which direction the robot is facing (in degrees or radians) | 45° |

> **θ (theta)** is just a Greek letter mathematicians use for angles. Don't let it scare you. It just means "which way is the front of the robot pointing?"

So the robot doesn't just know *where* it is — it also knows *which way it's facing*. This is crucial because a car facing north can't instantly move east. It has to turn first.

---

## The Bicycle Model

How does the robot actually move? We use something called the **bicycle model**.

Don't worry — we're not building a bicycle. It's just a simplified way to think about how a vehicle moves. Imagine a car as having two wheels: one in front (that steers) and one in back (that doesn't). That's the "bicycle model."

### The Three Things That Control Movement

| Parameter | Symbol | What It Does | Our Value |
|-----------|--------|-------------|-----------|
| **Velocity** | V | How fast the robot moves forward | 2.0 units/step |
| **Wheelbase** | L | Distance between front and back wheels | 3.0 units |
| **Steering angle** | δ (delta) | How much the front wheels are turned | -30°, 0°, or +30° |

### The Physics Equations (Explained Simply)

At each step, the robot calculates its new position using these formulas:

```
x_new = x + V × cos(θ) × dt
y_new = y + V × sin(θ) × dt
θ_new = θ + (V / L) × tan(steering_angle) × dt
```

**What this means in plain English:**

- **x_new**: "Move forward in the direction I'm facing." `cos(θ)` gives the horizontal component of your direction.
- **y_new**: Same thing, but the vertical component.
- **θ_new**: "Turn based on my steering angle." The sharper the steering, the more you turn. A longer wheelbase (bigger L) means you turn less sharply — think about how a long truck needs more room to turn than a small car.

> **dt** is the "time step" — how much time passes between each calculation. Think of it as the size of each tiny movement. Our value is 1.0.

---

## How the Algorithm Explores

At every position, the robot has **3 choices**:

1. **Turn left** (steering = -30°)
2. **Go straight** (steering = 0°)
3. **Turn right** (steering = +30°)

For each choice, it calculates where the robot would end up and whether that position is safe (not inside an obstacle and within the map boundaries).

Then it uses the A* cost function to decide which path to explore next:

```
f-cost = g-cost + h-cost + steering_penalty
```

| Part | What It Means |
|------|--------------|
| **g-cost** | Total distance traveled so far |
| **h-cost** | Straight-line distance from current position to the goal (the "guess") |
| **steering_penalty** | A small extra cost for turning — this encourages the robot to go straight when possible, because straight paths are smoother and more efficient |

The **steering penalty** is the clever part. Without it, the robot might zigzag unnecessarily. With it, the robot prefers smooth, straight-ish paths.

---

## Keeping Track: The "Visited" Set

The algorithm keeps a list of states it has already explored. This prevents it from going in circles.

But since positions are continuous (any real number), we can't just check "have I been at exactly (10.347, 10.891) before?" — we'd almost never find a match.

Instead, we **round** the position:
- x and y are rounded to the nearest integer
- θ is rounded to the nearest 15 degrees

So (10.347, 10.891, 45.2°) becomes (10, 10, 3) — where 3 means "3 × 15° = 45°."

This is called **discretization**. We're creating a rough "memory grid" on top of the smooth continuous space, just to remember where we've been.

---

## The Goal Check

How does the algorithm know it's done?

```python
if math.dist((x, y), goal) < 5.0:
    return path
```

**Translation:** "If I'm within 5 units of the goal, I'm close enough. Stop searching and give me the path."

We don't need to land on the exact pixel — being "close enough" is good enough for a real robot.

---

## From Path to Waypoints

The Hybrid A* algorithm produces a path with many tiny steps (in our case, 63 steps). Each step is very small — just one frame of movement.

For the next stage of our system, we don't need every single tiny step. We just need **checkpoints** along the way.

```python
waypoints = full_path[::5]
```

**Translation:** "Take every 5th step from the full path." This gives us a shorter list of key positions the robot should pass through on its way to the goal.

Think of it like a road trip: the full path is every single meter of the road. The waypoints are the cities you'll pass through. You don't need to plan every meter — just know which cities to aim for.

---

## What the Code Does (Step by Step)

Here's a walkthrough of the Hybrid A* section of `apf.py`:

### 1. Set up the map

```python
grid_size = 100
start_state = (10.0, 10.0, math.radians(45))
goal = (90.0, 90.0)
```

**Translation:** "The map is 100×100. The robot starts at position (10, 10) facing 45 degrees (northeast). The goal is at (90, 90)."

### 2. Define the obstacles

```python
obstacles = np.array([
    [20.0, 80.0],
    [50., 50.0],
    [80.0, 20.0],
    [80.0, 85.0]
])
safety_radius = 12.0
```

**Translation:** "There are 4 obstacles on the map. The robot must stay at least 12 units away from each one."

### 3. Define the vehicle

```python
V = 2.0
L = 3.0
dt = 1.0
max_steer = math.radians(30)
steer_angles = [-max_steer, 0.0, max_steer]
```

**Translation:** "The robot moves at speed 2.0, has a wheelbase of 3.0, and can steer up to 30 degrees left or right. At each step, it considers three options: turn left, go straight, or turn right."

### 4. The safety check

```python
def is_safe(x, y):
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return False
    for obs in obstacles:
        if math.dist((x, y), obs) < safety_radius + 2.0:
            return False
    return True
```

**Translation:** "A position is safe if it's inside the map AND far enough from all obstacles. We add an extra 2-unit buffer for the global plan to be extra cautious."

### 5. The search algorithm

The `hybrid_a_star()` function runs the full search:
- Uses a **priority queue** (a special list that always gives you the cheapest option first)
- Explores the cheapest option, generates 3 new options (left, straight, right)
- Keeps going until it reaches the goal or runs out of options

### 6. Extract waypoints

```python
waypoints = full_path[::5]
```

**Translation:** "Take every 5th step from the full path to create a shorter list of checkpoints."

---

## Real-World Examples

### Tesla Self-Parking

When you press the "auto park" button in a Tesla, the car uses something very similar to Hybrid A*. It:
1. Scans the parking space and nearby cars (obstacles)
2. Plans a smooth, drivable path into the spot
3. Considers its own turning radius and size
4. Executes the path smoothly

### Warehouse Robots

Amazon's warehouse robots navigate between shelves using pathfinding algorithms. They need to:
- Plan smooth paths through narrow aisles
- Avoid bumping into shelves or other robots
- Account for their own size and turning ability

### Mars Rovers

NASA's Mars rovers use pathfinding to navigate the Martian surface. The terrain is full of rocks and craters (obstacles), and the rover has physical limits on how sharply it can turn. Hybrid A*-style algorithms are perfect for this.

### Robot Lawnmowers

Modern robot lawnmowers map your yard, identify obstacles (trees, flower beds, pools), and plan efficient mowing paths that account for their turning radius.

---

## Key Terms Glossary

| Term | Simple Definition |
|------|-------------------|
| **Pathfinding** | Finding a route from A to B while avoiding obstacles |
| **A\*** | A classic algorithm that finds the shortest path using a cost function |
| **Hybrid A\*** | A* that plans in continuous space with vehicle physics |
| **State (x, y, θ)** | A robot's full position: where it is AND which way it's facing |
| **Bicycle model** | A simplified way to calculate how a vehicle moves |
| **Velocity (V)** | How fast the robot moves forward |
| **Wheelbase (L)** | Distance between front and back wheels |
| **Steering angle (δ)** | How much the front wheels are turned |
| **g-cost** | Distance already traveled |
| **h-cost** | Estimated distance remaining (the "guess") |
| **f-cost** | Total cost = g-cost + h-cost (+ penalties) |
| **Heuristic** | An educated guess used to guide the search |
| **Discretization** | Rounding continuous values into buckets for comparison |
| **Priority queue** | A list that always gives you the cheapest/lowest item first |
| **Waypoint** | A checkpoint along a path — a key position to aim for |
| **Kinematic** | Related to the physics of how things move |
| **Non-holonomic** | A robot that can't move in every direction instantly (like a car — it can't slide sideways) |

---

**Previous:** [Guide 1: Graph Theory Basics](01-graph-theory-basics.md)
**Next:** [Guide 3: Artificial Potential Fields](03-artificial-potential-fields.md) — Now that we have a global plan, how does the robot dodge moving obstacles in real time?
