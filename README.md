# Bots Swarm

A robotics path planning and multi-agent collision avoidance project featuring three complementary approaches:

1. **Hybrid A\* + APF** — Single robot with global planning and real-time obstacle avoidance
2. **ORCA/RVO2** — Multi-agent swarm navigation with velocity-based collision avoidance
3. **Graph Theory** — Node mapping with distance-weighted edges

![Simulation](swarm_simulation.gif)

## Project Structure

```
.
├── apf.py                  # Hybrid A* global planner + APF reactive local planner (single robot)
├── orca.py                 # ORCA/RVO2 multi-agent swarm navigation (10 agents)
├── node.py                 # Graph-based node mapping with distance calculations
├── swarm_simulation.gif    # Generated APF simulation output
├── swarm_simulation.mp4    # Generated APF simulation output (MP4)
├── Python-RVO2/            # RVO2 library (git submodule — dependency for orca.py)
├── docs/                   # Beginner-friendly concept guides
└── doc.txt                 # Concept notes
```

## Prerequisites

```bash
pip install numpy matplotlib networkx Pillow
```

> `networkx` is only required by `node.py`. `apf.py` uses only `numpy`, `matplotlib`, and `heapq` (stdlib).
>
> **GIF export** requires `Pillow`: `pip install Pillow`
>
> **`orca.py`** requires the RVO2 library — see the ORCA section below for installation.

---

## Scripts

### Hybrid A\* + APF (`apf.py`)

Two-stage path planning on a 100×100 continuous map with static and moving obstacles:

| Stage | Role |
|-------|------|
| **Hybrid A\*** (Global) | Offline kinematic planner using a bicycle model — expands states `(x, y, θ)` with steering constraints |
| **APF** (Local) | Real-time reactive controller that tracks waypoints while avoiding dynamic obstacles |

```bash
python3 apf.py
```

**Hybrid A\* Configuration**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `V` | 2.0 | Forward velocity |
| `L` | 3.0 | Wheelbase length |
| `dt` | 1.0 | Time step per expansion |
| `max_steer` | ±30° | Steering angle limits |
| `safety_radius` | 5.0 | Obstacle clearance buffer |

**APF Configuration**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `k_att` | 2.0 | Attractive force gain |
| `k_rep` | 50 000.0 | Repulsive force gain |
| `rho_0` | 15.0 | Repulsive influence radius |
| `step_size` | 0.5 | Max step per frame |
| `wp_threshold` | 3.0 | Distance to accept a waypoint |

**Animation Export**

The simulation automatically saves as `swarm_simulation.gif` (500 frames, 30 fps). To export as MP4 instead, swap the commented lines in the saving section.

**Obstacles:** 4 total — 2 static, 2 moving (sinusoidal horizontal & vertical trajectories).

---

### ORCA/RVO2 Swarm Navigation (`orca.py`)

Multi-agent collision avoidance using **Optimal Reciprocal Collision Avoidance (ORCA)** — a velocity-based approach where multiple robots navigate through each other without colliding.

```bash
python3 orca.py
```

#### The Setup

- **10 agents** arranged in a circle (radius 5)
- Each agent's goal is the **diametrically opposite point** on the circle
- All agents start simultaneously — paths all cross at the center
- This is the hardest test for any collision avoidance algorithm

#### How ORCA Differs from APF

| Aspect | APF | ORCA |
|--------|-----|------|
| **Approach** | Force-based (push/pull) | Velocity-based (safe speeds) |
| **Best for** | Single robot, dynamic obstacles | Multiple robots, mutual avoidance |
| **Coordination** | None (each acts alone) | Reciprocal (both share avoidance) |
| **Deadlock risk** | High (local minima) | Low (richer velocity space) |

#### RVO2 Simulator Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `timeStep` | 1/60 s | Simulation step duration (60 Hz) |
| `neighborDist` | 2.5 | How far each agent looks for neighbors |
| `maxNeighbors` | 10 | Max neighbors considered per agent |
| `timeHorizon` | 2.0 s | How far ahead to plan collision avoidance |
| `timeHorizonObst` | 2.0 s | How far ahead to plan obstacle avoidance |
| `radius` | 0.3 | Physical size of each agent |
| `maxSpeed` | 1.5 | Maximum agent speed |

#### Symmetry-Breaking Noise

A tiny random nudge (0–0.05 units) is added to each agent's preferred velocity to prevent deadlock. Without it, perfectly mirrored agents would make perfectly mirrored decisions and still collide — the classic "two people stepping the same way in a hallway" problem.

#### Installing Python-RVO2

The `Python-RVO2/` directory is included as a git submodule. To install:

```bash
# Initialize the submodule (if not already done)
git submodule update --init --recursive

# Build and install
cd Python-RVO2
pip install .
```

Or install directly from PyPI:

```bash
pip install rvo2
```

**Animation:** 400 frames at 16ms intervals with velocity vectors shown as arrows (quiver plot). Each arrow shows an agent's current speed and direction.

---

### Node Graph Mapping (`node.py`)

Graph theory visualization of 5 nodes (A–E) on a 100×100 coordinate map with a reference point at (0,0):

```bash
python3 node.py
```

- Complete graph — every node connected to every other node
- Euclidean distances stored as edge weights
- Prints distance matrix (node-to-node and reference-to-node)
- NetworkX visualization with labeled nodes and weighted edges

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Hybrid A\*** | Kinematic pathfinder that plans in continuous `(x, y, θ)` space using a bicycle model — produces feasible, smooth paths for non-holonomic robots |
| **Artificial Potential Fields** | Reactive navigation using attractive forces toward targets and repulsive forces from obstacles; handles dynamic obstacle avoidance in real time |
| **ORCA/RVO2** | Velocity-based multi-agent collision avoidance where agents share the responsibility of avoiding each other |
| **Graph Theory** | Mathematical modeling of spatial node relationships with NetworkX using distance-weighted edges |

---

## Documentation

For detailed, beginner-friendly explanations of every concept, see the [docs/](docs/) folder:

| Guide | Topic |
|-------|-------|
| [1. Graph Theory Basics](docs/01-graph-theory-basics.md) | Dots, lines, distances, and how `node.py` works |
| [2. Pathfinding & Hybrid A*](docs/02-pathfinding-and-hybrid-a-star.md) | How robots find drivable paths (not teleport) |
| [3. Artificial Potential Fields](docs/03-artificial-potential-fields.md) | Invisible magnet forces for real-time obstacle dodging |
| [4. How It All Works Together](docs/04-how-it-all-works-together.md) | The full pipeline from code to animation |
| [5. ORCA/RVO2 Swarm Navigation](docs/05-orca-rvo2-swarm-navigation.md) | Multi-agent velocity-based collision avoidance |

No coding experience, no robotics background, no math degree required — just simple explanations with analogies.
