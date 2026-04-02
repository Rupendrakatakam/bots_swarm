# Bots Swarm

A robotics path planning and graph theory visualization project featuring **Hybrid A\*** kinematic planning, **Artificial Potential Fields (APF)** for real-time obstacle avoidance, and graph-based node mapping.

<video src="swarm_simulation.mp4" autoplay loop muted playsinline width="100%"></video>

## Project Structure

```
.
├── apf.py                  # Hybrid A* global planner + APF reactive local planner
├── node.py                 # Graph-based node mapping with distance calculations
├── swarm_simulation.mp4    # Generated simulation output
└── doc.txt                 # Concept notes
```

## Prerequisites

```bash
pip install numpy matplotlib networkx
```

> `networkx` is only required by `node.py`. `apf.py` uses only `numpy`, `matplotlib`, and `heapq` (stdlib).
>
> **MP4 export** requires `ffmpeg`: `sudo apt install ffmpeg`

---

## Scripts

### Hybrid A\* + APF (`apf.py`)

Two-stage path planning on a 100×100 continuous map with static and moving obstacles:

| Stage | Role |
|-------|------|
| **Hybrid A\*** (Global) | Offline kinematic planner using a bicycle model — expands states `(x, y, θ)` with steering constraints |
| **APF** (Local) | Real-time reactive controller that tracks waypoints while avoiding dynamic obstacles |

```bash
python apf.py
```

**Hybrid A\* Configuration**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `V` | 2.0 | Forward velocity |
| `L` | 3.0 | Wheelbase length |
| `dt` | 1.0 | Time step per expansion |
| `max_steer` | ±30° | Steering angle limits |
| `safety_radius` | 12.0 | Obstacle clearance buffer |

**APF Configuration**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `k_att` | 2.0 | Attractive force gain |
| `k_rep` | 50 000.0 | Repulsive force gain |
| `rho_0` | 15.0 | Repulsive influence radius |
| `step_size` | 0.5 | Max step per frame |
| `wp_threshold` | 3.0 | Distance to accept a waypoint |

**Animation Export**

The simulation automatically saves as `swarm_simulation.mp4` (500 frames, 30 fps). To export as GIF instead, swap the commented lines in the saving section.

**Obstacles:** 4 total — 2 static, 2 moving (sinusoidal horizontal & vertical trajectories).

---

### Node Graph Mapping (`node.py`)

Graph theory visualization of 5 nodes (A–E) on a 100×100 coordinate map with a reference point at (0,0):

```bash
python node.py
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
| **Graph Theory** | Mathematical modeling of spatial node relationships with NetworkX using distance-weighted edges |
