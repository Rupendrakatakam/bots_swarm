# Bots Swarm

A robotics path planning and graph theory visualization project featuring A* global planning, Artificial Potential Fields (APF) for real-time obstacle avoidance, and graph-based node mapping.

## Project Structure

```
.
├── apf.py      # Real-time reactive path planning with APF
├── node.py     # Graph-based node mapping with distance calculations
└── doc.txt     # Concept notes
```

## Prerequisites

```bash
pip install networkx numpy matplotlib
```

## Usage

### A* + Artificial Potential Fields (`apf.py`)

Two-stage path planning on a 100x100 grid with static and moving obstacles:

1. **A* Global Planner** — Computes an offline path through a grid graph, extracting waypoints while avoiding obstacle zones
2. **APF Reactive Controller** — Real-time navigation between waypoints using attractive forces (toward target) and repulsive forces (away from obstacles)

```bash
python apf.py
```

**Features:**
- 4 obstacles (2 static, 2 moving with sine/cosine trajectories)
- Configurable safety radius, force constants, and step size
- Animated visualization showing global plan, actual path, and danger zones

### Node Graph Mapping (`node.py`)

Graph theory visualization of 5 nodes (A–E) on a 100x100 coordinate map with a reference point at (0,0):

```bash
python node.py
```

**Features:**
- Complete graph where every node connects to every other node
- Euclidean distances stored as edge weights
- Distance matrix output (node-to-node and reference-to-node)
- NetworkX-based visualization with labeled nodes and weighted edges

## Key Concepts

| Concept | Description |
|---------|-------------|
| **A\* Search** | Optimal pathfinding algorithm using heuristics to find the shortest path through a grid |
| **Artificial Potential Fields** | Reactive navigation method using attractive forces toward goals and repulsive forces from obstacles |
| **Graph Theory** | Mathematical modeling of node relationships using NetworkX with distance-weighted edges |
