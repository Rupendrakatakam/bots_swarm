# Bots Swarm — Documentation

Welcome! This folder breaks down the concepts behind this project in plain English. No coding experience, no robotics background, no math degree — just simple explanations.

## Reading Order

We recommend reading these in order, since each builds on the previous one:

| # | Guide | What You'll Learn |
|---|-------|-------------------|
| 1 | [Graph Theory Basics](01-graph-theory-basics.md) | What graphs are, how we use them to represent locations and distances, and how `node.py` works |
| 2 | [Pathfinding & Hybrid A*](02-pathfinding-and-hybrid-a-star.md) | How a robot finds a path from A to B when it has to actually *drive* (not teleport) |
| 3 | [Artificial Potential Fields](03-artificial-potential-fields.md) | How invisible "magnet" forces let a robot dodge moving obstacles in real time |
| 4 | [How It All Works Together](04-how-it-all-works-together.md) | The full pipeline — how both planners combine, what happens when you run the code, and real-world applications |

## Quick Start

- **Never heard of any of this?** Start with [Guide 1](01-graph-theory-basics.md).
- **Know some math/coding?** Jump to [Guide 2](02-pathfinding-and-hybrid-a-star.md).
- **Just want the big picture?** Read [Guide 4](04-how-it-all-works-together.md).

## Code Files

These docs explain the following files in the project root:

| File | Explained In |
|------|-------------|
| `node.py` | [Guide 1](01-graph-theory-basics.md) |
| `apf.py` | [Guides 2](02-pathfinding-and-hybrid-a-star.md), [3](03-artificial-potential-fields.md), [4](04-how-it-all-works-together.md) |
