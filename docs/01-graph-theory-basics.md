# Guide 1: Graph Theory Basics

> **What this explains:** How we represent locations and distances using dots and lines — and how `node.py` does exactly that.

---

## Imagine a Map

Picture a blank piece of paper. Now imagine you want to show someone where 5 different places are on that paper.

You could:
- Write down a list of addresses (boring, hard to visualize)
- Draw dots on the paper and label them (much better!)

**That's a graph.** It's literally just dots and lines connecting them. That's the entire concept. No magic.

---

## The Two Ingredients

### 1. Nodes (also called "vertices")

A **node** is just a dot. A point. A location.

Think of nodes as **cities on a map**. Each city has a name and a position.

In our project, we have these nodes:

| Name | Position (x, y) |
|------|-----------------|
| Ref  | (0, 0)          |
| A    | (10, 10)        |
| B    | (20, 80)        |
| C    | (50, 50)        |
| D    | (80, 20)        |
| E    | (90, 90)        |

The map is 100×100 units. The bottom-left corner is (0, 0) — that's our **reference point**. The top-right corner is (100, 100).

> **What is a reference point?** It's your "home base." Like saying "everything is measured from here." If you've ever used a map with a "You Are Here" marker — that's a reference point.

### 2. Edges

An **edge** is a line connecting two nodes.

Think of edges as **roads between cities**. If two cities have a road between them, we draw a line.

In our project, we connect **every node to every other node**. This is called a **complete graph**. It looks messy, but it means you can get from any location to any other location directly.

---

## Weights: Numbers on the Edges

Every edge has a **weight** — a number attached to it.

In our case, the weight is the **distance** between the two nodes.

> **Example:** The distance from node A (10, 10) to node C (50, 50) is about 56.6 units. So the edge A—C has a weight of 56.6.

Why does this matter? Because now the computer knows not just *that* two places are connected, but *how far apart* they are. This is essential for finding the shortest path later.

---

## How Do We Measure Distance?

We use something called **Euclidean distance**. It sounds fancy, but it's just the Pythagorean theorem you might remember from school.

### The Formula

```
distance = √((x₂ - x₁)² + (y₂ - y₁)²)
```

### What It Means in Plain English

Imagine two points on a piece of graph paper. Draw a right triangle where:
- One side goes horizontally from point 1 to point 2
- One side goes vertically from point 1 to point 2
- The hypotenuse (diagonal) is the straight-line distance between them

The formula just calculates the length of that diagonal.

### Example

Point A is at (10, 10). Point B is at (20, 80).

```
Horizontal difference: 20 - 10 = 10
Vertical difference:   80 - 10 = 70

distance = √(10² + 70²)
         = √(100 + 4900)
         = √5000
         = 70.71 units
```

That's it. The computer does this for every pair of nodes.

---

## What `node.py` Actually Does

Let's walk through the code step by step. You don't need to know Python to follow along — we'll explain what each part means.

### Step 1: Define the locations

```python
positions = {
    'Ref': (0, 0),
    'A': (10, 10),
    'B': (20, 80),
    'C': (50, 50),
    'D': (80, 20),
    'E': (90, 90)
}
```

**What this means:** "Here are 6 places on my map, and here's where each one is."

### Step 2: Create the graph

```python
G = nx.Graph()
```

**What this means:** "Hey computer, I want to build a graph. Give me an empty one to start with." (`nx` is short for NetworkX — a Python library for working with graphs.)

### Step 3: Add the nodes

```python
for node, pos in positions.items():
    G.add_node(node, pos=pos)
```

**What this means:** "Add each location as a dot on the graph, and remember its position."

### Step 4: Connect everything and measure distances

```python
for i in range(len(nodes_list)):
    for j in range(i + 1, len(nodes_list)):
        dist = math.dist(positions[node1], positions[node2])
        G.add_edge(node1, node2, weight=dist)
```

**What this means:** "For every pair of nodes, measure the distance between them and draw a line. Store that distance as the line's weight."

This is a **nested loop** — it goes through every possible pair. Node A connects to B, C, D, E. Then B connects to C, D, E (we already counted A—B). And so on.

### Step 5: Print the results

The code prints two things:

1. **Distances from the reference point** — How far each node is from (0, 0)
2. **Distance matrix** — A table showing the distance between every pair of nodes

```
--- Distances from Reference Point (0,0) ---
To Node A: 14.14 units
To Node B: 82.46 units
To Node C: 70.71 units
To Node D: 82.46 units
To Node E: 127.28 units

--- Distance Matrix (Node to Node) ---
     |     A |     B |     C |     D |     E
---------------------------------------------
   A |   0.0 |  70.7 |  56.6 |  78.1 | 113.1
   B |  70.7 |   0.0 |  53.9 |  72.1 |  78.1
   C |  56.6 |  53.9 |   0.0 |  50.0 |  56.6
   D |  78.1 |  72.1 |  50.0 |   0.0 |  99.0
   E | 113.1 |  78.1 |  56.6 |  99.0 |   0.0
```

### Step 6: Draw the picture

```python
plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(G, positions, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(G, positions, font_weight='bold')
```

**What this means:** "Open a drawing window, put blue dots at each node's position, and label them."

The edges to the reference point are drawn in red dashed lines. All other edges are drawn in faint gray so the picture stays readable.

---

## Real-World Examples

Graphs are everywhere. You use them every day without realizing it.

### GPS Navigation

- **Nodes** = Intersections, landmarks, addresses
- **Edges** = Roads between them
- **Weights** = Distance or travel time
- Your GPS finds the shortest-weight path from your location (node) to your destination (node)

### Social Networks

- **Nodes** = People
- **Edges** = Friendships or connections
- **Weights** = How close the relationship is (or just "connected" vs "not")
- "Friends of friends" is literally graph traversal

### Delivery Routes

- **Nodes** = Warehouses, stores, customer addresses
- **Edges** = Possible routes between them
- **Weights** = Distance, time, or cost
- Companies use graphs to figure out the cheapest way to deliver everything

### The Internet

- **Nodes** = Computers, routers, servers
- **Edges** = Network connections
- **Weights** = Bandwidth or latency
- Your data finds the fastest path through this graph to reach its destination

---

## Key Terms Glossary

| Term | Simple Definition |
|------|-------------------|
| **Graph** | A collection of dots (nodes) connected by lines (edges) |
| **Node (vertex)** | A point or location — like a city on a map |
| **Edge** | A connection between two nodes — like a road between cities |
| **Weight** | A number attached to an edge — in our case, distance |
| **Complete graph** | A graph where every node connects to every other node |
| **Reference point** | The origin (0, 0) — where all measurements start from |
| **Euclidean distance** | The straight-line distance between two points |
| **Position (x, y)** | A location on a 2D map — how far right and how far up |
| **NetworkX** | A Python library for creating and working with graphs |

---

**Next:** [Guide 2: Pathfinding & Hybrid A*](02-pathfinding-and-hybrid-a-star.md) — Now that we know how to represent locations, let's figure out how to get from one to another.
