import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np

# --- 1. Define the Map Parameters ---
grid_size = 100
start = (10, 10)
goal = (95, 95) # Changed slightly from 100 to fit nicely inside the 0-99 grid index

obstacles = [
    (20, 80),  
    (50, 50),  
    (80, 20),  
    (85, 85)   
]
safety_radius = 12.0 # How far the robot must stay away from the obstacle center

# --- 2. Create the Free Space Graph ---
print("Building 100x100 Grid Graph...")
# This creates a grid of 10,000 nodes, connected to their Up/Down/Left/Right neighbors
G = nx.grid_2d_graph(grid_size, grid_size)

# Add diagonal connections (optional, but makes path smoother)
# Connects (x, y) to (x+1, y+1), (x-1, y+1), etc.
edges_to_add = []
for x, y in G.nodes():
    if (x+1, y+1) in G.nodes(): edges_to_add.append(((x,y), (x+1, y+1)))
    if (x-1, y+1) in G.nodes(): edges_to_add.append(((x,y), (x-1, y+1)))
G.add_edges_from(edges_to_add)

# --- 3. Carve out Obstacles ---
print("Removing nodes inside obstacles...")
nodes_to_remove = []
for node in G.nodes():
    for obs in obstacles:
        # If a grid node is inside the safety radius, mark it for deletion
        if math.dist(node, obs) < safety_radius:
            nodes_to_remove.append(node)
            break # No need to check other obstacles

G.remove_nodes_from(nodes_to_remove)

# --- 4. Run A* Search Algorithm ---
print("Running A* Search...")
# A* requires a "heuristic" (an educated guess of distance to the goal)
# We use our standard Euclidean distance formula
def euclidean_heuristic(node1, node2):
    return math.dist(node1, node2)

try:
    # nx.astar_path calculates the shortest safe route automatically!
    full_path = nx.astar_path(G, source=start, target=goal, heuristic=euclidean_heuristic)
    print(f"Path found! Length: {len(full_path)} steps.")
except nx.NetworkXNoPath:
    print("Error: No safe path exists between Start and Goal.")
    full_path = []

# --- 5. Extract Waypoints ---
# The full path has hundreds of tiny steps. 
# We only need a few waypoints to feed to our Local Planner (APF).
# Let's take every 15th node on the path.
if full_path:
    waypoints = full_path[::15]
    if full_path[-1] not in waypoints:
        waypoints.append(full_path[-1]) # Ensure goal is the final waypoint
    waypoints = np.array(waypoints)
else:
    waypoints = np.array([])

# --- 6. Visualization ---
plt.figure(figsize=(8, 8))

# Plot the Free Space (Optional: plotting 10,000 nodes is slow, so we skip drawing the full graph)
# Instead, we just plot the results

# Plot Obstacles
obs_arr = np.array(obstacles)
plt.scatter(obs_arr[:, 0], obs_arr[:, 1], c='red', s=100, marker='X', label='Obstacles')
for obs in obstacles:
    circle = plt.Circle((obs[0], obs[1]), safety_radius, color='r', fill=True, alpha=0.3)
    plt.gca().add_patch(circle)

# Plot the Full A* Path
if full_path:
    path_arr = np.array(full_path)
    plt.plot(path_arr[:, 0], path_arr[:, 1], 'k-', alpha=0.3, linewidth=3, label='A* Full Path')

# Plot the Extracted Waypoints
if len(waypoints) > 0:
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'o--', color='orange', markersize=8, label='Extracted Waypoints')

# Plot Start and Goal
plt.scatter(start[0], start[1], c='green', s=150, zorder=5, label='Start')
plt.scatter(goal[0], goal[1], c='gold', s=200, marker='*', zorder=5, label='Goal')

# Map Styling
plt.xlim(0, grid_size - 1)
plt.ylim(0, grid_size - 1)
plt.grid(True, linestyle=':', alpha=0.5)
plt.title("A* Global Planner: Automatic Waypoint Generation")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend(loc='lower right')

plt.show()

print("\nGenerated Waypoints for APF:")
for i, wp in enumerate(waypoints):
    print(f"Waypoint {i+1}: ({wp[0]}, {wp[1]})")