import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# ==========================================
# 1. MAP & PLANNER SETUP
# ==========================================
grid_size = 100
start = (10, 10)
goal = (99, 99) # Safe goal coordinate

obstacles = [
    (20, 80),  
    (50, 50),  
    (80, 20),  
    (80, 85)   
]
safety_radius = 15.0 

print("1. Building Grid Graph...")
G = nx.grid_2d_graph(grid_size, grid_size)

# Add diagonal movements
edges_to_add = []
for x, y in G.nodes():
    if (x+1, y+1) in G.nodes(): edges_to_add.append(((x,y), (x+1, y+1)))
    if (x-1, y+1) in G.nodes(): edges_to_add.append(((x,y), (x-1, y+1)))
G.add_edges_from(edges_to_add)

print("2. Carving out Obstacles...")
nodes_to_remove = []
for node in G.nodes():
    for obs in obstacles:
        if math.dist(node, obs) < safety_radius:
            nodes_to_remove.append(node)
            break
G.remove_nodes_from(nodes_to_remove)

print("3. Running A* Search...")
def euclidean_heuristic(node1, node2):
    return math.dist(node1, node2)

full_path = nx.astar_path(G, source=start, target=goal, heuristic=euclidean_heuristic)

# Extract Waypoints (every 15th node)
waypoints = full_path[::15]
if full_path[-1] not in waypoints:
    waypoints.append(full_path[-1])
waypoints = np.array(waypoints)

# ==========================================
# 2. APF LOCAL PLANNER (PRE-CALCULATE PATH)
# ==========================================
print("4. Simulating APF Physics...")
k_att = 1.0       
k_rep = 50000.0   
rho_0 = 15.0      # Should be similar to safety_radius
step_size = 0.5   
wp_threshold = 2.0 

current_pos = np.array(start, dtype=float)
obs_arr = np.array(obstacles, dtype=float)
path_history = [np.copy(current_pos)]
current_wp_idx = 0 

# Run the simulation to generate the movement path
for _ in range(3000): # 3000 steps max
    target = waypoints[current_wp_idx]
    
    # Check if waypoint is reached
    if math.dist(current_pos, target) < wp_threshold:
        if current_wp_idx < len(waypoints) - 1:
            current_wp_idx += 1
            target = waypoints[current_wp_idx]
        else:
            path_history.append(np.copy(target)) # Reached Final Goal
            break
            
    # Calculate Forces
    F_att = k_att * (target - current_pos)
    F_rep = np.array([0.0, 0.0])
    
    for obs in obs_arr:
        d = math.dist(current_pos, obs)
        if 0 < d < rho_0:
            direction = (current_pos - obs) / d 
            magnitude = k_rep * (1.0/d - 1.0/rho_0) * (1.0 / (d**2))
            F_rep += direction * magnitude
            
    F_total = F_att + F_rep
    
    # Move Robot
    force_magnitude = np.linalg.norm(F_total)
    if force_magnitude > 0:
        step_vector = (F_total / force_magnitude) * step_size
    else:
        step_vector = np.array([0.0, 0.0])
        
    current_pos += step_vector
    path_history.append(np.copy(current_pos))

path_history = np.array(path_history)

# ==========================================
# 3. ANIMATION
# ==========================================
print(f"5. Animating {len(path_history)} frames...")
fig, ax = plt.subplots(figsize=(8, 8))

# Draw Obstacles
ax.scatter(obs_arr[:, 0], obs_arr[:, 1], c='red', s=100, marker='X', label='Obstacles')
for obs in obstacles:
    circle = plt.Circle((obs[0], obs[1]), safety_radius, color='r', fill=True, alpha=0.2)
    ax.add_patch(circle)

# Draw A* Waypoints
ax.plot(waypoints[:, 0], waypoints[:, 1], 'k--', alpha=0.3, label='A* Global Plan')
ax.scatter(waypoints[:-1, 0], waypoints[:-1, 1], c='orange', s=80, marker='s', label='Waypoints')

# Draw Start and Goal
ax.scatter(start[0], start[1], c='green', s=150, zorder=5, label='Start')
ax.scatter(goal[0], goal[1], c='gold', s=200, marker='*', zorder=5, label='Final Goal')

# Dynamic Elements
trail, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5, label='APF Actual Path')
robot, = ax.plot([], [], 'bo', markersize=8)

# Map Styling
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.grid(True, linestyle=':', alpha=0.5)
ax.set_title("Hybrid Navigation: A* Global Plan + APF Local Execution")
ax.legend(loc='lower right')

def init():
    trail.set_data([], [])
    robot.set_data([], [])
    return trail, robot

def animate(frame):
    trail.set_data(path_history[:frame, 0], path_history[:frame, 1])
    robot.set_data([path_history[frame, 0]], [path_history[frame, 1]]) 
    return trail, robot

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(path_history), interval=10, blit=True)

plt.show()