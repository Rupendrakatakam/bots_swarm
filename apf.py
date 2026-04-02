import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# ==========================================
# 1. MAP & A* GLOBAL PLANNER (OFFLINE)
# ==========================================
grid_size = 100
start = (10, 10)
goal = (99, 99)

# Initial positions of obstacles
obstacles = np.array([
    [20.0, 80.0],  # Static
    [50.0, 50.0],  # Moving (Horizontal)
    [80.0, 20.0],  # Moving (Vertical)
    [80.0, 85.0]   # Static
])
safety_radius = 12.0 

print("1. Running A* Global Planner...")
G = nx.grid_2d_graph(grid_size, grid_size)

edges_to_add = []
for x, y in G.nodes():
    if (x+1, y+1) in G.nodes(): edges_to_add.append(((x,y), (x+1, y+1)))
    if (x-1, y+1) in G.nodes(): edges_to_add.append(((x,y), (x-1, y+1)))
G.add_edges_from(edges_to_add)

nodes_to_remove = []
for node in G.nodes():
    for obs in obstacles:
        if math.dist(node, obs) < safety_radius:
            nodes_to_remove.append(node)
            break
G.remove_nodes_from(nodes_to_remove)

def euclidean_heuristic(node1, node2):
    return math.dist(node1, node2)

full_path = nx.astar_path(G, source=start, target=goal, heuristic=euclidean_heuristic)

# Extract Waypoints
waypoints = full_path[::15]
if full_path[-1] not in waypoints:
    waypoints.append(full_path[-1])
waypoints = np.array(waypoints)

# ==========================================
# 2. REAL-TIME APF & ANIMATION SETUP
# ==========================================
print("2. Initializing Real-Time APF Simulation...")
fig, ax = plt.subplots(figsize=(8, 8))

# Static Visuals
ax.plot(waypoints[:, 0], waypoints[:, 1], 'k--', alpha=0.3, label='A* Global Plan')
ax.scatter(waypoints[:-1, 0], waypoints[:-1, 1], c='orange', s=80, marker='s', label='Waypoints')
ax.scatter(start[0], start[1], c='green', s=150, zorder=5, label='Start')
ax.scatter(goal[0], goal[1], c='gold', s=200, marker='*', zorder=5, label='Final Goal')

# Dynamic Visuals (Updated every frame)
obs_scatter = ax.scatter(obstacles[:, 0], obstacles[:, 1], c='red', s=100, marker='X', label='Obstacles')
# We create a list of circle patches to represent the danger zones
obs_circles = [plt.Circle((obs[0], obs[1]), safety_radius, color='r', fill=True, alpha=0.2) for obs in obstacles]
for circle in obs_circles:
    ax.add_patch(circle)

trail_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5, label='APF Actual Path')
robot_dot, = ax.plot([], [], 'bo', markersize=8)

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.grid(True, linestyle=':', alpha=0.5)
ax.set_title("Real-Time Reactive Planning (Moving Obstacles)")
ax.legend(loc='lower right')

# Global State Variables for the robot
current_pos = np.array(start, dtype=float)
path_x, path_y = [start[0]], [start[1]]
current_wp_idx = 0 

# Physics Constants
k_att = 1.0       
k_rep = 50000.0   
rho_0 = 15.0      
step_size = 0.5   
wp_threshold = 2.0 

# ==========================================
# 3. REAL-TIME ANIMATION LOOP
# ==========================================
def animate(frame):
    global current_pos, current_wp_idx, obstacles, path_x, path_y
    
    # --- A. MOVE THE OBSTACLES ---
    # Obstacle 1 (Index 1) moves horizontally using a sine wave
    obstacles[1][0] = 50.0 + 20.0 * math.sin(frame * 0.05)
    # Obstacle 2 (Index 2) moves vertically using a cosine wave
    obstacles[2][1] = 20.0 + 15.0 * math.cos(frame * 0.05)
    
    # Update Obstacle Visuals
    obs_scatter.set_offsets(obstacles)
    for i, circle in enumerate(obs_circles):
        circle.set_center((obstacles[i][0], obstacles[i][1]))
    
    # --- B. APF LOGIC (Only run if goal not reached) ---
    if current_wp_idx < len(waypoints):
        target = waypoints[current_wp_idx]
        
        # Check if current waypoint is reached
        if math.dist(current_pos, target) < wp_threshold:
            current_wp_idx += 1
            if current_wp_idx < len(waypoints):
                target = waypoints[current_wp_idx]
            else:
                return trail_line, robot_dot, obs_scatter # Reached final goal
                
        # Calculate Real-Time Forces
        F_att = k_att * (target - current_pos)
        F_rep = np.array([0.0, 0.0])
        
        for obs in obstacles:
            d = math.dist(current_pos, obs)
            if 0 < d < rho_0:
                direction = (current_pos - obs) / d 
                magnitude = k_rep * (1.0/d - 1.0/rho_0) * (1.0 / (d**2))
                F_rep += direction * magnitude
                
        F_total = F_att + F_rep
        
        # Move the Robot
        force_magnitude = np.linalg.norm(F_total)
        if force_magnitude > 0:
            step_vector = (F_total / force_magnitude) * step_size
            current_pos += step_vector
            
        # Record history for trail
        path_x.append(current_pos[0])
        path_y.append(current_pos[1])

    # --- C. UPDATE ROBOT VISUALS ---
    trail_line.set_data(path_x, path_y)
    robot_dot.set_data([current_pos[0]], [current_pos[1]])
    
    return trail_line, robot_dot, obs_scatter

# Run animation at 20ms per frame
print("3. Starting Animation...")
ani = animation.FuncAnimation(fig, animate, frames=2000, interval=20, blit=False)

plt.show()