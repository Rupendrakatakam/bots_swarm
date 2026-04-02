import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import heapq

# ==========================================
# 1. MAP & HYBRID A* GLOBAL PLANNER
# ==========================================
grid_size = 100
start_state = (10.0, 10.0, math.radians(45)) 
goal = (90.0, 90.0)

# Obstacles
obstacles = np.array([
    [20.0, 80.0],  
    [50.0, 50.0],  
    [80.0, 20.0],  
    [80.0, 85.0]   
])
safety_radius = 5.0 

# Hybrid A* Kinematic Parameters
V = 2.0        
L = 3.0        
dt = 1.0       
max_steer = math.radians(30) 
steer_angles = [-max_steer, 0.0, max_steer] 

print("1. Running Hybrid A* Global Planner (This may take a few seconds)...")

def is_safe(x, y):
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return False
    for obs in obstacles:
        if math.dist((x, y), obs) < safety_radius + 2.0: 
            return False
    return True

def hybrid_a_star(start, goal):
    start_heuristic = math.dist((start[0], start[1]), goal)
    pq = [(start_heuristic, start, [start])]
    visited = set()
    
    while pq:
        f_cost, current_state, path = heapq.heappop(pq)
        x, y, theta = current_state
        
        if math.dist((x, y), goal) < 5.0:
            path.append((goal[0], goal[1], theta))
            return path
            
        discrete_state = (int(x), int(y), int(math.degrees(theta) // 15))
        if discrete_state in visited:
            continue
        visited.add(discrete_state)
        
        for steer in steer_angles:
            x_new = x + V * math.cos(theta) * dt
            y_new = y + V * math.sin(theta) * dt
            theta_new = theta + (V / L) * math.tan(steer) * dt
            
            theta_new = math.atan2(math.sin(theta_new), math.cos(theta_new))
            
            if is_safe(x_new, y_new):
                new_state = (x_new, y_new, theta_new)
                new_path = path + [new_state]
                
                g_cost = len(new_path) * (V * dt)
                h_cost = math.dist((x_new, y_new), goal)
                steer_penalty = abs(steer) * 10.0 
                
                heapq.heappush(pq, (g_cost + h_cost + steer_penalty, new_state, new_path))
                
    return [] 

full_path = hybrid_a_star(start_state, goal)

if not full_path:
    print("ERROR: Hybrid A* could not find a path!")
    exit()

print(f"Path found! {len(full_path)} kinematic steps.")

waypoints = full_path[::5]
if full_path[-1] not in waypoints:
    waypoints.append(full_path[-1])
waypoints = np.array(waypoints)[:, :2] 

# ==========================================
# 2. REAL-TIME APF & ANIMATION SETUP
# ==========================================
print("2. Initializing Real-Time APF Simulation...")
fig, ax = plt.subplots(figsize=(8, 8))

full_path_arr = np.array(full_path)
ax.plot(full_path_arr[:, 0], full_path_arr[:, 1], 'k--', alpha=0.5, label='Hybrid A* Smooth Plan')
ax.scatter(waypoints[:-1, 0], waypoints[:-1, 1], c='orange', s=50, marker='s', label='Waypoints')
ax.scatter(start_state[0], start_state[1], c='green', s=150, zorder=5, label='Start')
ax.scatter(goal[0], goal[1], c='gold', s=200, marker='*', zorder=5, label='Goal')

obs_scatter = ax.scatter(obstacles[:, 0], obstacles[:, 1], c='red', s=100, marker='X', label='Obstacles')
obs_circles = [plt.Circle((obs[0], obs[1]), safety_radius, color='r', fill=True, alpha=0.2) for obs in obstacles]
for circle in obs_circles:
    ax.add_patch(circle)

trail_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.8, label='APF Executed Path')
robot_dot, = ax.plot([], [], 'bo', markersize=8)

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.grid(True, linestyle=':', alpha=0.5)
ax.set_title("Hybrid A* (Global) + Dynamic APF (Local)")
ax.legend(loc='lower right')

current_pos = np.array([start_state[0], start_state[1]], dtype=float)
path_x, path_y = [current_pos[0]], [current_pos[1]]
current_wp_idx = 0 

k_att = 2.0       
k_rep = 50000.0   
rho_0 = 15.0      
step_size = 0.5   
wp_threshold = 3.0 

# ==========================================
# 3. REAL-TIME ANIMATION LOOP & SAVING
# ==========================================
def animate(frame):
    global current_pos, current_wp_idx, obstacles, path_x, path_y
    
    obstacles[1][0] = 50.0 + 20.0 * math.sin(frame * 0.05)
    obstacles[2][1] = 20.0 + 15.0 * math.cos(frame * 0.05)
    
    obs_scatter.set_offsets(obstacles)
    for i, circle in enumerate(obs_circles):
        circle.set_center((obstacles[i][0], obstacles[i][1]))
    
    if current_wp_idx < len(waypoints):
        target = waypoints[current_wp_idx]
        
        if math.dist(current_pos, target) < wp_threshold:
            current_wp_idx += 1
            if current_wp_idx < len(waypoints):
                target = waypoints[current_wp_idx]
            else:
                return trail_line, robot_dot, obs_scatter
                
        F_att = k_att * (target - current_pos)
        F_rep = np.array([0.0, 0.0])
        
        for obs in obstacles:
            d = math.dist(current_pos, obs)
            if 0 < d < rho_0:
                direction = (current_pos - obs) / d 
                magnitude = k_rep * (1.0/d - 1.0/rho_0) * (1.0 / (d**2))
                F_rep += direction * magnitude
                
        F_total = F_att + F_rep
        
        force_magnitude = np.linalg.norm(F_total)
        if force_magnitude > 0:
            step_vector = (F_total / force_magnitude) * step_size
            current_pos += step_vector
            
        path_x.append(current_pos[0])
        path_y.append(current_pos[1])

    trail_line.set_data(path_x, path_y)
    robot_dot.set_data([current_pos[0]], [current_pos[1]])
    
    return trail_line, robot_dot, obs_scatter

# Set frames to 500 so it captures the whole journey without creating an overly massive file
print("3. Generating Animation frames...")
ani = animation.FuncAnimation(fig, animate, frames=500, interval=20, blit=False)

# ---- SAVING THE ANIMATION ----
print("4. Saving as GIF... (Please wait, this takes a moment)")
writer = animation.PillowWriter(fps=30)
ani.save("swarm_simulation.gif", writer=writer)
print("-> Successfully saved as 'swarm_simulation.gif' in your current directory!")

# If you prefer MP4, comment out the GIF lines above and uncomment the line below:
# (Note: Requires ffmpeg installed on your system -> sudo apt install ffmpeg)
# ani.save("swarm_simulation.mp4", writer='ffmpeg', fps=30)
# print("-> Successfully saved as 'swarm_simulation.mp4' in your current directory!")

plt.show()