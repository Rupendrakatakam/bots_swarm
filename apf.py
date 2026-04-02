import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# --- 1. Setup Parameters ---
start = np.array([10.0, 10.0])      

# Instead of just a Goal, we define a sequential list of Waypoints ending at the Goal
waypoints = np.array([
    [30.0, 40.0],  # Waypoint 1
    [70.0, 60.0],  # Waypoint 2
    [100.0, 100.0] # Final Goal
])

obstacles = np.array([
    [20.0, 80.0],  
    [50.0, 50.0],  
    [80.0, 20.0],  
    [90.0, 90.0]   
])

k_att = 1.0       
k_rep = 50000.0   
rho_0 = 20.0      
step_size = 0.5   
wp_threshold = 2.0 # How close the robot must get to a waypoint to "collect" it

# --- 2. Calculate the Full Path First ---
current_pos = np.copy(start)
path = [np.copy(current_pos)]
current_wp_idx = 0 

for _ in range(2000): # Increased iterations since the path is longer
    
    # 1. Get the current target waypoint
    target = waypoints[current_wp_idx]
    
    # 2. Check if we reached the current waypoint
    if math.dist(current_pos, target) < wp_threshold:
        if current_wp_idx < len(waypoints) - 1:
            current_wp_idx += 1  # Move to the next waypoint
            target = waypoints[current_wp_idx]
        else:
            path.append(np.copy(target)) # Reached the Final Goal!
            break
            
    # 3. Calculate Forces (Attracted to the TARGET, not the final goal)
    F_att = k_att * (target - current_pos)
    
    F_rep = np.array([0.0, 0.0])
    for obs in obstacles:
        d = math.dist(current_pos, obs)
        if d < rho_0 and d > 0:
            direction = (current_pos - obs) / d 
            magnitude = k_rep * (1.0/d - 1.0/rho_0) * (1.0 / (d**2))
            F_rep += direction * magnitude
            
    F_total = F_att + F_rep
    
    # 4. Move the Robot
    force_magnitude = np.linalg.norm(F_total)
    if force_magnitude > 0:
        step_vector = (F_total / force_magnitude) * step_size
    else:
        step_vector = np.array([0.0, 0.0])
        
    current_pos += step_vector
    path.append(np.copy(current_pos))

path = np.array(path) 

# --- 3. Animation Setup ---
fig, ax = plt.subplots(figsize=(8, 8))

# Draw Obstacles
ax.scatter(obstacles[:, 0], obstacles[:, 1], c='red', s=300, marker='X', label='Obstacles')
for obs in obstacles:
    circle = plt.Circle((obs[0], obs[1]), rho_0, color='r', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)

# Draw Start, Waypoints, and Goal
ax.scatter(start[0], start[1], c='green', s=150, label='Start')
ax.scatter(waypoints[:-1, 0], waypoints[:-1, 1], c='orange', s=100, marker='s', label='Waypoints')
ax.scatter(waypoints[-1, 0], waypoints[-1, 1], c='gold', s=200, marker='*', label='Final Goal')

# Draw the "Global Plan" (straight lines between waypoints)
plan_x = [start[0]] + list(waypoints[:, 0])
plan_y = [start[1]] + list(waypoints[:, 1])
ax.plot(plan_x, plan_y, 'k--', alpha=0.3, label='Global Plan')

# Map boundaries and styling
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_title("APF Navigation using Waypoints")
ax.legend(loc='lower right')

trail, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5, label='Actual Path')
robot, = ax.plot([], [], 'bo', markersize=8)

# --- 4. Animation Execution ---
def init():
    trail.set_data([], [])
    robot.set_data([], [])
    return trail, robot

def animate(frame):
    trail.set_data(path[:frame, 0], path[:frame, 1])
    robot.set_data([path[frame, 0]], [path[frame, 1]]) 
    return trail, robot

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(path), interval=20, blit=True)
plt.show()