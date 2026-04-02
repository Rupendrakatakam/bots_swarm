import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# --- 1. Setup Parameters ---
start = np.array([10.0, 10.0])      
goal = np.array([100.0, 100.0])     

# We will make the second obstacle (index 1) move later!
obstacles = np.array([
    [20.0, 80.0],  
    [50.0, 50.0],  # Moving Obstacle
    [80.0, 20.0],  
    [90.0, 90.0]   
])

k_att = 1.0       
k_rep = 50000.0   
rho_0 = 20.0      
step_size = 0.5   

# --- 2. Animation Plot Setup ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_title("Real-Time APF with Moving Obstacle")

# Static Elements
ax.scatter(start[0], start[1], c='green', s=150, label='Start')
ax.scatter(goal[0], goal[1], c='gold', s=200, marker='*', label='Goal')

# Dynamic Elements (We will update these in the loop)
trail_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5, label='Robot Trail')
robot_dot, = ax.plot([], [], 'bo', markersize=8, label='Robot')
obs_scatter = ax.scatter(obstacles[:, 0], obstacles[:, 1], c='red', s=300, marker='X', label='Obstacles')

# --- 3. Global State Variables ---
# We track the robot's current position and history during the animation
current_pos = np.copy(start)
path_x, path_y = [start[0]], [start[1]]

# --- 4. The Real-Time Animation Loop ---
def animate(frame):
    global current_pos, obstacles, path_x, path_y
    
    # 1. Update the Moving Obstacle
    # We use a sine wave to make Node C oscillate horizontally
    obstacles[1][0] = 50.0 + 30.0 * math.sin(frame * 0.05)
    obs_scatter.set_offsets(obstacles) # Update scatter plot
    
    # 2. Check if Goal is Reached
    if math.dist(current_pos, goal) >= step_size:
        
        # 3. Calculate Real-Time Forces
        F_att = k_att * (goal - current_pos)
        F_rep = np.array([0.0, 0.0])
        
        for obs in obstacles:
            d = math.dist(current_pos, obs)
            if 0 < d < rho_0:
                direction = (current_pos - obs) / d 
                magnitude = k_rep * (1.0/d - 1.0/rho_0) * (1.0 / (d**2))
                F_rep += direction * magnitude
                
        # Combine and apply forces
        F_total = F_att + F_rep
        force_magnitude = np.linalg.norm(F_total)
        
        if force_magnitude > 0:
            current_pos += (F_total / force_magnitude) * step_size
            
        # Record the step for the trail
        path_x.append(current_pos[0])
        path_y.append(current_pos[1])
        
    # 4. Update the Visuals
    trail_line.set_data(path_x, path_y)
    robot_dot.set_data([current_pos[0]], [current_pos[1]])
    
    return trail_line, robot_dot, obs_scatter

# Run the animation
ani = animation.FuncAnimation(fig, animate, frames=600, interval=20, blit=True)
plt.legend(loc='lower right')
plt.show()