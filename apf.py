import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# --- 1. Setup Parameters (Same as before) ---
start = np.array([10.0, 10.0])      
goal = np.array([90.0, 90.0])     
obstacles = np.array([
    [20.0, 80.0],  
    [45.0, 50.0],  
    [80.0, 20.0],  
    [80.0, 80.0]   
])

k_att = 1.0       
k_rep = 50000.0   
rho_0 = 20.0      
step_size = 0.5   
max_iters = 1000  

# --- 2. Calculate the Full Path First ---
current_pos = np.copy(start)
path = [np.copy(current_pos)]

for _ in range(max_iters):
    if math.dist(current_pos, goal) < step_size:
        path.append(np.copy(goal)) # Snap to goal at the end
        break
        
    F_att = k_att * (goal - current_pos)
    
    F_rep = np.array([0.0, 0.0])
    for obs in obstacles:
        d = math.dist(current_pos, obs)
        if d < rho_0 and d > 0:
            direction = (current_pos - obs) / d 
            magnitude = k_rep * (1.0/d - 1.0/rho_0) * (1.0 / (d**2))
            F_rep += direction * magnitude
            
    F_total = F_att + F_rep
    
    force_magnitude = np.linalg.norm(F_total)
    if force_magnitude > 0:
        step_vector = (F_total / force_magnitude) * step_size
    else:
        step_vector = np.array([0.0, 0.0])
        
    current_pos += step_vector
    path.append(np.copy(current_pos))

path = np.array(path) # Convert to numpy array for easy slicing

# --- 3. Animation Setup ---
fig, ax = plt.subplots(figsize=(8, 8))

# Draw the static elements (Start, Goal, Obstacles)
ax.scatter(obstacles[:, 0], obstacles[:, 1], c='red', s=300, marker='X', label='Obstacles')
for obs in obstacles:
    circle = plt.Circle((obs[0], obs[1]), rho_0, color='r', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
ax.scatter(start[0], start[1], c='green', s=150, label='Start')
ax.scatter(goal[0], goal[1], c='gold', s=200, marker='*', label='Goal')

# Set map boundaries and styling
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_title("Artificial Potential Field Animation")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend(loc='lower right')

# Create empty line and point objects that we will update in the animation loop
trail, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5, label='Trail')
robot, = ax.plot([], [], 'bo', markersize=8, label='Robot')

# --- 4. Animation Functions ---
def init():
    """Initializes the empty frame"""
    trail.set_data([], [])
    robot.set_data([], [])
    return trail, robot

def animate(frame):
    """Updates the frame with the robot's current position and past trail"""
    # Slice the path array up to the current frame to create the trail
    trail.set_data(path[:frame, 0], path[:frame, 1])
    
    # Set the robot's position at the exact current frame
    # We pass it as a list [x] and [y] to satisfy matplotlib's formatting
    robot.set_data([path[frame, 0]], [path[frame, 1]]) 
    
    return trail, robot

# Create the animation object
# interval=20 means 20 milliseconds between frames (controls the speed)
ani = animation.FuncAnimation(
    fig, 
    animate, 
    init_func=init, 
    frames=len(path), 
    interval=20, 
    blit=True, 
    repeat=False # Set to True if you want it to loop endlessly
)

# Display it
plt.show()