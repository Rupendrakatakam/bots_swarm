import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. Setup Parameters ---
start = np.array([10.0, 10.0])      # Node A
goal = np.array([100.0, 100.0])     # The Goal
obstacles = np.array([
    [20.0, 80.0],  # Node B
    [50.0, 50.0],  # Node C
    [80.0, 20.0],  # Node D
    [90.0, 90.0]   # Node E
])

# APF Tuning Constants
k_att = 1.0       # Attractive force gain
k_rep = 50000.0   # Repulsive force gain
rho_0 = 20.0      # Radius of influence for obstacles
step_size = 0.5   # How far the robot moves per calculation
max_iters = 1000  # Failsafe to prevent infinite loops

# --- 2. APF Algorithm ---
current_pos = np.copy(start)
path = [np.copy(current_pos)]

for _ in range(max_iters):
    # Check if we reached the goal (within a small threshold)
    if math.dist(current_pos, goal) < step_size:
        break
        
    # Calculate Attractive Force Vector (pulls to goal)
    # F_att = k_att * (Goal - Current)
    F_att = k_att * (goal - current_pos)
    
    # Calculate Repulsive Force Vector (pushes from obstacles)
    F_rep = np.array([0.0, 0.0])
    for obs in obstacles:
        d = math.dist(current_pos, obs)
        if d < rho_0 and d > 0: # Inside influence radius
            # Direction vector pointing AWAY from obstacle
            direction = (current_pos - obs) / d 
            # Repulsive magnitude formula
            magnitude = k_rep * (1.0/d - 1.0/rho_0) * (1.0 / (d**2))
            F_rep += direction * magnitude
            
    # Combine forces
    F_total = F_att + F_rep
    
    # Normalize the total force vector to step_size
    force_magnitude = np.linalg.norm(F_total)
    if force_magnitude > 0:
        step_vector = (F_total / force_magnitude) * step_size
    else:
        step_vector = np.array([0.0, 0.0])
        
    # Update position and save to path
    current_pos += step_vector
    path.append(np.copy(current_pos))

# Convert path to numpy array for easy plotting
path = np.array(path)

# --- 3. Visualization ---
plt.figure(figsize=(8, 8))

# Plot path
plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Robot Path')

# Plot obstacles
plt.scatter(obstacles[:, 0], obstacles[:, 1], c='red', s=300, marker='X', label='Obstacles (B,C,D,E)')
# Plot influence radius around obstacles
for obs in obstacles:
    circle = plt.Circle((obs[0], obs[1]), rho_0, color='r', fill=False, linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)

# Plot Start and Goal
plt.scatter(start[0], start[1], c='green', s=150, label='Start (Node A)')
plt.scatter(goal[0], goal[1], c='gold', s=200, marker='*', label='Goal (100,100)')

# Map styling
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, linestyle=':', alpha=0.7)
plt.title("Artificial Potential Field Path Planning")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend(loc='lower right')

plt.show()