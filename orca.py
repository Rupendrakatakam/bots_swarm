import rvo2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random

# ==========================================
# 1. SETUP RVO2 SIMULATOR
# ==========================================
sim = rvo2.PyRVOSimulator(1/60.0, 2.5, 10, 2.0, 2.0, 0.3, 1.5)

# ==========================================
# 2. INITIALIZE SWARM (The Circle Cross)
# ==========================================
num_agents = 10
circle_radius = 5.0
agents = []
goals = []

print("Initializing ORCA Simulator...")

for i in range(num_agents):  
    angle = i * (2 * math.pi / num_agents)
    
    start_x = circle_radius * math.cos(angle)
    start_y = circle_radius * math.sin(angle)
    
    goal_x = -start_x
    goal_y = -start_y
    
    agent_id = sim.addAgent((start_x, start_y))
    agents.append(agent_id)
    goals.append((goal_x, goal_y))

# ==========================================
# 3. MATPLOTLIB ANIMATION SETUP
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_title("RVO2 (ORCA) Swarm Navigation with Velocity Vectors")

# --- THE FIX: Get initial positions instead of using empty arrays ---
init_x = [sim.getAgentPosition(a)[0] for a in agents]
init_y = [sim.getAgentPosition(a)[1] for a in agents]

colors = plt.cm.get_cmap('hsv', num_agents)
agent_colors = [colors(i) for i in range(num_agents)]

# Initialize WITH data so Matplotlib knows exactly how to render them
scat_agents = ax.scatter(init_x, init_y, s=400, c=agent_colors, edgecolors='black', zorder=3)
scat_goals = ax.scatter([g[0] for g in goals], [g[1] for g in goals], s=100, marker='X', c='black', alpha=0.5)

# Initialize the arrows with dummy zeros for their initial velocities
quiver = ax.quiver(init_x, init_y, [0]*num_agents, [0]*num_agents, 
                   angles='xy', scale_units='xy', scale=1, 
                   color='black', width=0.005, headwidth=4, zorder=4)

def init():
    return scat_agents, quiver

# ==========================================
# 4. THE ANIMATION LOOP
# ==========================================
def animate(frame):
    current_positions = []
    current_velocities = [] 
    
    for i, agent_id in enumerate(agents):
        pos = sim.getAgentPosition(agent_id)
        vel = sim.getAgentVelocity(agent_id)
        
        current_positions.append(pos)
        current_velocities.append(vel) 
        
        goal = goals[i]
        
        vector_x = goal[0] - pos[0]
        vector_y = goal[1] - pos[1]
        
        dist = math.hypot(vector_x, vector_y)
        
        if dist > 0.1: 
            pref_vx = (vector_x / dist) * 1.5 
            pref_vy = (vector_y / dist) * 1.5
            
            # Symmetry Breaking Noise
            noise_angle = random.uniform(0, 2 * math.pi)
            noise_dist = random.uniform(0, 0.05) 
            pref_vx += math.cos(noise_angle) * noise_dist
            pref_vy += math.sin(noise_angle) * noise_dist
            
        else:
            pref_vx, pref_vy = 0.0, 0.0 
            
        sim.setAgentPrefVelocity(agent_id, (pref_vx, pref_vy))
    
    sim.doStep()
    
    # Update Robot Dots (Notice we only need to update offsets now, not colors!)
    pos_arr = np.array(current_positions)
    scat_agents.set_offsets(pos_arr)
    
    # Update Arrows
    if len(pos_arr) > 0:
        vel_arr = np.array(current_velocities)
        quiver.set_offsets(pos_arr)
        quiver.set_UVC(vel_arr[:, 0], vel_arr[:, 1])
    
    return scat_agents, quiver

print("Starting Animation...")
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=400, interval=16, blit=True)
plt.show()