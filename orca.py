import rvo2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import math
import random

# ==========================================
# 1. MAP & A* GLOBAL PLANNER SETUP
# ==========================================
map_min, map_max = -10.0, 10.0
grid_res = 0.5  
grid_size = int((map_max - map_min) / grid_res)

obstacles = [(-7, -7), (7, 7), (-7, 7), (7, -7)]
safety_radius = 1.5

print("1. Building A* Grid Graph...")
G = nx.grid_2d_graph(grid_size, grid_size)
edges = [((x, y), (x+1, y+1)) for x, y in G.nodes() if (x+1, y+1) in G.nodes()] + \
        [((x, y), (x-1, y+1)) for x, y in G.nodes() if (x-1, y+1) in G.nodes()]
G.add_edges_from(edges)

def to_grid(x, y):
    gx = min(grid_size-1, max(0, int((x - map_min) / grid_res)))
    gy = min(grid_size-1, max(0, int((y - map_min) / grid_res)))
    return (gx, gy)

def to_cont(gx, gy):
    x = map_min + gx * grid_res
    y = map_min + gy * grid_res
    return (x, y)

nodes_to_remove = []
for node in G.nodes():
    cx, cy = to_cont(node[0], node[1])
    for obs in obstacles:
        if math.hypot(cx - obs[0], cy - obs[1]) < safety_radius:
            nodes_to_remove.append(node)
G.remove_nodes_from(nodes_to_remove)

def get_astar_path(start, goal):
    start_g = to_grid(start[0], start[1])
    goal_g = to_grid(goal[0], goal[1])
    
    if goal_g not in G: return [goal]
    if start_g not in G: start_g = min(G.nodes(), key=lambda n: math.hypot(n[0]-start_g[0], n[1]-start_g[1]))

    try:
        path = nx.astar_path(G, start_g, goal_g, heuristic=lambda a, b: math.hypot(a[0]-b[0], a[1]-b[1]))
        waypoints = [to_cont(p[0], p[1]) for p in path[::4]]
        waypoints.append(goal)
        return waypoints
    except nx.NetworkXNoPath:
        return [goal] 

# ==========================================
# 2. RVO2 SIMULATOR & FORMATIONS
# ==========================================
print("2. Initializing ORCA Simulator...")
agent_radius = 0.4 
sim = rvo2.PyRVOSimulator(1/60.0, 2.5, 10, 2.0, 2.0, agent_radius, 2.0)

print("Registering solid obstacles in physics engine...")
for obs in obstacles:
    vertices = []
    for j in range(8):
        angle = j * (2 * math.pi / 8)
        vx = obs[0] + safety_radius * math.cos(angle)
        vy = obs[1] + safety_radius * math.sin(angle)
        vertices.append((vx, vy))
    sim.addObstacle(vertices)

sim.processObstacles() 

num_agents = 10
agents = []
agent_waypoints = [] 

# Start agents clustered near the center
for _ in range(num_agents):
    rx, ry = np.random.uniform(-2, 2), np.random.uniform(-2, 2)
    agents.append(sim.addAgent((rx, ry)))
    agent_waypoints.append([])

def generate_formation(shape_name):
    targets = []
    if shape_name == 'Circle':
        for i in range(num_agents):
            angle = i * (2 * math.pi / num_agents)
            targets.append((5 * math.cos(angle), 5 * math.sin(angle))) 
    elif shape_name == 'Line':
        start_x = -7.5
        for i in range(num_agents):
            targets.append((start_x + (i * 1.6), 0))
    elif shape_name == 'V-Shape':
        targets.append((0, 8))
        for i in range(1, 5):
            targets.append((-i * 1.5, 8 - i * 1.5)) 
            targets.append((i * 1.5, 8 - i * 1.5))  
        targets.append((0, 2)) 
    elif shape_name == 'Grid':
        for x in [-4, 0, 4]:
            for y in [-4, 0, 4]:
                targets.append((x, y))
        targets.append((0, -8)) 
    return targets

# ==========================================
# 3. EXACT PHYSICAL VISUALIZATION SETUP
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_xlim(map_min, map_max)
ax.set_ylim(map_min, map_max)
ax.set_aspect('equal') 
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_title("Autonomous Swarm: A* Global & ORCA Local Planners")

for obs in obstacles:
    circle = plt.Circle((obs[0], obs[1]), safety_radius, color='red', alpha=0.3)
    ax.add_patch(circle)

colors = mpl.colormaps['tab10']

agent_circles = []
agent_paths = [] 

for i in range(num_agents):
    color = colors(i % 10)
    circle = plt.Circle((0, 0), agent_radius, color=color, ec='black', zorder=4)
    ax.add_patch(circle)
    agent_circles.append(circle)
    
    line, = ax.plot([], [], '--', color=color, alpha=0.5, linewidth=1.5, zorder=2)
    agent_paths.append(line)

scat_targets = ax.scatter([], [], s=100, marker='X', c='black', alpha=0.5, zorder=3)

zeros_arr = np.zeros(num_agents)
quiver = ax.quiver(zeros_arr, zeros_arr, zeros_arr, zeros_arr, 
                   angles='xy', scale_units='xy', scale=1, 
                   color='black', width=0.005, headwidth=4, zorder=5)

current_targets = generate_formation('Circle')

def update_global_plans(label):
    global current_targets
    current_targets = generate_formation(label)
    scat_targets.set_offsets(current_targets)
    
    for i, agent_id in enumerate(agents):
        pos = sim.getAgentPosition(agent_id)
        goal = current_targets[i]
        agent_waypoints[i] = get_astar_path(pos, goal)

def init():
    for circle in agent_circles:
        circle.center = (0, 0)
    for line in agent_paths:
        line.set_data([], [])
        
    scat_targets.set_offsets(current_targets)
    quiver.set_offsets(np.zeros((num_agents, 2)))
    quiver.set_UVC(np.zeros(num_agents), np.zeros(num_agents))
    
    update_global_plans('Circle')
    return agent_circles + agent_paths + [scat_targets, quiver]

# ==========================================
# 4. CHOREOGRAPHED ANIMATION LOOP
# ==========================================
def animate(frame):
    current_positions = []
    current_velocities = []
    
    # --- CINEMATIC CHOREOGRAPHY ---
    # Automatically switch formations at specific frames
    if frame == 200:
        print("\n--> Frame 200: Switching to Line Formation")
        update_global_plans('Line')
    elif frame == 450:
        print("\n--> Frame 450: Switching to V-Shape Formation")
        update_global_plans('V-Shape')
    elif frame == 700:
        print("\n--> Frame 700: Switching to Grid Formation")
        update_global_plans('Grid')
    
    # Progress tracker
    if frame % 50 == 0:
        print(f"Rendering frame {frame}/900...")
    # ------------------------------

    for i, agent_id in enumerate(agents):
        pos = sim.getAgentPosition(agent_id)
        vel = sim.getAgentVelocity(agent_id)
        
        current_positions.append(pos)
        current_velocities.append(vel)
        
        agent_circles[i].center = pos
        
        path_coords = [pos] 
        
        if len(agent_waypoints[i]) > 0:
            target = agent_waypoints[i][0]
            if math.hypot(target[0] - pos[0], target[1] - pos[1]) < 1.0:
                if len(agent_waypoints[i]) > 1:
                    agent_waypoints[i].pop(0)
                    target = agent_waypoints[i][0]
            
            path_coords.extend(agent_waypoints[i])
        else:
            target = current_targets[i] 
            path_coords.append(target)
            
        x_data = [p[0] for p in path_coords]
        y_data = [p[1] for p in path_coords]
        agent_paths[i].set_data(x_data, y_data)
        
        vector_x = target[0] - pos[0]
        vector_y = target[1] - pos[1]
        dist = math.hypot(vector_x, vector_y)
        
        if dist > 0.1:
            speed = min(2.0, dist) 
            pref_vx = (vector_x / dist) * speed
            pref_vy = (vector_y / dist) * speed
            
            noise_angle = random.uniform(0, 2 * math.pi)
            noise_dist = random.uniform(0, 0.05)
            pref_vx += math.cos(noise_angle) * noise_dist
            pref_vy += math.sin(noise_angle) * noise_dist
        else:
            pref_vx, pref_vy = 0.0, 0.0 
            
        sim.setAgentPrefVelocity(agent_id, (pref_vx, pref_vy))
    
    sim.doStep()
    
    pos_arr = np.array(current_positions)
    vel_arr = np.array(current_velocities)
    
    if len(pos_arr) == num_agents:
        quiver.set_offsets(pos_arr)
        quiver.set_UVC(vel_arr[:, 0], vel_arr[:, 1])
    
    return agent_circles + agent_paths + [scat_targets, quiver]

# ==========================================
# 5. RENDER AND SAVE TO MP4
# ==========================================
# We set total frames to 900 to give time for all formations to complete
total_frames = 900
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=total_frames, interval=33, blit=True)

print("\n3. Generating Video File. Please wait, this takes a minute...")
writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='SwarmBot'), bitrate=1800)

# This is the line that actually writes the file!
ani.save("swarm_cinematic.mp4", writer=writer)

print("\n===========================================")
print("SUCCESS! Video saved as 'swarm_cinematic.mp4'")
print("===========================================")