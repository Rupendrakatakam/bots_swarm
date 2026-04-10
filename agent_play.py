import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
from matplotlib.patches import Circle, Rectangle

from cbs import CBS
from orca import ORCASwarmManager  # Your new RVO2 wrapper!

# --- Environment Setup ---
WORLD_SIZE = 20.0

CIRC_OBSTACLES = []#(14.0, 15.0, 2.0)
RECT_OBSTACLES = [] #8.0, 8.0, 2.0, 0.25, 35.0)

# Define Multi-Agent Data
AGENTS_DATA = {
    'A': {'start': (2.0, 2.0, math.pi/4), 'goal': (18.0, 18.0, 0.0), 'color': 'blue'},
    'B': {'start': (18.0, 2.0, 3*math.pi/4), 'goal': (2.0, 18.0, math.pi), 'color': 'green'},
    'C': {'start': (18.0, 18.0, 3*math.pi/4), 'goal': (2.0, 2.0, math.pi), 'color': 'red'},
    'D': {'start': (2.0, 18.0, 3*math.pi/4), 'goal': (18.0, 2.0, math.pi), 'color': 'yellow'}
}

ROBOT_RADIUS = 1.0

def run_simulation():
    print("=== Multi-Agent CBS + ORCA Navigation ===")
    
    # 1. GLOBAL PLANNER (CBS with Hybrid A*)
    cbs_planner = CBS(
        agents_data=AGENTS_DATA,
        obstacles=CIRC_OBSTACLES,
        rect_obstacles=RECT_OBSTACLES,
        robot_radius=ROBOT_RADIUS
    )
    
    global_paths = cbs_planner.find_solution()
    if not global_paths:
        return None, None

    # 2. LOCAL PLANNER (ORCA INITIALIZATION)
    # ORCA uses integer IDs (0, 1, 2...). We must map our string IDs ('A', 'B') to them.
    agent_keys = list(AGENTS_DATA.keys())
    num_agents = len(agent_keys)

    # Initialize ORCA
    orca = ORCASwarmManager(
        num_agents=num_agents, 
        dt=0.2, 
        neighbor_dist=5.0, 
        max_neighbors=10, 
        radius=ROBOT_RADIUS + 0.2, # Add tiny buffer
        max_speed=1.5
    )

    # Spawn agents into ORCA in the exact order of agent_keys
    start_positions = [(AGENTS_DATA[k]['start'][0], AGENTS_DATA[k]['start'][1]) for k in agent_keys]
    orca.spawn_agents(start_positions)
    orca.finalize_obstacles() # Empty for now, as requested

    # Tracking states
    histories = {a: [(AGENTS_DATA[a]['start'][0], AGENTS_DATA[a]['start'][1], AGENTS_DATA[a]['start'][2])] for a in agent_keys}
    waypoint_indices = {a: 1 for a in agent_keys}

    max_steps = 1500
    for step in range(max_steps):
        all_reached = True
        current_targets = []
        current_positions = orca.get_positions()

        # Gather the current target waypoint for ALL agents
        for i, a_id in enumerate(agent_keys):
            agent_pos = np.array(current_positions[i])
            final_goal = np.array([AGENTS_DATA[a_id]['goal'][0], AGENTS_DATA[a_id]['goal'][1]])
            
            if np.linalg.norm(agent_pos - final_goal) < 0.5:
                # If reached, the target is just the goal itself (stay still)
                current_targets.append((final_goal[0], final_goal[1]))
            else:
                all_reached = False
                path = global_paths[a_id]
                idx = waypoint_indices[a_id]
                
                # Fetch waypoint
                target_x, target_y, _ = path[idx] if idx < len(path) else path[-1]
                current_waypoint = np.array([target_x, target_y])
                
                # Advance waypoint if close enough
                if np.linalg.norm(agent_pos - current_waypoint) < 1.0 and idx < len(path) - 1:
                    waypoint_indices[a_id] += 1
                    target_x, target_y, _ = path[waypoint_indices[a_id]]
                
                current_targets.append((target_x, target_y))

        if all_reached:
            print("All goals reached!")
            break

        # --- BATCH ORCA STEP ---
        # Pass all targets to ORCA, let it do the collision avoidance math, and return new states
        new_positions, new_velocities = orca.update_step(current_targets)

        # Record histories
        for i, a_id in enumerate(agent_keys):
            nx, ny = new_positions[i]
            vx, vy = new_velocities[i]

            # Since ORCA is holonomic, we calculate the robot's heading (theta) directly from its velocity vector
            if math.hypot(vx, vy) > 0.05:
                theta = math.atan2(vy, vx)
            else:
                theta = histories[a_id][-1][2] # Keep previous heading if stopped

            histories[a_id].append((nx, ny, theta))

    return global_paths, histories

if __name__ == '__main__':
    global_paths, histories = run_simulation()

    if global_paths is not None:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw Static Environments
        for obs in CIRC_OBSTACLES:
            ax.add_patch(Circle((obs[0], obs[1]), obs[2], fill=True, color='red', alpha=0.3))
        
        for rect in RECT_OBSTACLES:
            cx, cy, L, W, angle = rect
            rect_patch = Rectangle((cx - L/2, cy - W/2), L, W, fill=True, color='red', alpha=0.3)
            t = transforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
            rect_patch.set_transform(t)
            ax.add_patch(rect_patch)

        # Init Agent Visuals
        body_patches, heading_lines, trail_lines = {}, {}, {}
        left_wheels, right_wheels = {}, {}

        for a_id in AGENTS_DATA:
            color = AGENTS_DATA[a_id]['color']
            
            # Draw Global Path
            gx = [p[0] for p in global_paths[a_id]]
            gy = [p[1] for p in global_paths[a_id]]
            ax.plot(gx, gy, color=color, linestyle='--', linewidth=1.5, alpha=0.5)
            ax.scatter(AGENTS_DATA[a_id]['goal'][0], AGENTS_DATA[a_id]['goal'][1], s=200, c=color, marker='*')

            # Dynamic Artists
            body_patches[a_id] = Circle((0,0), ROBOT_RADIUS, fill=True, color=color, alpha=0.5)
            ax.add_patch(body_patches[a_id])
            heading_lines[a_id], = ax.plot([], [], 'k-', linewidth=2)
            trail_lines[a_id], = ax.plot([], [], color=color, linestyle='-', linewidth=2, alpha=0.7)
            left_wheels[a_id], = ax.plot([], [], 'k-', linewidth=4)
            right_wheels[a_id], = ax.plot([], [], 'k-', linewidth=4)

        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_aspect('equal')
        ax.set_title('Multi-Agent CBS + ORCA (RVO2)', fontsize=16)

        def update(frame):
            artists = []
            for a_id in AGENTS_DATA:
                f = min(frame, len(histories[a_id]) - 1)
                x, y, theta = histories[a_id][f]
                
                trail_lines[a_id].set_data([p[0] for p in histories[a_id][:f+1]], [p[1] for p in histories[a_id][:f+1]])
                body_patches[a_id].center = (x, y)
                heading_lines[a_id].set_data([x, x + ROBOT_RADIUS * np.cos(theta)], [y, y + ROBOT_RADIUS * np.sin(theta)])
                
                # Calculate Wheel Positions for visualization
                w_dist, w_len = 0.5625, 0.4
                lx, ly = x - w_dist * np.sin(theta), y + w_dist * np.cos(theta)
                rx, ry = x + w_dist * np.sin(theta), y - w_dist * np.cos(theta)
                
                left_wheels[a_id].set_data([lx - w_len * np.cos(theta), lx + w_len * np.cos(theta)], 
                                           [ly - w_len * np.sin(theta), ly + w_len * np.sin(theta)])
                right_wheels[a_id].set_data([rx - w_len * np.cos(theta), rx + w_len * np.cos(theta)], 
                                            [ry - w_len * np.sin(theta), ry + w_len * np.sin(theta)])
                
                artists.extend([trail_lines[a_id], body_patches[a_id], heading_lines[a_id], left_wheels[a_id], right_wheels[a_id]])
            return artists

        max_frames = max(len(h) for h in histories.values())
        ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=15, blit=True, repeat=False)
        plt.tight_layout()
        plt.show()