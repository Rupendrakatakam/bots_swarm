import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
from matplotlib.patches import Circle, Rectangle

from hybrid_astar import HybridAStar
from apf import APF
from robot import DiffDriveRobot  

# --- Environment Setup ---
WORLD_SIZE = 20.0
GOAL_TOLERANCE = 0.5

CIRC_OBSTACLES = []
RECT_OBSTACLES = []

# Define Multi-Agent Data
AGENTS_DATA = {
    'A': {'start': (2.0, 2.0, math.pi/3), 'goal': (18.0, 18.0, 0.0), 'color': 'blue'},
    'B': {'start': (2.0, 7.0, 3*math.pi/4), 'goal': (18.0, 2.0, math.pi), 'color': 'green'},
}

# Priority Ranking: Agent A (0) > B (1) > C (2) > D (3)
AGENT_KEYS = list(AGENTS_DATA.keys())
YIELD_THRESHOLD = 2.5 

def run_simulation():
    print("=== Multi-Agent Fast Hybrid Navigation (No CBS) ===")
    
    # 1. GLOBAL PLANNER (Independent Hybrid A*)
    print("1. Calculating independent global paths...")
    global_paths = {}
    
    for a_id, data in AGENTS_DATA.items():
        print(f"   -> Planning for Agent {a_id}...")
        h_astar = HybridAStar(
            start_pose=data['start'],
            goal_pose=data['goal'],
            obstacles=CIRC_OBSTACLES,
            rect_obstacles=RECT_OBSTACLES,
            robot_radius=1.0
        )
        
        result = h_astar.find_path()
        if not result:
            print(f"ERROR: Could not find path for Agent {a_id}")
            return None, None, None
            
        global_paths[a_id] = result[0]

    # 2. LOCAL PLANNER INITIALIZATION
    print("2. Initializing Hardware & Local APF Planners...")
    robots = {}
    apfs = {}
    histories = {a: [] for a in AGENTS_DATA}
    waypoint_indices = {a: 1 for a in AGENTS_DATA}

    for a_id, data in AGENTS_DATA.items():
        robots[a_id] = DiffDriveRobot(start_pos=[data['start'][0], data['start'][1]], start_theta=data['start'][2])
        robots[a_id].radius = 1.0
        robots[a_id].wheel_length = 0.8
        
        apfs[a_id] = APF(k_att=0.75, k_rep=16.0, rho_0=2.5, obstacles=CIRC_OBSTACLES, rect_obstacles=RECT_OBSTACLES)
        histories[a_id].append((robots[a_id].x, robots[a_id].y, robots[a_id].theta))

    max_steps = 1500
    dt = 0.2  

    prev_forces = {a: np.array([0.0, 0.0]) for a in AGENTS_DATA}

    for step in range(max_steps):
        all_reached = True
        current_positions = {a: np.array([robots[a].x, robots[a].y]) for a in AGENTS_DATA}

        for i, a_id in enumerate(AGENT_KEYS):
            robot = robots[a_id]
            pos = current_positions[a_id]
            final_goal = np.array([AGENTS_DATA[a_id]['goal'][0], AGENTS_DATA[a_id]['goal'][1]])
            
            # --- BUG FIX: THE MISSING GOAL CHECK ---
            if np.linalg.norm(pos - final_goal) < GOAL_TOLERANCE:
                robot.drive(0.0, 0.0, dt)
                histories[a_id].append((robot.x, robot.y, robot.theta))
                continue 
            else:
                all_reached = False

            # --- WAYPOINT TRACKING ---
            path = global_paths[a_id]
            current_idx = waypoint_indices[a_id]
            
            min_dist = float('inf')
            best_idx = current_idx
            search_range = min(current_idx + 15, len(path))
            
            for k in range(current_idx, search_range):
                dist = np.linalg.norm(pos - np.array([path[k][0], path[k][1]]))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = k
                    
            waypoint_indices[a_id] = best_idx
            
            lookahead_idx = min(best_idx + 3, len(path) - 1)
            target_x, target_y, _ = path[lookahead_idx]
            current_waypoint = np.array([target_x, target_y])

            # --- DYNAMIC APF INJECTION & PRIORITY ---
            yield_multiplier = 1.0
            is_yielding = False
            active_dynamic_obs = []

            for j, other_id in enumerate(AGENT_KEYS):
                if i == j: continue 
                
                dist = np.linalg.norm(pos - current_positions[other_id])
                
                # Priority Check: Lower Index = Higher Priority
                if j < i: 
                    if dist < YIELD_THRESHOLD:
                        is_yielding = True
                        yield_multiplier = 5.0 # Massive repulsion from superiors
                        active_dynamic_obs.append((current_positions[other_id][0], 
                                                   current_positions[other_id][1], 
                                                   robot.radius, 
                                                   yield_multiplier))
                else:
                    if dist < YIELD_THRESHOLD:
                        active_dynamic_obs.append((current_positions[other_id][0], 
                                                   current_positions[other_id][1], 
                                                   robot.radius, 
                                                   1.0))

            # --- CALCULATE FORCE & APPLY ANTI-JITTER FILTER ---
            raw_force = apfs[a_id].get_force(pos, current_waypoint, active_dynamic_obs)
            
            alpha = 0.2
            force = alpha * raw_force + (1.0 - alpha) * prev_forces[a_id]
            prev_forces[a_id] = force

            # --- P-CONTROLLER & VELOCITY SCALING ---
            fx, fy = force
            force_mag = np.linalg.norm(force)
            
            if force_mag < 0.01:
                cmd_v, cmd_omega = 0.0, 0.0
            else:
                target_theta = np.arctan2(fy, fx)
                error_theta = target_theta - robot.theta
                error_theta = (error_theta + np.pi) % (2 * np.pi) - np.pi
                
                cmd_omega = 2.5 * error_theta  
                
                base_v = force_mag * np.cos(error_theta) 
                base_v = max(0.0, base_v) 
                
                if is_yielding:
                    # Velocity Scaling: Drop max speed to 20%
                    max_allowed_v = robot.max_wheel_speed * 0.2
                    cmd_v = min(base_v, max_allowed_v)
                else:
                    cmd_v = min(base_v, robot.max_wheel_speed)

            # --- HARDWARE WHEEL COMMANDS ---
            half_wheelbase = robot.wheel_dist / 2.0
            cmd_vl = cmd_v - (cmd_omega * half_wheelbase)
            cmd_vr = cmd_v + (cmd_omega * half_wheelbase)

            robot.drive(cmd_vl, cmd_vr, dt)
            histories[a_id].append((robot.x, robot.y, robot.theta))

        if all_reached:
            print(f"All goals reached in {step} steps!")
            break

    return global_paths, histories, robots

if __name__ == '__main__':
    global_paths, histories, robots = run_simulation()

    if global_paths is not None:
        fig, ax = plt.subplots(figsize=(10, 10))

        INFLATION_BUFFER = 1.5  

        for obs in CIRC_OBSTACLES:
            cx, cy, radius = obs
            ax.add_patch(Circle((cx, cy), radius, fill=True, color='red', alpha=0.3))
            ax.add_patch(Circle((cx, cy), radius + INFLATION_BUFFER, fill=False, color='orange', linestyle='--', linewidth=1.5))
        
        for rect in RECT_OBSTACLES:
            cx, cy, L, W, angle = rect
            rect_patch = Rectangle((cx - L/2, cy - W/2), L, W, fill=True, color='red', alpha=0.3)
            t = transforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
            rect_patch.set_transform(t)
            ax.add_patch(rect_patch)
            
            inf_L = L + (2 * INFLATION_BUFFER)
            inf_W = W + (2 * INFLATION_BUFFER)
            inf_patch = Rectangle((cx - inf_L/2, cy - inf_W/2), inf_L, inf_W, fill=False, color='orange', linestyle='--', linewidth=1.5)
            inf_patch.set_transform(t) 
            ax.add_patch(inf_patch)
        
        body_patches, heading_lines, trail_lines = {}, {}, {}
        left_wheels, right_wheels = {}, {}

        for a_id in AGENTS_DATA:
            color = AGENTS_DATA[a_id]['color']
            
            gx = [p[0] for p in global_paths[a_id]]
            gy = [p[1] for p in global_paths[a_id]]
            ax.plot(gx, gy, color=color, linestyle='--', linewidth=1.5, alpha=0.5)
            
            goal = AGENTS_DATA[a_id]['goal']
            ax.scatter(goal[0], goal[1], s=200, c=color, marker='*')

            body_patches[a_id] = Circle((0,0), robots[a_id].radius, fill=True, color=color, alpha=0.5)
            ax.add_patch(body_patches[a_id])
            
            heading_lines[a_id], = ax.plot([], [], 'k-', linewidth=2)
            trail_lines[a_id], = ax.plot([], [], color=color, linestyle='-', linewidth=2, alpha=0.7)
            
            left_wheels[a_id], = ax.plot([], [], 'k-', linewidth=4)
            right_wheels[a_id], = ax.plot([], [], 'k-', linewidth=4)

        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_aspect('equal')
        ax.set_title('Multi-Agent Navigation (Asymmetric Priority APF)', fontsize=16)
        ax.set_xticks(np.arange(0, WORLD_SIZE + 1, 1))
        ax.set_yticks(np.arange(0, WORLD_SIZE + 1, 1))
        ax.grid(True, linestyle=':', alpha=0.6)

        def update(frame):
            artists = []
            for a_id in AGENTS_DATA:
                f = min(frame, len(histories[a_id]) - 1)
                x, y, theta = histories[a_id][f]
                
                trail_lines[a_id].set_data([p[0] for p in histories[a_id][:f+1]], [p[1] for p in histories[a_id][:f+1]])
                body_patches[a_id].center = (x, y)
                heading_lines[a_id].set_data([x, x + robots[a_id].radius * np.cos(theta)], [y, y + robots[a_id].radius * np.sin(theta)])
                
                lx = x - robots[a_id].wheel_dist * np.sin(theta)
                ly = y + robots[a_id].wheel_dist * np.cos(theta)
                rx = x + robots[a_id].wheel_dist * np.sin(theta)
                ry = y - robots[a_id].wheel_dist * np.cos(theta)
                
                wl = robots[a_id].wheel_length / 2.0
                left_wheels[a_id].set_data([lx - wl * np.cos(theta), lx + wl * np.cos(theta)],
                                           [ly - wl * np.sin(theta), ly + wl * np.sin(theta)])
                right_wheels[a_id].set_data([rx - wl * np.cos(theta), rx + wl * np.cos(theta)],
                                            [ry - wl * np.sin(theta), ry + wl * np.sin(theta)])
                
                artists.extend([
                    trail_lines[a_id], body_patches[a_id], heading_lines[a_id], 
                    left_wheels[a_id], right_wheels[a_id]
                ])
            return artists

        max_frames = max(len(h) for h in histories.values())
        ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=15, blit=True, repeat=False)
        plt.tight_layout()
        plt.show()