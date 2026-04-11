import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
from matplotlib.patches import Circle, Rectangle

from cbs import CBS
from apf import APF
from robot import DiffDriveRobot  

# --- Environment Setup ---
WORLD_SIZE = 20.0

CIRC_OBSTACLES = [(14.0, 15.0, 1.0)]
RECT_OBSTACLES = []

# Define Multi-Agent Data
AGENTS_DATA = {
    'A': {'start': (2.0, 2.0, math.pi/4), 'goal': (18.0, 18.0, 0.0), 'color': 'blue'},
    # 'B': {'start': (2.0, 7.0, 3*math.pi/4), 'goal': (18.0, 2.0, math.pi), 'color': 'green'},
    # 'C': {'start': (7.0, 3.0, 3*math.pi/4), 'goal': (2.0, 18.0, math.pi), 'color': 'red'},
    # 'D': {'start': (7.0, 7.0, 3*math.pi/4), 'goal': (2.0, 2.0, math.pi), 'color': 'yellow'},
}

def run_simulation():
    print("=== Multi-Agent CBS + Noisy Diff-Drive Navigation ===")
    
    # 1. GLOBAL PLANNER (CBS with Hybrid A*)
    cbs_planner = CBS(
        agents_data=AGENTS_DATA,
        obstacles=CIRC_OBSTACLES,
        rect_obstacles=RECT_OBSTACLES,
        robot_radius=1.0
    )
    
    global_paths = cbs_planner.find_solution()
    if not global_paths:
        return None, None, None

    # 2. LOCAL PLANNER INITIALIZATION
    robots = {}
    apfs = {}
    histories = {a: [] for a in AGENTS_DATA}
    waypoint_indices = {a: 1 for a in AGENTS_DATA}

    for a_id, data in AGENTS_DATA.items():
        # Initialize the new noisy diff-drive robot
        robots[a_id] = DiffDriveRobot(start_pos=[data['start'][0], data['start'][1]], start_theta=data['start'][2])
        
        # Inject dimensions missing from the new class for APF and Animation compatibility
        robots[a_id].radius = 1.0
        robots[a_id].wheel_length = 0.8
        
        # Each agent has its own APF
        apfs[a_id] = APF(k_att=1.0, k_rep=15.0, rho_0=2.5, obstacles=CIRC_OBSTACLES, rect_obstacles=RECT_OBSTACLES)
        histories[a_id].append((robots[a_id].x, robots[a_id].y, robots[a_id].theta))

    max_steps = 1500
    dt = 0.2  # Time step

    for step in range(max_steps):
        all_reached = True
        current_positions = {a: np.array([robots[a].x, robots[a].y]) for a in AGENTS_DATA}

        for a_id in AGENTS_DATA:
            robot = robots[a_id]
            agent_pos = current_positions[a_id]
            final_goal = np.array([AGENTS_DATA[a_id]['goal'][0], AGENTS_DATA[a_id]['goal'][1]])
            
            if np.linalg.norm(agent_pos - final_goal) < 0.5:
                # Stop the wheels if the goal is reached
                robot.drive(0.0, 0.0, dt)
                histories[a_id].append((robot.x, robot.y, robot.theta))
                continue 
            else:
                all_reached = False

            # --- START OF WAYPOINT TRACKING ---
            path = global_paths[a_id]
            current_idx = waypoint_indices[a_id]
            
            # 1. Find the closest waypoint currently ahead of the robot
            # We scan the next 15 points to see which one we are actually closest to
            min_dist = float('inf')
            best_idx = current_idx
            search_range = min(current_idx + 15, len(path))
            
            for i in range(current_idx, search_range):
                dist = np.linalg.norm(agent_pos - np.array([path[i][0], path[i][1]]))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
                    
            # Update our official index to this closest point
            waypoint_indices[a_id] = best_idx
            
            # 2. Place the "Carrot" a few steps ahead of the closest point
            # Lowered from +5 to +3 so it tracks corners a bit tighter
            lookahead_idx = min(best_idx + 3, len(path) - 1)
            target_x, target_y, _ = path[lookahead_idx]
            current_waypoint = np.array([target_x, target_y])
            # --- END OF WAYPOINT TRACKING ---


            # --- DYNAMIC APF INJECTION ---
            dynamic_obstacles = CIRC_OBSTACLES.copy()
            for other_id, other_pos in current_positions.items():
                if other_id != a_id:
                    dynamic_obstacles.append((other_pos[0], other_pos[1], robot.radius))
            apfs[a_id].obstacles = dynamic_obstacles

            # 1. Calculate APF Force
            force = apfs[a_id].get_force(agent_pos, current_waypoint)
            
            # 2. P-Controller: Convert Force to target v and omega
            fx, fy = force
            force_mag = np.linalg.norm(force)
            
            if force_mag < 0.01:
                cmd_v, cmd_omega = 0.0, 0.0
            else:
                target_theta = np.arctan2(fy, fx)
                error_theta = target_theta - robot.theta
                error_theta = (error_theta + np.pi) % (2 * np.pi) - np.pi
                
                cmd_omega = 2.5 * error_theta  
                cmd_v = force_mag * np.cos(error_theta) 
                cmd_v = max(0.0, cmd_v) # Don't drive backwards
            
            # 3. Kinematic mapping: Convert v and omega to left/right wheel speeds
            half_wheelbase = robot.wheel_dist / 2.0
            cmd_vl = cmd_v - (cmd_omega * half_wheelbase)
            cmd_vr = cmd_v + (cmd_omega * half_wheelbase)

            # 4. Drive the actual hardware simulation
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

        # Draw Environments
# Draw Environments and Inflation Boundaries
        INFLATION_BUFFER = 1.5  

        # 1. Circular Obstacles
        for obs in CIRC_OBSTACLES:
            cx, cy, radius = obs
            # Draw the solid red obstacle
            ax.add_patch(Circle((cx, cy), radius, fill=True, color='red', alpha=0.3))
            
            # Draw the dotted orange inflation boundary
            ax.add_patch(Circle((cx, cy), radius + INFLATION_BUFFER, 
                                fill=False, color='orange', linestyle='--', linewidth=1.5))
        
        # 2. Rectangular Obstacles
        for rect in RECT_OBSTACLES:
            cx, cy, L, W, angle = rect
            
            # Draw the solid red obstacle
            rect_patch = Rectangle((cx - L/2, cy - W/2), L, W, fill=True, color='red', alpha=0.3)
            t = transforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
            rect_patch.set_transform(t)
            ax.add_patch(rect_patch)
            
            # Draw the dotted orange inflation boundary
            # We expand the Length and Width by the buffer on all sides
            inf_L = L + (2 * INFLATION_BUFFER)
            inf_W = W + (2 * INFLATION_BUFFER)
            inf_patch = Rectangle((cx - inf_L/2, cy - inf_W/2), inf_L, inf_W, 
                                  fill=False, color='orange', linestyle='--', linewidth=1.5)
            inf_patch.set_transform(t) # Apply the exact same rotation!
            ax.add_patch(inf_patch)
        
        # Draw Global Paths and Initialize Agent Visuals
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
        ax.set_title('Multi-Agent CBS with Hardware Differential Drive', fontsize=16)
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