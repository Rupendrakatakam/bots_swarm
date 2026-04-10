import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# Import your custom modules
from hybrid_astar import HybridAStar
from apf import APF
from robot import DiffDriveRobot  # Importing the newly separated robot

# --- Continuous Environment Setup ---
WORLD_SIZE = 20.0
OBSTACLE_RADIUS = 1.0
ROBOT_RADIUS = 1.0

START = (2.0, 2.0, math.pi/4)  
GOAL = (18.0, 18.0, 0.0)       

# Obstacles: (x, y, radius)
OBSTACLES = [
    (10.0, 10.0, OBSTACLE_RADIUS),
    (14.0, 15.0, OBSTACLE_RADIUS)
]

def run_simulation():
    print("=== Continuous Hybrid Navigation ===")
    
    # 1. GLOBAL PLANNER (Hybrid A*)
    print("1. Calculating continuous kinematic path (Hybrid A*)...")
    h_astar = HybridAStar(
        start_pose=START, 
        goal_pose=GOAL, 
        obstacles=OBSTACLES,
        robot_radius=1.0
    )
    
    result = h_astar.find_path()
    if result is None:
        return None, None, None
        
    global_path, controls = result
    print(f"   -> Path found with {len(global_path)} kinematic states.")

    # 2. LOCAL PLANNER (APF & Robot)
    print("2. Running APF Local Planner tracking Hybrid A* waypoints...")
    
    # Initialize our separated robot
    robot = DiffDriveRobot(start_pos=(START[0], START[1]), start_theta=START[2])

    apf = APF(
        k_att=1.5,   
        k_rep=15.0, 
        rho_0=OBSTACLE_RADIUS + 1.0 + 0.5, 
        obstacles=OBSTACLES
    )

    final_goal = np.array([GOAL[0], GOAL[1]], dtype=float)
    history = [(robot.x, robot.y, robot.theta)]
    
    waypoint_index = 1 
    max_steps = 1500

    for step in range(max_steps):
        agent_pos = np.array([robot.x, robot.y])
        
        if np.linalg.norm(agent_pos - final_goal) < 0.5:
            history.append((robot.x, robot.y, robot.theta))
            print("Goal Reached!")
            break

        target_x, target_y, _ = global_path[waypoint_index]
        current_waypoint = np.array([target_x, target_y])
        
        if np.linalg.norm(agent_pos - current_waypoint) < 1.0:
            if waypoint_index < len(global_path) - 1:
                waypoint_index += 1
                target_x, target_y, _ = global_path[waypoint_index]
                current_waypoint = np.array([target_x, target_y])

        force = apf.get_force(agent_pos, current_waypoint)
        robot.step(force, dt=0.2)
        history.append((robot.x, robot.y, robot.theta))

    return global_path, history, robot

if __name__ == '__main__':
    global_path, history, robot_def = run_simulation()

    if global_path is not None and history is not None:
        # --- Visualization & Animation ---
        inches = WORLD_SIZE / 2.54 
        fig, ax = plt.subplots(figsize=(inches, inches), dpi=100)

        # Static Elements
        gx = [p[0] for p in global_path]
        gy = [p[1] for p in global_path]
        ax.plot(gx, gy, 'r--', linewidth=2.0, alpha=0.6, label='Global Path (Hybrid A*)')

        ax.scatter(START[0], START[1], s=200, c='green', marker='s', label='Start')
        ax.scatter(GOAL[0], GOAL[1], s=300, c='magenta', marker='*', label='Goal')

        for obs in OBSTACLES:
            obs_x, obs_y, radius = obs
            ax.scatter(obs_x, obs_y, s=100, c='black')
            ax.add_patch(plt.Circle((obs_x, obs_y), radius, fill=True, color='red', alpha=0.3))
            ax.add_patch(plt.Circle((obs_x, obs_y), radius + robot_def.radius + 0.5, fill=False, color='orange', linestyle='--'))

        # Dynamic Elements
        body_patch = Circle((history[0][0], history[0][1]), robot_def.radius, fill=True, color='blue', alpha=0.5, label='Diff-Drive Robot')
        ax.add_patch(body_patch)
        
        heading_line, = ax.plot([], [], 'k-', linewidth=2)
        left_wheel, = ax.plot([], [], 'k-', linewidth=4)
        right_wheel, = ax.plot([], [], 'k-', linewidth=4)
        trail_line, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.7, label='Actual Driven Path')

        # Formatting
        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_aspect('equal')
        ax.set_title('Advanced Hybrid Navigation: Hybrid A* + APF', fontsize=16)
        ax.grid(True, linestyle=':', color='gray', alpha=0.6)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1))

        ticks = np.arange(0, WORLD_SIZE + 1, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
        # Turn on the grid with solid lines to create clear squares
        ax.grid(True, which='major', linestyle='-', color='black', alpha=0.4, linewidth=1)

        # Animation Loop
        def update(frame):
            x, y, theta = history[frame]
            
            hx = [p[0] for p in history[:frame+1]]
            hy = [p[1] for p in history[:frame+1]]
            trail_line.set_data(hx, hy)
            
            body_patch.center = (x, y)
            heading_line.set_data([x, x + robot_def.radius * np.cos(theta)], 
                                  [y, y + robot_def.radius * np.sin(theta)])
            
            lx = x - robot_def.wheel_dist * np.sin(theta)
            ly = y + robot_def.wheel_dist * np.cos(theta)
            rx = x + robot_def.wheel_dist * np.sin(theta)
            ry = y - robot_def.wheel_dist * np.cos(theta)
            
            wl = robot_def.wheel_length / 2.0
            left_wheel.set_data([lx - wl * np.cos(theta), lx + wl * np.cos(theta)],
                                [ly - wl * np.sin(theta), ly + wl * np.sin(theta)])
            right_wheel.set_data([rx - wl * np.cos(theta), rx + wl * np.cos(theta)],
                                 [ry - wl * np.sin(theta), ry + wl * np.sin(theta)])
            
            return trail_line, body_patch, heading_line, left_wheel, right_wheel

        ani = animation.FuncAnimation(fig, update, frames=len(history), interval=15, repeat=False)
        plt.tight_layout()
        plt.show()