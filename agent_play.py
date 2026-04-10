import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astar import AStar
from apf import APF

# --- Environment Setup ---
WORLD_SIZE = 20
CELL_SIZE = 1
AGENT_DIAMETER = 1
AGENT_RADIUS = AGENT_DIAMETER / 2
OBSTACLE_RADIUS = 2

START = (1, 1)    # (row, col)
GOAL = (19, 19)   # (row, col)

# Obstacles: (row, col, radius)
OBSTACLES = [
    (10, 10, OBSTACLE_RADIUS)
]

def grid_to_world(row_col):
    """Converts matrix indices (row, col) to geometric (x, y)."""
    row, col = row_col
    return (col * CELL_SIZE, row * CELL_SIZE)

def run_simulation():
    print("=== Hybrid Navigation ===")
    
    # 1. GLOBAL PLANNER (A*)
    print("1. Running Global Planner (A*)...")
    astar = AStar(
        start=START,
        goal=GOAL,
        world_size=WORLD_SIZE,
        cell_size=CELL_SIZE,
        obstacles=OBSTACLES
    )
    
    global_path_grid = astar.find_path()
    if not global_path_grid:
        print("ERROR: A* could not find a path!")
        return None, None, None

    global_path = [grid_to_world(p) for p in global_path_grid]
    
    # 2. LOCAL PLANNER (APF)
    print("2. Running Local Planner (APF)...")
    
    apf_obstacles = []
    for obs in OBSTACLES:
        apf_obstacles.append(grid_to_world((obs[0], obs[1])))

    apf = APF(
        k_att=0.5, 
        k_rep=10.0, 
        rho_0=OBSTACLE_RADIUS + AGENT_RADIUS + 1, 
        obstacles=apf_obstacles
    )

    agent_pos = np.array(grid_to_world(START), dtype=float)
    final_goal = np.array(grid_to_world(GOAL), dtype=float)
    
    local_path = [agent_pos.copy()]
    waypoint_index = 1 
    
    max_steps = 1000
    step_size = 0.5

    for step in range(max_steps):
        if np.linalg.norm(agent_pos - final_goal) < step_size:
            local_path.append(final_goal)
            print("Goal Reached!")
            break

        current_waypoint = np.array(global_path[waypoint_index])
        
        if np.linalg.norm(agent_pos - current_waypoint) < 1.5:
            if waypoint_index < len(global_path) - 1:
                waypoint_index += 1
                current_waypoint = np.array(global_path[waypoint_index])

        force = apf.get_force(agent_pos, current_waypoint)
        
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > 0:
            force = (force / force_magnitude) * step_size

        agent_pos += force
        local_path.append(agent_pos.copy())

    return astar.grid, global_path, np.array(local_path)

if __name__ == '__main__':
    grid, global_path, local_path = run_simulation()

    if global_path is not None and local_path is not None:
        # --- Animation Setup ---
        fig, ax = plt.subplots(figsize=(10, 10))

        # 1. Plot Static Elements (Background, A* path, Start/Goal)
        gx, gy = zip(*global_path)
        ax.plot(gx, gy, 'r--', linewidth=2.5, alpha=0.5, label='Global Path (A*)')

        start_world = grid_to_world(START)
        goal_world = grid_to_world(GOAL)
        ax.scatter(start_world[0], start_world[1], s=200, c='green', marker='s', label='Start')
        ax.scatter(goal_world[0], goal_world[1], s=300, c='magenta', marker='*', label='Goal')

        # Draw Obstacles
        for obs in OBSTACLES:
            obs_x, obs_y = grid_to_world((obs[0], obs[1]))
            radius = obs[2]
            
            ax.scatter(obs_x, obs_y, s=100, c='black')
            circ = plt.Circle((obs_x, obs_y), radius, fill=True, color='red', alpha=0.3)
            ax.add_patch(circ)
            
            rho_0 = radius + AGENT_RADIUS + 2
            circ_inf = plt.Circle((obs_x, obs_y), rho_0, fill=False, color='orange', linestyle='--')
            ax.add_patch(circ_inf)

        # 2. Initialize Dynamic Elements (Agent and APF Trail)
        # The agent is drawn as a circle matching its physical diameter
        agent_patch = plt.Circle((start_world[0], start_world[1]), AGENT_RADIUS, fill=True, color='blue', alpha=0.7, label='Agent')
        ax.add_patch(agent_patch)
        
        # The trail the agent leaves behind
        trail_line, = ax.plot([], [], 'b-', linewidth=2.5, label='Actual Path (APF)')

        # Formatting
        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title('Hybrid Planner Animation', fontsize=16, pad=20)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5))
        
        ax.set_xticks(np.arange(0, WORLD_SIZE + 1, CELL_SIZE))
        ax.set_yticks(np.arange(0, WORLD_SIZE + 1, CELL_SIZE))
        ax.grid(True, linestyle=':', color='gray', alpha=0.6)

        # 3. Animation Update Function
        def update(frame):
            # Update the trail up to the current frame
            trail_x = local_path[:frame+1, 0]
            trail_y = local_path[:frame+1, 1]
            trail_line.set_data(trail_x, trail_y)
            
            # Update the agent's current position
            current_pos = local_path[frame]
            agent_patch.center = (current_pos[0], current_pos[1])
            
            return trail_line, agent_patch

        # 4. Run Animation
        ani = animation.FuncAnimation(
            fig, 
            update, 
            frames=len(local_path), 
            interval=30,     # Milliseconds between frames (adjust for speed)
            repeat=False,    # Stops when it reaches the goal
            blit=False       # Set to False because we are updating complex patches
        )

        plt.tight_layout()
        plt.show()