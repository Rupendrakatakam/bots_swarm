"""
Move agent from start (1,1) to goal (19,19) with obstacles
in a map of 20x20
cell size 1cmx1cm, agent diameter 5cm
obstacle at (10,15) with radius 2cm

Hybrid Planner:
- Global Planner (A*): Plan path on grid
- Local Planner (APF): Navigate around obstacles between waypoints
"""

import numpy as np
from astar import AStar
from apf import APF


WORLD_SIZE = 20
CELL_SIZE = 1
AGENT_DIAMETER = 5
AGENT_RADIUS = AGENT_DIAMETER / 2
OBSTACLE_RADIUS = 2
SAFE_MARGIN = 1
OBSTACLE_INFLUENCE = OBSTACLE_RADIUS + AGENT_RADIUS + SAFE_MARGIN

START = (1, 1)
GOAL = (19, 19)
OBSTACLE_CENTER = (10, 15)


def grid_to_world(row_col):
    row, col = row_col
    return (col * CELL_SIZE, row * CELL_SIZE)


def world_to_grid(x_y):
    x, y = x_y
    return (int(y / CELL_SIZE), int(x / CELL_SIZE))


def create_grid_with_obstacles():
    grid = np.zeros((WORLD_SIZE, WORLD_SIZE))

    ox, oy = grid_to_world(OBSTACLE_CENTER)
    obs_radius = OBSTACLE_INFLUENCE

    for row in range(WORLD_SIZE):
        for col in range(WORLD_SIZE):
            wx, wy = grid_to_world((row, col))
            dist = np.sqrt((wx - ox)**2 + (wy - oy)**2)
            if dist < obs_radius:
                grid[row, col] = 1

    return grid


def run_global_planner(grid, start, goal):
    astar = AStar(grid, start, goal)
    path = astar.find_path()
    return path


def run_local_planner(start, goal, obstacles):
    apf = APF(
        goal=goal,
        obstacles=obstacles,
        k_att=1.0,
        k_rep=2.0,
        rho_0=OBSTACLE_INFLUENCE
    )
    path = apf.compute_path(start, max_steps=500, step_size=0.5)
    return path


def move_agent():
    print("=== Hybrid Navigation ===")
    print(f"Map: {WORLD_SIZE}x{WORLD_SIZE}, Cell: {CELL_SIZE}cm")
    print(f"Agent: {AGENT_DIAMETER}cm dia, Start: {START}, Goal: {GOAL}")
    print(f"Obstacle: cell {OBSTACLE_CENTER}, radius {OBSTACLE_RADIUS}cm")
    print()

    grid = create_grid_with_obstacles()
    print("Grid created with obstacles marked")

    global_path = run_global_planner(grid, START, GOAL)
    if global_path is None:
        print("ERROR: No path found!")
        return

    print(f"Global path (A*): {len(global_path)} waypoints")
    print(f"  Path: {global_path[:5]} ... {global_path[-3:]}")
    print()

    obstacle_world = [grid_to_world(OBSTACLE_CENTER)]
    start_world = grid_to_world(START)
    goal_world = grid_to_world(GOAL)

    print("Running local planner (APF)...")
    local_path = run_local_planner(start_world, goal_world, obstacle_world)

    print(f"Local path (APF): {len(local_path)} points")

    return grid, global_path, local_path


if __name__ == '__main__':
    grid, global_path, local_path = move_agent()

    import matplotlib.pyplot as plt

    start_world = grid_to_world(START)
    goal_world = grid_to_world(GOAL)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.imshow(grid, cmap='binary', origin='upper')
    gr, gc = zip(*global_path) if global_path else ([], [])
    ax1.plot(gc, gr, 'r-', linewidth=2, label='Global Path')
    ax1.plot(START[1], START[0], 'gH', markersize=12, label='Start')
    ax1.plot(GOAL[1], GOAL[0], 'm*', markersize=15, label='Goal')
    ax1.set_title('Global Path (A*)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    lx, ly = local_path[:, 0], local_path[:, 1]
    ax2.plot(lx, ly, 'b-', linewidth=1.5, label='Local Path')
    ax2.scatter(OBSTACLE_CENTER[0], OBSTACLE_CENTER[1], s=200, c='red',
                label='Obstacle', marker='o')
    circ = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, fill=False,
                     color='red', linestyle='--')
    ax2.add_patch(circ)
    ax2.scatter(start_world[0], start_world[1], s=100, c='green',
                marker='H', label='Start')
    ax2.scatter(goal_world[0], goal_world[1], s=100, c='magenta',
                marker='*', label='Goal')
    ax2.set_xlim(0, WORLD_SIZE)
    ax2.set_ylim(0, WORLD_SIZE)
    ax2.set_aspect('equal')
    ax2.set_title('Local Path (APF)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()