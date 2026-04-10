"""
a-star
f = g + h
g = cost from start to current
h = cost from current to goal

h  = 0.5**((x2-x1)**2 + (y2-y1)**2)
"""

import heapq


class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


class AStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self):
        start_node = Node(self.start)
        goal_node = Node(self.goal)

        open_list = []
        heapq.heappush(open_list, start_node)

        open_set_coords = {self.start}
        closed_set_coords = set()
        node_map = {self.start: start_node}

        while open_list:
            current_node = heapq.heappop(open_list)
            open_set_coords.discard(current_node.position)
            closed_set_coords.add(current_node.position)

            if current_node.position == self.goal:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            for new_position in self.neighbors:
                node_pos = (current_node.position[0] + new_position[0],
                            current_node.position[1] + new_position[1])

                if (node_pos[0] > (self.rows - 1) or node_pos[0] < 0 or
                        node_pos[1] > (self.cols - 1) or node_pos[1] < 0):
                    continue

                if self.grid[node_pos[0]][node_pos[1]] != 0:
                    continue

                if node_pos in closed_set_coords:
                    continue

                child = Node(node_pos, current_node)
                child.g = current_node.g + 1
                child.h = self._heuristic(child.position, goal_node.position)
                child.f = child.g + child.h

                if node_pos in open_set_coords:
                    existing_node = node_map[node_pos]
                    if child.g >= existing_node.g:
                        continue
                    existing_node.g = child.g
                    existing_node.f = child.f
                    existing_node.parent = current_node
                    heapq.heapify(open_list)
                else:
                    heapq.heappush(open_list, child)
                    open_set_coords.add(node_pos)
                    node_map[node_pos] = child

        return None

    def get_animation_frames(self):
        start_node = Node(self.start)
        goal_node = Node(self.goal)

        open_list = []
        heapq.heappush(open_list, start_node)

        open_set_coords = {self.start}
        closed_set_coords = set()
        node_map = {self.start: start_node}

        while open_list:
            current_node = heapq.heappop(open_list)
            open_set_coords.discard(current_node.position)
            closed_set_coords.add(current_node.position)

            yield list(open_set_coords), list(closed_set_coords), []

            if current_node.position == self.goal:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                path = path[::-1]
                yield list(open_set_coords), list(closed_set_coords), path
                return

            for new_position in self.neighbors:
                node_pos = (current_node.position[0] + new_position[0],
                            current_node.position[1] + new_position[1])

                if (node_pos[0] > (self.rows - 1) or node_pos[0] < 0 or
                        node_pos[1] > (self.cols - 1) or node_pos[1] < 0):
                    continue

                if self.grid[node_pos[0]][node_pos[1]] != 0:
                    continue

                if node_pos in closed_set_coords:
                    continue

                child = Node(node_pos, current_node)
                child.g = current_node.g + 1
                child.h = self._heuristic(child.position, goal_node.position)
                child.f = child.g + child.h

                if node_pos in open_set_coords:
                    existing_node = node_map[node_pos]
                    if child.g >= existing_node.g:
                        continue
                    existing_node.g = child.g
                    existing_node.f = child.f
                    existing_node.parent = current_node
                    heapq.heapify(open_list)
                else:
                    heapq.heappush(open_list, child)
                    open_set_coords.add(node_pos)
                    node_map[node_pos] = child

        yield list(open_set_coords), list(closed_set_coords), None


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    grid = np.zeros((20, 20))

    grid[5:15, 5] = 1
    grid[5:15, 10] = 1
    grid[10, 5:10] = 1
    grid[2:7, 15] = 1
    grid[12:18, 15] = 1

    start = (2, 2)
    goal = (17, 18)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap='binary', origin='upper')
    ax.plot(start[1], start[0], 'gH', markersize=12, label='Start')
    ax.plot(goal[1], goal[0], 'r*', markersize=15, label='Goal')

    closed_plot, = ax.plot([], [], 's', color='lightgray', markersize=6, label='Closed Set')
    open_plot, = ax.plot([], [], 's', color='lightblue', markersize=6, label='Open Set')
    path_plot, = ax.plot([], [], '-', color='red', linewidth=4, label='Final Path')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.set_title("A* Pathfinding Animation", pad=30)
    ax.set_xticks(np.arange(-0.5, 20, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 20, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.2)
    ax.tick_params(which="minor", size=0)

    astar = AStar(grid, start, goal)
    astar_gen = astar.get_animation_frames()

    def update(frame_data):
        open_coords, closed_coords, path = frame_data

        if closed_coords:
            cy, cx = zip(*closed_coords)
            closed_plot.set_data(cx, cy)

        if open_coords:
            oy, ox = zip(*open_coords)
            open_plot.set_data(ox, oy)

        if path:
            py, px = zip(*path)
            path_plot.set_data(px, py)

        return closed_plot, open_plot, path_plot

    ani = animation.FuncAnimation(
        fig, update, frames=astar_gen, interval=30,
        repeat=False, blit=True, save_count=500
    )

    plt.tight_layout()
    plt.show()