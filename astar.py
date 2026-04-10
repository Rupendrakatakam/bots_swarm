import heapq
import numpy as np


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
    def __init__(self, start, goal, world_size=20, cell_size=1, obstacles=None):
        """
        Initializes the A* pathfinding algorithm.
        
        :param start: Tuple (row, col) representing the starting position.
        :param goal: Tuple (row, col) representing the goal position.
        :param world_size: Size of the grid (world_size x world_size).
        :param cell_size: Size of each cell in cm.
        :param obstacles: List of tuples (row, col, radius) for circular obstacles.
        """
        self.world_size = world_size
        self.cell_size = cell_size
        self.start = start
        self.goal = goal
        self.rows = world_size
        self.cols = world_size
        self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0) ,(1,1), (1,-1), (-1,1), (-1,-1)]

        self.grid = self._create_grid(obstacles)

    def _create_grid(self, obstacles):
        grid = np.zeros((self.rows, self.cols))

        if obstacles is None:
            return grid

        for obs_row, obs_col, radius in obstacles:
            obs_x = obs_col * self.cell_size
            obs_y = obs_row * self.cell_size
            influence = radius + self.cell_size

            for row in range(self.rows):
                for col in range(self.cols):
                    wx = col * self.cell_size
                    wy = row * self.cell_size
                    dist = np.sqrt((wx - obs_x)**2 + (wy - obs_y)**2)
                    if dist < influence:
                        grid[row, col] = 1

        return grid

    def _heuristic(self, a, b):
        # Euclidean distance
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def find_path(self):
        """
        Executes the A* search.
        
        :return: A list of tuples representing the path from start to goal, 
                 or None if no path is found.
        """
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

            # Goal reached
            if current_node.position == self.goal:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            # Check neighbors
            for new_position in self.neighbors:
                node_pos = (current_node.position[0] + new_position[0],
                            current_node.position[1] + new_position[1])

                # Check boundaries
                if (node_pos[0] > (self.rows - 1) or node_pos[0] < 0 or
                        node_pos[1] > (self.cols - 1) or node_pos[1] < 0):
                    continue

                # Check obstacles (assuming 0 is free space)
                if self.grid[node_pos[0]][node_pos[1]] != 0:
                    continue

                # Check if already evaluated
                if node_pos in closed_set_coords:
                    continue

                # Create child node
                child = Node(node_pos, current_node)
                # If the move is diagonal (both row and col change), cost is 1.414. Otherwise, 1.
                cost = 1.414 if new_position[0] != 0 and new_position[1] != 0 else 1
                child.g = current_node.g + cost
                child.h = self._heuristic(child.position, goal_node.position)
                child.f = child.g + child.h

                # Check if a better path exists in the open set
                if node_pos in open_set_coords:
                    existing_node = node_map[node_pos]
                    if child.g >= existing_node.g:
                        continue
                    
                    # Update existing node
                    existing_node.g = child.g
                    existing_node.f = child.f
                    existing_node.parent = current_node
                    heapq.heapify(open_list)
                else:
                    # Add new node to open set
                    heapq.heappush(open_list, child)
                    open_set_coords.add(node_pos)
                    node_map[node_pos] = child

        # Return None if no valid path exists
        return None