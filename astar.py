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
        """
        Initializes the A* pathfinding algorithm.
        
        :param grid: 2D list or numpy array where 0 is free space and non-zero is an obstacle.
        :param start: Tuple (row, col) representing the starting position.
        :param goal: Tuple (row, col) representing the goal position.
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])
        # 4-way movement (Right, Left, Down, Up)
        self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def _heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
                child.g = current_node.g + 1
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