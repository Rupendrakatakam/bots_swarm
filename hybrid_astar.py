import heapq
import numpy as np
import math

class KinematicNode:
    def __init__(self, x, y, theta, parent=None, v=0.0, w=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent
        
        # The control inputs used to reach this node
        self.v = v  
        self.w = w  
        
        self.g = 0.0
        self.h = 0.0
        self.f = 0.0

    def __lt__(self, other):
        return self.f < other.f

class HybridAStar:
    def __init__(self, start_pose, goal_pose, obstacles, robot_radius=1.0):
        """
        :param start_pose: (x, y, theta)
        :param goal_pose: (x, y, theta)
        :param obstacles: list of (x, y, radius)
        :param robot_radius: Physical radius of the robot
        """
        self.start = start_pose
        self.goal = goal_pose
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        
        # --- Discretization Settings ---
        # To prevent infinite states, we round positions to group them in the closed set
        self.XY_RESO = 0.5       # Group states within 0.5 cm
        self.YAW_RESO = math.radians(15) # Group states within 15 degrees

        # --- Motion Primitives (Differential Drive) ---
        # List of (linear_velocity v, angular_velocity w)
        self.controls = [
            (1.0, 0.0),    # Drive straight
            (1.0, 0.5),    # Curve left
            (1.0, -0.5),   # Curve right
            (0.0, 1.0),    # Spin left in place
            (0.0, -1.0)    # Spin right in place
        ]
        self.dt = 0.5  # Time step for simulating each motion primitive

    def _get_discrete_state(self, x, y, theta):
        """Converts continuous state to discrete index for the closed set lookup."""
        idx_x = int(round(x / self.XY_RESO))
        idx_y = int(round(y / self.XY_RESO))
        
        # Normalize theta to [-pi, pi] then discretize
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        idx_yaw = int(round(theta / self.YAW_RESO))
        
        return (idx_x, idx_y, idx_yaw)

    def _simulate_step(self, node, v, w):
        """Simulates robot kinematics over time dt."""
        new_theta = node.theta + w * self.dt
        new_x = node.x + v * math.cos(new_theta) * self.dt
        new_y = node.y + v * math.sin(new_theta) * self.dt
        return new_x, new_y, new_theta

    def _is_valid(self, x, y):
        """Checks if the (x,y) position collides with any obstacles."""
        # Check boundaries (assuming 20x20 world)
        if x < 0 or x > 20 or y < 0 or y > 20:
            return False
            
        # Check obstacles
        for obs in self.obstacles:
            obs_x, obs_y, obs_r = obs
            dist = math.hypot(x - obs_x, y - obs_y)
            # Add a small safety buffer (0.5)
            if dist < (obs_r + self.robot_radius + 0.5):
                return False
                
        return True

    def _heuristic(self, x, y):
        """Euclidean distance to goal."""
        return math.hypot(self.goal[0] - x, self.goal[1] - y)

    def find_path(self):
        start_node = KinematicNode(self.start[0], self.start[1], self.start[2])
        start_node.h = self._heuristic(start_node.x, start_node.y)
        start_node.f = start_node.h
        
        open_list = []
        heapq.heappush(open_list, start_node)
        
        # Dictionary to keep track of visited discrete states to avoid infinite loops
        closed_set = set()
        
        while open_list:
            current = heapq.heappop(open_list)
            
            # Check if we are close enough to the goal (e.g., within 1.0 cm)
            dist_to_goal = math.hypot(self.goal[0] - current.x, self.goal[1] - current.y)
            if dist_to_goal < 1.0:
                return self._reconstruct_path(current)
                
            discrete_state = self._get_discrete_state(current.x, current.y, current.theta)
            if discrete_state in closed_set:
                continue
                
            closed_set.add(discrete_state)
            
            # Expand node using motion primitives
            for v, w in self.controls:
                new_x, new_y, new_theta = self._simulate_step(current, v, w)
                
                if not self._is_valid(new_x, new_y):
                    continue
                    
                child = KinematicNode(new_x, new_y, new_theta, parent=current, v=v, w=w)
                
                # Cost is slightly higher for turning to encourage straight paths
                turn_penalty = abs(w) * 0.5 
                child.g = current.g + (v * self.dt) + turn_penalty
                child.h = self._heuristic(child.x, child.y)
                child.f = child.g + child.h
                
                heapq.heappush(open_list, child)
                
        print("Hybrid A* failed to find a path.")
        return None

    def _reconstruct_path(self, node):
        """Backtracks from the goal to the start, returning continuous states and controls."""
        path = []
        controls = []
        
        while node.parent is not None:
            path.append((node.x, node.y, node.theta))
            controls.append((node.v, node.w))
            node = node.parent
            
        path.append((node.x, node.y, node.theta)) # Add start node
        
        # Reverse to get start -> goal
        path.reverse()
        controls.reverse()
        
        return path, controls