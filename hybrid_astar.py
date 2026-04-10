import heapq
import numpy as np
import math
from geometry import get_closest_point_on_circle, get_closest_point_on_rect

class KinematicNode:
    def __init__(self, x, y, theta, time_step=0, parent=None, v=0.0, w=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.time_step = time_step  # NEW: Track time for CBS constraints
        self.parent = parent
        
        self.v = v  
        self.w = w  
        
        self.g = 0.0
        self.h = 0.0
        self.f = 0.0

    def __lt__(self, other):
        return self.f < other.f

class HybridAStar:
    # NEW: Added agent_id and constraints
    def __init__(self, start_pose, goal_pose, obstacles, rect_obstacles=None, robot_radius=1.0, agent_id=0, constraints=None):
        self.start = start_pose
        self.goal = goal_pose
        self.obstacles = obstacles
        self.rect_obstacles = rect_obstacles if rect_obstacles else []
        self.robot_radius = robot_radius
        
        self.agent_id = agent_id
        self.constraints = constraints if constraints else []  # List of (agent_id, x, y, time_step)
        
        self.XY_RESO = 0.5       
        self.YAW_RESO = math.radians(15) 
        self.dt = 0.5

        # Inside hybrid_astar.py -> HybridAStar.__init__
        self.controls = [
            (1.0, 0.0),   # Drive straight
            (1.0, 0.5),   # Curve left
            (1.0, -0.5),  # Curve right
            (0.0, 1.0),   # Spin left
            (0.0, -1.0),  # Spin right
            (0.0, 0.0)    # NEW: Wait in place (Brakes)
        ]

    def _get_discrete_state(self, x, y, theta):
        idx_x = int(round(x / self.XY_RESO))
        idx_y = int(round(y / self.XY_RESO))
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        idx_yaw = int(round(theta / self.YAW_RESO))
        return (idx_x, idx_y, idx_yaw)

    def _simulate_step(self, node, v, w):
        new_theta = node.theta + w * self.dt
        new_x = node.x + v * math.cos(new_theta) * self.dt
        new_y = node.y + v * math.sin(new_theta) * self.dt
        return new_x, new_y, new_theta

    def _is_valid(self, x, y, time_step):
        if x < 0 or x > 20 or y < 0 or y > 20:
            return False
            
        q = np.array([x, y])
        safety_buffer = self.robot_radius + 0.5
        
        # 1. Check Static Circles
        for obs in self.obstacles:
            closest_pt = get_closest_point_on_circle(q, obs)
            if np.linalg.norm(q - closest_pt) < safety_buffer: return False
                
        # 2. Check Static Rectangles
        for rect in self.rect_obstacles:
            closest_pt = get_closest_point_on_rect(q, rect)
            if np.linalg.norm(q - closest_pt) < safety_buffer: return False
                
        # 3. NEW: Check CBS Dynamic Constraints
        for const in self.constraints:
            c_agent, c_x, c_y, c_time = const
            if c_agent == self.agent_id and c_time == time_step:
                # If we are too close to the restricted coordinate at this specific time step
                dist = math.hypot(x - c_x, y - c_y)
                if dist < (self.robot_radius * 2.0 + 0.5): # 2x radius (both robots) + buffer
                    return False

        return True

    def _heuristic(self, x, y):
        return math.hypot(self.goal[0] - x, self.goal[1] - y)

    def find_path(self):
        start_node = KinematicNode(self.start[0], self.start[1], self.start[2], time_step=0)
        start_node.h = self._heuristic(start_node.x, start_node.y)
        start_node.f = start_node.h
        
        open_list = []
        heapq.heappush(open_list, start_node)
        closed_set = set()
        
        while open_list:
            current = heapq.heappop(open_list)
            
            if math.hypot(self.goal[0] - current.x, self.goal[1] - current.y) < 1.0:
                return self._reconstruct_path(current)
                
            discrete_state = self._get_discrete_state(current.x, current.y, current.theta)
            # Add time to discrete state to allow waiting/re-evaluating if blocked
            state_key = (discrete_state[0], discrete_state[1], discrete_state[2], current.time_step)
            
            if state_key in closed_set: continue
            closed_set.add(state_key)
            
            # --- Update this section inside find_path() in hybrid_astar.py ---
            for v, w in self.controls:
                new_x, new_y, new_theta = self._simulate_step(current, v, w)
                new_time = current.time_step + 1
                
                # 1. HARD CAP: Prevent infinite time loops
                if new_time > 200: 
                    continue

                if not self._is_valid(new_x, new_y, new_time):
                    continue
                    
                child = KinematicNode(new_x, new_y, new_theta, time_step=new_time, parent=current, v=v, w=w)
                
                # 2. COST CALCULATION FIX
                if v == 0.0 and w == 0.0:
                    # Heavy penalty for procrastinating (waiting)
                    step_cost = 2.0  
                else:
                    # Normal cost for driving + slight penalty for turning
                    step_cost = (v * self.dt) + (abs(w) * 0.5)

                child.g = current.g + step_cost
                child.h = self._heuristic(child.x, child.y)
                child.f = child.g + child.h
                
                heapq.heappush(open_list, child)
                
        return None

    def _reconstruct_path(self, node):
        path = []
        controls = []
        while node.parent is not None:
            path.append((node.x, node.y, node.theta))
            controls.append((node.v, node.w))
            node = node.parent
        path.append((node.x, node.y, node.theta))
        path.reverse()
        controls.reverse()
        return path, controls