import heapq
import math
import numpy as np
from hybrid_astar import HybridAStar

class CBSNode:
    def __init__(self, constraints, paths):
        self.constraints = constraints  
        self.paths = paths              
        self.cost = sum(len(p) for p in paths.values())

    def __lt__(self, other):
        return self.cost < other.cost

class CBS:
    def __init__(self, agents_data, obstacles, rect_obstacles, robot_radius=1.0):
        self.agents_data = agents_data
        self.obstacles = obstacles
        self.rect_obstacles = rect_obstacles
        self.robot_radius = robot_radius

    def get_first_conflict(self, paths):
        """Robust conflict detection preventing tunneling and handling goal parking."""
        max_t = max(len(p) for p in paths.values())
        agent_ids = list(paths.keys())

        # Sub-step resolution (Check 3 times between every time step to prevent tunneling)
        sub_steps = 3 

        for t in range(max_t):
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    a1 = agent_ids[i]
                    a2 = agent_ids[j]

                    # Fetch current and next positions (or hold at goal if finished)
                    pos1_curr = paths[a1][t] if t < len(paths[a1]) else paths[a1][-1]
                    pos2_curr = paths[a2][t] if t < len(paths[a2]) else paths[a2][-1]
                    
                    pos1_next = paths[a1][t+1] if (t+1) < len(paths[a1]) else paths[a1][-1]
                    pos2_next = paths[a2][t+1] if (t+1) < len(paths[a2]) else paths[a2][-1]

                    # Interpolate positions between T and T+1
                    for step in range(sub_steps):
                        alpha = step / float(sub_steps)
                        
                        interp_x1 = pos1_curr[0] + alpha * (pos1_next[0] - pos1_curr[0])
                        interp_y1 = pos1_curr[1] + alpha * (pos1_next[1] - pos1_curr[1])
                        
                        interp_x2 = pos2_curr[0] + alpha * (pos2_next[0] - pos2_curr[0])
                        interp_y2 = pos2_curr[1] + alpha * (pos2_next[1] - pos2_curr[1])

                        dist = math.hypot(interp_x1 - interp_x2, interp_y1 - interp_y2)
                        
                        # Collision threshold = 2 * radius + 0.5 cm safety buffer
                        if dist < (self.robot_radius * 2.0 + 0.5):
                            # Conflict returned exactly at integer time step T+1
                            return (a1, a2, pos1_next, pos2_next, t + 1)
        return None

    def run_low_level(self, agent_id, constraints):
        data = self.agents_data[agent_id]
        h_astar = HybridAStar(
            start_pose=data['start'],
            goal_pose=data['goal'],
            obstacles=self.obstacles,
            rect_obstacles=self.rect_obstacles,
            robot_radius=self.robot_radius,
            agent_id=agent_id,
            constraints=constraints
        )
        result = h_astar.find_path()
        return result[0] if result else None

    def find_solution(self):
        print("Initializing CBS Root...")
        initial_paths = {}
        for agent in self.agents_data:
            path = self.run_low_level(agent, [])
            if not path:
                print(f"Failed to find initial path for Agent {agent}")
                return None
            initial_paths[agent] = path

        root = CBSNode(constraints=[], paths=initial_paths)
        open_list = [root]

        iteration = 0
        max_iterations = 500 # Increased from 50 to allow deeper search trees
        
        while open_list and iteration < max_iterations:
            iteration += 1
            P = heapq.heappop(open_list)
            
            conflict = self.get_first_conflict(P.paths)
            if not conflict:
                print(f"CBS Solution Found in {iteration} iterations!")
                return P.paths

            a1, a2, pos1, pos2, time = conflict
            print(f"[{iteration}] Conflict found at T={time} between {a1} and {a2}")

            # Branch 1: Constrain Agent 1
            new_constraints1 = P.constraints + [(a1, pos2[0], pos2[1], time)]
            # Also add a constraint for T-1 to prevent them swapping positions
            new_constraints1 += [(a1, pos2[0], pos2[1], time - 1)] 
            
            new_paths1 = P.paths.copy()
            updated_path1 = self.run_low_level(a1, new_constraints1)
            
            if updated_path1:
                new_paths1[a1] = updated_path1
                heapq.heappush(open_list, CBSNode(new_constraints1, new_paths1))

            # Branch 2: Constrain Agent 2
            new_constraints2 = P.constraints + [(a2, pos1[0], pos1[1], time)]
            new_constraints2 += [(a2, pos1[0], pos1[1], time - 1)]
            
            new_paths2 = P.paths.copy()
            updated_path2 = self.run_low_level(a2, new_constraints2)
            
            if updated_path2:
                new_paths2[a2] = updated_path2
                heapq.heappush(open_list, CBSNode(new_constraints2, new_paths2))

        print("CBS Failed to find a collision-free path (Max Iterations Reached).")
        return None