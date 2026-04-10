import heapq
import math
from hybrid_astar import HybridAStar

class CBSNode:
    def __init__(self, constraints, paths):
        self.constraints = constraints  # List of (agent_id, x, y, time_step)
        self.paths = paths              # Dictionary {agent_id: [(x, y, theta), ...]}
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
        """Finds the first time step where two agents physically overlap."""
        max_t = max(len(p) for p in paths.values())
        agent_ids = list(paths.keys())

        for t in range(max_t):
            # Compare every pair of agents
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    a1 = agent_ids[i]
                    a2 = agent_ids[j]

                    # If an agent finishes its path, it stays at its goal coordinate
                    pos1 = paths[a1][t] if t < len(paths[a1]) else paths[a1][-1]
                    pos2 = paths[a2][t] if t < len(paths[a2]) else paths[a2][-1]

                    dist = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
                    
                    # If they are closer than their combined diameters (+ buffer)
                    if dist < (self.robot_radius * 2.0 + 0.5):
                        # Conflict: Agent 1, Agent 2, Pos of Agent 1, Pos of Agent 2, Time
                        return (a1, a2, pos1, pos2, t)
        return None

    def run_low_level(self, agent_id, constraints):
        """Runs Hybrid A* for a single agent with specific constraints."""
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
        while open_list and iteration < 50:  # Safety cap to prevent infinite loops
            iteration += 1
            P = heapq.heappop(open_list)
            
            conflict = self.get_first_conflict(P.paths)
            if not conflict:
                print(f"CBS Solution Found in {iteration} iterations!")
                return P.paths  # Success!

            a1, a2, pos1, pos2, time = conflict
            print(f"Conflict found at T={time} between Agent {a1} and Agent {a2}")

            # Branch 1: Force Agent 1 to avoid Agent 2's position at time T
            constraint1 = (a1, pos2[0], pos2[1], time)
            new_constraints1 = P.constraints + [constraint1]
            new_paths1 = P.paths.copy()
            updated_path1 = self.run_low_level(a1, new_constraints1)
            
            if updated_path1:
                new_paths1[a1] = updated_path1
                heapq.heappush(open_list, CBSNode(new_constraints1, new_paths1))

            # Branch 2: Force Agent 2 to avoid Agent 1's position at time T
            constraint2 = (a2, pos1[0], pos1[1], time)
            new_constraints2 = P.constraints + [constraint2]
            new_paths2 = P.paths.copy()
            updated_path2 = self.run_low_level(a2, new_constraints2)
            
            if updated_path2:
                new_paths2[a2] = updated_path2
                heapq.heappush(open_list, CBSNode(new_constraints2, new_paths2))

        print("CBS Failed to find a collision-free path.")
        return None