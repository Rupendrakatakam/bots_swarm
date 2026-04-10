import rvo2
import numpy as np
import math

class ORCASwarmManager:
    def __init__(self, num_agents, dt=1/60.0, neighbor_dist=2.5, max_neighbors=10, 
                 time_horizon=2.0, time_horizon_obst=2.0, radius=0.4, max_speed=2.0):
        """
        Initializes the RVO2 (ORCA) Simulator.
        """
        self.sim = rvo2.PyRVOSimulator(dt, neighbor_dist, max_neighbors, 
                                       time_horizon, time_horizon_obst, 
                                       radius, max_speed)
        self.agents = []
        self.num_agents = num_agents
        self.max_speed = max_speed

    def add_circular_obstacle(self, center_x, center_y, radius, num_points=8):
        """Adds a circular obstacle to the ORCA physics engine."""
        vertices = []
        for j in range(num_points):
            angle = j * (2 * math.pi / num_points)
            vx = center_x + radius * math.cos(angle)
            vy = center_y + radius * math.sin(angle)
            vertices.append((vx, vy))
        self.sim.addObstacle(vertices)

    def finalize_obstacles(self):
        """Must be called after all obstacles are added, before simulation steps."""
        self.sim.processObstacles()

    def spawn_agents(self, positions):
        """
        Registers agents in the simulation.
        positions: List of (x, y) tuples.
        """
        for pos in positions:
            agent_id = self.sim.addAgent(pos)
            self.agents.append(agent_id)

    def update_step(self, current_targets):
        """
        Calculates preferred velocities, runs one ORCA step, and returns new positions.
        current_targets: List of (x, y) coordinates the agents want to reach.
        """
        new_positions = []
        new_velocities = []

        for i, agent_id in enumerate(self.agents):
            pos = self.sim.getAgentPosition(agent_id)
            target = current_targets[i]
            
            # 1. Calculate Preferred Velocity (The 'Intent')
            vx = target[0] - pos[0]
            vy = target[1] - pos[1]
            dist = math.hypot(vx, vy)

            if dist > 0.1:
                # Normalize and scale to max_speed
                pref_vel = (vx / dist * min(self.max_speed, dist), 
                            vy / dist * min(self.max_speed, dist))
            else:
                pref_vel = (0, 0)

            # 2. Tell the simulator what the agent WANTS to do
            self.sim.setAgentPrefVelocity(agent_id, pref_vel)

        # 3. Perform the collision avoidance step (The 'ORCA' magic)
        self.sim.doStep()

        # 4. Collect results
        for agent_id in self.agents:
            new_positions.append(self.sim.getAgentPosition(agent_id))
            new_velocities.append(self.sim.getAgentVelocity(agent_id))

        return new_positions, new_velocities

    def get_positions(self):
        return [self.sim.getAgentPosition(a) for a in self.agents]