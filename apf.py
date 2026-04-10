"""
arificial potencial field :
F_att = -k_att(q-q_goal)
F_rep = k_rep(1/rho - 1/rho_0) * (1/rho^2) * (rho - rho_0)/rho
F_total = F_att + F_rep
"""

import numpy as np

class APF:
    def __init__(self, k_att, k_rep, rho_0, obstacles):
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
        self.obstacles = obstacles  # List of (x, y, radius)

    def get_force(self, q, current_goal):
        """Calculates the total force vector at position q."""
        q = np.array(q, dtype=float)
        current_goal = np.array(current_goal, dtype=float)

        # 1. Attractive Force towards the CURRENT waypoint
        F_att = -self.k_att * (q - current_goal)

        # 2. Repulsive Force from all obstacles
        F_rep = np.array([0.0, 0.0])
        for obs in self.obstacles:
            obs_pos = np.array([obs[0], obs[1]])
            
            # Distance from agent to obstacle center
            rho = np.linalg.norm(q - obs_pos)

            # If within influence radius, calculate repulsive push
            if 0 < rho < self.rho_0:
                magnitude = self.k_rep * (1.0/rho - 1.0/self.rho_0) * (1.0/(rho**2))
                direction = (q - obs_pos) / rho  # Unit vector away from obstacle
                F_rep += magnitude * direction

        return F_att + F_rep