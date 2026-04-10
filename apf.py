"""
arificial potencial field :
F_att = -k_att(q-q_goal)
F_rep = k_rep(1/rho - 1/rho_0) * (1/rho^2) * (rho - rho_0)/rho
F_total = F_att + F_rep
"""

import numpy as np

class APF:
    def __init__(self, k_att, k_rep, rho_0, goal, obstacles):
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
        self.goal = goal
        self.obstacles = obstacles

    def _attractive_force(self, q):
        return -self.k_att * (q - self.goal)

    def _repulsive_force(self, dist):
        F_rep = np.zeros_like(dist)
        for i in range(len(dist)):
            if dist[i] < self.rho_0:
                F_rep[i] = self.k_rep * (1/dist[i] - 1/self.rho_0) * (1/dist[i]**2) * (dist[i] - self.obstacles[i]) / dist[i]
            else:
                F_rep[i] = 0
        return F_rep

    def _total_force(self, q):
        F_att = self._attractive_force(q)
        F_rep = self._repulsive_force(q)
        return F_att + F_rep
        
                