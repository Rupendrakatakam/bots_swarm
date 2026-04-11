"""
Artificial Potential Field (APF) - Local Path Planning
F_att = -k_att * (q - goal)
F_rep = k_rep * (1/rho - 1/rho_0) * (1/rho^2) * (grad(rho))
F_total = F_att + F_rep



"""

import numpy as np
from geometry import get_closest_point_on_circle, get_closest_point_on_rect

class APF:
    def __init__(self, k_att, k_rep, rho_0, obstacles=None, rect_obstacles=None):
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
        self.obstacles = obstacles if obstacles else []
        self.rect_obstacles = rect_obstacles if rect_obstacles else []

    def get_force(self, q, current_goal):
        """Calculates the total force vector at position q."""
        q = np.array(q, dtype=float)
        current_goal = np.array(current_goal, dtype=float)

        # 1. Attractive Force towards the CURRENT waypoint
        F_att = -self.k_att * (q - current_goal)
        F_rep = np.array([0.0, 0.0])

        # 2. Gather all closest surface points from both circles and rectangles
        closest_points = []
        for obs in self.obstacles:
            closest_points.append(get_closest_point_on_circle(q, obs))
        for rect in self.rect_obstacles:
            closest_points.append(get_closest_point_on_rect(q, rect))

        # 3. Repulsive Force from all obstacles
        for surface_point in closest_points:
            rho = np.linalg.norm(q - surface_point)

            # Only push if within the influence radius
            if 0 < rho < self.rho_0:
                magnitude = self.k_rep * (1.0/rho - 1.0/self.rho_0) * (1.0/(rho**2))
                direction = (q - surface_point) / rho 
                F_rep += magnitude * direction

        return F_att + F_rep