import numpy as np
from geometry import get_closest_point_on_circle, get_closest_point_on_rect

class APF:
    def __init__(self, k_att, k_rep, rho_0, obstacles=None, rect_obstacles=None):
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
        self.obstacles = obstacles if obstacles else []
        self.rect_obstacles = rect_obstacles if rect_obstacles else []

    def get_force(self, q, current_goal, dynamic_obstacles=None):
        """
        dynamic_obstacles: List of (x, y, radius, multiplier)
        """
        q = np.array(q, dtype=float)
        current_goal = np.array(current_goal, dtype=float)

        # 1. Attractive Force
        F_att = -self.k_att * (q - current_goal)
        F_rep = np.array([0.0, 0.0])

        # 2. Static Obstacles
        static_pts = []
        for obs in self.obstacles:
            static_pts.append(get_closest_point_on_circle(q, obs))
        for rect in self.rect_obstacles:
            static_pts.append(get_closest_point_on_rect(q, rect))

        for pt in static_pts:
            rho = np.linalg.norm(q - pt)
            if 0 < rho < self.rho_0:
                mag = self.k_rep * (1.0/rho - 1.0/self.rho_0) * (1.0/(rho**2))
                F_rep += mag * (q - pt) / rho

        # 3. Dynamic Priority Obstacles (Step 3: Asymmetric Repulsion)
        if dynamic_obstacles:
            for (ox, oy, orad, multiplier) in dynamic_obstacles:
                obs_pos = np.array([ox, oy])
                dist = np.linalg.norm(q - obs_pos)

                # Distance to the surface of the other robot
                rho = dist - orad
                
                # CRITICAL SAFETY CATCH: Prevent Division by Zero if they touch!
                if rho < 0.05:
                    rho = 0.05
                
                if 0 < rho < self.rho_0:
                    # We multiply the k_rep by the priority multiplier
                    mag = (self.k_rep * multiplier) * (1.0/rho - 1.0/self.rho_0) * (1.0/(rho**2))
                    direction = (q - obs_pos) / (rho + orad)
                    F_rep += mag * direction
                    
                    # Add Vortex Force to break head-on symmetry
                    tangent = np.array([-direction[1], direction[0]])
                    F_rep += (mag * 0.5) * tangent

        return F_att + F_rep