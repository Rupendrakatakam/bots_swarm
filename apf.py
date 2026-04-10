"""
Artificial Potential Field (APF) - Local Path Planning
F_att = -k_att * (q - goal)
F_rep = k_rep * (1/rho - 1/rho_0)^2 * grad(rho)
F_total = F_att + F_rep
"""

import numpy as np


class APF:
    def __init__(self, goal, obstacles, k_att=1.0, k_rep=1.0, rho_0=2.0):
        self.goal = np.array(goal, dtype=float)
        self.obstacles = [np.array(obs, dtype=float) for obs in obstacles]
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0

    def _attractive_force(self, q):
        q = np.array(q, dtype=float)
        diff = q - self.goal
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.zeros(2)
        return -self.k_att * diff

    def _repulsive_force(self, q):
        q = np.array(q, dtype=float)
        F_rep = np.zeros(2)

        for obs in self.obstacles:
            diff = q - obs
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue

            if dist < self.rho_0:
                factor = self.k_rep * (1.0 / dist - 1.0 / self.rho_0) ** 2
                away = diff / dist
                F_rep += factor * away

        return F_rep

    def compute_force(self, q):
        F_att = self._attractive_force(q)
        F_rep = self._repulsive_force(q)
        return F_att + F_rep

    def get_next_position(self, q, step_size=0.5):
        q = np.array(q, dtype=float)
        force = self.compute_force(q)
        force_mag = np.linalg.norm(force)

        if force_mag > 1e-6:
            direction = force / force_mag
            new_pos = q + direction * step_size
        else:
            new_pos = q.copy()

        return new_pos

    def compute_path(self, start, max_steps=500, step_size=0.5):
        path = [np.array(start, dtype=float)]
        current = np.array(start, dtype=float)

        for _ in range(max_steps):
            next_pos = self.get_next_position(current, step_size)
            path.append(next_pos)
            current = next_pos

            if np.linalg.norm(current - self.goal) < step_size:
                break

        return np.array(path)
        
                