import numpy as np

class DiffDriveRobot:
    def __init__(self, start_pos, start_theta=0.0):
        # State
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.theta = start_theta
        
        # Dimensions
        self.radius = 1.0         
        self.wheel_dist = 0.5625  
        self.wheel_width = 0.125  
        self.wheel_length = 0.8   
        
        # Kinematic Limits
        self.max_v = 1.5          
        self.max_omega = np.pi    
        
    def step(self, force, dt=0.2):
        """Translates an X/Y force vector into diff-drive kinematics."""
        fx, fy = force
        force_mag = np.linalg.norm(force)
        
        if force_mag < 0.01:
            return np.array([self.x, self.y])

        target_theta = np.arctan2(fy, fx)
        
        error_theta = target_theta - self.theta
        error_theta = (error_theta + np.pi) % (2 * np.pi) - np.pi
        
        omega = 2.5 * error_theta 
        omega = np.clip(omega, -self.max_omega, self.max_omega)
        
        v = force_mag * np.cos(error_theta)
        v = np.clip(max(0, v), 0, self.max_v)
        
        self.theta += omega * dt
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        
        return np.array([self.x, self.y])