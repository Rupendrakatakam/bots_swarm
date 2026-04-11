import numpy as np

class DiffDriveRobot:
    def __init__(self, start_pos=[0.0, 0.0], start_theta=0.0):
        # --- Pose State ---
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.theta = start_theta
        
        # --- Current Actual Wheel Speeds ---
        self.vl = 0.0          
        self.vr = 0.0      
        
        # --- Robot Dimensions ---
        self.wheel_dist = 0.5      # Distance between wheels (L)
        
        # --- Hardware Constraints ---
        self.max_wheel_speed = 2.0 # Max speed of a single wheel (m/s)
        self.max_wheel_accel = 3.0 # Max acceleration of a single wheel (m/s^2)
        
        # --- Noise Parameters ---
        # Standard deviation for wheel slip/motor noise
        # 0.02 means the wheel speed will randomly fluctuate by roughly +/- 2cm/s
        self.slip_noise_std = 0.02 

    def drive(self, cmd_vl, cmd_vr, dt=0.05):
        """
        Drives the robot using individual wheel velocities.
        Applies acceleration limits, simulates wheel slip (noise), 
        and calculates exact circular arc geometry.
        """
        # 1. Clip commands to absolute maximum hardware limits
        cmd_vl = np.clip(cmd_vl, -self.max_wheel_speed, self.max_wheel_speed)
        cmd_vr = np.clip(cmd_vr, -self.max_wheel_speed, self.max_wheel_speed)
        
        # 2. Apply Acceleration Limits (Velocity Ramping per wheel)
        max_dv = self.max_wheel_accel * dt
        
        # Ramp Left Wheel
        if cmd_vl > self.vl:
            self.vl = min(cmd_vl, self.vl + max_dv)
        elif cmd_vl < self.vl:
            self.vl = max(cmd_vl, self.vl - max_dv)
            
        # Ramp Right Wheel
        if cmd_vr > self.vr:
            self.vr = min(cmd_vr, self.vr + max_dv)
        elif cmd_vr < self.vr:
            self.vr = max(cmd_vr, self.vr - max_dv)
            
        # 3. Apply Noise (Simulating wheel slip / uneven terrain)
        # We use a Gaussian distribution for noise. 
        # We only apply noise if the wheel is actually trying to move.
        eff_vl = self.vl + np.random.normal(0, self.slip_noise_std) if abs(self.vl) > 0.001 else 0.0
        eff_vr = self.vr + np.random.normal(0, self.slip_noise_std) if abs(self.vr) > 0.001 else 0.0

        # 4. Forward Kinematics (Convert effective wheel speeds back to chassis speeds)
        v = (eff_vr + eff_vl) / 2.0
        omega = (eff_vr - eff_vl) / self.wheel_dist
            
        # 5. Exact Arc Integration for Pose Update
        if abs(omega) < 1e-6:
            # Driving straight
            self.x += v * np.cos(self.theta) * dt
            self.y += v * np.sin(self.theta) * dt
            self.theta += omega * dt
        else:
            # Turning along a circular arc: R = v / omega
            R = v / omega
            
            self.x += R * (np.sin(self.theta + omega * dt) - np.sin(self.theta))
            self.y -= R * (np.cos(self.theta + omega * dt) - np.cos(self.theta))
            self.theta += omega * dt
            
        # Keep theta normalized between -pi and pi
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([self.x, self.y, self.theta])