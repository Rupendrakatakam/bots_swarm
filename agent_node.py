import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rvo2
import math

class SingleAgentController(Node):
    def __init__(self):
        super().__init__('single_agent_controller')
        
        # --- 1. CONFIGURATION ---
        self.robot_name = 'turtlebot3' # Change this if your Gazebo namespace is different
        self.target = (2.0, 2.0)       # Milestone 1 Hardcoded Target
        self.wp_threshold = 0.1        # Stop when within 10cm
        
        # --- 2. ORCA SETUP ---
        self.get_logger().info("Initializing ORCA Simulator for 1 Agent...")
        # timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed
        self.sim = rvo2.PyRVOSimulator(1/20.0, 2.5, 10, 2.0, 2.0, 0.3, 0.22) # Max speed 0.22m/s
        self.agent_id = self.sim.addAgent((0.0, 0.0))
        
        # --- 3. STATE VARIABLES ---
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.goal_reached = False

        # --- 4. ROS 2 PUBLISHERS & SUBSCRIBERS ---
        self.cmd_pub = self.create_publisher(Twist, f'/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, f'/odom', self.odom_callback, 10)
        
        # --- 5. CONTROL LOOP (Runs at 20Hz) ---
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(f"Node Started. Target set to {self.target}")

    def euler_from_quaternion(self, x, y, z, w):
        """Converts 3D Quaternion to 2D Yaw (Heading)"""
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def odom_callback(self, msg):
        """Extracts X, Y, and Yaw from Gazebo's Odometry message"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        self.current_yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)

    def control_loop(self):
        if self.goal_reached:
            return

        # 1. Update ORCA with our exact physical position
        self.sim.setAgentPosition(self.agent_id, (self.current_x, self.current_y))

        # 2. Calculate vector to target
        vector_x = self.target[0] - self.current_x
        vector_y = self.target[1] - self.current_y
        dist = math.hypot(vector_x, vector_y)

        # 3. Check if we arrived
        if dist < self.wp_threshold:
            self.get_logger().info("Target Reached! Stopping.")
            self.cmd_pub.publish(Twist()) # Publish empty Twist to brake
            self.goal_reached = True
            return

        # 4. Calculate Preferred Velocity (Slow down smoothly as we approach)
        speed = min(0.22, dist) # Max speed of TurtleBot3 is 0.22 m/s
        pref_vx = (vector_x / dist) * speed
        pref_vy = (vector_y / dist) * speed
        
        self.sim.setAgentPrefVelocity(self.agent_id, (pref_vx, pref_vy))

        # 5. Run ORCA Physics (Even for 1 agent, this is required to output the safe vector)
        self.sim.doStep()
        safe_vx, safe_vy = self.sim.getAgentVelocity(self.agent_id)

        # 6. TRANSLATE: Holonomic ORCA -> Non-Holonomic Differential Drive
        target_yaw = math.atan2(safe_vy, safe_vx)
        
        # Calculate heading error and normalize between -PI and PI
        yaw_error = target_yaw - self.current_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        # Create Motor Commands
        cmd = Twist()
        
        # Proportional Steering (Turns harder the further away the angle is)
        cmd.angular.z = 1.5 * yaw_error 
        
        # Forward speed (Cos ensures we slow down our forward movement if we need to make a sharp turn)
        cmd.linear.x = math.hypot(safe_vx, safe_vy) * max(0.0, math.cos(yaw_error))

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = SingleAgentController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Guarantee the robot stops if we kill the script
        node.cmd_pub.publish(Twist()) 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()