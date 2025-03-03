import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from .ur5e_trajectory_planner import demo_motion_gen
import numpy as np

class UR5eJointStatePublisher(Node):
    def __init__(self):
        super().__init__('ur5e_joint_state_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 8)
        self.traj = demo_motion_gen()
        self.traj_points = self.traj.position.cpu().numpy().reshape(-1, 7)
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        self.joint_names = [
            "base_x",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "finger_joint"
        ]

        self.current_point = 0

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        if self.current_point < len(self.traj_points):
            msg.position = self.traj_points[self.current_point].tolist()
            msg.position.append(0.0)
            print(msg.position)
            msg.velocity = [0.0] * 8
            msg.effort = [0.0] * 8
            self.publisher_.publish(msg)
            self.current_point += 1
        else:
            self.get_logger().info('Trajectory completed')
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    ur5e_joint_state_publisher = UR5eJointStatePublisher()
    rclpy.spin(ur5e_joint_state_publisher)
    ur5e_joint_state_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
