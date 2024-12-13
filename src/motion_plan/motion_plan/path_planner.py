#PYTHON IMPORTS
import os
import yaml

#ROS2 IMPORTS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory

#CUROBO IMPORTS
import torch
from curobo.types.math import Pose
from curobo.types.robot import JointState as RoboJointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.joint_positions = self.generate_joint_positions()
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        self.current_step = 0

    def generate_joint_positions(self):
        robot_config_file = "ur5e.yml"
        current_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        goal_position = [-0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_config_file,
            interpolation_dt=0.01,
        )
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup()

        start_state = RoboJointState.from_position(torch.tensor([current_position], device="cuda:0"))
        goal_pose = Pose.from_list(goal_position)

        result = motion_gen.plan_single(
            start_state, goal_pose, MotionGenPlanConfig(max_attempts=1)
        )

        if result.success.item():
            joint_positions = [state.position.tolist() for state in result.get_interpolated_plan()]
            self.get_logger().info("Trajectory Generated Successfully!")
            return joint_positions
        else:
            self.get_logger().error("Trajectory Generation Failed.")
            return []

    def collision_checker(self, joint_states):
        # Load YAML file
        package_share_directory = get_package_share_directory("motion_plan")
        yaml_file = os.path.join(package_share_directory, "configs", "world", "collision_test.yml")

        self.get_logger().info(f"Loading obstacles from {yaml_file}")
        with open(yaml_file, "r") as f:
            self.obstacles = yaml.safe_load(f)

        robot_file = "ur5e.yml"
        self.tensor_args = TensorDeviceType()
        self.world_config = RobotWorldConfig.load_from_config(
            robot_file,
            yaml_file,
            collision_activation_distance=0.0
        )
        self.curobo_fn = RobotWorld(self.world_config)

        # Perform collision checks
        q_sph = torch.tensor([joint_states], device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        d_world, d_self = self.curobo_fn.get_world_self_collision_distance_from_joints(q_sph)

        if d_world.item() < 0.0 or d_self.item() < 0.0:
            self.get_logger().warn(f"Collision Detected! World: {d_world.item()}, Self: {d_self.item()}")
        else:
            self.get_logger().info(f"No collision. World: {d_world.item():.3f}, Self: {d_self.item():.3f}")


def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

