#PYTHON IMPORTS
import os
import yaml
import time

#ROS2 IMPORTS
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

#CUROBO IMPORTS
import torch
from curobo.types.math import Pose
from curobo.types.robot import JointState as RoboJointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.types import Mesh, WorldConfig, Cuboid
from curobo.geom.sdf.world import CollisionCheckerType

#LOCAL IMPORTS
from motion_plan.rviz_publisher import RvizPublisher

class CuroboMotionPlanner(Node):
    def __init__(self):
        super().__init__('curobo_motion_planner')
        joint_names = [
            ""
            "linear_rail"
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.robot_config_file = "ur5e.yml"
        self.obstacles = []
        self.get_logger().info("Motion Planner Initialized")
        self.init_world_config()

        # RViz Publisher for visualization
        self.rviz_publisher = RvizPublisher(joint_names=joint_names, obstacles=self.obstacles, ParentNode=self)
        self.rviz_publisher.publish_obstacles()

        # Start and goal positions
        current_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        goal_position = [1.57, -1.57/2, 0.0, 0.0, 0.0, 0.0]
        self.joint_positions = self.generate_joint_positions(current_position, goal_position)

    def init_world_config(self):
        self.get_logger().info("Loading World")
        self.world_config = WorldConfig()

        stl_file_path = "/home/jetson/ros2_ws/src/motion_plan/motion_plan/CAD/Tray.stl"
        mesh = Mesh(file_path=stl_file_path, name="example_mesh", pose=[0, 0, 0, 0, 0, 0, 1.0])
        mesh.file_path = stl_file_path
        self.world_config.add_obstacle(mesh)
        self.obstacles.append(mesh)

        # self.tensor_args = TensorDeviceType()
        # self.world_config = RobotWorldConfig.load_from_config(
        #     self.robot_config_file,
        #     self.obstacles,
        #     collision_activation_distance=0.0
        # )
        # self.curobo_fn = RobotWorld(self.world_config)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config_file,
            interpolation_dt=0.01,
            world_model=self.world_config,
            collision_checker_type=CollisionCheckerType.MESH
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()

        # config = RobotWorldConfig.load_from_config(
        # self.robot_config_file, world_model=self.world_config, collision_activation_distance=0.0
        # )
        # # Initialize RobotWorld for collision checking
        # self.robot_world = RobotWorld(self.world_config)

    def generate_joint_positions(self, current_position, goal_position):
        start_state = RoboJointState.from_position(torch.tensor([current_position], device="cuda:0"))
        goal_pose = RoboJointState.from_position(torch.tensor([goal_position], device="cuda:0"))

        # Plan the trajectory from start to goal position
        result = self.motion_gen.plan_single_js(
            start_state, goal_pose, MotionGenPlanConfig(max_attempts=1)
        )

        if result.success.item():
            '''
            resulting_plan contains the following info in steps
            joint angles
            acceleration
            velocity
            jerk
            joint names
            '''
            resulting_plan = result.get_interpolated_plan()
            joint_positions = resulting_plan.position.tolist()
            
            self.get_logger().info("Trajectory Generated Successfully!")

            # Iterate through the generated joint positions to check for collisions
            for joint_position in joint_positions:
                self.rviz_publisher.current_joint_state = joint_position
                self.rviz_publisher.publish_joint_states(joint_position)
                time.sleep(0.01)  # Small delay between publishing states

            # Return the trajectory if no collisions were found
            return joint_positions
        else:
            self.get_logger().error("Trajectory Generation Failed.")
            return []

    def collision_checker(self, joint_states):
        q_sph = torch.tensor([joint_states], device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        d_world, d_self = self.curobo_fn.get_world_self_collision_distance_from_joints(q_sph)

        if d_world.item() < 0.0 or d_self.item() < 0.0:
            self.get_logger().warn(f"Collision Detected! World: {d_world.item()}, Self: {d_self.item()}")
        else:
            self.get_logger().info(f"No collision. World: {d_world.item():.3f}, Self: {d_self.item():.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = CuroboMotionPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

