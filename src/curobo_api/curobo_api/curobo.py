import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
import torch
import numpy as np
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.types.math import Pose
from curobo_action.action import MoveJ, MoveL

class CuroboNode(Node):
    def __init__(self):
        super().__init__('curobo')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_config_file', 'ur5e.yml'),
                ('world_config_file', 'collision_table.yml'),
                ('tensor_device', 'cuda:0'),
                ('n_obstacle_cuboids', 20),
                ('n_obstacle_mesh', 2)
            ]
        )
        
        self.robot_config_file = self.get_parameter('robot_config_file').get_parameter_value().string_value
        self.world_config_file = self.get_parameter('world_config_file').get_parameter_value().string_value
        self.tensor_device = self.get_parameter('tensor_device').get_parameter_value().string_value
        self.n_obstacle_cuboids = self.get_parameter('n_obstacle_cuboids').get_parameter_value().integer_value
        self.n_obstacle_mesh = self.get_parameter('n_obstacle_mesh').get_parameter_value().integer_value
        
        self.tensor_args = TensorDeviceType(device=self.tensor_device, dtype=torch.float32)
        self.init_curobo()
        
        self.movej_server = ActionServer(
            self, MoveJ, 'movej', self.execute_movej)
        self.movel_server = ActionServer(
            self, MoveL, 'movel', self.execute_movel)
        
        self.get_logger().info("Curobo Action Server Ready.")
    
    def init_curobo(self):
        world_config = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), self.world_config_file))
        )
        
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config_file,
            interpolation_dt=0.01,
            world_model=world_config,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": self.n_obstacle_cuboids, "mesh": self.n_obstacle_mesh},
        )
        
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        
        self.get_logger().info("Curobo MotionGen initialized and warmed up.")
    
    def plan_motion_js(self, start_state, goal_state):
        current_state = JointState.from_position(torch.tensor([start_state], device="cuda:0", dtype=torch.float32))
        goal_state = JointState.from_position(torch.tensor([goal_state], device="cuda:0", dtype=torch.float32))

        result = self.motion_gen.plan_single_js(current_state, goal_state, MotionGenPlanConfig(max_attempts=10000))

        if result.success.item():
            trajectory = result.get_interpolated_plan().position.tolist()
            joints = len(trajectory[0]) if trajectory else 0  # Number of joints per waypoint
            flattened_trajectory = [item for sublist in trajectory for item in sublist]  # Flatten list
            return True, flattened_trajectory, joints, str(result)
        else:
            return False, [], 0, str(result)


    def plan_motion(self, start_position, goal_position):
        goal_pose = Pose(
            position=self.tensor_args.to_device([goal_position[:3]]),
            quaternion=self.tensor_args.to_device([goal_position[3:]])  
        )
        current_position = JointState.from_position(
            torch.tensor([start_position], device="cuda:0", dtype=torch.float32))

        result = self.motion_gen.plan_single(current_position, goal_pose, MotionGenPlanConfig(max_attempts=10000))

        if result.success.item():
            trajectory = result.get_interpolated_plan().position.tolist()
            joints = len(trajectory[0]) if trajectory else 0  # Number of joints per waypoint
            flattened_trajectory = [item for sublist in trajectory for item in sublist]  # Flatten list
            return True, flattened_trajectory, joints, str(result)
        else:
            return False, [], 0, str(result)

    async def execute_movej(self, goal_handle):
        self.get_logger().info("Executing MoveJ action request")
        success, trajectory, joints, result = self.plan_motion_js(goal_handle.request.start_state, goal_handle.request.goal_pose)
        goal_handle.succeed()
        return MoveJ.Result(success=success, result=result, trajectory=trajectory, joints=joints)


    async def execute_movel(self, goal_handle):
        self.get_logger().info("Executing MoveL action request")
        success, trajectory, joints, result = self.plan_motion(goal_handle.request.start_state, goal_handle.request.goal_pose)
        goal_handle.succeed()
        return MoveL.Result(success=success, result=result, trajectory=trajectory, joints=joints)

def main(args=None):
    rclpy.init(args=args)
    node = CuroboNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
