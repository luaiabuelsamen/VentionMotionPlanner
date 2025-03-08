import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
import torch
import numpy as np
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.types.math import Pose
from curobo_action.action import MoveJ, MoveL, PublishJoints
import time

class CuroboNode(Node):
    def __init__(self):
        super().__init__('curobo')
        self.callback_group = ReentrantCallbackGroup()
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_config_file', 'ur5e.yml'),
                ('world_config_file', 'collision_table.yml'),
                ('tensor_device', 'cuda:0'),
                ('n_obstacle_cuboids', 20),
                ('n_obstacle_mesh', 2),
                ('trajectory_execution_dt', 0.01),
                ('wait_for_completion', True)
            ]
        )
        
        self.robot_config_file = self.get_parameter('robot_config_file').get_parameter_value().string_value
        self.world_config_file = self.get_parameter('world_config_file').get_parameter_value().string_value
        self.tensor_device = self.get_parameter('tensor_device').get_parameter_value().string_value
        self.n_obstacle_cuboids = self.get_parameter('n_obstacle_cuboids').get_parameter_value().integer_value
        self.n_obstacle_mesh = self.get_parameter('n_obstacle_mesh').get_parameter_value().integer_value
        self.trajectory_execution_dt = self.get_parameter('trajectory_execution_dt').get_parameter_value().double_value
        self.wait_for_completion = self.get_parameter('wait_for_completion').get_parameter_value().bool_value
        
        self.tensor_args = TensorDeviceType(device=self.tensor_device, dtype=torch.float32)
        self.init_curobo()
        
        self._action_client = ActionClient(
            self,
            PublishJoints,
            'publish_joints',
            callback_group=self.callback_group
        )
        
        self.movej_server = ActionServer(
            self, 
            MoveJ, 
            'movej', 
            self.execute_movej,
            callback_group=self.callback_group
        )
        
        self.movel_server = ActionServer(
            self, 
            MoveL, 
            'movel', 
            self.execute_movel,
            callback_group=self.callback_group
        )
        
        self.get_logger().info("Waiting for PublishJoints action server...")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("PublishJoints action server not available after 5 seconds")
        else:
            self.get_logger().info("PublishJoints action server connected")
        
        self.get_logger().info("Curobo Action Server Ready.")
    
    def init_curobo(self):
        world_config = WorldConfig.from_dict(
            load_yaml(self.world_config_file)
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
            joints = len(trajectory[0]) if trajectory else 0
            flattened_trajectory = [item for sublist in trajectory for item in sublist]
            return True, trajectory, joints, str(result)
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
            joints = len(trajectory[0]) if trajectory else 0
            flattened_trajectory = [item for sublist in trajectory for item in sublist]
            return True, trajectory, joints, str(result)
        else:
            return False, [], 0, str(result)
    
    async def execute_trajectory(self, trajectory):
        
        self.get_logger().info(f"Executing trajectory with {len(trajectory)} waypoints")
        
        for i, waypoint in enumerate(trajectory):
            goal_msg = PublishJoints.Goal()
            goal_msg.positions = waypoint
            goal_msg.indices = list(range(len(waypoint)))
            
            self.get_logger().debug(f"Sending waypoint {i+1}/{len(trajectory)}")
            
            send_goal_future = self._action_client.send_goal_async(goal_msg)
            goal_handle = await send_goal_future
            
            if not goal_handle.accepted:
                self.get_logger().error(f"Goal for waypoint {i+1} was rejected")
                return False
            if self.wait_for_completion:
                get_result_future = goal_handle.get_result_async()
                result = await get_result_future
                
                if not result.result.success:
                    self.get_logger().error(f"Failed to execute waypoint {i+1}: {result.result.message}")
                    return False
        self.get_logger().info("Trajectory execution completed successfully")
        return True

    async def execute_movej(self, goal_handle):
        self.get_logger().info("Executing MoveJ action request")
        
        start_state = goal_handle.request.start_state
        goal_state = goal_handle.request.goal_pose
        success, trajectory, joints, result_str = self.plan_motion_js(start_state, goal_state)
        
        if success:
            self.get_logger().info("Motion planning successful, executing trajectory")
            execution_success = await self.execute_trajectory(
                trajectory
            )
            
            if not execution_success:
                success = False
                result_str += "\nTrajectory execution failed"
        else:
            self.get_logger().error("Motion planning failed")
        
        goal_handle.succeed()
        return MoveJ.Result(
            success=success, 
            result=result_str, 
            trajectory=[item for sublist in trajectory for item in sublist], 
            joints=joints
        )

    async def execute_movel(self, goal_handle):
        self.get_logger().info("Executing MoveL action request")

        start_state = goal_handle.request.start_state
        goal_pose = goal_handle.request.goal_pose
        success, trajectory, joints, result_str = self.plan_motion(start_state, goal_pose)
        
        if success:
            self.get_logger().info("Motion planning successful, executing trajectory")
            execution_success = await self.execute_trajectory(
                trajectory
            )
            
            if not execution_success:
                success = False
                result_str += "\nTrajectory execution failed"
        else:
            self.get_logger().error("Motion planning failed")
        
        goal_handle.succeed()
        return MoveL.Result(
            success=success, 
            result=result_str, 
            trajectory=[item for sublist in trajectory for item in sublist], 
            joints=joints
        )

def main(args=None):
    rclpy.init(args=args)
    node = CuroboNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()