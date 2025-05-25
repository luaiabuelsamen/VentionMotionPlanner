import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState as RosJointState
from std_msgs.msg import Header

import torch
import numpy as np
import time

from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.rollout.rollout_base import Goal
from curobo.geom.types import WorldConfig
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.types.math import Pose
from curobo_action.action import MoveJ, MoveL, PublishJoints
from curobo.geom.types import Mesh
from std_msgs.msg import String
import json
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
            self.joint_state_publisher = self.create_publisher(
                RosJointState,
                'joint_states',
                10
            )
            self.mujoco = False    
            self.latest_joint_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            self.get_logger().info("PublishJoints action server connected")
            self.joint_state_publisher = None
            self.mujoco = True
                
            self.latest_joint_state = None
        
        self.get_logger().info("Curobo Action Server Ready.")

        self.joint_states_subscription = self.create_subscription(
            RosJointState,
            'joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )
    
    def joint_state_callback(self, msg):
        self.latest_joint_state = list(msg.position)

    def init_curobo(self):
        self.world_config_initial = WorldConfig.from_dict(
            load_yaml(self.world_config_file)
        )
        
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config_file,
            interpolation_dt=0.01,
            world_model=self.world_config_initial,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_max_outside_distance = 0.0001,
            collision_activation_distance = 0.0001,
            collision_cache={"obb": self.n_obstacle_cuboids, "mesh": self.n_obstacle_mesh},
        )
        mpc_config = MpcSolverConfig.load_from_robot_config(
            self.robot_config_file,
            self.world_config_initial,
            store_rollouts=True,
            step_dt=0.03,
        )

        self.mpc = MpcSolver(mpc_config)
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        self.subscription = self.create_subscription(String, 'world_state_json', self.world_callback, 10)
        self.get_logger().info("Curobo MotionGen initialized and warmed up.")
    
    def world_callback(self, msg):
        mujoco_dict = json.loads(msg.data)
        #self.get_logger().info(str(mujoco_dict))
        meshes = {}
        if 'mesh' in mujoco_dict:
            meshes = mujoco_dict['mesh']
            del mujoco_dict['mesh']
        new_world = WorldConfig.from_dict(mujoco_dict)
        for mesh in meshes:
            cur_mesh = Mesh(file_path=meshes[mesh]['file_path'], name=mesh, pose=meshes[mesh]['pose'])
            cur_mesh.file_path = meshes[mesh]['file_path']
            new_world.add_obstacle(cur_mesh)
        new_world.add_obstacle(self.world_config_initial.cuboid[0])
        self.motion_gen.update_world(new_world)
        self.mpc.update_world(new_world)
    
    def plan_motion_js(self, start_state, goal_state, constrained = False, constrained_input = [0, 0, 0, 0, 0, 0]):
        current_state = JointState.from_position(torch.tensor([start_state], device="cuda:0", dtype=torch.float32))
        goal_state = JointState.from_position(torch.tensor([goal_state], device="cuda:0", dtype=torch.float32))
        if constrained:
            #trying constrained motion (LU)
            pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.tensor_args.to_device(constrained_input), # this should set cost on EE
            )
            result = self.motion_gen.plan_single_js(current_state, goal_state, MotionGenPlanConfig(max_attempts=10000, pose_cost_metric= pose_cost_metric))
        else:
            result = self.motion_gen.plan_single_js(current_state, goal_state, MotionGenPlanConfig(max_attempts=10000))
        if result.success.item():
            trajectory = result.get_interpolated_plan().position.tolist()
            joints = len(trajectory[0]) if trajectory else 0
            flattened_trajectory = [item for sublist in trajectory for item in sublist]
            return True, trajectory, joints, str(result)
        else:
            return False, [], 0, str(result)

    async def plan_motion(self, start_position, goal_position, mpc = False, constrained = False, constrained_input = [0 , 0, 0, 0, 0, 0]):
        goal_pose = Pose(
            position=self.tensor_args.to_device([goal_position[:3]]),
            quaternion=self.tensor_args.to_device([goal_position[3:]])  
        )
        start_state = JointState.from_position(
            torch.tensor([start_position], device="cuda:0", dtype=torch.float32))

        if mpc:

            goal = Goal(
                current_state=start_state,
                goal_pose=goal_pose
            )
            goal_buffer = self.mpc.setup_solve_single(goal, 1)
            self.mpc.update_goal(goal_buffer)
            converged = False
            tstep = 0
            traj_list = []
            mpc_time = []
            
            while not converged:
                start_state = JointState.from_position(
                    torch.tensor([self.latest_joint_state], device="cuda:0", dtype=torch.float32))
                st_time = time.time()
                result = self.mpc.step(start_state, 1)
                torch.cuda.synchronize()
                if tstep > 5:
                    mpc_time.append(time.time() - st_time)
                traj_list.append(result.action.get_state_tensor())
                tstep += 1
                if result.metrics.pose_error.item() < 0.05:
                    converged = True
                if tstep > 300:
                    break
                execution_success = await self.execute_trajectory(
                   result.action.position.cpu().numpy()
                )
            return True, [], 0, str(result)
        # need later modified with mpc. cause now if MPC is True it will get into MPC first.
        if constrained:
            #trying constrained motion
            pose_cost = PoseCostMetric(
                hold_partial_pose= True,
                #hold_vec_weight=self.tensor_args.to_device([0, 0, 0, 0, 1, 0]), # this should constrained y movement
                #hold_vec_weight=self.tensor_args.to_device([0, 0, 0, 1, 1, 0]), # this should constrained x and y movement

                #hold_vec_weight=self.tensor_args.to_device([1, 1, 1, 1, 1, 0]), # this should set cost on EE

                hold_vec_weight=self.tensor_args.to_device(constrained_input), # this should set cost on EE
            )
            self.get_logger().info(f"I am using Constrained")
            result = self.motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=10000, pose_cost_metric= pose_cost))
        else:
            self.get_logger().info(f"I am using non Constrained")
            result = self.motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=10000))

        if result.success.item():
            trajectory = result.get_interpolated_plan().position.tolist()
            joints = len(trajectory[0]) if trajectory else 0
            flattened_trajectory = [item for sublist in trajectory for item in sublist]
            return True, trajectory, joints, str(result)
        else:
            return False, [], 0, str(result)
    
    async def execute_trajectory(self, trajectory):
        for i, waypoint in enumerate(trajectory):
            if self.mujoco:
                goal_msg = PublishJoints.Goal()
                goal_msg.positions = [float(x) for x in waypoint]
                goal_msg.indices = list(range(len(waypoint)))
                
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
            else:
                self.publish_joint_states(waypoint)
                time.sleep(0.01)
        return True

    async def execute_movej(self, goal_handle):
        self.get_logger().info("Executing MoveJ action request")
        if not goal_handle.request.start_state:
            if self.latest_joint_state is None:
                goal_handle.succeed()
                return MoveJ.Result(
                    success=False,
                    result="No joint state received yet. Cannot use current state.",
                    trajectory=[],
                    joints=0
                )
            start_state = self.latest_joint_state
        else:
            start_state = goal_handle.request.start_state
        
        goal_state = goal_handle.request.goal_pose
        constrained = goal_handle.request.constrained
        constrained_input = goal_handle.request.constrained_input
        success, trajectory, joints, result_str = self.plan_motion_js(start_state, goal_state, constrained=constrained, constrained_input=constrained_input)
        
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

        if not goal_handle.request.start_state:
            if self.latest_joint_state is None:
                goal_handle.succeed()
                return MoveL.Result(
                    success=False,
                    result="No joint state received yet. Cannot use current state.",
                    trajectory=[],
                    joints=0
                )
            start_state = self.latest_joint_state
        else:
            start_state = goal_handle.request.start_state
        
        goal_pose = goal_handle.request.goal_pose
        mpc = goal_handle.request.mpc
        constrained = goal_handle.request.constrained
        constrained_input = goal_handle.request.constrained_input
        success, trajectory, joints, result_str = await self.plan_motion(start_state, goal_pose, mpc = mpc, constrained = constrained, constrained_input=constrained_input)
        
        if success and not mpc:
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

    def publish_joint_states(self, positions, velocities = []):

        joint_state_msg = RosJointState()
        joint_state_msg.header = Header()
        names = [
            "base_x",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = names

        joint_state_msg.position = positions
        joint_state_msg.velocity = velocities
        self.joint_state_publisher.publish(joint_state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CuroboNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()