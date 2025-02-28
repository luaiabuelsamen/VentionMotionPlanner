import rclpy
from rclpy.node import Node
import torch
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType

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


def main(args=None):
    rclpy.init(args=args)
    node = CuroboNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()