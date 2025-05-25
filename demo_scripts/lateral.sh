source install/setup.bash
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.55, -0.3, 0.5, 0.5, -0.5, 0.5, 0.5], mpc: false, constrained: false}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.55, 0.4, 0.5, 0.5, -0.5, 0.5, 0.5], mpc: false, constrained: false}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.55, -0.3, 0.5, 0.5, -0.5, 0.5, 0.5], mpc: false, constrained: false}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.55, 0.4, 0.5, 0.5, -0.5, 0.5, 0.5], mpc: false, constrained: false}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.55, -0.3, 0.5, 0.5, -0.5, 0.5, 0.5], mpc: false, constrained: true}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.55, 0.4, 0.5, 0.5, -0.5, 0.5, 0.5], mpc: false, constrained: true}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.55, -0.3, 0.5, 0.5, -0.5, 0.5, 0.5], mpc: false, constrained: true}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.55, 0.4, 0.5, 0.5, -0.5, 0.5, 0.5], mpc: false, constrained: true}"

# the EE facing up
#ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0, -0.5, 0.2, 1.0, 0.0, 0.0, 0.0], mpc: false, constrained: false}"
#ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0], mpc: false, constrained: false}"
