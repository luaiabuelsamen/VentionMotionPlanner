source install/setup.bash
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.5, 0, 0.2,0 , 0.0, -1.0, 0.0]}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.5, 0, 0.1,0 , 0.0, -1.0, 0.0]}"
ros2 action send_goal /publish_joints curobo_action/action/PublishJoints "{positions: [1.0], indices: [6]}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.5, 0, 0.5,0 , 0.0, -1.0, 0.0]}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.5, 0, 0.2,0 , 0.0, -1.0, 0.0]}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.5, 0, 0.1,0 , 0.0, -1.0, 0.0]}"
ros2 action send_goal /publish_joints curobo_action/action/PublishJoints "{positions: [0.0], indices: [6]}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.5, 0, 0.2,0 , 0.0, -1.0, 0.0]}"