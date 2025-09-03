import grpc
import time
import proto.motion_planner_pb2 as mpb2
import proto.motion_planner_pb2_grpc as mpb2_grpc

def main():
    channel = grpc.insecure_channel('localhost:50051')
    stub = mpb2_grpc.MotionPlannerStub(channel)

    # Example: SetConfig (YAML for curobo, URDF for moveit)
    config_yaml = """
robot_name: test_robot
"""
    set_config_resp = stub.SetConfig(mpb2.SetConfigRequest(config_yaml=config_yaml))
    print("SetConfig response:", set_config_resp)

    # Example: MoveL
    movel_req = mpb2.MoveLRequest(
        target_pose=mpb2.Pose(x=0.1, y=0.2, z=0.3, qx=0, qy=0, qz=0, qw=1),
        start_joints=mpb2.JointValues(values=[0, 0, 0, 0, 0, 0])
    )
    movel_resp = stub.MoveL(movel_req)
    print("MoveL response:", movel_resp)

    # Example: MoveJ
    movej_req = mpb2.MoveJRequest(
        target_joints=mpb2.JointValues(values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        start_joints=mpb2.JointValues(values=[0, 0, 0, 0, 0, 0])
    )
    movej_resp = stub.MoveJ(movej_req)
    print("MoveJ response:", movej_resp)

if __name__ == "__main__":
    main()