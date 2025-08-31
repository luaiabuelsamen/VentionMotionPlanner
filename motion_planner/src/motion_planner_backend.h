// motion_planner_backend.h
// Abstract interface for modular motion planner backends
#pragma once
#include <string>
#include "proto/motion_planner.pb.h"

class MotionPlannerBackend {
public:
    virtual ~MotionPlannerBackend() = default;
    virtual bool planMoveL(const motionplanner::MoveLRequest& request, motionplanner::MotionResponse* response) = 0;
    virtual bool planMoveJ(const motionplanner::MoveJRequest& request, motionplanner::MotionResponse* response) = 0;
};
