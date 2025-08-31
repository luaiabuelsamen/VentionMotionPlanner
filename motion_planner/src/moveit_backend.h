// moveit_backend.h
// MoveIt backend stub for demonstration
#pragma once
#include "motion_planner_backend.h"

class MoveItBackend : public MotionPlannerBackend {
public:
    MoveItBackend() {}
    ~MoveItBackend() override {}
    bool planMoveL(const motionplanner::MoveLRequest& request, motionplanner::MotionResponse* response) override {
        response->set_success(true);
        response->set_message("MoveL planned by MoveIt (stub)");
        return true;
    }
    bool planMoveJ(const motionplanner::MoveJRequest& request, motionplanner::MotionResponse* response) override {
        response->set_success(true);
        response->set_message("MoveJ planned by MoveIt (stub)");
        return true;
    }
};