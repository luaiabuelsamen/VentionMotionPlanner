// motion_planner_backend.h
// Abstract interface for modular motion planner backends
// Each backend should handle its own config format (YAML, URDF, etc.)
#pragma once
#include <string>
#include "proto/motion_planner.pb.h"

class MotionPlannerBackend {
public:
    virtual ~MotionPlannerBackend() = default;
    // Load config from YAML string
    virtual bool loadConfigString(const std::string& config_yaml, std::string* error_msg) = 0;
    // Optionally load config file (legacy)
    virtual bool loadConfig(const std::string& config_path) = 0;
    virtual bool planMoveL(const motionplanner::MoveLRequest& request, motionplanner::MotionResponse* response) = 0;
    virtual bool planMoveJ(const motionplanner::MoveJRequest& request, motionplanner::MotionResponse* response) = 0;
};
