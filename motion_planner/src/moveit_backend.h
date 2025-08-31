// moveit_backend.h
// MoveIt backend stub for demonstration
#pragma once
#include "motion_planner_backend.h"

class MoveItBackend : public MotionPlannerBackend {
public:
    MoveItBackend() : config_loaded_(false) {}
    ~MoveItBackend() override {}
    // MoveIt expects URDF as config string, not YAML
    bool loadConfigString(const std::string& config_urdf, std::string* error_msg) override {
        // In a real implementation, parse and validate URDF here
        config_urdf_ = config_urdf;
        config_loaded_ = true;
        return true;
    }
    bool loadConfig(const std::string& config_path) override {
        // Example: just mark as loaded
        config_path_ = config_path;
        config_loaded_ = true;
        return true;
    }
    bool planMoveL(const motionplanner::MoveLRequest& request, motionplanner::MotionResponse* response) override {
        if (!config_loaded_) {
            response->set_success(false);
            response->set_message("Config not loaded");
            return false;
        }
        response->set_success(true);
        response->set_message("MoveL planned by MoveIt (stub)");
        return true;
    }
    bool planMoveJ(const motionplanner::MoveJRequest& request, motionplanner::MotionResponse* response) override {
        if (!config_loaded_) {
            response->set_success(false);
            response->set_message("Config not loaded");
            return false;
        }
        response->set_success(true);
        response->set_message("MoveJ planned by MoveIt (stub)");
        return true;
    }
private:
    std::string config_path_;
    std::string config_urdf_; // MoveIt expects URDF
    bool config_loaded_;
};