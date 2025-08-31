// curobo_backend.h
// curobo backend implementation using Python embedding
#pragma once
#include "motion_planner_backend.h"
#include <Python.h>
#include <yaml-cpp/yaml.h>

// CuroboBackend expects YAML config
class CuroboBackend : public MotionPlannerBackend {
public:
    CuroboBackend() : config_loaded_(false) {
        Py_Initialize();
        PyRun_SimpleString("import sys; sys.path.append('../../src')");
    }
    ~CuroboBackend() override {
        Py_Finalize();
    }
    bool loadConfigString(const std::string& config_yaml, std::string* error_msg) override {
        try {
            config_ = YAML::Load(config_yaml);
            config_loaded_ = true;
            return true;
        } catch (const std::exception& e) {
            config_loaded_ = false;
            if (error_msg) *error_msg = e.what();
            return false;
        }
    }
    bool loadConfig(const std::string& config_path) override {
        try {
            config_ = YAML::LoadFile(config_path);
            config_loaded_ = true;
            return true;
        } catch (const std::exception& e) {
            config_loaded_ = false;
            return false;
        }
    }
    bool planMoveL(const motionplanner::MoveLRequest& request, motionplanner::MotionResponse* response) override {
        if (!config_loaded_) {
            response->set_success(false);
            response->set_message("Config not loaded");
            return false;
        }
        // Example: access config value
        // auto robot_name = config_["robot_name"].as<std::string>("unknown");
        PyObject *pName, *pModule, *pFunc, *pValue;
        pName = PyUnicode_DecodeFSDefault("curobo.examples.kinematics_example");
        pModule = PyImport_Import(pName);
        if (pModule != nullptr) {
            pFunc = PyObject_GetAttrString(pModule, "main");
            if (pFunc && PyCallable_Check(pFunc)) {
                pValue = PyObject_CallObject(pFunc, nullptr);
                response->set_success(true);
                response->set_message("MoveL planned by curobo");
                Py_XDECREF(pValue);
            } else {
                response->set_success(false);
                response->set_message("Failed to call curobo main()");
            }
            Py_XDECREF(pFunc);
            Py_DECREF(pModule);
        } else {
            response->set_success(false);
            response->set_message("Failed to import curobo example");
        }
        Py_XDECREF(pName);
        return response->success();
    }
    bool planMoveJ(const motionplanner::MoveJRequest& request, motionplanner::MotionResponse* response) override {
        return planMoveL(*(reinterpret_cast<const motionplanner::MoveLRequest*>(&request)), response);
    }
private:
    YAML::Node config_;
    bool config_loaded_;
};
