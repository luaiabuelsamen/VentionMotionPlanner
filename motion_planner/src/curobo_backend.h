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
    // Set up Python path for curobo_api (dynamic path)
    PyRun_SimpleString("import sys, os; sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))");
        PyObject *pName = PyUnicode_DecodeFSDefault("curobo_api.curobo");
        PyObject *pModule = PyImport_Import(pName);
        if (pModule != nullptr) {
            PyObject *pFunc = PyObject_GetAttrString(pModule, "plan_motion");
            if (pFunc && PyCallable_Check(pFunc)) {
                // Prepare arguments for plan_motion: start_position, goal_position
                PyObject *pArgs = PyTuple_New(2);
                PyObject *start_joints = PyList_New(request.start_joints().values_size());
                for (int i = 0; i < request.start_joints().values_size(); ++i) {
                    PyList_SetItem(start_joints, i, PyFloat_FromDouble(request.start_joints().values(i)));
                }
                PyObject *goal_pose = PyList_New(7);
                goal_pose = PyList_New(7);
                PyList_SetItem(goal_pose, 0, PyFloat_FromDouble(request.target_pose().x()));
                PyList_SetItem(goal_pose, 1, PyFloat_FromDouble(request.target_pose().y()));
                PyList_SetItem(goal_pose, 2, PyFloat_FromDouble(request.target_pose().z()));
                PyList_SetItem(goal_pose, 3, PyFloat_FromDouble(request.target_pose().qx()));
                PyList_SetItem(goal_pose, 4, PyFloat_FromDouble(request.target_pose().qy()));
                PyList_SetItem(goal_pose, 5, PyFloat_FromDouble(request.target_pose().qz()));
                PyList_SetItem(goal_pose, 6, PyFloat_FromDouble(request.target_pose().qw()));
                PyTuple_SetItem(pArgs, 0, start_joints);
                PyTuple_SetItem(pArgs, 1, goal_pose);
                PyObject *pResult = PyObject_CallObject(pFunc, pArgs);
                if (pResult) {
                    // Expecting tuple: (success, trajectory, joints, result_str)
                    int success = PyObject_IsTrue(PyTuple_GetItem(pResult, 0));
                    response->set_success(success);
                    response->set_message(PyUnicode_AsUTF8(PyTuple_GetItem(pResult, 3)));
                    if (success) {
                        PyObject *traj = PyTuple_GetItem(pResult, 1);
                        for (Py_ssize_t i = 0; i < PyList_Size(traj); ++i) {
                            PyObject *jv = PyList_GetItem(traj, i);
                            motionplanner::JointValues *jv_msg = response->add_trajectory();
                            for (Py_ssize_t j = 0; j < PyList_Size(jv); ++j) {
                                jv_msg->add_values(PyFloat_AsDouble(PyList_GetItem(jv, j)));
                            }
                        }
                    }
                    Py_DECREF(pResult);
                } else {
                    response->set_success(false);
                    response->set_message("Python curobo_api.curobo.plan_motion failed");
                }
                Py_DECREF(pArgs);
            } else {
                response->set_success(false);
                response->set_message("Failed to find curobo_api.curobo.plan_motion");
            }
            Py_XDECREF(pFunc);
            Py_DECREF(pModule);
        } else {
            response->set_success(false);
            response->set_message("Failed to import curobo_api.curobo");
        }
        Py_XDECREF(pName);
        return response->success();
    }
    bool planMoveJ(const motionplanner::MoveJRequest& request, motionplanner::MotionResponse* response) override {
        if (!config_loaded_) {
            response->set_success(false);
            response->set_message("Config not loaded");
            return false;
        }
    PyRun_SimpleString("import sys, os; sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))");
        PyObject *pName = PyUnicode_DecodeFSDefault("curobo_api.curobo");
        PyObject *pModule = PyImport_Import(pName);
        if (pModule != nullptr) {
            PyObject *pFunc = PyObject_GetAttrString(pModule, "plan_motion_js");
            if (pFunc && PyCallable_Check(pFunc)) {
                // Prepare arguments for plan_motion_js: start_state, goal_state
                PyObject *pArgs = PyTuple_New(2);
                PyObject *start_joints = PyList_New(request.start_joints().values_size());
                for (int i = 0; i < request.start_joints().values_size(); ++i) {
                    PyList_SetItem(start_joints, i, PyFloat_FromDouble(request.start_joints().values(i)));
                }
                PyObject *goal_joints = PyList_New(request.target_joints().values_size());
                for (int i = 0; i < request.target_joints().values_size(); ++i) {
                    PyList_SetItem(goal_joints, i, PyFloat_FromDouble(request.target_joints().values(i)));
                }
                PyTuple_SetItem(pArgs, 0, start_joints);
                PyTuple_SetItem(pArgs, 1, goal_joints);
                PyObject *pResult = PyObject_CallObject(pFunc, pArgs);
                if (pResult) {
                    int success = PyObject_IsTrue(PyTuple_GetItem(pResult, 0));
                    response->set_success(success);
                    response->set_message(PyUnicode_AsUTF8(PyTuple_GetItem(pResult, 3)));
                    if (success) {
                        PyObject *traj = PyTuple_GetItem(pResult, 1);
                        for (Py_ssize_t i = 0; i < PyList_Size(traj); ++i) {
                            PyObject *jv = PyList_GetItem(traj, i);
                            motionplanner::JointValues *jv_msg = response->add_trajectory();
                            for (Py_ssize_t j = 0; j < PyList_Size(jv); ++j) {
                                jv_msg->add_values(PyFloat_AsDouble(PyList_GetItem(jv, j)));
                            }
                        }
                    }
                    Py_DECREF(pResult);
                } else {
                    response->set_success(false);
                    response->set_message("Python curobo_api.curobo.plan_motion_js failed");
                }
                Py_DECREF(pArgs);
            } else {
                response->set_success(false);
                response->set_message("Failed to find curobo_api.curobo.plan_motion_js");
            }
            Py_XDECREF(pFunc);
            Py_DECREF(pModule);
        } else {
            response->set_success(false);
            response->set_message("Failed to import curobo_api.curobo");
        }
        Py_XDECREF(pName);
        return response->success();
    }
private:
    YAML::Node config_;
    bool config_loaded_;
};
