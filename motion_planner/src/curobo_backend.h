// curobo_backend.h
// curobo backend implementation using Python embedding
#pragma once
#include "motion_planner_backend.h"
#include <Python.h>

class CuroboBackend : public MotionPlannerBackend {
public:
    CuroboBackend() {
        Py_Initialize();
        PyRun_SimpleString("import sys; sys.path.append('../../src')");
    }
    ~CuroboBackend() override {
        Py_Finalize();
    }
    bool planMoveL(const motionplanner::MoveLRequest& request, motionplanner::MotionResponse* response) override {
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
        // Same as planMoveL for now
        return planMoveL(*(reinterpret_cast<const motionplanner::MoveLRequest*>(&request)), response);
    }
};
