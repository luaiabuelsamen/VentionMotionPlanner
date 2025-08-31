// motion_planner_server.cpp
// Async gRPC server skeleton for curobo-based motion planning

#include "proto/motion_planner.pb.h"
#include "proto/motion_planner.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/alarm.h>
#include <memory>
#include <thread>
#include <iostream>
#include "motion_planner_backend.h"
#include "curobo_backend.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerAsyncResponseWriter;
using grpc::CompletionQueue;
using grpc::ServerCompletionQueue;
using grpc::Status;
using motionplanner::MotionPlanner;
using motionplanner::MoveLRequest;
using motionplanner::MoveJRequest;
using motionplanner::MotionResponse;

class MotionPlannerServiceImpl final {
public:
    MotionPlannerServiceImpl(MotionPlannerBackend* backend) : backend_(backend) {}
    ~MotionPlannerServiceImpl() {
        server_->Shutdown();
        cq_->Shutdown();
    }

    void Run(const std::string& server_address) {
        ServerBuilder builder;
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service_);
        cq_ = builder.AddCompletionQueue();
        server_ = builder.BuildAndStart();
        std::cout << "Server listening on " << server_address << std::endl;
        HandleRpcs();
    }
public:
    ~MotionPlannerServiceImpl() { 
        server_->Shutdown(); 
        cq_->Shutdown(); 
    }

    void Run(const std::string& server_address) {
        ServerBuilder builder;
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service_);
        cq_ = builder.AddCompletionQueue();
        server_ = builder.BuildAndStart();
        std::cout << "Server listening on " << server_address << std::endl;
    Py_Initialize();
        HandleRpcs();
    Py_Finalize();
    }

private:
    class CallDataBase {
    public:
        virtual void Proceed(bool ok) = 0;
        virtual ~CallDataBase() = default;
    };

    class MoveLCallData : public CallDataBase {
    public:
        MoveLCallData(MotionPlanner::AsyncService* service, ServerCompletionQueue* cq, MotionPlannerBackend* backend)
            : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE), backend_(backend) {
            Proceed(true);
        }
        void Proceed(bool ok) override {
            if (status_ == CREATE) {
                status_ = PROCESS;
                service_->RequestMoveL(&ctx_, &request_, &responder_, cq_, cq_, this);
            } else if (status_ == PROCESS) {
                new MoveLCallData(service_, cq_, backend_); // Spawn new handler
                backend_->planMoveL(request_, &response_);
                status_ = FINISH;
                responder_.Finish(response_, Status::OK, this);
            } else {
                delete this;
            }
        }
    private:
        MotionPlanner::AsyncService* service_;
        ServerCompletionQueue* cq_;
        ServerContext ctx_;
        MoveLRequest request_;
        MotionResponse response_;
        ServerAsyncResponseWriter<MotionResponse> responder_;
        enum CallStatus { CREATE, PROCESS, FINISH } status_;
        MotionPlannerBackend* backend_;
    };

    class MoveJCallData : public CallDataBase {
    public:
        MoveJCallData(MotionPlanner::AsyncService* service, ServerCompletionQueue* cq, MotionPlannerBackend* backend)
            : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE), backend_(backend) {
            Proceed(true);
        }
        void Proceed(bool ok) override {
            if (status_ == CREATE) {
                status_ = PROCESS;
                service_->RequestMoveJ(&ctx_, &request_, &responder_, cq_, cq_, this);
            } else if (status_ == PROCESS) {
                new MoveJCallData(service_, cq_, backend_); // Spawn new handler
                backend_->planMoveJ(request_, &response_);
                status_ = FINISH;
                responder_.Finish(response_, Status::OK, this);
            } else {
                delete this;
            }
        }
    private:
        MotionPlanner::AsyncService* service_;
        ServerCompletionQueue* cq_;
        ServerContext ctx_;
        MoveJRequest request_;
        MotionResponse response_;
        ServerAsyncResponseWriter<MotionResponse> responder_;
        enum CallStatus { CREATE, PROCESS, FINISH } status_;
        MotionPlannerBackend* backend_;
    };

    void HandleRpcs() {
        new MoveLCallData(&service_, cq_.get(), backend_);
        new MoveJCallData(&service_, cq_.get(), backend_);
        void* tag;
        bool ok;
        while (cq_->Next(&tag, &ok)) {
            static_cast<CallDataBase*>(tag)->Proceed(ok);
        }
    }

    std::unique_ptr<ServerCompletionQueue> cq_;
    MotionPlanner::AsyncService service_;
    std::unique_ptr<Server> server_;
    MotionPlannerBackend* backend_;
};

int main(int argc, char** argv) {
    std::string server_address("0.0.0.0:50051");
    // Choose backend here (can be made configurable)
    CuroboBackend curobo_backend;
    MotionPlannerServiceImpl server(&curobo_backend);
    server.Run(server_address);
    return 0;
}
