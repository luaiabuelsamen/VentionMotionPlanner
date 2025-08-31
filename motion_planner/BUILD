load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_proto_library", "cc_library")

proto_library(
    name = "motion_planner_proto",
    srcs = ["proto/motion_planner.proto"],
)

cc_proto_library(
    name = "motion_planner_cc_proto",
    deps = [":motion_planner_proto"],
)

# Generate gRPC C++ files using the correct protoc command
genrule(
    name = "motion_planner_grpc_codegen",
    srcs = ["proto/motion_planner.proto"],
    outs = [
        "proto/motion_planner.grpc.pb.cc",
        "proto/motion_planner.grpc.pb.h",
    ],
    cmd = "$(location @protobuf//:protoc) --cpp_out=$(@D) --grpc_out=$(@D) --plugin=protoc-gen-grpc=$(location @grpc//src/compiler:grpc_cpp_plugin) $(location proto/motion_planner.proto)",
    tools = [
        "@protobuf//:protoc",
        "@grpc//src/compiler:grpc_cpp_plugin",
    ],
)

cc_library(
    name = "motion_planner_grpc_lib",
    srcs = ["proto/motion_planner.grpc.pb.cc"],
    hdrs = ["proto/motion_planner.grpc.pb.h"],
    deps = [
        ":motion_planner_cc_proto",
        "@grpc//:grpc++",
    ],
    strip_include_prefix = "",
)

cc_binary(
    name = "motion_planner_server",
    srcs = ["src/motion_planner_server.cpp"],
    deps = [
        ":motion_planner_cc_proto",
        ":motion_planner_grpc_lib",
        "@grpc//:grpc++",
    ],
    linkopts = ["-lpython3.8"],  # Adjust if using a different Python version
    copts = ["-I/usr/include/python3.8"],  # Adjust if using a different Python version
)
