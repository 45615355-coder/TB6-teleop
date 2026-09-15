#include "command_ik.hpp"
#include "tb6_ik_solver.hpp"
#include "shared_data.hpp"
#include "system_state_reader.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

namespace
{

std::atomic<int> g_movejp_var_counter{0};

std::vector<double> toJointTarget10(const std::vector<double>& arm6)
{
    std::vector<double> q(10, 0.0);
    for (size_t i = 0; i < arm6.size() && i < 6; ++i) {
        q[i] = arm6[i];
    }
    return q;
}

}  // namespace

Tb6CommandIk::Tb6CommandIk(
    Command& cmd,
    std::string zmq_host,
    int model_idx,
    int joint_index_offset)
    : cmd_(cmd)
    , zmq_host_(std::move(zmq_host))
    , model_idx_(model_idx)
    , joint_index_offset_(joint_index_offset)
{
}

void Tb6CommandIk::ensureTelemetry()
{
    if (subscriber_started_) {
        return;
    }
    start_subscriber(zmq_host_);
    subscriber_started_ = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

bool Tb6CommandIk::resolveIkSeed(std::vector<double>& seed_out, std::string* error_msg)
{
    seed_out.assign(6, 0.0);

    ensureTelemetry();
    const auto rt = SystemStateReader::snapshotRt();
    if (!rt.valid() || static_cast<size_t>(model_idx_) >= rt.modelCount()) {
        if (error_msg) {
            *error_msg =
                "IK seed unavailable: ensure ZMQ state topic from " + zmq_host_;
        }
        return false;
    }

    const size_t model = static_cast<size_t>(model_idx_);
    for (size_t j = 0; j < 6; ++j) {
        seed_out[j] = rt.jointPosition(model, static_cast<size_t>(joint_index_offset_) + j);
    }
    return true;
}

bool Tb6CommandIk::solveIkPe321(
    const std::vector<double>& pe321,
    std::vector<double>& joints_out,
    std::string* error_msg)
{
    joints_out.clear();
    if (pe321.size() < 6) {
        if (error_msg) {
            *error_msg = "pe321 must have 6 elements [x,y,z,rz,ry,rx]";
        }
        return false;
    }

    std::vector<double> seed;
    if (!resolveIkSeed(seed, error_msg)) {
        return false;
    }

    tb6_ik::Pe321 pe{};
    for (size_t i = 0; i < 6; ++i) {
        pe[i] = pe321[i];
    }

    const auto ik = tb6_ik::solveIkPe321(pe, seed);
    if (!ik.success) {
        if (error_msg) {
            *error_msg = ik.message;
        }
        return false;
    }

    joints_out = ik.joints_rad;
    if (error_msg) {
        error_msg->clear();
    }
    return true;
}

bool Tb6CommandIk::MovejP(const std::vector<double>& pe321, int speed)
{
    std::cout << "====== Sending MovejP command (KDL IK + MoveAbsj) ======" << std::endl;

    std::vector<double> joints_rad;
    std::string ik_error;
    if (!solveIkPe321(pe321, joints_rad, &ik_error)) {
        std::cerr << "MovejP IK failed: " << ik_error << std::endl;
        return false;
    }

    const std::string var_name = "movejp_" + std::to_string(g_movejp_var_counter.fetch_add(1));
    cmd_.DeleteVar(var_name);
    if (!cmd_.setJointTargetVar(var_name, toJointTarget10(joints_rad))) {
        std::cerr << "MovejP setJointTargetVar failed" << std::endl;
        return false;
    }
    if (!cmd_.MoveAbsj(var_name, speed)) {
        std::cerr << "MovejP MoveAbsj failed" << std::endl;
        return false;
    }

    std::cout << "MovejP completed" << std::endl;
    return true;
}
