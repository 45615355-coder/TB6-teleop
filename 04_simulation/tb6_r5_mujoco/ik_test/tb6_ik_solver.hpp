#pragma once

#include <array>
#include <string>
#include <vector>

namespace tb6_ik
{

constexpr size_t kArmDof = 6;

/** PE321 teach pendant / SDK ee_pe321: [x,y,z, rz,ry,rx] in m and rad (ZYX). */
using Pe321 = std::array<double, kArmDof>;

struct IkSolveResult
{
  bool success{false};
  std::vector<double> joints_rad;
  std::string message;
  size_t raw_solution_count{0};
};

/** KDL numerical IK ; picks solution closest to seed joints. */
IkSolveResult solveIkPe321(const Pe321& pe321, const std::vector<double>& seed_joints_rad);

/** Unwrap to seed ± π for shortest path, then clamp to limits (e.g. >π → π). */
void unwrapAndClampJointsNearSeed(
  std::vector<double>& joints_rad, const std::vector<double>& seed_joints_rad);

}  // namespace tb6_ik
