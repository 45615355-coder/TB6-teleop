#include "tb6_ik_solver.hpp"

#include <kdl/chain.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <kdl_parser/kdl_parser.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace tb6_ik
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

constexpr double kJointLimits[6][2] = {
  {-kPi, kPi},
  {-kPi, kPi},
  {-2.8623, 2.8623},
  {-kPi, kPi},
  {-kPi, kPi},
  {-kPi, kPi},
};

constexpr int kIkMaxAttempts = 5;

struct KdlChainContext
{
  bool ok{false};
  std::string message;
  KDL::Chain chain;
};

std::string fileExists(const std::string& path)
{
  std::ifstream file(path);
  return file.good() ? path : std::string{};
}

std::string resolveUrdfPath()
{
  if (const char* env = std::getenv("TB6_IK_URDF")) {
    if (const auto p = fileExists(env); !p.empty()) {
      return p;
    }
  }

#ifdef TB6_IK_URDF_DEFAULT
  if (const auto p = fileExists(TB6_IK_URDF_DEFAULT); !p.empty()) {
    return p;
  }
#endif

  const std::vector<std::string> fallbacks = {
    "src/tb6_description/urdf/tb6_r5.urdf",
    "../tb6_description/urdf/tb6_r5.urdf",
    "../../tb6_description/urdf/tb6_r5.urdf",
    "src/tb6_arm_control/urdf/tb6_r5.urdf",
    "../tb6_arm_control/urdf/tb6_r5.urdf",
  };
  for (const auto& candidate : fallbacks) {
    if (const auto p = fileExists(candidate); !p.empty()) {
      return p;
    }
  }

  return {};
}

KdlChainContext loadChain()
{
  KdlChainContext ctx;
  const std::string urdf_path = resolveUrdfPath();
  if (urdf_path.empty()) {
    ctx.message =
      "URDF not found; set TB6_IK_URDF or build with TB6_IK_URDF_DEFAULT";
    return ctx;
  }

  KDL::Tree tree;
  if (!kdl_parser::treeFromFile(urdf_path, tree)) {
    ctx.message = "kdl_parser::treeFromFile failed for " + urdf_path;
    return ctx;
  }

  if (!tree.getChain("base_link", "ee_Link", ctx.chain)) {
    ctx.message = "KDL getChain(base_link, ee_Link) failed for " + urdf_path;
    return ctx;
  }

  if (ctx.chain.getNrOfJoints() < static_cast<unsigned int>(kArmDof)) {
    ctx.message = "Expected at least 6 joints in chain, got " +
      std::to_string(ctx.chain.getNrOfJoints());
    return ctx;
  }

  ctx.ok = true;
  ctx.message = "loaded " + urdf_path;
  return ctx;
}

const KdlChainContext& chainContext()
{
  static const KdlChainContext ctx = loadChain();
  return ctx;
}

double wrapToPi(double a)
{
  while (a > kPi) {
    a -= 2.0 * kPi;
  }
  while (a < -kPi) {
    a += 2.0 * kPi;
  }
  return a;
}

double jointDelta(double a, double b)
{
  return wrapToPi(a - b);
}

KDL::Frame pe321ToFrame(const Pe321& pe321)
{
  KDL::Frame frame;
  frame.p = KDL::Vector(pe321[0], pe321[1], pe321[2]);
  frame.M = KDL::Rotation::RPY(pe321[5], pe321[4], pe321[3]);
  return frame;
}

bool withinJointLimits(const std::vector<double>& q)
{
  if (q.size() < kArmDof) {
    return false;
  }
  for (size_t i = 0; i < kArmDof; ++i) {
    if (q[i] < kJointLimits[i][0] || q[i] > kJointLimits[i][1]) {
      return false;
    }
  }
  return true;
}

double jointDistanceCost(const std::vector<double>& q, const std::vector<double>& seed)
{
  double cost = 0.0;
  for (size_t i = 0; i < kArmDof; ++i) {
    const double d = jointDelta(q[i], seed[i]);
    cost += d * d;
  }
  return cost;
}

std::vector<double> jntArrayToVector(const KDL::JntArray& joints)
{
  std::vector<double> q(kArmDof, 0.0);
  const unsigned int n = std::min(
    static_cast<unsigned int>(kArmDof), joints.rows());
  for (unsigned int i = 0; i < n; ++i) {
    q[i] = joints(i);
  }
  return q;
}

void clampJointsToLimits(std::vector<double>& q)
{
  for (size_t i = 0; i < kArmDof && i < q.size(); ++i) {
    if (q[i] > kJointLimits[i][1]) {
      q[i] = kJointLimits[i][1];
    } else if (q[i] < kJointLimits[i][0]) {
      q[i] = kJointLimits[i][0];
    }
  }
}

void unwrapAndClampJointsNearSeedImpl(
  std::vector<double>& joints_rad, const std::vector<double>& seed_joints_rad)
{
  for (size_t i = 0; i < kArmDof && i < joints_rad.size() && i < seed_joints_rad.size(); ++i) {
    joints_rad[i] =
      seed_joints_rad[i] + jointDelta(joints_rad[i], seed_joints_rad[i]);
  }
  clampJointsToLimits(joints_rad);
}

}  // namespace

void unwrapAndClampJointsNearSeed(
  std::vector<double>& joints_rad, const std::vector<double>& seed_joints_rad)
{
  unwrapAndClampJointsNearSeedImpl(joints_rad, seed_joints_rad);
}

IkSolveResult solveIkPe321(const Pe321& pe321, const std::vector<double>& seed_joints_rad)
{
  IkSolveResult result;

  if (seed_joints_rad.size() < kArmDof) {
    result.message = "seed_joints_rad must have at least 6 elements";
    return result;
  }

  const KdlChainContext& ctx = chainContext();
  if (!ctx.ok) {
    result.message = ctx.message;
    return result;
  }

  const KDL::Frame goal = pe321ToFrame(pe321);
  KDL::ChainIkSolverPos_LMA ik_solver(ctx.chain, 1e-5, 500);

  KDL::JntArray q_seed(static_cast<unsigned int>(kArmDof));
  for (size_t i = 0; i < kArmDof; ++i) {
    q_seed(static_cast<unsigned int>(i)) = seed_joints_rad[i];
  }

  double best_cost = std::numeric_limits<double>::infinity();
  bool found = false;

  for (int attempt = 0; attempt < kIkMaxAttempts; ++attempt) {
    KDL::JntArray q_try = q_seed;
    if (attempt > 0) {
      const double scale = 0.02 * static_cast<double>(attempt);
      for (unsigned int i = 0; i < q_try.rows(); ++i) {
        q_try(i) = wrapToPi(q_seed(i) + scale * std::sin(0.7 * attempt + i));
      }
    }

    KDL::JntArray q_out(static_cast<unsigned int>(kArmDof));
    const int rc = ik_solver.CartToJnt(q_try, goal, q_out);
    if (rc < 0) {
      continue;
    }

    ++result.raw_solution_count;
    const std::vector<double> q = jntArrayToVector(q_out);
    if (!withinJointLimits(q)) {
      continue;
    }

    const double cost = jointDistanceCost(q, seed_joints_rad);
    if (cost < best_cost) {
      best_cost = cost;
      result.joints_rad = q;
      found = true;
    }
  }

  if (!found) {
    if (result.raw_solution_count == 0) {
      result.message = "KDL IK failed (no converged solution)";
    } else {
      result.message = "KDL IK solutions out of joint limits";
    }
    return result;
  }

  // Unwrap to seed ± π, then clamp to joint limits (>|π| → ±π).
  unwrapAndClampJointsNearSeed(result.joints_rad, seed_joints_rad);

  result.success = true;
  std::ostringstream ss;
  ss << "KDL selected 1/" << result.raw_solution_count
     << " attempts, cost=" << best_cost << " (" << ctx.message << ")";
  result.message = ss.str();
  return result;
}

}  // namespace tb6_ik
