#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <kdl_parser/kdl_parser.hpp>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{

constexpr double kPi = 3.14159265358979323846;

double degToRad(double value)
{
    return value * kPi / 180.0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 7) {
        std::cerr
            << "Usage: fk_kdl_test "
            << "joint1_deg joint2_deg joint3_deg "
            << "joint4_deg joint5_deg joint6_deg\n";
        return 1;
    }

    const char* urdf_env = std::getenv("TB6_IK_URDF");
    if (urdf_env == nullptr) {
        std::cerr << "TB6_IK_URDF is not set\n";
        return 1;
    }

    KDL::Tree tree;
    if (!kdl_parser::treeFromFile(urdf_env, tree)) {
        std::cerr << "Failed to load URDF\n";
        return 1;
    }

    KDL::Chain chain;
    if (!tree.getChain("base_link", "ee_Link", chain)) {
        std::cerr << "Failed to build base_link -> ee_Link chain\n";
        return 1;
    }

    if (chain.getNrOfJoints() != 6) {
        std::cerr << "Expected 6 joints, got "
                  << chain.getNrOfJoints() << '\n';
        return 1;
    }

    KDL::JntArray joints(6);

    for (int i = 0; i < 6; ++i) {
        joints(i) = degToRad(std::stod(argv[i + 1]));
    }

    KDL::ChainFkSolverPos_recursive fk_solver(chain);
    KDL::Frame frame;

    if (fk_solver.JntToCart(joints, frame) < 0) {
        std::cerr << "KDL FK failed\n";
        return 1;
    }

    double rx = 0.0;
    double ry = 0.0;
    double rz = 0.0;
    frame.M.GetRPY(rx, ry, rz);

    std::cout << std::fixed << std::setprecision(8);

    std::cout << "KDL PE321 [x, y, z, rz, ry, rx]:\n";
    std::cout
        << frame.p.x() << ' '
        << frame.p.y() << ' '
        << frame.p.z() << ' '
        << rz << ' '
        << ry << ' '
        << rx << '\n';

    return 0;
}
