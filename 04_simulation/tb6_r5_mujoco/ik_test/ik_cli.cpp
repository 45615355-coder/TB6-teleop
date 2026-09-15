#include "tb6_ik_solver.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;

double degToRad(double degree)
{
    return degree * kPi / 180.0;
}

double radToDeg(double radian)
{
    return radian * 180.0 / kPi;
}

}  // namespace

int main(int argc, char** argv)
{
    // 参数：
    // x y z rz ry rx
    // seed_j1_deg ... seed_j6_deg
    if (argc != 13) {
        std::cerr
            << "Usage:\n"
            << "  ik_cli x y z rz ry rx "
            << "j1_deg j2_deg j3_deg j4_deg j5_deg j6_deg\n";
        return 1;
    }

    try {
        tb6_ik::Pe321 target{};

        for (int i = 0; i < 6; ++i) {
            target[static_cast<std::size_t>(i)] =
                std::stod(argv[i + 1]);
        }

        std::vector<double> seed(6, 0.0);

        for (int i = 0; i < 6; ++i) {
            seed[static_cast<std::size_t>(i)] =
                degToRad(std::stod(argv[i + 7]));
        }

        const tb6_ik::IkSolveResult result =
            tb6_ik::solveIkPe321(target, seed);

        std::cout << "IK success: "
                  << (result.success ? "true" : "false")
                  << '\n';

        std::cout << "Message: "
                  << result.message
                  << '\n';

        if (!result.success) {
            return 2;
        }

        std::cout << std::fixed << std::setprecision(8);

        std::cout << "q_target_deg:";

        for (double joint_rad : result.joints_rad) {
            std::cout << ' ' << radToDeg(joint_rad);
        }

        std::cout << '\n';

        std::cout << "q_target_rad:";

        for (double joint_rad : result.joints_rad) {
            std::cout << ' ' << joint_rad;
        }

        std::cout << '\n';
    }
    catch (const std::exception& exception) {
        std::cerr << "Invalid argument: "
                  << exception.what()
                  << '\n';
        return 3;
    }

    return 0;
}