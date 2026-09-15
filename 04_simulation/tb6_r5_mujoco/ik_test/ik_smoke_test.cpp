#include "tb6_ik_solver.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
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

int main()
{
    // 这是之前由 MuJoCo 根据关节角
    // [0, 20, -30, 0, 10, 0] 度算出的末端位姿。
    const tb6_ik::Pe321 target_pe321 = {
        -0.82299887,
        -0.17747419,
        -0.05691273,
        -0.17716919,
         0.17185604,
         1.54019064,
    };

    // IK 初值使用原来的六个关节角。
    const std::vector<double> seed_joints_rad = {
        degToRad(0.0),
        degToRad(20.0),
        degToRad(-30.0),
        degToRad(0.0),
        degToRad(10.0),
        degToRad(0.0),
    };

    const tb6_ik::IkSolveResult result =
        tb6_ik::solveIkPe321(target_pe321, seed_joints_rad);

    std::cout << "IK success: "
              << (result.success ? "true" : "false")
              << '\n';

    std::cout << "Message: " << result.message << '\n';

    if (!result.success) {
        return 1;
    }

    std::cout << "\nIK output in degrees:\n";

    std::cout << std::fixed << std::setprecision(6);

    for (std::size_t i = 0; i < result.joints_rad.size(); ++i) {
        std::cout
            << "joint" << i + 1
            << " = "
            << radToDeg(result.joints_rad[i])
            << " deg\n";
    }

    return 0;
}