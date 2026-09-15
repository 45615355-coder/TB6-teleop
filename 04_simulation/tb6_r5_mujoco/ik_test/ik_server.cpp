#include "tb6_ik_solver.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;

double degToRad(double degree)
{
    return degree * kPi / 180.0;
}

}  // namespace

int main()
{
    std::ios::sync_with_stdio(false);

    std::cout << "READY" << std::endl;

    std::string line;

    while (std::getline(std::cin, line)) {
        if (line == "quit" || line == "exit") {
            break;
        }

        if (line.empty()) {
            continue;
        }

        std::istringstream input(line);

        tb6_ik::Pe321 target{};
        std::vector<double> seed_joints_rad(6, 0.0);

        bool valid = true;

        // 读取目标位姿：
        // x y z rz ry rx
        for (std::size_t i = 0; i < 6; ++i) {
            if (!(input >> target[i]) || !std::isfinite(target[i])) {
                valid = false;
                break;
            }
        }

        // 读取六个初始关节角，单位为度
        for (std::size_t i = 0; valid && i < 6; ++i) {
            double angle_deg = 0.0;

            if (!(input >> angle_deg) || !std::isfinite(angle_deg)) {
                valid = false;
                break;
            }

            seed_joints_rad[i] = degToRad(angle_deg);
        }

        if (!valid) {
            std::cout
                << "ERR expected 12 finite numbers: "
                << "x y z rz ry rx j1_deg ... j6_deg"
                << std::endl;
            continue;
        }

        const tb6_ik::IkSolveResult result =
            tb6_ik::solveIkPe321(target, seed_joints_rad);

        if (!result.success) {
            std::cout << "ERR " << result.message << std::endl;
            continue;
        }

        std::cout << std::setprecision(17) << "OK";

        // 输出弧度，后续可直接写入 MuJoCo qpos
        for (double joint_rad : result.joints_rad) {
            std::cout << ' ' << joint_rad;
        }

        std::cout << std::endl;
    }

    return 0;
}