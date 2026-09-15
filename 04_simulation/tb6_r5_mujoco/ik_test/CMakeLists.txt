cmake_minimum_required(VERSION 3.8)

project(tb6_ik_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)

add_executable(
    ik_smoke_test
    ik_smoke_test.cpp
    tb6_ik_solver.cpp
)

add_executable(
    fk_kdl_test
    fk_kdl_test.cpp
)

add_executable(
    ik_cli
    ik_cli.cpp
    tb6_ik_solver.cpp
)

add_executable(
    ik_server
    ik_server.cpp
    tb6_ik_solver.cpp
)

target_include_directories(
    ik_smoke_test
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    ik_cli
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    ik_server
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

ament_target_dependencies(
    ik_smoke_test
    orocos_kdl
    kdl_parser
)

ament_target_dependencies(
    fk_kdl_test
    orocos_kdl
    kdl_parser
)

ament_target_dependencies(
    ik_cli
    orocos_kdl
    kdl_parser
)

ament_target_dependencies(
    ik_server
    orocos_kdl
    kdl_parser
)

ament_package()
