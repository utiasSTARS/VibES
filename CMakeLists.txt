cmake_minimum_required(VERSION 3.26)
project(project)

find_package(Eigen3 REQUIRED)
find_package(dv-processing REQUIRED)
find_package(Open3D REQUIRED)


include_directories(${EIGEN3_INCLUDE_DIR})


set(CMAKE_CXX_STANDARD 20)

add_executable(project main.cpp)
add_executable(main_ekf main_ekf.cpp)
add_executable(main_ekf_ini main_ekf_ini.cpp)
add_executable(features features.cpp)
add_executable(freq_est freq_est.cpp)
add_executable(freq_est_multi freq_est_multi.cpp)

target_link_libraries(main_ekf_ini dv::processing Open3D::Open3D)
target_link_libraries(features dv::processing)
target_link_libraries(freq_est dv::processing)
target_link_libraries(freq_est_multi dv::processing)
