cmake_minimum_required(VERSION 3.26)
project(project)

find_package(Eigen3 REQUIRED)
find_package(dv-processing REQUIRED)
find_package(Open3D REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(Torch REQUIRED)

find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

add_executable(project main.cpp)
add_executable(test_event_generation test_event_generation.cpp)
add_executable(main_ekf main_ekf.cpp)
add_executable(main_ekf_ini main_ekf_ini.cpp)
add_executable(features features.cpp)
add_executable(freq_est freq_est.cpp)
add_executable(freq_est_multi freq_est_multi.cpp)
add_executable(RANSAC_fitting RANSAC_fitting.cpp)
add_executable(fft_fitting fft_fitting.cpp)
add_executable(prophesee_ekf_fitting prophesee_ekf_fitting.cpp)
add_executable(learned_approach learned_approach.cpp)
add_executable(learned_approach_motion_comp learned_approach_motion_comp.cpp)
add_executable(fft_fitting2d fft_fitting2d.cpp)

target_link_libraries(main_ekf_ini dv::processing Open3D::Open3D)
target_link_libraries(test_event_generation dv::processing Open3D::Open3D)
target_link_libraries(features dv::processing)
target_link_libraries(freq_est dv::processing Open3D::Open3D)
target_link_libraries(freq_est_multi dv::processing)
target_link_libraries(RANSAC_fitting dv::processing Open3D::Open3D)
target_link_libraries(fft_fitting dv::processing Open3D::Open3D fftw3)
target_link_libraries(prophesee_ekf_fitting Open3D::Open3D MetavisionSDK::core MetavisionSDK::driver MetavisionSDK::ui)
target_link_libraries(learned_approach dv::processing Open3D::Open3D fftw3 ${TORCH_LIBRARIES})
target_link_libraries(learned_approach_motion_comp dv::processing ${TORCH_LIBRARIES} Open3D::Open3D)
target_link_libraries(fft_fitting2d dv::processing Open3D::Open3D fftw3)


