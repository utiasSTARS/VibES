cmake_minimum_required(VERSION 3.26)
project(project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

include(cmake/CPM.cmake)
# short version
CPMAddPackage("gh:flatironinstitute/finufft@2.3.1")

# alternative in case custom options are needed
CPMAddPackage(
        NAME Finufft
        GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git
        GIT_TAG 2.3.1
        GIT_SHALLOW Yes
        GIT_PROGRESS Yes
        EXCLUDE_FROM_ALL Yes
        SYSTEM
)

find_package(Eigen3 REQUIRED)
find_package(dv-processing REQUIRED)
find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)
find_package(Open3D REQUIRED)
find_package(FFTW3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})


include(tests/CMakeLists.txt)


add_executable(project demo/main.cpp)
target_link_libraries(project PRIVATE Open3D::Open3D finufft dv::processing)

add_executable(helix_vis demo/helix_vis.cpp)
target_link_libraries(helix_vis PRIVATE Open3D::Open3D finufft dv::processing)

add_executable(helix_vis_thread demo/helix_vis_thread.cpp)
target_link_libraries(helix_vis_thread PRIVATE Open3D::Open3D finufft dv::processing)

add_executable(helix_vis_ocv_vis demo/helix_vis_ocv_vis.cpp)
target_link_libraries(helix_vis_ocv_vis PRIVATE Open3D::Open3D finufft dv::processing)


add_executable(test_event_generation test_event_generation.cpp)
add_executable(main_ekf main_ekf.cpp)
add_executable(main_ekf_ini main_ekf_ini.cpp)
add_executable(features features.cpp)
add_executable(freq_est freq_est.cpp)
add_executable(freq_est_multi freq_est_multi.cpp)
add_executable(RANSAC_fitting RANSAC_fitting.cpp)
add_executable(fft_fitting fft_fitting.cpp)
add_executable(prophesee_ekf_fitting prophesee_ekf_fitting.cpp)

add_executable(finufft_fitting finufft_fitting.cpp)

target_link_libraries(main_ekf_ini dv::processing Open3D::Open3D)
target_link_libraries(test_event_generation dv::processing Open3D::Open3D)
target_link_libraries(features dv::processing)
target_link_libraries(freq_est dv::processing Open3D::Open3D)
target_link_libraries(freq_est_multi dv::processing Open3D::Open3D)
target_link_libraries(RANSAC_fitting dv::processing Open3D::Open3D)
target_link_libraries(fft_fitting dv::processing Open3D::Open3D fftw3)
target_link_libraries(prophesee_ekf_fitting Open3D::Open3D MetavisionSDK::core MetavisionSDK::driver MetavisionSDK::ui)
target_link_libraries(finufft_fitting PRIVATE finufft dv::processing Open3D::Open3D)
# target_link_libraries(finufft_fitting PRIVATE -Wl,-rpath,/usr/local/lib dv::processing Open3D::Open3D)
