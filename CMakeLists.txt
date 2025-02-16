cmake_minimum_required(VERSION 3.22)
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

# find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(dv-processing REQUIRED)
find_package(Open3D REQUIRED)
find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

include(tests/CMakeLists.txt)

add_executable(helix_vis_thread demo/helix_vis_thread.cpp)
target_link_libraries(helix_vis_thread PRIVATE Open3D::Open3D finufft dv::processing)

add_executable(helix_vis_thread_vis demo/helix_vis_thread_vis.cpp)
target_link_libraries(helix_vis_thread_vis PRIVATE Open3D::Open3D finufft dv::processing)

add_executable(motion_comp demo/motion_comp.cpp)
target_link_libraries(motion_comp PRIVATE Open3D::Open3D finufft dv::processing)

add_executable(harmeda demo/harmeda.cpp)
target_link_libraries(harmeda PRIVATE Open3D::Open3D finufft dv::processing PUBLIC OpenMP::OpenMP_CXX) # TBB::tbb)

add_executable(harmeda_metav demo/harmeda_metavision.cpp)
target_link_libraries(harmeda_metav PRIVATE Open3D::Open3D finufft MetavisionSDK::core MetavisionSDK::driver MetavisionSDK::ui PUBLIC OpenMP::OpenMP_CXX) # TBB::tbb)

add_executable(harmeda_motion_comp demo/harmeda_motion_comp.cpp)
target_link_libraries(harmeda_motion_comp PRIVATE Open3D::Open3D finufft MetavisionSDK::core MetavisionSDK::driver MetavisionSDK::ui PUBLIC OpenMP::OpenMP_CXX) # TBB::tbb)
