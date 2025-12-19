cmake_minimum_required(VERSION 3.22)
project(project)

option(BUILD_METRICS "Build metrics executables" OFF)

set(CMAKE_CXX_STANDARD 20)

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
set(CMAKE_INSTALL_RPATH "/usr/local/lib")

# find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED)
find_package(Boost 1.74 REQUIRED COMPONENTS filesystem system program_options)
find_package(MetavisionSDK COMPONENTS base cv driver ui REQUIRED analytics)


include_directories(${EIGEN3_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Build HASTE AND INTEGRATE IT INTO THE PROJECT
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
add_subdirectory(thirdparty/haste/thirdparty/quickgui)
include_directories(thirdparty/haste/include)


add_executable(compensation evaluation/compensation.cpp)
target_link_libraries(compensation PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(depth_estimation_iter evaluation/depth_estimation_iter.cpp)
target_link_libraries(depth_estimation_iter PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(depth_estimation demo/depth_estimation.cpp)
target_link_libraries(depth_estimation PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(multi_tracker demo/multi_tracker.cpp)
target_link_libraries(multi_tracker PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(amiev_visualization evaluation/amiev_visualization.cpp)
target_link_libraries(amiev_visualization PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(frequency_estimation_visualizer evaluation/frequency_estimation_visualizer.cpp)
target_link_libraries(frequency_estimation_visualizer PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(frequency_estimation evaluation/frequency_estimation.cpp)
target_link_libraries(frequency_estimation PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)


if (BUILD_METRICS)

    add_executable(edges_metrics evaluation/edges_metrics.cpp)
    target_link_libraries(edges_metrics PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

    add_executable(slicer evaluation/file_slicer.cpp)
    target_link_libraries(slicer PRIVATE ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

    add_executable(variance_grad_metrics evaluation/variance_grad_metrics.cpp)
    target_link_libraries(variance_grad_metrics PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

endif ()
