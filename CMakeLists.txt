cmake_minimum_required(VERSION 3.22)
project(project)

option(BUILD_TESTS "Build test programs" OFF)

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
set(CMAKE_INSTALL_RPATH "/usr/local/lib")

# find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED)
find_package(Boost 1.74 REQUIRED COMPONENTS filesystem system program_options)
find_package(MetavisionSDK COMPONENTS base cv driver ui REQUIRED analytics)

#find_package(dv-processing REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

if (BUILD_TESTS)
    include(tests/CMakeLists.txt)
endif (BUILD_TESTS)

# Build HASTE AND INTEGRATE IT INTO THE PROJECT

find_package(gflags REQUIRED)
find_package(glog REQUIRED)
add_subdirectory(thirdparty/haste/thirdparty/quickgui)
include_directories(thirdparty/haste/include)

# Generate the lib using the instructions in the thirdparty repo
#if (NOT EXISTS "${PROJECT_SOURCE_DIR}/thirdparty/VecKM_flow_cpp/libSliceNormalFlowEstimator.so" AND NOT EXISTS "${PROJECT_SOURCE_DIR}/thirdparty/VecKM_flow_cpp/libSliceNormalFlowEstimator.dll")
#    message(FATAL_ERROR "The VecKM_flow_cpp library is not built. Please follow the instructions in the thirdparty/VecKM_flow_cpp README to build it.")
#endif ()
#if (UNIX)
#    set(FLOW_LIB_PATH "${PROJECT_SOURCE_DIR}/thirdparty/VecKM_flow_cpp/libSliceNormalFlowEstimator.so")
#elseif (WIN32)
#    set(FLOW_LIB_PATH "${PROJECT_SOURCE_DIR}/thirdparty/VecKM_flow_cpp/libSliceNormalFlowEstimator.dll")
#else ()
#    message(FATAL_ERROR "Unsupported platform for VecKM_flow_cpp library")
#endif ()

#add_executable(helix_vis_thread demo_old/helix_vis_thread.cpp)
#target_link_libraries(helix_vis_thread PRIVATE Open3D::Open3D finufft dv::processing)
#
#add_executable(helix_vis_thread_vis demo_old/helix_vis_thread_vis.cpp)
#target_link_libraries(helix_vis_thread_vis PRIVATE Open3D::Open3D finufft dv::processing)
#
#add_executable(motion_comp demo_old/motion_comp.cpp)
#target_link_libraries(motion_comp PRIVATE Open3D::Open3D finufft dv::processing)
#
#add_executable(harmeda demo_old/harmeda.cpp)
#target_link_libraries(harmeda PRIVATE Open3D::Open3D finufft dv::processing PUBLIC OpenMP::OpenMP_CXX) # TBB::tbb)
#
#add_executable(harmeda_metav demo_old/harmeda_metavision.cpp)
#target_link_libraries(harmeda_metav PRIVATE Open3D::Open3D finufft MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system PUBLIC OpenMP::OpenMP_CXX) # TBB::tbb)
#
#add_executable(harmeda_motion_comp demo_old/harmeda_motion_comp.cpp)
#target_link_libraries(harmeda_motion_comp PRIVATE Open3D::Open3D finufft MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system PUBLIC OpenMP::OpenMP_CXX) # TBB::tbb)
#
#add_executable(harmeda_filter demo_old/harmeda_filter.cpp)
#target_link_libraries(harmeda_filter PRIVATE Open3D::Open3D finufft MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system PUBLIC OpenMP::OpenMP_CXX) # TBB::tbb)
#
#add_executable(harmeda_motion_comp_filter demo_old/harmeda_motion_comp_filter.cpp)
#target_link_libraries(harmeda_motion_comp_filter PRIVATE Open3D::Open3D finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system PUBLIC OpenMP::OpenMP_CXX)
#
#
#add_executable(viewer demo_old/viewer.cpp)
#target_link_libraries(viewer PRIVATE Open3D::Open3D finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system PUBLIC OpenMP::OpenMP_CXX)
#
#add_executable(centroid_ema demo_old/centroid_ema.cpp)
#target_link_libraries(centroid_ema PRIVATE Open3D::Open3D finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX)
#
#add_executable(centroid_ema_comp demo_old/centroid_ema_with_compensation.cpp)
#target_link_libraries(centroid_ema_comp PRIVATE Open3D::Open3D finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX)
#
#add_executable(centroid_arma demo_old/centroid_arma.cpp)
#target_link_libraries(centroid_arma PRIVATE Open3D::Open3D finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX)
#
#add_executable(centroid_ema_flow demo_old/centroid_ema_with_flow.cpp)
#target_link_libraries(centroid_ema_flow PRIVATE Open3D::Open3D finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX)
#
#add_executable(motion_estimation demo_old/motion_estimation.cpp)
#target_link_libraries(motion_estimation PRIVATE Open3D::Open3D finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX)

add_executable(haste_comp demo/haste_comp.cpp)
target_link_libraries(haste_comp PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(compensation evaluation/compensation.cpp)
target_link_libraries(compensation PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(compensation_harm_test evaluation/compensation_harm_test.cpp)
target_link_libraries(compensation_harm_test PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(compensation_store_imgs evaluation/compensation_store_imgs.cpp)
target_link_libraries(compensation_store_imgs PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)


add_executable(depth_estimation demo/depth_estimation.cpp)
target_link_libraries(depth_estimation PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(multi_tracker demo/multi_tracker.cpp)
target_link_libraries(multi_tracker PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(undistort_and_store demo/undistort_and_store.cpp)
target_link_libraries(undistort_and_store PRIVATE ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options)

add_executable(recorder demo/recorder.cpp)
target_link_libraries(recorder PRIVATE MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options)

add_executable(compensation_open3d demo/compensation_open3d.cpp)
target_link_libraries(compensation_open3d PRIVATE Open3D::Open3D finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(feature_tracker demo/feature_tracker.cpp)
target_link_libraries(feature_tracker PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(frequency_filter evaluation/frequency_filter.cpp)
target_link_libraries(frequency_filter PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)

add_executable(amiev_visualization evaluation/amiev_visualization.cpp)
target_link_libraries(amiev_visualization PRIVATE finufft ${OpenCV_LIBS} MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::driver MetavisionSDK::ui Boost::filesystem Boost::system Boost::program_options PUBLIC OpenMP::OpenMP_CXX gflags glog QuickGui)
