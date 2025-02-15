# CMAKE FOR TESTS

# GTEST
include(FetchContent)
FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(test_ekf ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ekf.cpp)
target_link_libraries(test_ekf PRIVATE gtest_main)
add_test(NAME test_ekf COMMAND test_ekf)

add_executable(test_bin ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bin.cpp)
target_link_libraries(test_bin PRIVATE gtest_main)
add_test(NAME test_bin COMMAND test_bin)


add_executable(test_sim_bins ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sim_bins.cpp)
target_link_libraries(test_sim_bins PRIVATE gtest_main dv::processing)

add_executable(test_dvs_bins ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dvs_bins.cpp)
target_link_libraries(test_dvs_bins PRIVATE gtest_main dv::processing)

add_executable(test_finuttf ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_finuttf.cpp)
target_link_libraries(test_finuttf PRIVATE gtest_main dv::processing finufft)

add_executable(test_centroid ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_centroid.cpp)
target_link_libraries(test_centroid PRIVATE gtest_main dv::processing)

add_executable(test_centroid_metav ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_centroid_metav.cpp)
target_link_libraries(test_centroid_metav PRIVATE gtest_main MetavisionSDK::core MetavisionSDK::driver MetavisionSDK::ui PUBLIC OpenMP::OpenMP_CXX) # TBB::tbb)

add_executable(test_centroid_nufft ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_centroid_nufft.cpp)
target_link_libraries(test_centroid_nufft PRIVATE gtest_main dv::processing finufft)

add_executable(test_centroid_nufft_metav ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_centroid_nufft_metav.cpp)
target_link_libraries(test_centroid_nufft_metav PRIVATE gtest_main finufft MetavisionSDK::driver MetavisionSDK::ui)

add_executable(test_finuttf_standalone ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_finuttf_standalone.cpp)
target_link_libraries(test_finuttf_standalone PRIVATE gtest_main dv::processing finufft)

add_executable(test_centroid_nufft_ekf ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_centroid_nufft_ekf.cpp)
target_link_libraries(test_centroid_nufft_ekf PRIVATE gtest_main dv::processing finufft)
