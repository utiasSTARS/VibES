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
