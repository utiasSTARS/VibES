//
// Created by viciopoli on 22/07/25.
//

#ifndef PROJECT_PROFILER_HPP
#define PROJECT_PROFILER_HPP

#include <chrono>
#include <iostream>
#include <string>

class ProfileTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string scope_name;

public:
    ProfileTimer(const std::string &name) : scope_name(name) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~ProfileTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        std::cout << "[PROFILE] " << scope_name << ": " << duration.count() << " ns" << std::endl;
    }
};


#define PROFILE_SCOPE(name) ProfileTimer timer_##name(#name)


#endif //PROJECT_PROFILER_HPP
