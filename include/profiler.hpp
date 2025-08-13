#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <iomanip>

class ProfileTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string scope_name;

    struct TimerData {
        long long total_ns = 0;
        size_t call_count = 0;

        void add_measurement(long long ns) {
            total_ns += ns;
            ++call_count;
        }
    };

    // Thread-safe access to timer data
    static std::mutex timer_mutex;
    static std::unordered_map<std::string, TimerData> full_timer;

public:
    explicit ProfileTimer(const std::string &name) : scope_name(name) {
        // Ensure timer entry exists
        {
            std::lock_guard<std::mutex> lock(timer_mutex);
            if (full_timer.find(scope_name) == full_timer.end()) {
                full_timer[scope_name] = TimerData{};
            }
        }
        start_time = std::chrono::high_resolution_clock::now();
    }

    // Disable copy/move to prevent timing issues
    ProfileTimer(const ProfileTimer &) = delete;

    ProfileTimer &operator=(const ProfileTimer &) = delete;

    ProfileTimer(ProfileTimer &&) = delete;

    ProfileTimer &operator=(ProfileTimer &&) = delete;

    static void add_name(const std::string &name) {
        std::lock_guard<std::mutex> lock(timer_mutex);
        if (full_timer.find(name) == full_timer.end()) {
            full_timer[name] = TimerData{};
        }
    }

    static void print_results(const std::string &name, bool show_average = false, long long total_events = 0) {
        std::lock_guard<std::mutex> lock(timer_mutex);

        auto it = full_timer.find(name);
        if (it == full_timer.end()) {
            std::cout << "[PROFILE] Timer '" << name << "' not found" << std::endl;
            return;
        }

        const auto &data = it->second;

        if (total_events > 0) {
            std::cout << "[PROFILE] " << name << ": "
                      << std::fixed << std::setprecision(2)
                      << (data.total_ns / 1e6) << " ms, "
                      << data.call_count << " calls, "
                      << (data.total_ns / total_events) << " ns/event"
                      << std::endl;
            return;
        }

        if (show_average && data.call_count > 0) {
            long long avg_ns = data.total_ns / data.call_count;
            std::cout << "[PROFILE] " << name << ": "
                      << std::fixed << std::setprecision(2)
                      << avg_ns << " ns/call (avg over " << data.call_count << " calls)"
                      << std::endl;
        } else {
            std::cout << "[PROFILE] " << name << ": "
                      << std::fixed << std::setprecision(2)
                      << data.total_ns << " ns, "
                      << (data.total_ns / 1e6) << " ms, "
                      << data.call_count << " calls"
                      << std::endl;
        }
    }

    static void print_all_results() {
        std::lock_guard<std::mutex> lock(timer_mutex);

        if (full_timer.empty()) {
            std::cout << "[PROFILE] No timing data available" << std::endl;
            return;
        }

        std::cout << "[PROFILE] All timing results:" << std::endl;
        for (const auto &[name, data]: full_timer) {
            std::cout << "  " << name << ": "
                      << std::fixed << std::setprecision(2)
                      << (data.total_ns / 1e6) << " ms ("
                      << data.call_count << " calls, "
                      << (data.call_count > 0 ? data.total_ns / data.call_count : 0)
                      << " ns/call avg)" << std::endl;
        }
    }

    static void reset(const std::string &name = "") {
        std::lock_guard<std::mutex> lock(timer_mutex);

        if (name.empty()) {
            full_timer.clear();
        } else {
            auto it = full_timer.find(name);
            if (it != full_timer.end()) {
                it->second = TimerData{};
            }
        }
    }

    static size_t get_call_count(const std::string &name) {
        std::lock_guard<std::mutex> lock(timer_mutex);
        auto it = full_timer.find(name);
        return (it != full_timer.end()) ? it->second.call_count : 0;
    }

    static long long get_total_time_ns(const std::string &name) {
        std::lock_guard<std::mutex> lock(timer_mutex);
        auto it = full_timer.find(name);
        return (it != full_timer.end()) ? it->second.total_ns : 0;
    }

    ~ProfileTimer() noexcept {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

        std::lock_guard<std::mutex> lock(timer_mutex);
        full_timer[scope_name].add_measurement(duration.count());
    }
};

// Static member definitions
std::mutex ProfileTimer::timer_mutex;
std::unordered_map<std::string, ProfileTimer::TimerData> ProfileTimer::full_timer;

// Convenience macro for easier usage
#define PROFILE_SCOPE(name) ProfileTimer _profile_timer(name)
#define PROFILE_FUNCTION() ProfileTimer _profile_timer(__FUNCTION__)

// RAII helper for conditional profiling
class ConditionalProfiler {
    std::unique_ptr<ProfileTimer> timer;

public:
    explicit ConditionalProfiler(const std::string &name, bool enabled) {
        if (enabled) {
            timer = std::make_unique<ProfileTimer>(name);
        }
    }
};