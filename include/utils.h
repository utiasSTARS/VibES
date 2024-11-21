//
// Created by viciopoli on 18/11/24.
//
#ifndef PROJECT_UTILS_H
#define PROJECT_UTILS_H

#include <algorithm>
#include <vector>
#include <tuple>

// Define EventStruct to store individual events
struct EventStruct {
    int64_t timestamp;
    int16_t x;
    int16_t y;
    bool polarity;
};

enum Colors {
    RED = 0,
    GREEN = 1,
    BLUE = 2
};

std::map<Colors, std::tuple<int, int, int>> color_map = {
        {RED,   std::tuple<int, int, int>(0, 0, 255)},
        {GREEN, std::tuple<int, int, int>(0, 255, 0)},
        {BLUE,  std::tuple<int, int, int>(255, 0, 0)}
};


template<typename T>
inline T linear_interp(T alpha, T x0, T x1) {
    return x0 + alpha * (x1 - x0);
}

template<typename T>
inline T rad2Hz(T rad) {
    return rad / (2 * M_PI);
}

template<typename T>
inline T Hz2rad(T rad) {
    return rad * (2 * M_PI);
}

template<typename T>
T clipAngle(T angle) {
    while (angle < 0) angle += 2 * M_PI;
    while (angle > 2 * M_PI) angle -= 2 * M_PI;
    return angle;
}

std::optional<std::vector<std::tuple<double, double, int64_t>>>
interpolate_events(std::vector<std::tuple<double, double, int64_t>> &events,
                   double dt) {
    if (events.empty()) return std::nullopt;
    std::vector<std::tuple<double, double, int64_t>> out;
    out.push_back(events[0]);
    size_t i = 1;
    for (; i < events.size(); ++i) {
        auto [x0, y0, t0] = events[i - 1];
        auto [x1, y1, t1] = events[i];

        // Ensure the time difference is valid
        if (t1 <= t0) continue;

        // Add interpolated points
        for (double t = t0 + dt; t < t1; t += dt) {
            double alpha = (t - t0) / (t1 - t0); // Normalized interpolation factor
            double x = linear_interp(alpha, x0, x1);
            double y = linear_interp(alpha, y0, y1);
            out.emplace_back(x, y, static_cast<int64_t>(t));
        }
        // Add the next event
        out.push_back(events[i]);
    }
    // remove the i events from the buffer
    events.erase(events.begin(), events.begin() + i - 1);

    return out;
}

#endif //PROJECT_UTILS_H
