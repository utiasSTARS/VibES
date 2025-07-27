//
// Created by viciopoli on 18/11/24.
//
#ifndef PROJECT_UTILS_HPP
#define PROJECT_UTILS_HPP

#include <algorithm>
#include <sstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <optional>


struct Centroid {
    float t;
    unsigned short x;
    unsigned short y;
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
inline T Hz2rad(T hz) {
    return hz * (2 * M_PI);
}

template<typename T>
T clipAngle(T angle) {
    while (angle < 0) { angle += 2 * M_PI; }
    while (angle > 2 * M_PI) { angle -= 2 * M_PI; }
    return angle;
}

#endif  // PROJECT_UTILS_HPP
