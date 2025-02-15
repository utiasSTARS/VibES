//
// Created by viciopoli on 02/02/25.
//

#ifndef PROJECT_CENTROID_HPP
#define PROJECT_CENTROID_HPP

#include "utils.hpp"

#include <optional>
#include <tuple>

class CMassCalculation {
public:
    explicit CMassCalculation(int counter_threshold = 50, double time_window = 1e-4) : _counter_threshold(
            counter_threshold), _time_window(time_window) {}

    std::optional<std::tuple<double, double, double>> feed(double x, double y, double t) {
        if (_init_t < 0) _init_t = t; // Initialize start time

        double delta_t = t - _init_t;
        if (delta_t > _time_window) {  // Time window exceeded
            auto centroid = getCMass();
            reset(x, y, t);
            return centroid;
        }

        // Accumulate values
        _x += x;
        _y += y;
        _t += t;
        _counter++;
        return std::nullopt;
    }

    std::optional<std::tuple<double, double, double>> getCMass() {
        if (_counter < _counter_threshold) return std::nullopt;
        double cc = static_cast<double>(_counter);
        return std::make_tuple(_x / cc, _y / cc, _t / cc);
    }

private:
    void reset(double x, double y, double t) {
        _x = x;
        _y = y;
        _t = t; // Keep last point
        _counter = 1;
        _init_t = t;
    }

    double _x = 0, _y = 0, _t = 0;
    double _init_t = -1;
    unsigned long int _counter = 0;
    const int _counter_threshold = 50;
    const double _time_window = 1e-4;
};


using CentroidPtr = std::shared_ptr<CMassCalculation>;
using CentroidsVec = std::vector<CentroidPtr>;

#endif //PROJECT_CENTROID_HPP
