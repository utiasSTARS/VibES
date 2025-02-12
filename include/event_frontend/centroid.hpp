//
// Created by viciopoli on 02/02/25.
//

#ifndef PROJECT_CENTROID_HPP
#define PROJECT_CENTROID_HPP

#include "utils.hpp"

class CentroidCalculation {
public:
    CentroidCalculation() = default;

    explicit CentroidCalculation(const int counter_threshold) : _counter_threshold(counter_threshold) {}

    std::optional<std::tuple<double, double, double>> feed(double x, double y, double t) {
        const auto t_avg = _t / (_counter + 1e-8);
        const double delta_t = t - t_avg;
        if (delta_t > 1e-5) {
            auto centroid = getCentroid();
            if (centroid) {
                _x = x;
                _y = y;
                _t = t;
                _counter = 1;
                return centroid;
            }
            reset();
        }

        _x += x;
        _y += y;
        _t += t;
        _counter++;
        return std::nullopt;
    }

    void reset() {
        _x = 0;
        _y = 0;
        _t = 0;
        _counter = 0;
    }

    std::optional<std::tuple<double, double, double>> getCentroid() {
        if (_counter < _counter_threshold) return std::nullopt;  // Early return

        double cc = static_cast<double>(_counter);  // Convert only once
        return std::make_tuple(_x / cc, _y / cc, _t / cc);
    }


private:
    double _x = 0, _y = 0;
    double _t = 0.0;
    unsigned long int _counter = 0;
    const int _counter_threshold = 50;
};

using CentroidPtr = std::shared_ptr<CentroidCalculation>;
using CentroidsVec = std::vector<CentroidPtr>;

#endif //PROJECT_CENTROID_HPP
