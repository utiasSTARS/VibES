//
// Created by viciopoli on 02/02/25.
//

#ifndef PROJECT_CENTROID_HPP
#define PROJECT_CENTROID_HPP

#include "utils.hpp"

#include <optional>
#include <tuple>
#include <vector>
#include <memory>

using opt_tuple = std::optional<std::tuple<double, double, double>>;

class CMassCalculation {
public:
    CMassCalculation(int width, int height, int counter_threshold = 50, double time_window = 1e-4, int n_bins_x = 10,
                     int n_bins_y = 10)
            : _width(width), _height(height),
              _counter_threshold(counter_threshold), _time_window(time_window),
              _n_bins_x(n_bins_x), _n_bins_y(n_bins_y),
              _bin_width(int(width / n_bins_x)), _bin_height(int(height / n_bins_y)) {
        // init bins and weights with 0
        _bins_x.reserve(_n_bins_x);
        _bins_x.assign(_n_bins_x, 0);
        _bins_y.reserve(_n_bins_y);
        _bins_y.assign(_n_bins_y, 0);
        _weights.reserve(_n_bins_x * _n_bins_y);
        _weights.assign(_n_bins_x * _n_bins_y, 0);
    }

    opt_tuple feed(int x, int y, double t) {
        // bounds check
        if (x < 0 || x >= _width || y < 0 || y >= _height) return std::nullopt;
        if (_init_t < 0) _init_t = t; // Initialize start time

        opt_tuple centroid;
        const double delta_t = t - _init_t;
        if (delta_t > _time_window) {  // Time window exceeded
            centroid = getCMass();
            reset(x, y, t);
            return centroid;
        }

        const auto x_idx = static_cast<int>(x / _bin_width);
        const auto y_idx = static_cast<int>(y / _bin_height);
        const auto idx = _n_bins_x * y_idx + x_idx;
        _weights.at(idx) += 1;
        _bins_x.at(x_idx) += x;
        _bins_y.at(y_idx) += y;
        _t += t;
        _counter++;
        return centroid;
    }

    opt_tuple getCMass() {
        if (_counter < _counter_threshold) return std::nullopt;

        const auto icc = 1. / static_cast<double>(_counter);
        double _x = 0, _y = 0;
        for (int i = 0; i < _n_bins_x; i++) {
            _weights[i] = _weights[i] * icc;
            _x += _bins_x[i] * _weights[i];
        }
        for (int i = 0; i < _n_bins_y; i++) {
            _y += _bins_y[i] * _weights[i];
        }
        return std::make_tuple(_x * icc, _y * icc, _t * icc);
    }

private:

    inline void reset(int x, int y, double t) {
        _bins_x.assign(_n_bins_x, 0);
        _bins_y.assign(_n_bins_x, 0);
        _weights.assign(_n_bins_x * _n_bins_y, 0);
        const auto idx = _n_bins_x * int(y / _bin_height) + int(x / _bin_width);
        _weights[idx] += 1;
        _bins_x[idx] += x;
        _bins_y[idx] += y;
        _t = t;
        _counter = 1;
    }

    double _t = 0;
    double _init_t = -1;
    unsigned long int _counter = 0;
    const int _counter_threshold;
    const double _time_window;
    const double alpha = 0.1;
    const int _bin_width, _bin_height;
    const int _width, _height;

    const int _n_bins_x = 10, _n_bins_y = 10;
    std::vector<double> _weights;
    std::vector<int> _bins_x, _bins_y;

};


using CentroidPtr = std::shared_ptr<CMassCalculation>;
using CentroidsVec = std::vector<CentroidPtr>;

#endif //PROJECT_CENTROID_HPP