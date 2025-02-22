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
    CMassCalculation(int width, int height, int counter_threshold = 50, double time_window = 1e-4,
                     int n_bins_x = 10,
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

        _img = cv::Mat::zeros(_height, _width, CV_8UC3);
        cv::namedWindow("Centroid", cv::WINDOW_NORMAL);
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

        // _img = cv::Mat::zeros(_height, _width, CV_8UC3);
        // draw_bins_lines();
        // draw_counter_per_bin();
        // cv::imshow("Centroid", _img);
        // cv::waitKey(0);

        return centroid;
    }

    void draw_bins_lines() {
        for (int i = 0; i < _n_bins_x; i++) {
            cv::line(_img, cv::Point(i * _bin_width, 0), cv::Point(i * _bin_width, _height), cv::Scalar(255, 255, 255),
                     1);
        }
        for (int i = 0; i < _n_bins_y; i++) {
            cv::line(_img, cv::Point(0, i * _bin_height), cv::Point(_width, i * _bin_height), cv::Scalar(255, 255, 255),
                     1);
        }
    }

    void draw_counter_per_bin() {
        // color the text in the highest bin
        const auto green = cv::Scalar(0, 255, 0);
        const auto blue = cv::Scalar(255, 0, 0);
        const auto orange = cv::Scalar(0, 165, 255);
        const auto white = cv::Scalar(255, 255, 255);

        int max_idx = std::distance(_weights.begin(), std::max_element(_weights.begin(), _weights.end()));
        int second_max_idx = 0;
        for (int i = 0; i < _weights.size(); i++) {
            if (i == max_idx) continue;
            if (_weights[i] > _weights[second_max_idx]) {
                second_max_idx = i;
            }
        }

        for (int i = 0; i < _n_bins_x; i++) {
            for (int j = 0; j < _n_bins_y; j++) {
                const auto idx = _n_bins_x * j + i;
                const auto x = i * _bin_width + _bin_width / 2;
                const auto y = j * _bin_height + _bin_height / 2;
                double w_i = _n_bins_x * (1 - std::exp(-_weights[idx] / _counter));
                if (max_idx == idx) {
                    cv::putText(_img, fp2str(w_i, 2), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                green, 1);
                } else if (second_max_idx == idx) {
                    cv::putText(_img, fp2str(w_i, 2), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                orange, 1);
                } else {
                    cv::putText(_img, fp2str(w_i, 2), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                w_i > 0 ? blue : white, 1);
                }
            }
        }
    }

    opt_tuple getCMass() {
        if (_counter < _counter_threshold) return std::nullopt;

        const auto icc = 1. / static_cast<double>(_counter);
        const auto nicc = -icc;
        double _x = 0, _y = 0;
        for (int i = 0; i < _n_bins_x; i++) {
            _weights[i] = std::exp(-1. + _weights[i] * nicc);
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

        const auto x_idx = static_cast<int>(x / _bin_width);
        const auto y_idx = static_cast<int>(y / _bin_height);
        const auto idx = _n_bins_x * y_idx + x_idx;

        _weights[idx] += 1;
        _bins_x[x_idx] += x;
        _bins_y[y_idx] += y;

        _t = t;
        _init_t = t;
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

    cv::Mat _img;
};


using CentroidPtr = std::shared_ptr<CMassCalculation>;
using CentroidsVec = std::vector<CentroidPtr>;

#endif //PROJECT_CENTROID_HPP