#ifndef PROJECT_CENTROID_HPP
#define PROJECT_CENTROID_HPP

#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <tuple>
#include <optional>
#include "event_frontend/centroid_base.h"


class CMassCalculation : CentroidBase {
public:
    CMassCalculation(int width, int height, int counter_threshold = 50, double time_window = 1e-4,
                     int n_bins_x = 10, int n_bins_y = 10)
            : _width(width), _height(height),
              _counter_threshold(counter_threshold), _time_window(time_window),
              _n_bins_x(n_bins_x), _n_bins_y(n_bins_y),
              _bin_width(width / n_bins_x), _bin_height(height / n_bins_y) {

        // Initialize bin structures
        _sum_x.resize(_n_bins_x * _n_bins_y, 0.0);
        _sum_y.resize(_n_bins_x * _n_bins_y, 0.0);
        _sum_t.resize(_n_bins_x * _n_bins_y, 0.0);
        _sum_w.resize(_n_bins_x * _n_bins_y, 0.0);

        _event_queues.resize(_n_bins_x * _n_bins_y);
        _img = cv::Mat::zeros(_height, _width, CV_8UC3);
    }

    std::optional<Metavision::EventCD> feed(const Metavision::EventCD &event) override {
        _counter++;

        if (_init_t < 0) _init_t = event.t; // Initialize start time

        // Determine bin index
        int x_idx = event.x / _bin_width;
        int y_idx = event.y / _bin_height;
        int bin_idx = _n_bins_x * y_idx + x_idx;

        // Add event to the bin's queue
        _event_queues[bin_idx].emplace_back(event.x, event.y, event.t);

        // Update running sums for this bin
        _sum_x[bin_idx] += event.x;
        _sum_y[bin_idx] += event.y;
        _sum_t[bin_idx] += event.t;
        _sum_w[bin_idx] += 1;

        // Remove old events (sliding window)
        while (!_event_queues[bin_idx].empty() &&
               (event.t - std::get<2>(_event_queues[bin_idx].front()) > _time_window)) {
            auto [old_x, old_y, old_t] = _event_queues[bin_idx].front();
            _event_queues[bin_idx].pop_front();
            _sum_x[bin_idx] -= old_x;
            _sum_y[bin_idx] -= old_y;
            _sum_t[bin_idx] -= old_t;
            _sum_w[bin_idx] -= 1;
            _counter--;
        }

        // Check if it's time to generate a new centroid
        if (event.t - _last_centroid_time >= _time_window) {
            auto centroid = getCMass(event.t);

            // Remove the most recent time_window / 4 events after computing centroid
//            double discard_threshold = t - (_time_window / 4);
//            for (auto &queue: _event_queues) {
//                while (!queue.empty() && std::get<2>(queue.back()) >= discard_threshold) {
//                    queue.pop_back();
//                }
//            }

            // Update last centroid generation time

            return centroid;
        }
        return std::nullopt;
    }

    std::optional<Metavision::EventCD> getCMass(double t) {
        if (_counter < _counter_threshold) return std::nullopt;

        double total_x = 0.0, total_y = 0.0, total_t = 0.0, total_w = 0.0;
        double alpha = 0.1; // Smoothing factor, adjust as needed

        for (int i = 0; i < _n_bins_x * _n_bins_y; i++) {
            if (_sum_w[i] > 0) {
                total_w += _sum_w[i];
                total_x = alpha * (_sum_x[i] / _sum_w[i]) + (1 - alpha) * total_x;
                total_y = alpha * (_sum_y[i] / _sum_w[i]) + (1 - alpha) * total_y;
                total_t = alpha * (_sum_t[i] / _sum_w[i]) + (1 - alpha) * total_t;
            }
        }
        _last_centroid_time = t;

        return Metavision::EventCD{
                static_cast<unsigned short>(total_x),
                static_cast<unsigned short>(total_y),
                0, // polarity is not used in this context
                static_cast<Metavision::timestamp>(total_t)
        };
    }

private:
    int _width, _height;
    int _n_bins_x, _n_bins_y;
    int _bin_width, _bin_height;
    int _counter_threshold;
    double _time_window;
    double _init_t = -1;

    std::vector<std::deque<std::tuple<int, int, double>>> _event_queues;
    std::vector<double> _sum_x, _sum_y, _sum_t, _sum_w;

    cv::Mat _img;

    double _last_centroid_time = 0;
    int _counter = 0;
};

using CentroidPtr = std::shared_ptr<CMassCalculation>;
using CentroidsVec = std::vector<CentroidPtr>;

#endif // PROJECT_CENTROID_HPP
