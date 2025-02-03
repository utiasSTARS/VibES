//
// Created by viciopoli on 02/02/25.
//

#ifndef PROJECT_CENTROID_HPP
#define PROJECT_CENTROID_HPP

#include "utils.hpp"

namespace harmeda {
    class CentroidCalculation {
    public:
        CentroidCalculation() = default;

        CentroidCalculation(const double time_resolution) : time_resolution(time_resolution) {}

        CentroidCalculation(const int counter_threshold) : _counter_threshold(counter_threshold) {}

        CentroidCalculation(const double time_resolution, const int counter_threshold) : time_resolution(
                time_resolution), _counter_threshold(counter_threshold) {}

        void feed(double x, double y, double t) {
            if (!almost_equal(t, _t / _counter)) {
                if (auto centroid = getCentroid(); centroid.has_value()) {
                    _centroids.emplace_back(centroid.value());
                }
                reset();
            }
            _x += x;
            _y += y;
            _t += t;
            _counter++;
        }

        void reset() {
            _x = 0;
            _y = 0;
            _t = 0;
            _counter = 0;
        }

        std::optional<std::tuple<double, double, double>> getCentroid() {
            if (_counter < _counter_threshold) {
                return std::nullopt;
            }
            return std::make_tuple(_x / _counter, _y / _counter, _t / _counter);
        }

        std::vector<std::tuple<double, double, double>> getCentroids() {
            // return centroids and reset the centroids
            auto centroids = _centroids;
            _centroids.clear();
            return centroids;
        }

        int size() {
            return _centroids.size();
        }

    private:
        bool almost_equal(double x, double y) {
            return std::abs(x - y) < time_resolution; // time resolution of 1 us
        }

        double _x = 0, _y = 0, _t = 0;
        unsigned long int _counter = 0;
        const double time_resolution = 0.000'001; // 1 us
        const int _counter_threshold = 100;

        std::vector<std::tuple<double, double, double>> _centroids;
    };
}

#endif //PROJECT_CENTROID_HPP
