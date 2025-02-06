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

        CentroidCalculation(const int counter_threshold) : _counter_threshold(counter_threshold) {}

        std::optional<std::tuple<double, double, Time>> feed(double x, double y, Time t) {
            if (_counter > 0 && t != _t/_counter) {
                if (auto centroid = getCentroid(); centroid.has_value()) {
                    _centroids.emplace_back(centroid.value());

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

        std::optional<std::tuple<double, double, Time>> getCentroid() {
            if (_counter < _counter_threshold) {
                return std::nullopt;
            }
            return std::make_tuple(
                    _x / static_cast<double>(_counter),
                    _y / static_cast<double>(_counter),
                    _t
            );
        }

        std::vector<std::tuple<double, double, double>> getCentroids() {
            // return centroids and reset the centroids
            auto centroids = _centroids;
            _centroids.clear();
            return centroids;
        }

        size_t size() {
            return _centroids.size();
        }

    private:
        double _x = 0, _y = 0;
        Time _t = 0;
        unsigned long int _counter = 0;
        const int _counter_threshold = 100;

        std::vector<std::tuple<double, double, double>> _centroids;
    };
}

#endif //PROJECT_CENTROID_HPP
