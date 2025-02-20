//
// Created by viciopoli on 17/02/25.
//

#ifndef PROJECT_FREQ_FILTER_H
#define PROJECT_FREQ_FILTER_H

#include <memory>
#include <limits>
#include <vector>

class FreqFilter {
public:
    FreqFilter() = delete;

    FreqFilter(int width, int height, double f_min, double f_max)
            : _width(width), _dt_min(1e6 / f_max), _dt_max(1e6 / f_min),
              _freq_map(std::make_unique<long long[]>(width * height)) {

        // Initialize with -infinity (avoids zero-initialization overhead)
        std::fill_n(_freq_map.get(), width * height, -std::numeric_limits<double>::infinity());
    }

    bool check(unsigned short x, unsigned short y, long long t) {
        auto &last_t = _freq_map[x + y * _width];
        const long long dt_t = t - last_t;
        last_t = t;
        return dt_t >= _dt_min && dt_t <= _dt_max;
    }

//    // The check function now inspects a region around (x, y).
//    bool check(short x, short y, double t) {
//        // Iterate over the square region centered at (x, y)
//        for (int dx = -_region_radius; dx <= _region_radius; ++dx) {
//            for (int dy = -_region_radius; dy <= _region_radius; ++dy) {
//                int nx = x + dx;
//                int ny = y + dy;
//                if (nx < 0 || nx >= _width || ny < 0 || ny >= _height)
//                    continue; // Skip out-of-bounds pixels
//
//                double &last_t = _freq_map[nx + ny * _width];
//                double dt = t - last_t;
//                if (dt > 0) {
//                    double freq = 1.0 / dt;
//                    if (freq >= _f_min && freq <= _f_max) {
//                        last_t = t;  // Update this pixel's timestamp.
//                        return true;
//                    }
//                }
//            }
//        }
//        // Optionally, if none in the region passed the filter, you can update the center pixel.
//        double &center_last_t = _freq_map[x + y * _width];
//        center_last_t = t;
//        return false;
//    }

private:
    int _width;
    long long _dt_min, _dt_max;
    std::unique_ptr<long long[]> _freq_map; // Faster than vector, safer than raw pointers
};

#endif //PROJECT_FREQ_FILTER_H
