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

// Define EventStruct to store individual events
struct EventStruct {
    int64_t timestamp;
    int16_t x;
    int16_t y;
    bool polarity;
};

enum EventRepresentation {
    BINARY = 0,
    COUNT = 1,
    TS = 2,
    AVERAGE_TS = 3
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


template<typename T>
std::string fp2str(const T a_value, const int n = 6) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return std::move(out).str();
}

// TIME CLASS
class Time {
public:
    template<typename T>
    Time(T time, double time_resolution = 1e-6) : _resolution(time_resolution) {
        static_assert(std::is_arithmetic_v<T>, "Time can only be constructed from arithmetic types.");
        if constexpr (std::is_integral_v<T>) {
            _time = time / 1e6;
        } else {
            _time = static_cast<double>(time);
        }
    }

    // Member arithmetic operators for Time with Time operands.
    Time operator+(const Time &other) const {
        double new_res = _choose_resolution(other);
        return Time(_time + other._time, new_res);
    }

    Time operator-(const Time &other) const {
        double new_res = _choose_resolution(other);
        return Time(_time - other._time, new_res);
    }

    Time operator*(const Time &other) const {
        double new_res = _choose_resolution(other);
        return Time(_time * other._time, new_res);
    }

    Time operator/(const Time &other) const {
        double new_res = _choose_resolution(other);
        return Time(_time / other._time, new_res);
    }

    // Compound assignment operators for Time with Time operands.
    Time &operator+=(const Time &other) {
        _time += other._time;
        _resolution = _choose_resolution(other);
        return *this;
    }

    Time &operator-=(const Time &other) {
        _time -= other._time;
        _resolution = _choose_resolution(other);
        return *this;
    }

    // Overloads for operations with double
    // These assume the double is in the same units as _time.
    Time operator+(double val) const {
        return Time(_time + val, _resolution);
    }

    Time operator-(double val) const {
        return Time(_time - val, _resolution);
    }

    Time operator*(double val) const {
        return Time(_time * val, _resolution);
    }

    Time operator/(double val) const {
        return Time(_time / val, _resolution);
    }

    // Compound assignment operators with double.
    Time &operator+=(double val) {
        _time += val;
        return *this;
    }

    Time &operator-=(double val) {
        _time -= val;
        return *this;
    }

    Time &operator*=(double val) {
        _time *= val;
        return *this;
    }

    Time &operator/=(double val) {
        _time /= val;
        return *this;
    }

    // Comparison operators (using fuzzy equality for ==)
    bool operator==(const Time &other) const {
        return std::abs(_time - other._time) < _choose_resolution(other);
    }

    bool operator!=(const Time &other) const {
        return !(*this == other);
    }

    bool operator<(const Time &other) const {
        return _time < other._time;
    }

    bool operator>(const Time &other) const {
        return _time > other._time;
    }

    bool operator<=(const Time &other) const {
        return _time <= other._time;
    }

    bool operator>=(const Time &other) const {
        return _time >= other._time;
    }

    // Conversion operator to double.
    explicit operator double() const {
        return _time;
    }

    // Friend for output.
    friend std::ostream &operator<<(std::ostream &os, const Time &time) {
        os << time._time << " s (resolution: " << time._resolution << ")";
        return os;
    }

    // Friend functions to allow double on the left-hand side.
    friend Time operator+(double lhs, const Time &rhs) {
        return Time(lhs + rhs._time, rhs._resolution);
    }

    friend Time operator-(double lhs, const Time &rhs) {
        return Time(lhs - rhs._time, rhs._resolution);
    }

    friend Time operator*(double lhs, const Time &rhs) {
        return Time(lhs * rhs._time, rhs._resolution);
    }

    friend Time operator/(double lhs, const Time &rhs) {
        return Time(lhs / rhs._time, rhs._resolution);
    }

private:
    double _time;         // Underlying time in seconds.
    double _resolution;   // Tolerance for equality comparisons.

    // Helper: choose the lower (i.e., finer) resolution.
    double _choose_resolution(const Time &other) const {
        return (other._resolution < _resolution ? other._resolution : _resolution);
    }
};


#endif //PROJECT_UTILS_HPP
