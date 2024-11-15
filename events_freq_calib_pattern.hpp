//
// Created by viciopoli on 14/11/24.
//

#ifndef PROJECT_EVENTS_FREQ_CALIB_PATTERN_H
#define PROJECT_EVENTS_FREQ_CALIB_PATTERN_H

#include <dv-processing/core/event.hpp>

class EventsFreqCalibPattern {
public:
    EventsFreqCalibPattern() = delete;

    EventsFreqCalibPattern(const int64_t &timestamp, const cv::Size &size, double omega, double amplitude_x,
                           double amplitude_y, double phi) :
            timestamp(timestamp), size(size), omega(omega), amplitude_x(amplitude_x), amplitude_y(amplitude_y),
            phi(phi) {

        // vertical line centered
        for (int i = 0; i < size.height; i++) {
            auto x = static_cast<int16_t>(size.width / 2);
            auto y = static_cast<int16_t>(i);
            bool polarity = true;
            dv::Event event(timestamp, x, y, polarity);
            vertical_line.push_back(event);
        }

        // horizontal line centered
        for (int i = 0; i < size.width; i++) {
            auto x = static_cast<int16_t>(i);
            auto y = static_cast<int16_t>(size.height / 2);
            bool polarity = true;
            dv::Event event(timestamp, x, y, polarity);
            horizontal_line.push_back(event);
        }
    }

    [[nodiscard]] int64_t get_timestamp() const {
        return timestamp;
    }

    [[nodiscard]] cv::Size get_size() const {
        return size;
    }

    std::optional<dv::EventStore> get_events() {
        dv::EventStore event_store;
        // event_store.reserve(9'999);

        while (event_store.size() < 10'000) {
            // Calculate d_x and d_y only once per loop iteration
            double angle = omega * static_cast<double>(timestamp) / 1e6 + phi;
            auto d_x = static_cast<int16_t>(amplitude_x * std::cos(angle));
            auto d_y = static_cast<int16_t>(amplitude_y * std::sin(angle));

            // Iterate over both lines and add events
            for (const auto &line : {vertical_line, horizontal_line}) {
                for (const auto &event : line) {
                    // Shift event and check if it is within bounds
                    int16_t new_x = event.x() + d_x;
                    int16_t new_y = event.y() + d_y;
                    if (new_x >= 0 && new_x < size.width && new_y >= 0 && new_y < size.height) {
                        event_store.emplace_back(timestamp, new_x, new_y, event.polarity());
                    }
                }
            }
            timestamp += 10;  // Increment timestamp by 1 us
        }

        return event_store;
    }

private:
    int64_t timestamp;
    cv::Size size;

    double omega = 0;
    double phi = 0;
    double amplitude_x = 0, amplitude_y = 0;

    std::vector<dv::Event> vertical_line, horizontal_line;

};

#endif //PROJECT_EVENTS_FREQ_CALIB_PATTERN_H
