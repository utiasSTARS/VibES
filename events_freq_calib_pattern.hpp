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

        auto d_x = static_cast<int16_t>(amplitude_x * std::cos(omega * static_cast<double>(timestamp) / 1e6 + phi));
        auto d_y = static_cast<int16_t>(amplitude_y * std::sin(omega * static_cast<double>(timestamp) / 1e6 + phi));
        while (event_store.size() < 10'000) {
            for (const auto &event: vertical_line) {
                // shift the vertical line
                dv::Event shifted_event(timestamp, event.x() + d_x, event.y() + d_y, event.polarity());
                // make sure the event is whitin the image
                if (shifted_event.x() < 0 || shifted_event.x() >= size.width || shifted_event.y() < 0 ||
                    shifted_event.y() >= size.height) {
                    continue;
                }
                event_store.push_back(shifted_event);
            }
            for (const auto &event: horizontal_line) {
                // shift the horizontal line
                dv::Event shifted_event(timestamp, event.x() + d_x, event.y() + d_y, event.polarity());
                // make sure the event is whitin the image
                if (shifted_event.x() < 0 || shifted_event.x() >= size.width || shifted_event.y() < 0 ||
                    shifted_event.y() >= size.height) {
                    continue;
                }
                event_store.push_back(shifted_event);
            }
            timestamp += 10; // 10 us
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
