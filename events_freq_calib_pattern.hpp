#ifndef PROJECT_EVENTS_FREQ_CALIB_PATTERN_H
#define PROJECT_EVENTS_FREQ_CALIB_PATTERN_H

#include <vector>
#include <cmath>
#include <optional>
#include <opencv2/core.hpp>

#include <dv-processing/core/core.hpp>

// Define EventStruct to store individual events
struct EventStruct {
    int64_t timestamp;
    int16_t x;
    int16_t y;
    bool polarity;
};

class EventsFreqCalibPattern {
public:
    EventsFreqCalibPattern() = delete;

    EventsFreqCalibPattern(const int64_t &timestamp, const cv::Size &size, double omega, double amplitude_x,
                           double amplitude_y, double phi)
            : timestamp(timestamp), size(size), omega(omega), amplitude_x(amplitude_x), amplitude_y(amplitude_y),
              phi(phi) {
        initialize_lines();
    }

    [[nodiscard]] int64_t get_timestamp() const {
        return timestamp;
    }

    [[nodiscard]] cv::Size get_size() const {
        return size;
    }

    std::optional<dv::EventStore> get_events() {
        dv::EventStore event_store;
        int delta_t = 5;
        int time_all = 0;

        while (time_all < 10'000) {  // Generate events for 10 ms
            double angle = omega * static_cast<double>(timestamp) / 1e6 + phi;
            auto d_x = static_cast<int16_t>(amplitude_x * std::sin(angle));
            auto d_y = static_cast<int16_t>(amplitude_y * std::cos(angle));
            // Generate events for both lines
            for (const auto &line: {vertical_line, horizontal_line}) {
                bool is_vertical = line.size() == vertical_line.size();

                for (const auto &event: line) {
                    int16_t new_x = event.x + d_x;
                    int16_t new_y = event.y + d_y;

                    if (is_vertical) {
                        vertical_x = new_x;
                    } else {
                        horizontal_y = new_y;
                    }

                    // Ensure the event is within bounds
                    if (new_x >= 0 && new_x < size.width && new_y >= 0 && new_y < size.height) {
                        bool is_left_half = (new_x < vertical_x) ? d_x < 0 : d_x > 0;
                        bool is_top_half = (new_y < horizontal_y) ? d_y < 0 : d_y > 0;

                        bool polarity = (is_vertical ? is_top_half : is_left_half);
                        event_store.push_back({timestamp, new_x, new_y, polarity});
                    }
                }
            }
            timestamp += delta_t;  // Increment timestamp by 1 μs
            time_all += delta_t;
        }

        return event_store;
    }

private:
    int64_t timestamp, initial_timestamp;
    cv::Size size;

    double omega = 0;
    double phi = 0;
    double amplitude_x = 0, amplitude_y = 0;

    int vertical_x = 0, horizontal_y = 0;

    std::vector<EventStruct> vertical_line, horizontal_line;

    void initialize_lines() {

        vertical_x = size.width / 2, horizontal_y = size.height / 2;
        // Create a vertical line centered at the width
        for (int i = 0; i < size.height; i++) {
            vertical_line.push_back({timestamp, static_cast<int16_t>(size.width / 2), static_cast<int16_t>(i), true});
        }

        // Create a horizontal line centered at the height
        for (int i = 0; i < size.width; i++) {
            horizontal_line.push_back(
                    {timestamp, static_cast<int16_t>(i), static_cast<int16_t>(size.height / 2), true});
        }
    }
};

#endif //PROJECT_EVENTS_FREQ_CALIB_PATTERN_H
