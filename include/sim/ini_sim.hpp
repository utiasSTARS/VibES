#ifndef PROJECT_EVENTS_FREQ_CALIB_PATTERN_H
#define PROJECT_EVENTS_FREQ_CALIB_PATTERN_H

#include <vector>
#include <cmath>
#include <optional>
#include <opencv2/core.hpp>

#include <dv-processing/core/core.hpp>
#include <dv-processing/io/camera_input_base.hpp>
#include "../utils.hpp"


class EventsFreqCalibPattern : public dv::io::CameraInputBase {
public:
    EventsFreqCalibPattern() = delete;

    EventsFreqCalibPattern(const int64_t &timestamp, const cv::Size &size, double omega, double amplitude_x,
                           double amplitude_y, double phi, bool noise = true)
            : timestamp(timestamp), size(size), omega(omega), amplitude_x(amplitude_x), amplitude_y(amplitude_y),
              phi(phi), delta_t(static_cast<int>((500'000 / rad2Hz(omega)) / 10)), noise(noise) {
        initialize_lines();

        // generate random noise over the image plane

        x_dist = std::uniform_int_distribution<int16_t>(0, size.width - 1);
        y_dist = std::uniform_int_distribution<int16_t>(0, size.height - 1);
        polarity_dist = std::uniform_int_distribution<int>(0, 1);
        t_dist = std::uniform_int_distribution<int>(0, 10);

    }

    [[nodiscard]] int64_t get_timestamp() const {
        return timestamp;
    }

    [[nodiscard]] cv::Size get_size() const {
        return size;
    }

    std::optional<dv::cvector<dv::IMU>> getNextImuBatch() override {
        return std::nullopt;
    }

    std::optional<dv::cvector<dv::Trigger>> getNextTriggerBatch() override {
        return std::nullopt;
    }

    [[nodiscard]] std::optional<cv::Size> getEventResolution() const override {
        return size;
    }

    [[nodiscard]] std::optional<cv::Size> getFrameResolution() const override {
        return std::nullopt;
    }

    [[nodiscard]] bool isEventStreamAvailable() const override {
        return true;
    }

    [[nodiscard]] bool isFrameStreamAvailable() const override {
        return false;
    }

    [[nodiscard]] bool isImuStreamAvailable() const override {
        return false;
    }

    [[nodiscard]] bool isTriggerStreamAvailable() const override {
        return false;
    }

    [[nodiscard]] std::string getCameraName() const override {
        return "EventsFreqCalibPattern";
    }

    [[nodiscard]] bool isRunning() const override {
        return running;
    }

    std::optional<dv::Frame> getNextFrame() override {
        return std::nullopt;
    }

    void shiftX(double x) {
        shift_x = x;
    }

    void shiftY(double y) {
        shift_y = y;
    }


    void stop() {
        running = false;
    }

    std::optional<dv::EventStore> getNextEventBatch() override {
        dv::EventStore event_store;
        int time_all = 0;

        while (time_all < 10'000) {  // Generate events for 10 ms
            double angle = omega * static_cast<double>(timestamp) / 1e6 + phi;
            auto d_x = static_cast<int16_t>(amplitude_x * std::sin(angle)) + shift_x;
            auto d_y = static_cast<int16_t>(amplitude_y * std::cos(angle)) + shift_y;
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
            if (noise)generate_noise_events(timestamp, delta_t, event_store);
            timestamp += delta_t;  // Increment timestamp by 1 μs
            time_all += delta_t;
        }

        return event_store;
    }

private:
    bool noise = true;
    bool running = true;
    int64_t timestamp, initial_timestamp{};
    cv::Size size;
    int delta_t = 1;  // 1 μs

    double omega = 0;
    double phi = 0;
    double amplitude_x = 0, amplitude_y = 0;

    int vertical_x = 0, horizontal_y = 0;

    double shift_x = 0, shift_y = 0;

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


    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<int16_t> x_dist;
    std::uniform_int_distribution<int16_t> y_dist;
    std::uniform_int_distribution<int> t_dist;
    std::uniform_int_distribution<int> polarity_dist;

    void generate_noise_events(int64_t start_time, int64_t delta_time, dv::EventStore &events) {


        // Generate events
        for (int i = 0; i < delta_time; ++i) {
            for (int j = 0; j < t_dist(gen); ++j) {
                int16_t x = x_dist(gen); // Random x-coordinate
                int16_t y = y_dist(gen); // Random y-coordinate
                bool polarity = polarity_dist(gen) == 1; // Random polarity

                events.push_back({start_time + i, x, y, polarity});
            }
        }

    }
};

#endif //PROJECT_EVENTS_FREQ_CALIB_PATTERN_H
