//
// Created by viciopoli on 06/06/25.
//

#ifndef PROJECT_EMA_H
#define PROJECT_EMA_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <atomic>
#include <optional>
#include "event_frontend/centroid_base.h"

/**
 * @brief Calculates the exponential moving average (EMA) of centroids for event data.
 * This class maintains separate centroids for different polarities.
 */
class CentroidEMA : CentroidBase {
public:
    /**
     * @brief Initializes the CentroidEMA calculator.
     * @param tau The time constant for the EMA. A smaller tau gives more weight to recent events.
     * @param t_window The time window in microseconds to group events together.
     */
    CentroidEMA(double tau, Metavision::timestamp t_window, int w, int h) : tau_(tau), t_window_(t_window),
                                                                            window_start_timestamps_(-1),
                                                                            density_(h * w) {}

    /**
     * @brief Updates the centroid calculation with a new event.
     * A new centroid is computed and returned only when a time window is completed.
     * @param event The new event.
     * @return An optional containing the updated centroid and its average timestamp if a window completed.
     */
    std::optional<Centroid> feed(const Metavision::EventCD &event) override {
        if (window_start_timestamps_ < 0) {
            window_start_timestamps_ = event.t;
        }

        // check if events are ordered by time
        if (last_timestamps_ >= 0 && event.t < last_timestamps_) {
            std::cerr << "Warning: Non-increasing event timestamps detected. Previous: "
                      << last_timestamps_ << ", Current: " << event.t << std::endl;
            throw std::runtime_error("Event timestamps are not increasing.");
        }

        event_buffers_.push_back(event);

        if (event.t - window_start_timestamps_ >= t_window_) {
            double sum_x = 0, sum_y = 0, sum_t = 0;
            for (const auto &ev: event_buffers_) {
                sum_x += ev.x;
                sum_y += ev.y;
                sum_t += (ev.t * 1e-6); // Convert timestamp to seconds
            }
            std::array<float, 2> buffer_centroid = {
                    static_cast<float>(sum_x / event_buffers_.size()),
                    static_cast<float>(sum_y / event_buffers_.size())};

            auto new_centroid_timestamp = static_cast<long long>((sum_t /
                                                                  event_buffers_.size()) *
                                                                 1e6);

            if (last_timestamps_ < 0) {
                centroids_ = buffer_centroid;
            } else {
                double delta_t = new_centroid_timestamp - last_timestamps_;
                double alpha = 1.0;
                if (tau_ > 0 && delta_t > 0) {
                    auto d = 500. * static_cast<double>(event_buffers_.size()) / (density_);
                    alpha = d * (1.0 - std::exp(-delta_t / tau_));
                }
//                double count_weight = 1.0 - std::exp(-event_buffers_.size() / (0.9 * density_));
//                alpha *= count_weight; // Adjust alpha based on event count
                centroids_[0] = (1.0 - alpha) * centroids_[0] + alpha * buffer_centroid[0];
                centroids_[1] = (1.0 - alpha) * centroids_[1] + alpha * buffer_centroid[1];
            }
            last_timestamps_ = new_centroid_timestamp;

            event_buffers_.clear();
            window_start_timestamps_ = -1;

//            Metavision::EventCD new_event{
//                    static_cast<unsigned short>(centroids_[0]),
//                    static_cast<unsigned short>(centroids_[1]),
//                    0, // polarity is not used in this context
//                    new_centroid_timestamp
//            };
            return Centroid(new_centroid_timestamp, centroids_[0], centroids_[1]);
        }

        return std::nullopt;
    }

    /**
     * @brief Gets the current centroids.
     * @return A const reference to the map of centroids.
     */
    [[nodiscard]] const std::array<float, 2> &get_centroids() const {
        return centroids_;
    }

private:
    double tau_;
    int density_;
    Metavision::timestamp t_window_;
    std::array<float, 2> centroids_{};
    Metavision::timestamp last_timestamps_ = -1;
    Metavision::timestamp window_start_timestamps_ = -1;
    std::vector<Metavision::EventCD> event_buffers_;
};


#endif //PROJECT_EMA_H
