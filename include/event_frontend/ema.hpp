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
    CentroidEMA(double tau, Metavision::timestamp t_window) : tau_(tau), t_window_(t_window),
                                                              window_start_timestamps_(-1) {}

    /**
     * @brief Updates the centroid calculation with a new event.
     * A new centroid is computed and returned only when a time window is completed.
     * @param event The new event.
     * @return An optional containing the updated centroid and its average timestamp if a window completed.
     */
    std::optional<Metavision::EventCD> feed(const Metavision::EventCD &event) override {
        if (window_start_timestamps_ < 0) {
            window_start_timestamps_ = event.t;
        }

        event_buffers_.push_back(event);

        if (event.t - window_start_timestamps_ >= t_window_) {
            if (event_buffers_.empty()) {
                window_start_timestamps_ = -1;
                return std::nullopt;
            }

            double sum_x = 0, sum_y = 0;
            double sum_t = 0;
            for (const auto &ev: event_buffers_) {
                sum_x += ev.x;
                sum_y += ev.y;
                sum_t += ev.t;
            }
            std::array<float, 2> buffer_centroid = {
                    static_cast<float>(sum_x / event_buffers_.size()),
                    static_cast<float>(sum_y / event_buffers_.size())};

            Metavision::timestamp new_centroid_timestamp = static_cast<Metavision::timestamp>(sum_t /
                                                                                              event_buffers_.size());

            if (last_timestamps_ < 0) {
                centroids_ = buffer_centroid;
            } else {
                double delta_t = new_centroid_timestamp - last_timestamps_;
                double alpha = 1.0;
                if (tau_ > 0 && delta_t > 0) {
                    alpha = 1.0 - std::exp(-delta_t / tau_);
                }
                centroids_[0] = (1.0 - alpha) * centroids_[0] + alpha * buffer_centroid[0];
                centroids_[1] = (1.0 - alpha) * centroids_[1] + alpha * buffer_centroid[1];
            }
            last_timestamps_ = new_centroid_timestamp;

            event_buffers_.clear();
            window_start_timestamps_ = -1;

            return Metavision::EventCD{
                    static_cast<unsigned short>(centroids_[0]),
                    static_cast<unsigned short>(centroids_[1]),
                    0, // polarity is not used in this context
                    new_centroid_timestamp
            };
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
    Metavision::timestamp t_window_;
    std::array<float, 2> centroids_{};
    Metavision::timestamp last_timestamps_ = -1;
    Metavision::timestamp window_start_timestamps_;
    std::vector<Metavision::EventCD> event_buffers_;
};


#endif //PROJECT_EMA_H
