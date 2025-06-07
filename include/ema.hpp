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

/**
 * @brief Calculates the exponential moving average (EMA) of centroids for event data.
 * This class maintains separate centroids for different polarities.
 */
class CentroidEMA {
public:
    /**
     * @brief Initializes the CentroidEMA calculator.
     * @param tau The time constant for the EMA. A smaller tau gives more weight to recent events.
     * @param t_window The time window in microseconds to group events together.
     */
    CentroidEMA(double tau, Metavision::timestamp t_window) : tau_(tau), t_window_(t_window) {}

    /**
     * @brief Updates the centroid calculation with a new event.
     * A new centroid is computed and returned only when a time window is completed.
     * @param event The new event.
     * @return An optional containing the updated centroid for the event's polarity if a window completed.
     */
    std::optional<std::array<float, 2>> update(const Metavision::EventCD &event) {
        int p = 0; //event.p;

        if (window_start_timestamps_.find(p) == window_start_timestamps_.end()) {
            window_start_timestamps_[p] = event.t;
        }

        event_buffers_[p].push_back(event);

        if (event.t - window_start_timestamps_[p] >= t_window_) {
            if (event_buffers_[p].empty()) {
                window_start_timestamps_.erase(p);
                return std::nullopt;
            }

            double sum_x = 0, sum_y = 0;
            for (const auto &ev: event_buffers_[p]) {
                sum_x += ev.x;
                sum_y += ev.y;
            }
            std::array<float, 2> buffer_centroid = {
                    static_cast<float>(sum_x / event_buffers_[p].size()),
                    static_cast<float>(sum_y / event_buffers_[p].size())};

            Metavision::timestamp new_centroid_timestamp = event_buffers_[p].back().t;

            if (centroids_.find(p) == centroids_.end()) {
                centroids_[p] = buffer_centroid;
            } else {
                double delta_t = new_centroid_timestamp - last_timestamps_[p];
                double alpha = 1.0;
                if (tau_ > 0 && delta_t > 0) {
                    alpha = 1.0 - std::exp(-delta_t / tau_);
                }
                centroids_[p][0] = (1.0 - alpha) * centroids_[p][0] + alpha * buffer_centroid[0];
                centroids_[p][1] = (1.0 - alpha) * centroids_[p][1] + alpha * buffer_centroid[1];
            }
            last_timestamps_[p] = new_centroid_timestamp;

            event_buffers_[p].clear();
            window_start_timestamps_.erase(p);
            prev_centroids_ = centroids_[p];

            return centroids_[p];
        }

        return prev_centroids_;
    }

    /**
     * @brief Gets the current centroids.
     * @return A const reference to the map of centroids.
     */
    [[nodiscard]] const std::map<int, std::array<float, 2>> &get_centroids() const {
        return centroids_;
    }

private:
    double tau_;
    Metavision::timestamp t_window_;
    std::map<int, std::array<float, 2>> centroids_;
    std::optional<std::array<float, 2>> prev_centroids_ = std::nullopt;
    std::map<int, Metavision::timestamp> last_timestamps_;
    std::map<int, Metavision::timestamp> window_start_timestamps_;
    std::map<int, std::vector<Metavision::EventCD>> event_buffers_;
};


#endif //PROJECT_EMA_H
