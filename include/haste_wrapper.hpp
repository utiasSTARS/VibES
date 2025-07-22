//
// Created by viciopoli on 13/07/25.
//

#ifndef PROJECT_HASTE_WRAPPER_H
#define PROJECT_HASTE_WRAPPER_H

#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/base_window.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <boost/lockfree/spsc_queue.hpp>
#include <algorithm>
#include <optional>

#include "haste/app/command_parser.hpp"
#include "haste/tracking.hpp"
#include "estimator/iekf_sinusoid_fitter.hpp"

using TrackerPtr = std::shared_ptr<haste::HypothesisPatchTracker>;

namespace {
    constexpr double MIN_FREQUENCY = 5.0;   // Hz
    constexpr double MAX_FREQUENCY = 80.0;  // Hz
    constexpr int MAX_HARMONICS = 1;
    constexpr double TRACKER_RATE = 0.01;
    constexpr int TRACKER_MARGIN = haste::HypothesisPatchTracker::kPatchSize / 2 + 15;
}

#define STORE

template<typename T>
class HasteWrapper {

public:
    HasteWrapper(int x, int y, double tracker_rate, Metavision::timestamp init_time = 0,
                 std::string output_folder = "output") : initial_x_(x),
                                                         initial_y_(y),
                                                         init_time_(init_time) {
        if constexpr (!(std::is_same_v<T, Metavision::EventCD> || std::is_same_v<T, Centroid>)) {
            throw std::invalid_argument("HasteWrapper can only be used with Metavision::EventCD or Centroid types.");
        }

        tracker_ = std::make_shared<haste::HasteDifferenceStarTracker>(
                tracker_rate, static_cast<haste::HypothesisPatchTracker::Scalar>(x),
                static_cast<haste::HypothesisPatchTracker::Scalar>(y), 0.0f);

        last_x_ = tracker_->x();
        last_y_ = tracker_->y();

#ifdef STORE
        file_centroid_ = std::ofstream(output_folder + "/centroids.txt");
#endif
        startEvents();
    }

    ~HasteWrapper() {
        stop();
#ifdef STORE
        file_centroid_.close();
#endif
    }

    bool feed(const T &event) {
//        std::lock_guard<std::mutex> lock(mtx_);
//        event_stack_.push(event);
        if (inTracker(event)) {
            event_stack_.push(event);
            return true;
        }
        return false;
    }

    bool inTracker(const T &event) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (last_x_ == 0 && last_y_ == 0)
            return false;

        return event.x >= last_x_ - TRACKER_MARGIN &&
               event.x <= last_x_ + TRACKER_MARGIN &&
               event.y >= last_y_ - TRACKER_MARGIN &&
               event.y <= last_y_ + TRACKER_MARGIN;
    }


    void stop() {
        run_.store(false);
        if (tracker_thread_.joinable()) {
            tracker_thread_.join();
        }
    }

    std::optional<std::pair<double, double>> getEstimate(double t) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!x_fitter || !y_fitter) {
            return std::nullopt;  // No fitters available
        }
        return std::make_pair(x_fitter->predict(t), y_fitter->predict(t));
    }

    std::optional<std::pair<double, double>> getRelEstimate(double t) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!x_fitter || !y_fitter) {
            return std::nullopt;  // No fitters available
        }
        return std::make_pair(x_fitter->predict_rel(t), y_fitter->predict_rel(t));
    }

    int color() const {
        return color_.load();
    }

    void addFitters(std::unique_ptr<IEKFSinusoidFitter> x_fitter_ptr,
                    std::unique_ptr<IEKFSinusoidFitter> y_fitter_ptr) {
        std::lock_guard<std::mutex> lock(mtx_);
        x_fitter = std::move(x_fitter_ptr);
        y_fitter = std::move(y_fitter_ptr);
    }


    // Check if tracker is running
    bool isRunning() const {
        return run_.load();
    }

    // Get current tracker position (thread-safe)
    std::pair<double, double> getCurrentPosition() {
        std::lock_guard<std::mutex> lock(mtx_);
        return {last_x_, last_y_};
    }

    std::optional<std::pair<double, double>> getShift() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (x_fitter && y_fitter) {
            std::pair<double, double> pair(x_fitter->getShift(), y_fitter->getShift());
            return pair;
        }
        return std::nullopt;  // No shift available
    }

    // Get initial position
    std::pair<int, int> getInitialPosition() const {
        return {initial_x_, initial_y_};
    }

    std::vector<Centroid> getCentroids() {
        std::lock_guard<std::mutex> lock(mtx_);
        // copy the centroids vector to return a snapshot
        if (centroids_.empty()) {
            return {};  // Return empty vector if no centroids
        }
        // Return a copy of the centroids
        std::vector<Centroid> centroids_copy = centroids_;
        // Clear the centroids vector to avoid memory leaks
        centroids_.clear();
        return centroids_copy;
    }

private:
    TrackerPtr tracker_;
    std::thread tracker_thread_;
    boost::lockfree::spsc_queue<T, boost::lockfree::capacity<1000>> event_stack_;
    std::atomic<bool> run_{true};
    std::atomic<int> color_{0};
    mutable std::mutex mtx_;  // Made mutable for const methods
    std::unique_ptr<IEKFSinusoidFitter> x_fitter, y_fitter;

    Metavision::timestamp init_time_{0};

    std::ofstream file_centroid_;

    // Tracking state
    double last_x_{0.0};
    double last_y_{0.0};

    // Initial position
    const int initial_x_;
    const int initial_y_;

    std::vector<Centroid> centroids_;

    static constexpr int MAX_AMPLITUDE = 5;  // Made const and static



    void startEvents() {
        // Start the tracker thread
        tracker_thread_ = std::thread([this]() {
            while (run_.load()) {
                event_stack_.consume_all([this](const T &event) {
                    // Convert timestamp to seconds relative to first event
                    double current_t_sec = 0;
                    if constexpr (std::is_same_v<T, Centroid>) {
                        current_t_sec = event.t;
                    } else {
                        current_t_sec = static_cast<double>(event.t - init_time_) / 1e6;
                    }
                    // Feed events to the tracker
                    auto update_type = tracker_->pushEvent(current_t_sec, event.x, event.y);
                    if (update_type == haste::HypothesisPatchTracker::EventUpdate::kStateEvent) {
                        std::lock_guard<std::mutex> lock(mtx_);

                        // Update tracking state
                        last_x_ = tracker_->x();
                        last_y_ = tracker_->y();

                        if (last_x_ == 0 && last_y_ == 0) { return; }

#ifdef STORE
                        file_centroid_ << std::fixed << std::setprecision(4)
                                       << tracker_->t() << " " << last_x_ << " " << last_y_ << "\n";
#endif

                        centroids_.emplace_back(
                                tracker_->t(), last_x_, last_y_
                        );

                        // Update fitters if available
                        if (x_fitter) {
                            x_fitter->update(tracker_->t(), tracker_->x());
                            // Calculate color based on amplitude
                            double amplitude = x_fitter->getAmplitude();
                            color_ = static_cast<int>(std::min(
                                    255.0,
                                    std::max(0.0, amplitude / static_cast<double>(MAX_AMPLITUDE) * 255.0)));
                        }
                        if (y_fitter) {
                            y_fitter->update(tracker_->t(), tracker_->y());
                        }
                    }
                });

                // Small sleep to prevent 100% CPU usage
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }


};


#endif //PROJECT_HASTE_WRAPPER_H