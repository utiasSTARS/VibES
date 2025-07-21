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
}

class HasteWrapper {

public:
    HasteWrapper(int x, int y, double tracker_rate) : initial_x_(x), initial_y_(y) {
        counter_++;
        tracker_ = std::make_shared<haste::HasteDifferenceStarTracker>(
                tracker_rate, static_cast<haste::HypothesisPatchTracker::Scalar>(x),
                static_cast<haste::HypothesisPatchTracker::Scalar>(y), 0.0f);

        if (!tracker_) {
            throw std::runtime_error("Tracker initialization failed");
        }

        // Start the tracker thread
        tracker_thread_ = std::thread([this]() {
            while (run_.load()) {
                event_stack_.consume_all([this](const Metavision::EventCD &event) {
                    // Convert timestamp to seconds relative to first event
                    float current_t_sec = static_cast<float>(event.t) / 1e6f;

                    // Feed events to the tracker
                    auto update_type = tracker_->pushEvent(current_t_sec, event.x, event.y);
                    if (update_type == haste::HypothesisPatchTracker::EventUpdate::kStateEvent) {
                        std::lock_guard<std::mutex> lock(mtx_);

                        // Update tracking state
                        last_update_time_ = current_t_sec;
                        last_x_ = tracker_->x();
                        last_y_ = tracker_->y();

                        centroids_.emplace_back(
                                tracker_->t(), last_x_, last_y_
                        );

                        // Update fitters if available
                        if (x_fitter || y_fitter) {
                            x_fitter->update(tracker_->t(), tracker_->x());
                            // Calculate color based on amplitude
                            double amplitude = x_fitter->getAmplitude();
                            color_ = static_cast<int>(std::min(
                                    255.0,
                                    std::max(0.0, amplitude / static_cast<double>(MAX_AMPLITUDE) * 255.0)));

                            y_fitter->update(tracker_->t(), tracker_->y());
                        }
                    }
                });

                // Small sleep to prevent 100% CPU usage
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }

    ~HasteWrapper() {
        stop();
    }

    void feed(const Metavision::EventCD &event) {
        // Push event to the lock-free stack
        if (!event_stack_.push(event)) {
            // Stack is full - could log this or handle overflow
            // For now, just drop the event
//            std::cerr << "Event stack is full, dropping event." << std::endl;
        }
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
        if (!tracker_) {
            return {0.0, 0.0};
        }
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

    // Get tracker time
    double getTrackerTime() {
        std::lock_guard<std::mutex> lock(mtx_);
        return last_update_time_;
    }

    // Get initial position
    std::pair<int, int> getInitialPosition() const {
        return {initial_x_, initial_y_};
    }

    // Get raw tracker pointer (use with caution)
    std::shared_ptr<haste::HypothesisPatchTracker> getTracker() {
        return tracker_;
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
    boost::lockfree::spsc_queue<Metavision::EventCD, boost::lockfree::capacity<10000>> event_stack_;
    std::atomic<bool> run_{true};
    std::atomic<int> color_{0};
    std::atomic<int> error_count_{0};
    std::atomic<bool> state_updated_{false};
    mutable std::mutex mtx_;  // Made mutable for const methods
    std::unique_ptr<IEKFSinusoidFitter> x_fitter, y_fitter;

    // Tracking state
    double last_update_time_{0.0};
    double last_x_{0.0};
    double last_y_{0.0};

    // Initial position
    const int initial_x_;
    const int initial_y_;

    std::vector<Centroid> centroids_;

    static int counter_;

    static constexpr int MAX_AMPLITUDE = 5;  // Made const and static
};

int HasteWrapper::counter_ = 0;


#endif //PROJECT_HASTE_WRAPPER_H