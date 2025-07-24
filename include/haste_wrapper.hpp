//
// Created by viciopoli on 13/07/25.
// Optimized version with performance improvements
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
#include <condition_variable>
#include <stdexcept>
#include <sstream>
#include <queue>
#include <boost/lockfree/spsc_queue.hpp>
#include <algorithm>
#include <optional>
#include <iomanip>

#include "haste/app/command_parser.hpp"
#include "haste/tracking.hpp"
#include "estimator/iekf_sinusoid_fitter.hpp"

using TrackerPtr = std::shared_ptr<haste::HypothesisPatchTracker>;

namespace {
    constexpr double MIN_FREQUENCY = 5.0;   // Hz
    constexpr double MAX_FREQUENCY = 80.0;  // Hz
    constexpr int MAX_HARMONICS = 1;
    constexpr double TRACKER_RATE = 0.001;
    constexpr int TRACKER_MARGIN = haste::HypothesisPatchTracker::kPatchSize / 2 + 15;
    constexpr double TRACKER_MARGIN_SQ = TRACKER_MARGIN * TRACKER_MARGIN;
    constexpr size_t MAX_CENTROIDS_QUEUE = 1000;
    constexpr int MAX_AMPLITUDE = 5000;
}

//#define STORE

// Base class for template-independent code
class HasteWrapperBase {
protected:
    // Hot data - frequently accessed together for cache efficiency
    std::atomic<double> last_x_{0.0};
    std::atomic<double> last_y_{0.0};
    std::atomic<bool> run_{true};
    std::atomic<int> color_{0};

    // Threading synchronization
    std::thread tracker_thread_;
    std::condition_variable cv_;
    std::mutex cv_mutex_;

    // Tracker and fitters
    TrackerPtr tracker_;
    std::unique_ptr<IEKFSinusoidFitter> x_fitter_;
    std::unique_ptr<IEKFSinusoidFitter> y_fitter_;

    // Centroid queue with mutex (accessed less frequently)
    mutable std::mutex centroids_mutex_;
    std::queue<Centroid> centroids_queue_;

    // Cold data - less frequently accessed
    const int initial_x_;
    const int initial_y_;
    Metavision::timestamp init_time_;

#ifdef STORE
    std::ofstream file_centroid_;
#endif

public:
    HasteWrapperBase(int x, int y, double tracker_rate, Metavision::timestamp init_time,
                     const std::string &output_folder, std::unique_ptr<IEKFSinusoidFitter> x_fitter_ptr = nullptr,
                     std::unique_ptr<IEKFSinusoidFitter> y_fitter_ptr = nullptr)
            : initial_x_(x), initial_y_(y), init_time_(init_time),
              x_fitter_(std::move(x_fitter_ptr)), y_fitter_(std::move(y_fitter_ptr)) {

        tracker_ = std::make_shared<haste::HasteDifferenceStarTracker>(
                tracker_rate,
                static_cast<haste::HypothesisPatchTracker::Scalar>(x),
                static_cast<haste::HypothesisPatchTracker::Scalar>(y),
                0.0f);

        last_x_.store(tracker_->x(), std::memory_order_relaxed);
        last_y_.store(tracker_->y(), std::memory_order_relaxed);

#ifdef STORE
        file_centroid_.open(output_folder + "/centroids.txt");
#endif
    }

    virtual ~HasteWrapperBase() {
        stop();
#ifdef STORE
        file_centroid_.close();
#endif
    }

    void stop() {
        run_.store(false, std::memory_order_release);
        cv_.notify_one();
        if (tracker_thread_.joinable()) {
            tracker_thread_.join();
        }
    }

    std::optional<std::pair<double, double>> getEstimate(double t) {
        // Only lock when both fitters exist
        if (!x_fitter_ || !y_fitter_) {
            return std::nullopt;
        }

        std::lock_guard<std::mutex> lock(centroids_mutex_);
        return std::make_pair(x_fitter_->predict(t), y_fitter_->predict(t));
    }

    std::optional<std::pair<double, double>> getRelEstimate(double t) {
        if (!x_fitter_ || !y_fitter_) {
            return std::nullopt;
        }

        std::lock_guard<std::mutex> lock(centroids_mutex_);
        return std::make_pair(x_fitter_->predict_rel(t), y_fitter_->predict_rel(t));
    }

    bool getRelEstimate(float t, double &x, double &y) {
        if (!x_fitter_ || !y_fitter_) {
            return false;
        }

        std::lock_guard<std::mutex> lock(centroids_mutex_);
        x = x_fitter_->predict_rel(t);
        y = y_fitter_->predict_rel(t);
        return true;
    }

    bool getRelEstimate(Metavision::timestamp t_query, double t, double &x, double &y) {
        if (!x_fitter_ || !y_fitter_) {
            return false;
        }

        static Metavision::timestamp last_t_query = std::numeric_limits<Metavision::timestamp>::max();
        static double last_t = std::numeric_limits<double>::quiet_NaN();
        static double last_x = 0.0;
        static double last_y = 0.0;

        if (last_t_query == t_query && last_t == t) {
            x = last_x;
            y = last_y;
            return true;
        }

        std::lock_guard<std::mutex> lock(centroids_mutex_);
        x = x_fitter_->predict_rel(t);
        y = y_fitter_->predict_rel(t);

        last_t_query = t_query;
        last_t = t;
        last_x = x;
        last_y = y;

        return true;
    }


    int color() const {
        return color_.load(std::memory_order_relaxed);
    }

    void addFitters(std::unique_ptr<IEKFSinusoidFitter> x_fitter_ptr,
                    std::unique_ptr<IEKFSinusoidFitter> y_fitter_ptr) {
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        x_fitter_ = std::move(x_fitter_ptr);
        y_fitter_ = std::move(y_fitter_ptr);
    }

    bool isRunning() const {
        return run_.load(std::memory_order_acquire);
    }

    std::pair<double, double> getCurrentPosition() {
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        return {last_x_.load(std::memory_order_relaxed),
                last_y_.load(std::memory_order_relaxed)};
    }

    void getCurrentPosition(unsigned short &x, unsigned short &y) {
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        x = last_x_.load(std::memory_order_relaxed);
        y = last_y_.load(std::memory_order_relaxed);
    }

    std::optional<std::pair<double, double>> getShift() {
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        if (x_fitter_ && y_fitter_) {
            return std::make_pair(x_fitter_->getShift(), y_fitter_->getShift());
        }
        return std::nullopt;
    }

    std::pair<int, int> getInitialPosition() const {
        return {initial_x_, initial_y_};
    }

    std::queue<Centroid> getCentroids() {
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        std::queue<Centroid> result;
        result.swap(centroids_queue_);  // Efficient swap instead of copy
        return result;
    }

protected:
    void updateTrackingState(double t, double x, double y) {
        // Update atomic positions

        if (x == 0 && y == 0) return;

#ifdef STORE
        file_centroid_ << std::fixed << std::setprecision(6)
                       << t << " " << x << " " << y << "\n";
        file_centroid_.flush();
#endif

        // Add to centroid queue with size limit
        {
            std::lock_guard<std::mutex> lock(centroids_mutex_);
            centroids_queue_.emplace(t, x, y);

            // Maintain queue size limit
            while (centroids_queue_.size() > MAX_CENTROIDS_QUEUE) {
                centroids_queue_.pop();
            }

            // Update fitters
            if (x_fitter_) {
                x_fitter_->update(t, x);
                double amplitude = x_fitter_->getAmplitude();
                color_.store(static_cast<int>(std::min(255.0,
                                                       std::max(0.0, amplitude*1000 / static_cast<double>(MAX_AMPLITUDE) *
                                                                     255.0))),
                             std::memory_order_relaxed);
//                color_.store(static_cast<int>(amplitude*1000),
//                             std::memory_order_relaxed);
                last_x_.store(x_fitter_->getShift(), std::memory_order_relaxed);
            }
            if (y_fitter_) {
                y_fitter_->update(t, y);
                last_y_.store(y_fitter_->getShift(), std::memory_order_relaxed);
            }
        }
    }
};

template<typename T>
class HasteWrapper : public HasteWrapperBase {
private:
    boost::lockfree::spsc_queue<T, boost::lockfree::capacity<1000>> event_stack_;

public:
    HasteWrapper(int x, int y, double tracker_rate, Metavision::timestamp init_time = 0,
                 std::string output_folder = "output", std::unique_ptr<IEKFSinusoidFitter> x_fitter_ptr = nullptr,
                 std::unique_ptr<IEKFSinusoidFitter> y_fitter_ptr = nullptr)
            : HasteWrapperBase(x, y, tracker_rate, init_time, output_folder, std::move(x_fitter_ptr),
                               std::move(y_fitter_ptr)) {

        if constexpr (!(std::is_same_v<T, Metavision::EventCD> || std::is_same_v<T, Centroid>)) {
            throw std::invalid_argument("HasteWrapper can only be used with Metavision::EventCD or Centroid types.");
        }

        startEvents();
    }

    bool feed(const T &event) {
        if (inTracker(event)) {
            return event_stack_.push(event);
        }
        return false;
    }

    bool feed(const T &event, unsigned short &x, unsigned short &y) {
        if (inTracker(event, x, y)) {
            return event_stack_.push(event);
        }
        return false;
    }

    bool inTracker(const T &event) {
        double x = last_x_.load(std::memory_order_acquire);
        double y = last_y_.load(std::memory_order_acquire);

        if (x == 0 && y == 0) {
            return false;
        }

        // Use circular distance check for better performance
        double dx = event.x - x;
        double dy = event.y - y;
        return (dx * dx + dy * dy) <= TRACKER_MARGIN_SQ;
    }

    bool inTracker(const T &event, unsigned short &x, unsigned short &y) {
        x = last_x_.load(std::memory_order_relaxed);
        y = last_y_.load(std::memory_order_relaxed);

        if (x == 0 && y == 0) {
            return false;
        }

        // Use circular distance check for better performance
        double dx = event.x - x;
        double dy = event.y - y;
        return (dx * dx + dy * dy) <= TRACKER_MARGIN_SQ;
    }

private:

    void startEvents() {
        tracker_thread_ = std::thread([this]() {
            while (run_.load(std::memory_order_acquire)) {
                bool processed_events = false;

                event_stack_.consume_all([&](const T &event) {
                    processed_events = true;

                    // Convert timestamp to seconds
                    double current_t_sec;
                    if constexpr (std::is_same_v<T, Centroid>) {
                        current_t_sec = event.t;
                    } else {
                        current_t_sec = static_cast<double>(event.t - init_time_) / 1e6;
                    }

                    // Feed events to the tracker
                    auto update_type = tracker_->pushEvent(current_t_sec, event.x, event.y);
                    if (update_type == haste::HypothesisPatchTracker::EventUpdate::kStateEvent) {
                        updateTrackingState(tracker_->t(), tracker_->x(), tracker_->y());
                    }
                });

                // If no events were processed, wait efficiently
                if (!processed_events) {
                    std::unique_lock<std::mutex> lock(cv_mutex_);
                    cv_.wait_for(lock, std::chrono::microseconds(100),
                                 [this] { return !run_.load(std::memory_order_acquire) || !event_stack_.empty(); });
                }
            }
        });
    }
};

#endif //PROJECT_HASTE_WRAPPER_H