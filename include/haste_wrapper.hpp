/**
 * @file haste_wrapper.h
 * @brief Thread-safe wrapper for the HASTE tracker with integrated Kalman Filtering.
 *
 * This file implements a high-performance wrapper around the HASTE (Hypothesis-based
 * Algorithm for Super-fast Tracking of Events) library. It uses a producer-consumer
 * architecture to decouple high-frequency event ingestion from the tracking logic.
 *
 * Key Features:
 * - **Lock-Free Buffering:** Uses a Single-Producer Single-Consumer (SPSC) queue
 * to ingest events from the camera callback without blocking.
 * - **Asynchronous Processing:** Runs the HASTE tracker in a dedicated thread.
 * - **Integrated Estimation:** Feeds tracker outputs into an IEKF (Iterated Extended
 * Kalman Filter) to estimate smooth trajectory parameters (amplitude, frequency).
 * - **Spatial Filtering:** Filters incoming events spatially to only process
 * those relevant to the target, reducing computational load.
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

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
#include "estimator/iekf_sinusoid_fitter_multi_harmonic.hpp"

// Define shared pointer type for convenience
using TrackerPtr = std::shared_ptr<haste::HypothesisPatchTracker>;

namespace {
    // Tracking Parameters
    constexpr double MIN_FREQUENCY = 5.0;   // Hz
    constexpr double MAX_FREQUENCY = 80.0;  // Hz
    constexpr int MAX_HARMONICS = 1;
    constexpr double TRACKER_RATE = 0.001;  // Temporal resolution for tracker updates
    constexpr int TRACKER_MARGIN = haste::HypothesisPatchTracker::kPatchSize / 2 + 15; // ROI radius
    constexpr double TRACKER_MARGIN_SQ = TRACKER_MARGIN * TRACKER_MARGIN; // ROI radius squared for fast comparison
    constexpr size_t MAX_CENTROIDS_QUEUE = 1000;
    constexpr int MAX_AMPLITUDE = 5000;
}


/**
 * @class HasteWrapperBase
 * @brief Base class containing the tracking logic independent of the event type.
 *
 * This class manages the state of the tracker, the IEKF estimators, and the
 * dedicated processing thread. It holds the "Hot Data" (atomic coordinates)
 * that need to be accessed frequently by the UI or other consumers.
 */
class HasteWrapperBase {
protected:
    // --- Hot Data ---
    // atomic variables for lock-free access to the latest tracker position
    std::atomic<double> last_x_{0.0};
    std::atomic<double> last_y_{0.0};
    std::atomic<bool> run_{true};
    std::atomic<int> color_{0};

    // --- Synchronization ---
    std::thread tracker_thread_;
    std::condition_variable cv_;
    std::mutex cv_mutex_;

    // --- Tracking Engines ---
    TrackerPtr tracker_;                           ///< The core HASTE tracker instance.
    std::unique_ptr<IEKFSinusoidFitter> x_fitter_; ///< IEKF for X-axis motion.
    std::unique_ptr<IEKFSinusoidFitter> y_fitter_; ///< IEKF for Y-axis motion.

    // --- Data Storage ---
    mutable std::mutex centroids_mutex_;
    std::queue<Centroid> centroids_queue_; ///< History of tracked centroids for initialization/debugging.

    // --- Configuration ---
    const int initial_x_;
    const int initial_y_;
    Metavision::timestamp init_time_;

    bool NUFFT_not_init = true; ///< Flag indicating if the NUFFT estimator has been run yet.

public:
    /**
     * @brief Constructor for the base wrapper.
     * @param x Initial X position.
     * @param y Initial Y position.
     * @param tracker_rate Temporal update rate for the tracker.
     * @param init_time Timestamp of initialization (t0).
     * @param output_folder Path for logging (unused in current impl).
     * @param x_fitter_ptr Optional pre-configured X-axis IEKF.
     * @param y_fitter_ptr Optional pre-configured Y-axis IEKF.
     */
    HasteWrapperBase(int x, int y, double tracker_rate, Metavision::timestamp init_time,
                     const std::string &output_folder, std::unique_ptr<IEKFSinusoidFitter> x_fitter_ptr = nullptr,
                     std::unique_ptr<IEKFSinusoidFitter> y_fitter_ptr = nullptr)
            : initial_x_(x), initial_y_(y), init_time_(init_time),
              x_fitter_(std::move(x_fitter_ptr)), y_fitter_(std::move(y_fitter_ptr)) {

        // Initialize the HASTE tracker
        tracker_ = std::make_shared<haste::HasteDifferenceStarTracker>(
                tracker_rate,
                static_cast<haste::HypothesisPatchTracker::Scalar>(x),
                static_cast<haste::HypothesisPatchTracker::Scalar>(y),
                0.0f);

        last_x_.store(tracker_->x(), std::memory_order_relaxed);
        last_y_.store(tracker_->y(), std::memory_order_relaxed);
    }

    virtual ~HasteWrapperBase() {
        stop();
    }

    /**
     * @brief Stops the tracking thread gracefully.
     */
    void stop() {
        run_.store(false, std::memory_order_release);
        cv_.notify_one();
        if (tracker_thread_.joinable()) {
            tracker_thread_.join();
        }
    }

    /**
     * @brief Prints the current state of the IEKF fitters to stdout.
     */
    void printFittersStatus() {
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        if (x_fitter_) {
            std::cout << "X Fitter: " << x_fitter_->to_string() << std::endl;
        } else {
            std::cout << "X Fitter: Not initialized" << std::endl;
        }
        if (y_fitter_) {
            std::cout << "Y Fitter: " << y_fitter_->to_string() << std::endl;
        } else {
            std::cout << "Y Fitter: Not initialized" << std::endl;
        }
    }

    // --- Accessors & Estimators ---

    /**
     * @brief Returns the estimated *absolute* position at time t.
     * Includes the DC offset (center of motion).
     */
    std::optional<std::pair<double, double>> getEstimate(double t) {
        if (!x_fitter_ || !y_fitter_) return std::nullopt;
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        return std::make_pair(x_fitter_->predict(t), y_fitter_->predict(t));
    }

    /**
     * @brief Returns the estimated *relative* position (AC component) at time t.
     * Ideal for motion compensation (stabilization).
     */
    std::optional<std::pair<double, double>> getRelEstimate(double t) {
        if (!x_fitter_ || !y_fitter_) return std::nullopt;
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        return std::make_pair(x_fitter_->predict_rel(t), y_fitter_->predict_rel(t));
    }

    /**
     * @brief Fast, thread-safe retrieval of relative estimates via reference.
     * @return true if fitters are initialized, false otherwise.
     */
    bool getRelEstimate(float t, double &x, double &y) {
        if (!x_fitter_ || !y_fitter_) return false;
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        x = x_fitter_->predict_rel(t);
        y = y_fitter_->predict_rel(t);
        return true;
    }

    /**
     * @brief Caching wrapper for relative estimates.
     * Useful when multiple consumers query the same timestamp in the same frame loop.
     */
    bool getRelEstimate(Metavision::timestamp t_query, double t, double &x, double &y) {
        if (!x_fitter_ || !y_fitter_) return false;

        static Metavision::timestamp last_t_query = std::numeric_limits<Metavision::timestamp>::max();
        static double last_t = std::numeric_limits<double>::quiet_NaN();
        static double last_x = 0.0;
        static double last_y = 0.0;

        // Cache hit check
        if (last_t_query == t_query && last_t == t) {
            x = last_x;
            y = last_y;
            return true;
        }

        std::lock_guard<std::mutex> lock(centroids_mutex_);
        x = x_fitter_->predict_rel(t);
        y = y_fitter_->predict_rel(t);

        // Update cache
        last_t_query = t_query;
        last_t = t;
        last_x = x;
        last_y = y;

        return true;
    }

    /**
     * @brief Returns the internal state of the HASTE tracker (raw tracking data).
     * @return tuple {x, y, t}
     */
    std::tuple<haste::HypothesisPatchTracker::Scalar, haste::HypothesisPatchTracker::Scalar, haste::HypothesisPatchTracker::Scalar>
    getTrackerState() const {
        if (tracker_) {
            return {tracker_->x(), tracker_->y(), tracker_->t()};
        }
        return {0.0, 0.0, 0.0};
    }

    /** @return Combined magnitude of oscillation (sqrt(Ax^2 + Ay^2)). */
    double getAmplitude() const {
        if (x_fitter_ && y_fitter_) {
            return std::sqrt(std::pow(x_fitter_->getAmplitude(), 2) + std::pow(y_fitter_->getAmplitude(), 2));
        }
        return 0.0;
    }

    int color() const {
        return color_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Injects initialized fitters into the wrapper (e.g., after NUFFT initialization).
     */
    void addFitters(std::unique_ptr<IEKFSinusoidFitter> x_fitter_ptr,
                    std::unique_ptr<IEKFSinusoidFitter> y_fitter_ptr) {
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        x_fitter_ = std::move(x_fitter_ptr);
        y_fitter_ = std::move(y_fitter_ptr);
    }

    bool isRunning() const {
        return run_.load(std::memory_order_acquire);
    }

    /** @return Current tracker position (thread-safe atomic load). */
    std::pair<double, double> getCurrentPosition() {
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        return {last_x_.load(std::memory_order_relaxed),
                last_y_.load(std::memory_order_relaxed)};
    }

    void getCurrentPosition(unsigned short &x, unsigned short &y) {
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        x = static_cast<unsigned short>(last_x_.load(std::memory_order_relaxed));
        y = static_cast<unsigned short>(last_y_.load(std::memory_order_relaxed));
    }

    /** @return The DC offset (center of oscillation) if fitters are valid. */
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

    /**
     * @brief Retrieves and clears the queue of collected centroids.
     * Used to pass data to the NUFFT estimator.
     */
    std::queue<Centroid> getCentroids() {
        std::lock_guard<std::mutex> lock(centroids_mutex_);
        std::queue<Centroid> result;
        result.swap(centroids_queue_);  // Efficient swap
        return result;
    }

protected:
    /**
     * @brief Internal method called when the HASTE tracker updates its state.
     *
     * This method:
     * 1. Updates the atomic current position variables.
     * 2. Pushes the new centroid to the history queue (if NUFFT not done).
     * 3. Feeds the new position into the IEKF fitters (if active).
     */
    void updateTrackingState(double t, double x, double y) {
        if (x == 0 && y == 0) return;

        std::lock_guard<std::mutex> lock(centroids_mutex_);

        // Store history for NUFFT initialization
        if (NUFFT_not_init) {
            centroids_queue_.emplace(t, x, y);
            // Circular buffer behavior
            while (centroids_queue_.size() > MAX_CENTROIDS_QUEUE) {
                centroids_queue_.pop();
            }
        }

        // Update Kalman Filters
        if (x_fitter_) {
            x_fitter_->update(t, x);
            // We store the FILTERED DC offset as the position for display stability,
            // or we could store the raw 'x'. Here we store the shift/center.
            // NOTE: Logic might vary depending on whether we want to track the *center* or the *object*.
            last_x_.store(x_fitter_->getShift(), std::memory_order_relaxed);
        } else {
            // If no filter, just store the raw tracker position
            last_x_.store(x, std::memory_order_relaxed);
        }

        if (y_fitter_) {
            y_fitter_->update(t, y);
            last_y_.store(y_fitter_->getShift(), std::memory_order_relaxed);
        } else {
            last_y_.store(y, std::memory_order_relaxed);
        }
    }
};

/**
 * @class HasteWrapper
 * @brief Template wrapper handling event ingestion for the HASTE tracker.
 *
 * @tparam T Event type (must be `Metavision::EventCD` or `Centroid`).
 * Uses a lock-free queue to buffer events from the camera callback to the tracking thread.
 */
template<typename T>
class HasteWrapper : public HasteWrapperBase {
private:
    // Single-Producer Single-Consumer lock-free queue
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

    ~HasteWrapper() override {
        this->stop();
    }

    /**
     * @brief Ingests an event if it falls within the tracker's Region of Interest (ROI).
     * @param event The event to check and potentially track.
     * @return true if the event was accepted (in ROI and queue space available).
     */
    bool feed(const T &event) {
        if (inTracker(event)) {
            return event_stack_.push(event);
        }
        return false;
    }

    /**
     * @brief Overload that returns the current tracker center via reference.
     */
    bool feed(const T &event, unsigned short &x, unsigned short &y) {
        if (inTracker(event, x, y)) {
            return event_stack_.push(event);
        }
        return false;
    }

    /**
     * @brief Checks if an event is within the spatial ROI of the tracker.
     * Uses squared distance to avoid expensive sqrt() calls.
     */
    bool inTracker(const T &event) {
        double x = last_x_.load(std::memory_order_acquire);
        double y = last_y_.load(std::memory_order_acquire);

        if (x == 0 && y == 0) return false;

        double dx = event.x - x;
        double dy = event.y - y;
        return (dx * dx + dy * dy) <= TRACKER_MARGIN_SQ;
    }

    /**
     * @brief Overload of inTracker that outputs the center coordinates.
     */
    bool inTracker(const T &event, unsigned short &x, unsigned short &y) {
        x = static_cast<unsigned short>(last_x_.load(std::memory_order_relaxed));
        y = static_cast<unsigned short>(last_y_.load(std::memory_order_relaxed));

        if (x == 0 && y == 0) return false;

        double dx = event.x - x;
        double dy = event.y - y;
        return (dx * dx + dy * dy) <= TRACKER_MARGIN_SQ;
    }

    /**
     * @brief Marks that the NUFFT initialization is complete, stopping history collection.
     */
    void setNUFFTInitialized() {
        NUFFT_not_init = false;
    }

private:

    /**
     * @brief The main loop for the tracking thread.
     * Consumes events from the SPSC queue and pushes them to the HASTE tracker.
     */
    void startEvents() {
        tracker_thread_ = std::thread([this]() {
            while (run_.load(std::memory_order_acquire)) {
                bool processed_events = false;

                // Bulk consume all available events
                event_stack_.consume_all([&](const T &event) {
                    processed_events = true;

                    // Normalize timestamp to seconds relative to init_time
                    double current_t_sec;
                    if constexpr (std::is_same_v<T, Centroid>) {
                        current_t_sec = event.t;
                    } else {
                        current_t_sec = static_cast<double>(event.t - init_time_) / 1e6;
                    }

                    // Feed to HASTE Engine
                    auto update_type = tracker_->pushEvent(current_t_sec, event.x, event.y);

                    // If HASTE decides the object has moved enough to update state
                    if (update_type == haste::HypothesisPatchTracker::EventUpdate::kStateEvent) {
                        updateTrackingState(tracker_->t(), tracker_->x(), tracker_->y());
                    }
                });

                // Yield CPU if queue was empty to prevent busy-waiting
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