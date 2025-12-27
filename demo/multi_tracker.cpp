/**
 * @file main.cpp
 * @brief Entry point for the VibES Event-based Motion Tracking System.
 *
 * This application implements a complete pipeline for tracking high-frequency
 * oscillating targets using Event Cameras. It combines:
 * 1. HASTE (Hypothesis-based Algorithm for Super-fast Tracking of Events) for spatial ROI tracking.
 * 2. NUFFT (Non-Uniform FFT) for initial frequency discovery.
 * 3. IEKF (Iterated Extended Kalman Filter) for real-time phase and amplitude estimation.
 * 4. Real-time Motion Compensation to stabilize the view of the vibrating target.
 *
 * Pipeline Flow:
 * [Events] -> [Undistort] -> [HASTE Tracker] -> [Buffer] -> [NUFFT (One-shot)] -> [IEKF (Stream)] -> [Compensator] -> [Display]
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

// Toggle compilation flags
#define FANCY_VISUALIZATION ///< Enable rich OSD (On-Screen Display).
//#define STORE             ///< Uncomment to save raw data to disk.
//#define TIMING            ///< Uncomment to enable performance profiling.

#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/core/utils/rate_estimator.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#ifndef OPENEB
#include <metavision/sdk/core/pipeline/stage.h>
#endif

#include <metavision/sdk/core/utils/misc.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <mutex>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <csignal>
#include <filesystem>

// Project headers
#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
#include "estimator/iekf_sinusoid_fitter_multi_harmonic.hpp"
#include "event_frontend/undistort.hpp"
#include "haste_wrapper.hpp"
#include "profiler.hpp"

// Constants & Configuration
namespace {
    constexpr double DEFAULT_FPS = 100.0;            ///< Visualization frame rate.
    constexpr std::uint32_t DEFAULT_ACCUMULATION = 5000; ///< Accumulation time (us) per frame.
    constexpr int ESC_KEY = 27;                      ///< OpenCV key code for ESC.
    constexpr int POLL_TIMEOUT_MS = 10;              ///< UI event polling timeout.

    // Global state flag: transitions system from Initialization (NUFFT) to Tracking (IEKF)
    bool NUFFT_ESTIMATION_DONE = false;
}

// Global Timing & Stats
static std::chrono::steady_clock::time_point end_time, start_time;
static Metavision::timestamp first_event_t = 0, last_event_t = 0;

/**
 * @brief Cleanup function called on program exit.
 * Prints processing statistics to the console.
 */
void print_on_exit() {
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "\033[1;33mProcessing complete.\033[0m" << std::endl;
    std::cout << "\033[1;33mTotal processing time: " << elapsed_time << " ms\033[0m" << std::endl;
    std::cout << "\033[1;33mTotal events duration: " << static_cast<double>(last_event_t - first_event_t) / 1e3
              << " ms\033[0m" << std::endl;
    std::cout << "\033[1;34mExiting VibES demo...\033[0m" << std::endl;
}

/**
 * @brief Intercepts OS signals (Ctrl+C) to ensure clean shutdown.
 */
void signal_handler(int signal) {
    const char *signal_name = (signal == SIGTERM) ? "SIGTERM" : "SIGINT";
    std::cout << "\n" << signal_name << " received, printing final results..." << std::endl;
    print_on_exit();
    std::exit(signal);
}

int main(int argc, char *argv[]) {
    // 1. Setup signal handlers and exit hooks
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);
    std::atexit(print_on_exit);

    // 2. Load Configuration
    VibES::ParamsLoader params(argc, argv);
    std::cout << params; // Print active configuration

    // 3. Camera Geometry Setup
#ifdef OPENEB
    const auto width = params.camera.geometry().get_width();
    const auto height = params.camera.geometry().get_height();
#else
    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();
#endif


    // HASTE Tracker settings
    const int size = haste::HypothesisPatchTracker::kPatchSize;
    const int half_size = size / 2;

    // 4. Initialize Lens Undistortion
    // Uses the optimized O(1) LUT implementation
    Undistort undistort(params.params->calib_file);

    // 5. Setup Visualization Pipeline (Frame Generator)
    // Generates standard frames from event streams for display
    std::mutex cd_frame_mutex;
    cv::Mat cd_frame;
    Metavision::timestamp cd_frame_ts{0};

    Metavision::CDFrameGenerator cd_frame_generator(width, height);
    cd_frame_generator.set_color_palette(Metavision::ColorPalette::Light);
    cd_frame_generator.set_display_accumulation_time_us(DEFAULT_ACCUMULATION);

    cd_frame_generator.start(DEFAULT_FPS,
                             [&cd_frame_mutex, &cd_frame, &cd_frame_ts](const Metavision::timestamp &ts,
                                                                        const cv::Mat &frame) {
                                 std::unique_lock<std::mutex> lock(cd_frame_mutex);
                                 cd_frame_ts = ts;
                                 frame.copyTo(cd_frame);
                             });

    // 6. Setup Rate Estimator (for OSD statistics)
    double avg_rate = 0, peak_rate = 0;
    Metavision::RateEstimator cd_rate_estimator(
            [&avg_rate, &peak_rate](Metavision::timestamp ts, double arate, double prate) {
                avg_rate = arate;
                peak_rate = prate;
            },
            100000, 1000000, true);

    // 7. Initialize Estimators
    // NUFFT for initial frequency discovery (One instance for the whole system in this demo)
    NUFFTHelixEstimator nufft_estimator(MIN_FREQUENCY, MAX_FREQUENCY, MAX_HARMONICS);

    // Trackers container (Multi-tracker support)
    std::vector<std::shared_ptr<HasteWrapper<Metavision::EventCD>>> trackers;

    // Harmonic parameters storage
    std::vector<double> Ax, Ay, Bx, By, omegas, offsets;
    std::mutex processing_mutex;

    // 8. Setup GUI Window
    std::string window_name("VibES Event Tracking");
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::resizeWindow(window_name, width, height);
    cv::moveWindow(window_name, 0, 0);

    // User Interaction State
    std::vector<std::pair<unsigned short, unsigned short>> tracker_centers;

    // --- Mouse Callback: Tracker Initialization ---
    // Handles creating new trackers when the user clicks on the video feed
    std::function<void(int, int)> mouse_callback = [&](const int x, const int y) {
        std::lock_guard<std::mutex> lock(processing_mutex);

        // Prevent adding new trackers during the critical NUFFT estimation phase
        if (trackers.size() > 0 && !NUFFT_ESTIMATION_DONE) {
            std::cout << "[UI] Estimation in progress, please wait..." << std::endl;
            return;
        }

        tracker_centers.emplace_back(x, y);

        // CASE A: System is already calibrated (NUFFT Done)
        // Initialize new trackers immediately with the known frequency/parameters
        if (NUFFT_ESTIMATION_DONE) [[likely]] {
            // Soft Start: Initialize IEKF offset at the click position (x, y)
            // instead of the global offset derived from the first tracker.
            auto of_0 = std::vector<double>{static_cast<double>(x)};
            auto of_1 = std::vector<double>{static_cast<double>(y)};

            trackers.push_back(
                    std::make_shared<HasteWrapper<Metavision::EventCD>>(x, y, TRACKER_RATE, first_event_t, "output",
                                                                        std::make_unique<IEKFSinusoidFitter>(
                                                                                IEKFSinusoidFitter::createFromHarmonicEstimates(Ax, Bx,
                                                                                                                                omegas, of_0,
                                                                                                                                params.params->iekf_iterations)),
                                                                        std::make_unique<IEKFSinusoidFitter>(
                                                                                IEKFSinusoidFitter::createFromHarmonicEstimates(Ay, By,
                                                                                                                                omegas, of_1,
                                                                                                                                params.params->iekf_iterations))));
        }
            // CASE B: First Tracker (System Uncalibrated)
            // Initialize a raw tracker to collect data for NUFFT
        else {
            trackers.push_back(std::make_shared<HasteWrapper<Metavision::EventCD>>(x, y, TRACKER_RATE, first_event_t));
        }
    };

    cv::setMouseCallback(window_name, receiveMouseEvent, &mouse_callback);

    // UI flags
    bool osd = false;
    bool in_tracker = true, tracker_enable = true;

    // Motion compensation variables
    double x_pred = 0, y_pred = 0;
   std::vector<Metavision::EventCD> compensated_events;
    unsigned short x_undistorted, y_undistorted, t_centre_x, t_centre_y;
    std::once_flag init_flag;

    // ========================================================================================
    // 9. Main Event Processing Loop
    // ========================================================================================
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {

        // One-time initialization
        std::call_once(init_flag, [&]() {
            start_time = std::chrono::steady_clock::now();
            first_event_t = begin->t;
        });

        std::lock_guard<std::mutex> lock(processing_mutex);
        compensated_events.clear();
        compensated_events.reserve(std::distance(begin, end));

        // Iterate over events
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            last_event_t = ev->t;

            // 1. Lens Correction
            undistort(ev->x, ev->y, x_undistorted, y_undistorted);

            // Filter out-of-bounds
            if (x_undistorted < 0 || x_undistorted >= width || y_undistorted < 0 || y_undistorted >= height) {
                continue;
            }

            // Create working event
            auto &event_to_build = compensated_events.emplace_back();
            event_to_build.x = x_undistorted;
            event_to_build.y = y_undistorted;
            event_to_build.t = ev->t;
            event_to_build.p = ev->p;

            const float current_t_sec = static_cast<float>(ev->t - first_event_t) / 1e6f;

            // 2. Multi-Tracker Processing
            for (auto &tracker: trackers) {
                // Feed event to tracker (returns true if event is inside ROI)
                if (!tracker->feed(event_to_build)) {
                    continue;
                }

                // Phase A: Estimation Complete (Compensating)
                if (NUFFT_ESTIMATION_DONE) [[likely]] {

                    // Get estimated motion from IEKF
                    if (tracker->getRelEstimate(event_to_build.t, current_t_sec, x_pred, y_pred)) {

                        // Apply Compensation
                        auto x_new = static_cast<unsigned short>(x_undistorted - x_pred);
                        if (x_new < 0 || x_new >= width) break;

                        auto y_new = static_cast<unsigned short>(y_undistorted - y_pred);
                        if (y_new < 0 || y_new >= height) break;

                        event_to_build.x = x_new;
                        event_to_build.y = y_new;
                        break; // Handled by this tracker, move to next event
                    }

                }
                    // Phase B: Initialization (Accumulating data)
                else {
                    // Feed centroids to NUFFT
                    if (nufft_estimator.feed(std::move(tracker->getCentroids()))) {

                        // NUFFT Batch Complete
                        nufft_estimator.printResults();
                        NUFFTHelixEstimator::extractHarmonicParameters(nufft_estimator.getHarmonics(), Ax, Ay, Bx, By, omegas, offsets);

                        if (!Ax.empty()) {
                            std::cout << "\033[1;34m[Estimator] Initializing IEKF with parameters:\033[0m" << std::endl;
                            std::cout << "  Ax: " << Ax[0] << ", Omega: " << omegas[0] << std::endl;

                            // Initialize IEKF for the FIRST tracker
                            // Use current tracker position for "Soft Start"
                            auto [tx, ty, tt] = tracker->getTrackerState();
                            auto of_0 = std::vector<double>{tx};
                            auto of_1 = std::vector<double>{ty};

                            tracker->addFitters(
                                    std::make_unique<IEKFSinusoidFitter>(
                                            IEKFSinusoidFitter::createFromHarmonicEstimates(Ax, Bx, omegas, of_0,
                                                                                            params.params->iekf_iterations)),
                                    std::make_unique<IEKFSinusoidFitter>(
                                            IEKFSinusoidFitter::createFromHarmonicEstimates(Ay, By, omegas, of_1,
                                                                                            params.params->iekf_iterations)));
                        }
                        NUFFT_ESTIMATION_DONE = nufft_estimator.done();
                    }
                }
            }
        }

        // 3. Visualization Updates
        const auto *begin_comp = compensated_events.data();
        const auto *end_comp = begin_comp + compensated_events.size();
        cd_frame_generator.add_events(begin_comp, end_comp);
        cd_rate_estimator.add_data(std::prev(end_comp)->t, std::distance(begin_comp, end_comp));
    });

    // Start camera
    params.camera.start();
    start_time = std::chrono::steady_clock::now();

    // ========================================================================================
    // 10. Main UI Loop
    // ========================================================================================
    while (params.camera.is_running()) {
        {
            std::unique_lock<std::mutex> lock(cd_frame_mutex);
            if (!cd_frame.empty()) {
                cv::Mat display_frame;
                cd_frame.copyTo(display_frame);

                if (osd) {
                    // Draw Statistics
                    std::string text = Metavision::getHumanReadableTime(cd_frame_ts);
                    text += "     ";
                    text += Metavision::getHumanReadableRate(avg_rate);

                    cv::putText(display_frame, text, cv::Point(10, 20),
                                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(108, 143, 255), 1, cv::LINE_AA);

                    // Draw Trackers
                    if (!trackers.empty()) {
                        for (const auto &tracker: trackers) {
                            tracker->getCurrentPosition(t_centre_x, t_centre_y);
                            const int c = tracker->color();
                            cv::rectangle(display_frame,
                                          cv::Point(t_centre_x - half_size, t_centre_y - half_size + 1),
                                          cv::Point(t_centre_x + half_size, t_centre_y + half_size + 1),
                                          cv::Scalar(c, 255 - c, 255 - int(0.2 * c)), 2);
                        }

                        cv::putText(display_frame, "System: ACTIVE", cv::Point(10, 40),
                                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

                        if (NUFFT_ESTIMATION_DONE) {
                            cv::putText(display_frame, "Estimation: LOCKED", cv::Point(10, 60),
                                        cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        }
                    } else {
                        cv::putText(display_frame, "Click to initialize tracker", cv::Point(10, 40),
                                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                    }
                }

                cv::imshow(window_name, display_frame);
            }
        }

        // Handle Input
        int key = processUI(POLL_TIMEOUT_MS);
        switch (key) {
            case ESC_KEY:
            case 'q':
                params.camera.stop();
                break;
            case 'o':
                osd = !osd;
                std::cout << "[UI] OSD: " << (osd ? "ON" : "OFF") << std::endl;
                break;
            case 't': {
                std::lock_guard<std::mutex> lock(processing_mutex);
                tracker_enable = !tracker_enable;
                std::cout << "[UI] Tracker " << (tracker_enable ? "enabled" : "disabled") << std::endl;
            }
                break;
            case 'h':
                std::cout << "Controls:\n"
                          << "  ESC/q: Exit\n"
                          << "  o: Toggle OSD\n"
                          << "  h: This help\n"
                          << "  Mouse click: Initialize tracker\n";
                break;
            default:
                break;
        }

        // Poll SDK events
        Metavision::EventLoop::poll_and_dispatch(1);
    }

    end_time = std::chrono::steady_clock::now();

    // Cleanup
    cd_frame_generator.stop();
    if (params.camera.is_running()) {
        params.camera.stop();
    }

    return 0;
}