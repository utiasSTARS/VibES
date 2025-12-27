/**
 * @file main.cpp
 * @brief Entry point for the VibES (Vibration Estimation System) application.
 *
 * This application captures data from a Metavision event camera, tracks a high-frequency
 * oscillating target, estimates its trajectory parameters using a hybrid NUFFT-IEKF approach,
 * and visualizes the stabilized (motion-compensated) output in real-time.
 *
 * Pipeline Stages:
 * 1. **Acquisition:** Raw events from camera or file.
 * 2. **Undistortion:** Lens correction using pre-calibrated LUT.
 * 3. **Tracking (HASTE):** Fast, asynchronous tracking of the feature ROI.
 * 4. **Initialization (NUFFT):** Frequency analysis to find harmonic parameters.
 * 5. **Estimation (IEKF):** Real-time phase/amplitude tracking.
 * 6. **Compensation:** Counter-shifting events to stabilize the view.
 * 7. **Visualization:** Rendering compensated events to video frames.
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/core/utils/rate_estimator.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/core/pipeline/stage.h>
#include <metavision/sdk/core/utils/misc.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <mutex>
#include <memory>
#include <chrono>
#include <sstream>
#include <csignal>

// Project Headers
#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
#include "event_frontend/undistort.hpp"
#include "haste_wrapper.hpp"
#include "profiler.hpp"

// Constants
namespace {
    constexpr double DEFAULT_FPS = 100.0;            ///< Visualization frame rate.
    constexpr std::uint32_t DEFAULT_ACCUMULATION = 5000; ///< Accumulation time (us) per frame.
    constexpr int ESC_KEY = 27;                      ///< OpenCV key code for ESC.
    constexpr int POLL_TIMEOUT_MS = 10;              ///< UI polling timeout.

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
    std::cout << params;

    // 3. Camera Geometry Setup
    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    const int size = haste::HypothesisPatchTracker::kPatchSize;
    const int half_size = size / 2;

    // 4. Initialize Lens Undistortion
    Undistort undistort(params.params->calib_file);

    // 5. Setup Visualization Pipeline (Frame Generator)
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
    // NUFFT Estimator (shared pointer allows easy reset)
    auto nufft_estimator = std::make_shared<NUFFTHelixEstimator>(MIN_FREQUENCY, MAX_FREQUENCY, MAX_HARMONICS);

    // HASTE Tracker (initially empty)
    std::shared_ptr<HasteWrapper<Metavision::EventCD>> tracker;

    // Harmonic parameters storage
    std::vector<double> Ax, Ay, Bx, By, omegas, offsets;
    std::mutex processing_mutex;

    // 8. Setup GUI Window
    std::string window_name("VibES Event Tracking");
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::resizeWindow(window_name, width, height);
    cv::moveWindow(window_name, 0, 0);

    // State variables for visualization and tracking
    Metavision::Stage::EventBuffer compensated_events;
    unsigned short x_undistorted, y_undistorted;
    unsigned short t_centre_x = 0, t_centre_y = 0;

    std::vector<std::pair<unsigned short, unsigned short>> tracker_centers;

    // This flag ensures we wait for a click before trying to track
    NUFFT_ESTIMATION_DONE = true;

    // --- Interaction Callback ---
    // Handles mouse clicks to initialize or move the tracker
    std::function<void(int, int)> mouse_callback = [&](const int x, const int y) {
        std::lock_guard<std::mutex> lock(processing_mutex);

        // Prevent re-clicking if estimation is currently running
        if (!NUFFT_ESTIMATION_DONE) {
            std::cout << "[UI] Estimation in progress, please wait..." << std::endl;
            // return; // Uncomment to strict block re-clicks
        }

        tracker_centers.emplace_back(x, y);

        // Update visualization target
        t_centre_x = x;
        t_centre_y = y;

        // Initialize new HASTE tracker at the CLICKED position
        tracker = std::make_shared<HasteWrapper<Metavision::EventCD>>(x, y, TRACKER_RATE, first_event_t);

        // Reset estimation flag to trigger data collection for NUFFT
        NUFFT_ESTIMATION_DONE = false;
        std::cout << "[UI] Tracker initialized at (" << x << ", " << y << ")" << std::endl;
    };

    cv::setMouseCallback(window_name, receiveMouseEvent, &mouse_callback);

    bool osd = false; // On-Screen Display toggle
    std::once_flag init_flag;

    // ========================================================================================
    // 9. Main Event Processing Loop
    // ========================================================================================
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {

        std::lock_guard<std::mutex> lock(processing_mutex);

        // One-time initialization on first event batch
        std::call_once(init_flag, [&]() {
            start_time = std::chrono::steady_clock::now();
            first_event_t = begin->t;
        });

        compensated_events.clear();
        compensated_events.reserve(std::distance(begin, end));

        // Iterate over events in the current batch
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            last_event_t = ev->t;

            // 1. Lens Correction
            undistort(ev->x, ev->y, x_undistorted, y_undistorted);

            // Filter out-of-bounds events
            if (x_undistorted < 0 || x_undistorted >= width || y_undistorted < 0 || y_undistorted >= height) {
                continue;
            }

            // Create working event
            auto &event_to_build = compensated_events.emplace_back();
            event_to_build.x = x_undistorted;
            event_to_build.y = y_undistorted;
            event_to_build.t = ev->t;
            event_to_build.p = ev->p;

            // 2. Tracking & Estimation Logic
            // If estimation is needed (NUFFT_ESTIMATION_DONE == false) AND we have a tracker...
            if (!NUFFT_ESTIMATION_DONE && tracker && tracker->feed(event_to_build)) {

                // ...and if the NUFFT buffer is full...
                if (nufft_estimator->feed(std::move(tracker->getCentroids()))) {

                    // NUFFT Batch Complete: Extract parameters
                    nufft_estimator->printResults();
                    NUFFTHelixEstimator::extractHarmonicParameters(nufft_estimator->getHarmonics(), Ax, Ay, Bx, By,
                                                                   omegas, offsets);

                    if (!Ax.empty()) {
                        std::cout << "\033[1;34m[Estimator] Initializing IEKF with parameters:\033[0m" << std::endl;
                        std::cout << "  Ax: " << Ax[0] << ", Ay: " << Ay[0]
                                  << ", Bx: " << Bx[0] << ", By: " << By[0]
                                  << ", Omega: " << omegas[0] << ", OffsetX: " << offsets[0] << std::endl;


                    }

                    // Mark estimation as done to stop feeding NUFFT
                    NUFFT_ESTIMATION_DONE = nufft_estimator->done();
                }
            }
        }

        // 3. Visualization Updates
        const auto *begin_comp = compensated_events.data();
        const auto *end_comp = begin_comp + compensated_events.size();
        cd_frame_generator.add_events(begin_comp, end_comp);
        cd_rate_estimator.add_data(std::prev(end_comp)->t, std::distance(begin_comp, end_comp));
    });

    // Start camera stream
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
                    std::lock_guard<std::mutex> lock(processing_mutex);

                    // Draw Statistics
                    std::string text = Metavision::getHumanReadableTime(cd_frame_ts);
                    text += "     ";
                    text += Metavision::getHumanReadableRate(avg_rate);

                    cv::putText(display_frame, text, cv::Point(10, 20),
                                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(108, 143, 255), 1, cv::LINE_AA);

                    // Draw Tracker Box
                    if (tracker) {
                        const cv::Scalar main_color(128, 200, 128);

                        // If tracking is live (accumulating data), update the box position
                        if(!NUFFT_ESTIMATION_DONE) {
                            tracker->getCurrentPosition(t_centre_x, t_centre_y);
                        }
                        // If estimation is done, box stays at the last known or clicked position
                        // (depending on desired behavior - usually we want it to follow the tracker)

                        cv::rectangle(display_frame,
                                      cv::Point(t_centre_x - half_size, t_centre_y - half_size + 1),
                                      cv::Point(t_centre_x + half_size, t_centre_y + half_size + 1),
                                      cv::Scalar(
                                              std::clamp(main_color[0], 0.0, 255.0),
                                              std::clamp(main_color[1], 0.0, 255.0),
                                              std::clamp(main_color[2], 0.0, 255.0)
                                      ), 2);


                        cv::putText(display_frame, "Tracker: ACTIVE", cv::Point(10, 40),
                                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

                        if (NUFFT_ESTIMATION_DONE) {
                            cv::putText(display_frame, "Estimation: COMPLETE", cv::Point(10, 60),
                                        cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        } else {
                            cv::putText(display_frame, "Estimation: GATHERING", cv::Point(10, 60),
                                        cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
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
            case 'r': {
                std::lock_guard<std::mutex> lock(processing_mutex);
                // Hard reset of the NUFFT estimator
                nufft_estimator = std::make_shared<NUFFTHelixEstimator>(MIN_FREQUENCY, MAX_FREQUENCY, MAX_HARMONICS);
                tracker.reset(); // Clear tracker to require new click
                NUFFT_ESTIMATION_DONE = true; // Set to true to wait for click
                std::cout << "[UI] Reset tracker" << std::endl;
            }
                break;
            case 'h':
                std::cout << "Controls:\n"
                          << "  ESC/q: Exit\n"
                          << "  o: Toggle OSD\n"
                          << "  r: Reset tracker\n"
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