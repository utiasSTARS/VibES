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

// Toggle compilation flags
#define FANCY_VISUALIZATION ///< Enable rich OSD (On-Screen Display).
//#define STORE             ///< Uncomment to save raw data to disk.
//#define TIMING            ///< Uncomment to enable performance profiling.

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

// Constants & Configuration
namespace {
    constexpr double DEFAULT_FPS = 1000.0;           ///< Frame rate for the visualization window.
    constexpr std::uint32_t DEFAULT_ACCUMULATION = 5000; ///< Accumulation time (us) for frame generation.
    constexpr int ESC_KEY = 27;                      ///< OpenCV key code for ESC.
    constexpr int POLL_TIMEOUT_MS = 1;               ///< UI event polling timeout.

    // Global state flag for estimation status
    bool NUFFT_ESTIMATION_DONE = false;
}

// Global Timing & Stats
static std::chrono::steady_clock::time_point end_time, start_time;
static Metavision::timestamp first_event_t = 0, last_event_t = 0;

/**
 * @brief cleanup function called on program exit.
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
    std::cout << "\n" << signal_name << " received, shutting down..." << std::endl;
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

    // Ensure output directory exists for compensated images
    std::string output_images = params.params->output_folder + "/compensated_events/";
    if (!std::filesystem::exists(output_images)) {
        std::filesystem::create_directories(output_images);
    }

    // 3. Camera Geometry Setup
    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    // HASTE Tracker parameters
    const int size = haste::HypothesisPatchTracker::kPatchSize;
    const int half_size = size / 2;
    const cv::Scalar color_tracker(0, 255, 0); // Green color for tracker bounding box
    unsigned short t_centre_x = 0, t_centre_y = 0; // Current tracker center

    // 4. Initialize Lens Undistortion
    // Uses the optimized O(1) LUT implementation
    Undistort undistort(params.params->calib_file);

    // 5. Setup Visualization Pipeline (Frame Generator)
    // Renders event streams into standard CV Mats
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
    // NUFFT for initial frequency discovery
    NUFFTHelixEstimator nufft_estimator(MIN_FREQUENCY, MAX_FREQUENCY, MAX_HARMONICS);

    // HASTE Wrapper for object tracking (initially empty until user click or config)
    std::shared_ptr<HasteWrapper<Metavision::EventCD>> tracker;

    // Harmonic parameters storage
    std::vector<double> Ax, Ay, Bx, By, omegas, offsets;
    std::mutex processing_mutex;

    // 8. Setup GUI Window
    std::string window_name("VibES Event Tracking");
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::resizeWindow(window_name, width, height);
    cv::moveWindow(window_name, 0, 0);

    // Split-screen visualization parameter (draws compensated events only on one side)
    int visualization_cut_off = width / 2;

    // --- Interaction Callback ---
    // Handles mouse clicks to initialize or move the tracker
    std::function<void(int, int)> mouse_callback = [&](const int x, const int y) {
        std::lock_guard<std::mutex> lock(processing_mutex);

        // If tracker already exists or compensation is disabled, just move the split-screen line
        if (tracker || params.params->nocompensation) {
            visualization_cut_off = x;
            return;
        }

        // Initialize new tracker at click position
        tracker = std::make_shared<HasteWrapper<Metavision::EventCD>>(x, y, TRACKER_RATE, first_event_t);
        t_centre_x = x;
        t_centre_y = y;
        std::cout << "[UI] Tracker initialized at (" << x << ", " << y << ")" << std::endl;
    };

    cv::setMouseCallback(window_name, receiveMouseEvent, &mouse_callback);

    // UI State flags
    bool osd = false;            // On-Screen Display
    bool in_tracker = true;      // Is current event inside tracker ROI?
    bool tracker_enable = true;  // Master toggle

    // Motion compensation variables
    double x_pred = 0, y_pred = 0;
    Metavision::Stage::EventBuffer compensated_events;
    unsigned short x_undistorted, y_undistorted;

    long long total_events = 0;
    std::once_flag init_flag;

    // ========================================================================================
    // 9. Main Event Processing Loop
    // ========================================================================================
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {

        // One-time initialization on first event batch
        std::call_once(init_flag, [&]() {
            start_time = std::chrono::steady_clock::now();
            first_event_t = begin->t;

            // Auto-start tracker if command line arguments were provided
            if (!params.params->nocompensation && params.params->tracker_x != 0 && params.params->tracker_y != 0) {
                tracker = std::make_shared<HasteWrapper<Metavision::EventCD>>(params.params->tracker_x,
                                                                              params.params->tracker_y,
                                                                              TRACKER_RATE, first_event_t);
                t_centre_x = params.params->tracker_x;
                t_centre_y = params.params->tracker_y;
                std::cout << "[Auto] Tracker initialized at (" << t_centre_x << ", " << t_centre_y << ")" << std::endl;
            }
        });

        // Prepare compensation buffer
        compensated_events.clear();
        compensated_events.reserve(std::distance(begin, end));

        // Iterate over events in the current batch
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            last_event_t = ev->t;

            // 1. Lens Correction
            undistort(ev->x, ev->y, x_undistorted, y_undistorted);

            // Filter out-of-bounds events after undistortion
            if (x_undistorted < 0 || x_undistorted >= width || y_undistorted < 0 || y_undistorted >= height) {
                continue;
            }

            // Copy event to working buffer
            auto &event_to_build = compensated_events.emplace_back();
            event_to_build.x = x_undistorted;
            event_to_build.y = y_undistorted;
            event_to_build.t = ev->t;
            event_to_build.p = ev->p;

            const float current_t_sec = static_cast<float>(ev->t - first_event_t) / 1e6f;

            // 2. Tracking & Compensation Logic
            if (tracker) {
                // Feed event to HASTE tracker (returns true if event is inside ROI)
                in_tracker = tracker->feed(event_to_build);

                // Phase A: Estimation Complete (Tracking + Compensating)
                if (NUFFT_ESTIMATION_DONE) [[likely]] {

                    // Skip visualization for left side of split-screen (shows original motion)
                    if (event_to_build.x < visualization_cut_off) {
                        continue;
                    }

                    // Get estimated displacement from IEKF
                    if (tracker->getRelEstimate(event_to_build.t, current_t_sec, x_pred, y_pred)) {

                        // Apply Compensation: New_Pos = Old_Pos - Estimated_Motion
                        auto x_new = static_cast<unsigned short>(x_undistorted - x_pred);
                        auto y_new = static_cast<unsigned short>(y_undistorted - y_pred);

                        // Bounds check
                        if (x_new < 0 || x_new >= width || y_new < 0 || y_new >= height) {
                            continue;
                        }

                        // Update event position to stabilized coordinates
                        event_to_build.x = x_new;
                        event_to_build.y = y_new;
                        continue;
                    }
                }
                    // Phase B: Initialization (Accumulating data for NUFFT)
                else {
                    // Feed tracked centroids to NUFFT estimator
                    if (in_tracker && nufft_estimator.feed(std::move(tracker->getCentroids()))) {

                        // NUFFT finished processing a batch
                        nufft_estimator.printResults();
                        NUFFTHelixEstimator::extractHarmonicParameters(nufft_estimator.getHarmonics(), Ax, Ay, Bx,
                                                                       By, omegas, offsets);

                        if (!Ax.empty()) {
                            std::cout << "\033[1;34m[Estimator] Initializing IEKF with parameters:\033[0m" << std::endl;
                            std::cout << "  Ax: " << Ax[0] << ", Ay: " << Ay[0]
                                      << ", Bx: " << Bx[0] << ", By: " << By[0]
                                      << ", Omega: " << omegas[0] << std::endl;

                            // Initialize IEKF Fitters
                            // Important: Use current tracker state as the DC offset for "Soft Start" stability
                            auto [x, y, t] = tracker->getTrackerState();
                            std::vector<double> offsets_x = {x};
                            std::vector<double> offsets_y = {y};

                            tracker->addFitters(
                                    std::make_unique<IEKFSinusoidFitter>(
                                            IEKFSinusoidFitter::createFromHarmonicEstimates(Ax, Bx, omegas,
                                                                                            offsets_x)),
                                    std::make_unique<IEKFSinusoidFitter>(
                                            IEKFSinusoidFitter::createFromHarmonicEstimates(Ay, By, omegas,
                                                                                            offsets_y)));
                        }

                        // Mark initialization as done
                        NUFFT_ESTIMATION_DONE = nufft_estimator.done();
                        tracker->setNUFFTInitialized();
                    }
                }
            }
        }

        // 3. Visualization Updates
        const auto *begin_comp = compensated_events.data();
        const auto *end_comp = begin_comp + compensated_events.size();
        total_events += std::distance(begin_comp, end_comp);

        // Feed processed events to frame generator
        cd_frame_generator.add_events(begin_comp, end_comp);
        // Feed raw timestamps to rate estimator
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
                    // Draw Tracker Box
                    if (tracker) {
                        tracker->getCurrentPosition(t_centre_x, t_centre_y);
                        cv::rectangle(display_frame,
                                      cv::Point(t_centre_x - half_size, t_centre_y - half_size + 1),
                                      cv::Point(t_centre_x + half_size, t_centre_y + half_size + 1),
                                      color_tracker, 2);
                    }

                    // Draw Statistics
                    std::string text = Metavision::getHumanReadableTime(cd_frame_ts);
                    text += "     ";
                    text += Metavision::getHumanReadableRate(avg_rate);

                    cv::putText(display_frame, text, cv::Point(10, 20),
                                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(108, 143, 255), 1, cv::LINE_AA);

                    // Draw System Status
                    if (tracker) {
                        cv::putText(display_frame, "Tracker: ACTIVE", cv::Point(10, 40),
                                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

                        if (NUFFT_ESTIMATION_DONE) {
                            cv::putText(display_frame, "Motion Comp: ON", cv::Point(10, 60),
                                        cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        } else {
                            cv::putText(display_frame, "Motion Comp: INIT...", cv::Point(10, 60),
                                        cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                        }
                    } else {
                        cv::putText(display_frame, "Click object to track", cv::Point(10, 40),
                                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                    }
                }

                cv::imshow(window_name, display_frame);
            }
        }

        // Handle Keyboard Input
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
                tracker.reset();
                NUFFT_ESTIMATION_DONE = false;
                std::cout << "[UI] Reset tracker" << std::endl;
                break;
            }
            case 't': {
                std::lock_guard<std::mutex> lock(processing_mutex);
                tracker_enable = !tracker_enable;
                std::cout << "[UI] Tracker " << (tracker_enable ? "enabled" : "disabled") << std::endl;
                break;
            }
            case 'h':
                std::cout << "Controls:\n"
                          << "  ESC/q: Exit\n"
                          << "  o: Toggle OSD\n"
                          << "  r: Reset tracker\n"
                          << "  t: Enable/Disable Tracker\n"
                          << "  h: This help\n"
                          << "  Mouse click: Initialize tracker\n";
                break;
            default:
                break;
        }

        // Poll Metavision SDK events
        Metavision::EventLoop::poll_and_dispatch(1);
    }

    end_time = std::chrono::steady_clock::now();

    // Cleanup
    if (params.camera.is_running()) {
        params.camera.stop();
    }

    return 0;
}