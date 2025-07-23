//
// Enhanced Event-based Motion Tracking with Proper Visualization
// Based on Metavision SDK patterns
//

#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
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
#include <thread>

#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
#include "estimator/iekf_sinusoid_fitter.hpp"
#include "event_frontend/undistort.hpp"
#include "haste_wrapper.hpp"

#define FANCY_VISUALIZATION

// Constants
namespace {
    constexpr double DEFAULT_FPS = 100.0;
    constexpr std::uint32_t DEFAULT_ACCUMULATION = 5000;
    constexpr double DEFAULT_MEASUREMENT_NOISE = 0.5;
    constexpr int ESC_KEY = 27;
    constexpr int POLL_TIMEOUT_MS = 10;
    bool NUFFT_ESTIMATION_DONE = false;
}

// UI processing function similar to original
int processUI(int delay_ms) {
    auto then = std::chrono::high_resolution_clock::now();
    int key = cv::waitKey(delay_ms);
    auto now = std::chrono::high_resolution_clock::now();

    // Ensure consistent timing
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
    if (elapsed < delay_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms - elapsed));
    }

    return key;
}

// Mouse callback for tracker selection
void receiveMouseEvent(int event, int x, int y, int flags, void *userdata) {
    auto *callback = reinterpret_cast<std::function<void(int, int)> *>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN && callback) {
        (*callback)(x, y);
    }
}

/**
 * Creates and configures an IEKF sinusoid fitter with given parameters
 */
IEKFSinusoidFitter createIEKFFitter(double A, double B, double omega, double C, int iterations = 1) {
    IEKFSinusoidFitter::StateVector initial_state;
    initial_state << A, B, omega, C;

    IEKFSinusoidFitter::StateCovariance initial_covariance;
    initial_covariance.setIdentity();
    initial_covariance(0, 0) = 1e2;  // A amplitude
    initial_covariance(1, 1) = 1e2;  // B amplitude
    initial_covariance(2, 2) = 1e1;  // omega frequency
    initial_covariance(3, 3) = 1e3;  // C DC offset

    IEKFSinusoidFitter::StateCovariance process_noise;
    process_noise.setIdentity();
    process_noise(0, 0) = 1e0;
    process_noise(1, 1) = 1e0;
    process_noise(2, 2) = 1e-3;
    process_noise(3, 3) = 1e0;

    return {initial_state, initial_covariance, process_noise,
            DEFAULT_MEASUREMENT_NOISE, iterations};
}

/**
 * Extracts harmonic parameters from NUFFT estimator results
 */
void extractHarmonicParameters(const NUFFTHelixEstimator &estimator,
                               std::vector<double> &Ax, std::vector<double> &Ay,
                               std::vector<double> &Bx, std::vector<double> &By,
                               std::vector<double> &omegas, std::vector<double> &offsets) {

    for (const auto &harmonic: estimator.getHarmonics()) {
        double phase_x = std::atan2(harmonic.amplitude_x, harmonic.amplitude_y);
        double phase_y = std::atan2(harmonic.amplitude_y, harmonic.amplitude_x);

        double ax = harmonic.amplitude_x * std::cos(phase_x);
        double ay = harmonic.amplitude_y * std::sin(phase_y);
        double bx = harmonic.amplitude_x * std::sin(phase_x);
        double by = harmonic.amplitude_y * std::cos(phase_y);

        Ax.push_back(ax);
        Ay.push_back(ay);
        Bx.push_back(bx);
        By.push_back(by);
        omegas.push_back(harmonic.frequency);
        offsets.push_back(harmonic.offset_x);
        offsets.push_back(harmonic.offset_y);
    }
}

static std::chrono::steady_clock::time_point end_time, start_time;
static Metavision::timestamp first_event_t = 0, last_event_t = 0;

void print_on_exit() {
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "\033[1;33mProcessing complete.\033[0m" << std::endl;
    std::cout << "\033[1;33mTotal processing time: " << elapsed_time << " ms\033[0m" << std::endl;
    std::cout << "\033[1;33mTotal events time: " << static_cast<double>(last_event_t - first_event_t) / 1e3
              << " ms\033[0m" << std::endl;
    std::cout << "\033[1;34mExiting HARMEDA demo...\033[0m" << std::endl;
}

void signal_handler(int signal) {
    const char *signal_name = (signal == SIGTERM) ? "SIGTERM" : "SIGINT";
    std::cout << "\n" << signal_name << " received, printing final results..." << std::endl;
    print_on_exit();
    std::exit(signal);
}

int main(int argc, char *argv[]) {
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);
    std::atexit(print_on_exit);

    // Initialize parameters and camera
    HARMEDA::ParamsLoader params(argc, argv);
    std::cout << params;

    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    // Initialize undistortion
    Undistort undistort(params.params->calib_file);

    // Setup CD frame generator (similar to original)
    std::mutex cd_frame_mutex;
    cv::Mat cd_frame;
    Metavision::timestamp cd_frame_ts{0};

    Metavision::CDFrameGenerator cd_frame_generator(width, height);
    cd_frame_generator.set_display_accumulation_time_us(DEFAULT_ACCUMULATION);

    // Start frame generator with callback
    cd_frame_generator.start(DEFAULT_FPS,
                             [&cd_frame_mutex, &cd_frame, &cd_frame_ts](const Metavision::timestamp &ts,
                                                                        const cv::Mat &frame) {
                                 std::unique_lock<std::mutex> lock(cd_frame_mutex);
                                 cd_frame_ts = ts;
                                 frame.copyTo(cd_frame);
                             });

    // Setup event rate estimator
    double avg_rate = 0, peak_rate = 0;
    Metavision::RateEstimator cd_rate_estimator(
            [&avg_rate, &peak_rate](Metavision::timestamp ts, double arate, double prate) {
                avg_rate = arate;
                peak_rate = prate;
            },
            100000, 1000000, true);

    // Initialize fitters and estimator
    NUFFTHelixEstimator nufft_estimator(MIN_FREQUENCY, MAX_FREQUENCY, MAX_HARMONICS);
    std::shared_ptr<HasteWrapper<Centroid>> tracker;

    std::vector<double> Ax, Ay, Bx, By, omegas, offsets;
    std::mutex processing_mutex;

    // Setup display window (similar to original)
    std::string window_name("HARMEDA Event Tracking");
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::resizeWindow(window_name, width, height);
    cv::moveWindow(window_name, 0, 0);
#ifdef FANCY_VISUALIZATION
    int visualization_cut_off = width / 2; // Cut-off for visualization
#endif
    // Mouse callback for tracker initialization
    std::function<void(int, int)> mouse_callback = [&](const int x, const int y) {
        std::lock_guard<std::mutex> lock(processing_mutex);
        if (tracker) {
#ifdef FANCY_VISUALIZATION
            visualization_cut_off = x; // Update cut-off for visualization
#endif
            return;
        }
        tracker = std::make_shared<HasteWrapper<Centroid>>(x, y, TRACKER_RATE);
        std::cout << "Tracker initialized at (" << x << ", " << y << ")" << std::endl;
    };

    cv::setMouseCallback(window_name, receiveMouseEvent, &mouse_callback);

    bool not_init = true;
    bool osd = true; // On-screen display toggle

    double x_pred = 0, y_pred = 0;

    Metavision::Stage::EventBuffer compensated_events;
    float x_undistorted, y_undistorted;

    // Main event processing callback
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        compensated_events.clear();
        if (NUFFT_ESTIMATION_DONE) { compensated_events.reserve(std::distance(begin, end)); }
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            if (not_init) {
                start_time = std::chrono::steady_clock::now();
                first_event_t = ev->t;
                not_init = false;
                continue;
            }

            last_event_t = ev->t;

            std::tie(x_undistorted, y_undistorted) = undistort(ev->x, ev->y);
//             make sure the undistorted coordinates are within the image bounds
            if (x_undistorted < 0 || x_undistorted >= width || y_undistorted < 0 || y_undistorted >= height) {
                continue; // Skip events that are out of bounds
            }
            Metavision::EventCD undist_event(
                    static_cast<unsigned short>(x_undistorted),
                    static_cast<unsigned short>(y_undistorted),
                    ev->p,
                    ev->t
            );

            const float current_t_sec = static_cast<float>(ev->t - first_event_t) / 1e6f;

            // Process with tracker
            if (tracker) {
                const bool in_tracker = tracker->feed({current_t_sec, x_undistorted, y_undistorted});
                if (NUFFT_ESTIMATION_DONE) {

#ifdef FANCY_VISUALIZATION
                    // if the event is in the left half of the image skip
                    if (undist_event.x < visualization_cut_off) {
                        compensated_events.emplace_back(undist_event);
                        continue;
                    }
#endif
                    if (auto value = tracker->getRelEstimate(current_t_sec); value.has_value()) {
                        std::tie(x_pred, y_pred) = value.value();
                        auto x_new = static_cast<unsigned short>(x_undistorted - x_pred);
                        auto y_new = static_cast<unsigned short>(y_undistorted - y_pred);

                        if (x_new < 0 || x_new >= width ||
                            y_new < 0 || y_new >= height) {
                            continue; // Skip out-of-bounds events
                        }
                        compensated_events.emplace_back(
                                x_new, y_new,
                                0, ev->t
                        );
                        continue;
                    }
                } else {
                    if (!in_tracker) {
                        continue;
                    }

                    if (nufft_estimator.feed(std::move(tracker->getCentroids()))) {
                        nufft_estimator.printResults();
                        extractHarmonicParameters(nufft_estimator, Ax, Ay, Bx, By, omegas, offsets);

                        if (!Ax.empty()) {
                            std::cout << "\033[1;34mTracker initialized with parameters:\033[0m" << std::endl;
                            std::cout << "Ax: " << Ax[0] << ", Ay: " << Ay[0]
                                      << ", Bx: " << Bx[0] << ", By: " << By[0]
                                      << ", omega: " << omegas[0] << ", offset_x: " << offsets[0]
                                      << ", offset_y: " << offsets[1] << std::endl;
                            tracker->addFitters(
                                    std::make_unique<IEKFSinusoidFitter>(
                                            createIEKFFitter(Ax[0], Bx[0], omegas[0], offsets[0],
                                                             params.params->iekf_iterations)),
                                    std::make_unique<IEKFSinusoidFitter>(
                                            createIEKFFitter(Ay[0], By[0], omegas[0], offsets[1],
                                                             params.params->iekf_iterations)));
                        }
                        NUFFT_ESTIMATION_DONE = nufft_estimator.done();
                    }
                }
            }
            compensated_events.emplace_back(undist_event);
        }
        // Feed events to frame generator and rate estimator
        const auto *begin_comp = compensated_events.data();
        const auto *end_comp = begin_comp + compensated_events.size();
        cd_frame_generator.add_events(begin_comp, end_comp);
        cd_rate_estimator.add_data(std::prev(end_comp)->t, std::distance(begin_comp, end_comp));
    });

    // Start camera
    params.camera.start();
    start_time = std::chrono::steady_clock::now();

    // Main processing loop (similar to original)
    while (params.camera.is_running()) {
        // Display frame with thread safety
        {
            std::unique_lock<std::mutex> lock(cd_frame_mutex);
            if (!cd_frame.empty()) {
                cv::Mat display_frame;
                cd_frame.copyTo(display_frame);

                if (osd) {
                    // Add on-screen display info
                    std::string text = Metavision::getHumanReadableTime(cd_frame_ts);
                    text += "     ";
                    text += Metavision::getHumanReadableRate(avg_rate);

                    cv::putText(display_frame, text, cv::Point(10, 20),
                                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(108, 143, 255), 1, cv::LINE_AA);

                    // Add tracker info if available
                    if (tracker) {
                        cv::putText(display_frame, "Tracker: Initialized", cv::Point(10, 40),
                                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

                        if (NUFFT_ESTIMATION_DONE) {
                            cv::putText(display_frame, "NUFFT: Complete", cv::Point(10, 60),
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

        // Process UI with consistent timing
        int key = processUI(POLL_TIMEOUT_MS);
        switch (key) {
            case ESC_KEY:
            case 'q':
                params.camera.stop();
                break;
            case 'o':
                osd = !osd;
                std::cout << "OSD: " << (osd ? "ON" : "OFF") << std::endl;
                break;
            case 'r': {
                std::lock_guard<std::mutex> lock(processing_mutex);
                tracker.reset();
                NUFFT_ESTIMATION_DONE = false;
                std::cout << "Reset tracker" << std::endl;
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

        // Poll Metavision events
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