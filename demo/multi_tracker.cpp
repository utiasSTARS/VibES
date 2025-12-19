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
#include "profiler.hpp"

//#define STORE

// Constants
namespace {
    constexpr double DEFAULT_FPS = 100.0;
    constexpr std::uint32_t DEFAULT_ACCUMULATION = 5000;
    constexpr int ESC_KEY = 27;
    constexpr int POLL_TIMEOUT_MS = 10;
    bool NUFFT_ESTIMATION_DONE = false;
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
    process_noise(0, 0) = 1e1;
    process_noise(1, 1) = 1e1;
    process_noise(2, 2) = 1e-2;
    process_noise(3, 3) = 1e1;

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

    const int size = haste::HypothesisPatchTracker::kPatchSize;
    const int half_size = size / 2;

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
    std::vector<std::shared_ptr<HasteWrapper<Metavision::EventCD>>> trackers;

    std::vector<double> Ax, Ay, Bx, By, omegas, offsets;
    std::mutex processing_mutex;

    // Setup display window (similar to original)
    std::string window_name("HARMEDA Event Tracking");
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::resizeWindow(window_name, width, height);
    cv::moveWindow(window_name, 0, 0);

    std::vector<std::pair<unsigned short, unsigned short>> tracker_centers;

    // Mouse callback for tracker initialization
    std::function<void(int, int)> mouse_callback = [&](const int x, const int y) {
        std::lock_guard<std::mutex> lock(processing_mutex);
        if (trackers.size() > 0 && !NUFFT_ESTIMATION_DONE) {
            std::cout << "Working on the first tracker, please wait..." << std::endl;
            return;
        }
        tracker_centers.emplace_back(x, y);
        if (NUFFT_ESTIMATION_DONE) [[likely]] {
            trackers.push_back(
                    std::make_shared<HasteWrapper<Metavision::EventCD>>(x, y, TRACKER_RATE, first_event_t, "output",
                                                                        std::make_unique<IEKFSinusoidFitter>(
                                                                                createIEKFFitter(Ax[0], Bx[0],
                                                                                                 omegas[0], offsets[0],
                                                                                                 params.params->iekf_iterations)),
                                                                        std::make_unique<IEKFSinusoidFitter>(
                                                                                createIEKFFitter(Ay[0], By[0],
                                                                                                 omegas[0], offsets[1],
                                                                                                 params.params->iekf_iterations))));
        } else {
            trackers.push_back(std::make_shared<HasteWrapper<Metavision::EventCD>>(x, y, TRACKER_RATE, first_event_t));
        }
    };

    cv::setMouseCallback(window_name, receiveMouseEvent, &mouse_callback);

    bool osd = false; // On-screen display toggle
    bool in_tracker = true, tracker_enable = true;

    double x_pred = 0, y_pred = 0;

    Metavision::Stage::EventBuffer compensated_events;
    unsigned short x_undistorted, y_undistorted, t_centre_x, t_centre_y;


    std::once_flag init_flag;
    // Main event processing callback
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::call_once(init_flag, [&]() {
            start_time = std::chrono::steady_clock::now();
            first_event_t = begin->t;
        });
        std::lock_guard<std::mutex> lock(processing_mutex); // TODO: find a way to remove this one
        compensated_events.clear();
        compensated_events.reserve(std::distance(begin, end));
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            last_event_t = ev->t;

            undistort(ev->x, ev->y, x_undistorted, y_undistorted);
//             make sure the undistorted coordinates are within the image bounds
            if (x_undistorted < 0 || x_undistorted >= width || y_undistorted < 0 || y_undistorted >= height) {
                continue; // Skip events that are out of bounds
            }
            auto &event_to_build = compensated_events.emplace_back();
            event_to_build.x = x_undistorted;
            event_to_build.y = y_undistorted;
            event_to_build.t = ev->t;
            event_to_build.p = ev->p;

            const float current_t_sec = static_cast<float>(ev->t - first_event_t) / 1e6f;

            // Process with tracker
            for (auto &tracker: trackers) {
                if (!tracker->feed(event_to_build)) {
                    continue;
                }

                if (NUFFT_ESTIMATION_DONE) [[likely]] {

                    if (tracker->getRelEstimate(event_to_build.t, current_t_sec, x_pred, y_pred)) {
                        auto x_new = static_cast<unsigned short>(x_undistorted - x_pred);
                        if (x_new < 0 || x_new >= width) {
                            break; // Skip if out of bounds
                        }
                        auto y_new = static_cast<unsigned short>(y_undistorted - y_pred);
                        if (y_new < 0 || y_new >= height) {
                            break; // Skip if out of bounds
                        }
                        event_to_build.x = x_new;
                        event_to_build.y = y_new;
                        break; // if already modified by a tracker skip, (no overlapping trackers)
                    }

                } else {
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
//                    std::lock_guard<std::mutex> lock(processing_mutex);
                    // Add on-screen display info
                    std::string text = Metavision::getHumanReadableTime(cd_frame_ts);
                    text += "     ";
                    text += Metavision::getHumanReadableRate(avg_rate);

                    cv::putText(display_frame, text, cv::Point(10, 20),
                                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(108, 143, 255), 1, cv::LINE_AA);

                    // Add tracker info if available
                    if (!trackers.empty()) {
                        // draw the tracker as a square
//                        for (const auto &centre: tracker_centers) {
//                            cv::rectangle(display_frame,
//                                          cv::Point(centre.first - half_size, centre.second - half_size + 1),
//                                          cv::Point(centre.first + half_size, centre.second + half_size + 1),
//                                          cv::Scalar(0, 255, 0), 1);
//                        }
                        int i = 0;
                        for (const auto &tracker: trackers) {
                            i++;
                            tracker->getCurrentPosition(t_centre_x, t_centre_y);
//                            std::cout << i << " " << tracker->color() << std::endl;
                            const int c = tracker->color();
                            cv::rectangle(display_frame,
                                          cv::Point(t_centre_x - half_size, t_centre_y - half_size + 1),
                                          cv::Point(t_centre_x + half_size, t_centre_y + half_size + 1),
                                          cv::Scalar(c, 255 - c, 255 - int(0.2 * c)), 2);
                        }

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
            case 't': {
                std::lock_guard<std::mutex> lock(processing_mutex);
                tracker_enable = !tracker_enable;
                std::cout << "Tracker " << (tracker_enable ? "enabled" : "disabled") << std::endl;
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