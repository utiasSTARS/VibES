//
// Event-based Motion Tracking with Sinusoidal Fitting
// Created by viciopoli on 07/07/25.
//

#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/core/utils/rate_estimator.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <mutex>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <csignal>

#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
#include "slice_visualizer.hpp"
#include "estimator/iekf_sinusoid_fitter.hpp"
#include "event_frontend/centroid.hpp"
#include "haste/app/command_parser.hpp"
#include "haste/tracking.hpp"
#include "event_frontend/undistort.hpp"
#include "visualizer/interactive_frame_visualizer.hpp"
#include "haste_wrapper.hpp"

// Configuration flags
#define VISUALIZE
//#define VISUALIZE_SLICES
//#define STORE_RESULTS

// Constants
namespace {
    constexpr double DEFAULT_FPS = 1000.0;
    constexpr std::uint32_t DEFAULT_ACCUMULATION = 10000;
    constexpr double DEFAULT_MEASUREMENT_NOISE = 0.5;
    constexpr int ESC_KEY = 27;
    constexpr int POLL_TIMEOUT_MS = 2;
    constexpr int SLEEP_DURATION_MS = 2;
    bool NUFFT_ESTIMATION_DONE = false;
}


/**
 * Creates and configures an IEKF sinusoid fitter with given parameters
 */
IEKFSinusoidFitter createIEKFFitter(double A, double B, double omega, double C, int iterations = 1) {
    IEKFSinusoidFitter::StateVector initial_state;
    initial_state << A, B, omega, C;

    // Initial covariance matrix
    IEKFSinusoidFitter::StateCovariance initial_covariance;
    initial_covariance.setIdentity();
    initial_covariance(0, 0) = 1e2;  // A amplitude
    initial_covariance(1, 1) = 1e2;  // B amplitude
    initial_covariance(2, 2) = 1e1;  // omega frequency
    initial_covariance(3, 3) = 1e3;  // C DC offset

    // Process noise covariance
    IEKFSinusoidFitter::StateCovariance process_noise;
    process_noise.setIdentity();
    process_noise(0, 0) = 1e0;   // A can vary
    process_noise(1, 1) = 1e0;   // B can vary
    process_noise(2, 2) = 1e-3;  // omega changes slowly
    process_noise(3, 3) = 1e0;   // C can vary

    return IEKFSinusoidFitter(initial_state, initial_covariance, process_noise,
                              DEFAULT_MEASUREMENT_NOISE, iterations);
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


static std::unique_ptr<IEKFSinusoidFitter> x_fitter, y_fitter;
static std::chrono::steady_clock::time_point end_time, start_time;
static Metavision::timestamp first_event_t = 0, last_event_t = 0;

void print_on_exit() {
    if (x_fitter) {
        std::cout << "X " << *x_fitter << std::endl;
    }
    if (y_fitter) {
        std::cout << "Y " << *y_fitter << std::endl;
    }

    // Calculate elapsed time in ms
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    // print orange
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
    // Print the filter state when exiting
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);

    std::atexit(print_on_exit);

    // Initialize parameters and camera
    HARMEDA::ParamsLoader params(argc, argv);
    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    // Initialize undistortion
    Undistort undistort(params.params->calib_file);

    // Create frame generators
    auto frame_gen_undist = Metavision::PeriodicFrameGenerationAlgorithm(
            width, height, DEFAULT_ACCUMULATION, DEFAULT_FPS);

    // Initialize fitters and estimator
    NUFFTHelixEstimator nufft_estimator(MIN_FREQUENCY, MAX_FREQUENCY, MAX_HARMONICS);

    // Initialize tracker
    std::vector<std::shared_ptr<HasteWrapper>> trackers;

    std::vector<double> Ax, Ay, Bx, By, omegas, offsets;
    std::mutex processing_mutex;

    EventFrameVisualizer visualizer("Event Frame Visualizer");

    // everytime there is a new point added, add a new tracker
    visualizer.set_tracker_point_callback([&](const int x, const int y) {
        std::lock_guard<std::mutex> lock(processing_mutex);
        trackers.push_back(std::make_shared<HasteWrapper>(x, y, TRACKER_RATE));
        if (NUFFT_ESTIMATION_DONE) {
            trackers.back()->addFitters(std::make_unique<IEKFSinusoidFitter>(
                                                createIEKFFitter(Ax[0], Bx[0], omegas[0], offsets[0],
                                                                 params.params->iekf_iterations)),
                                        std::make_unique<IEKFSinusoidFitter>(
                                                createIEKFFitter(Ay[0], By[0], omegas[0], offsets[1],
                                                                 params.params->iekf_iterations)));
        } else {
            visualizer.block();
        }
    });

    // Setup frame generation callbacks
    frame_gen_undist.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
#ifdef VISUALIZE
        if (!trackers.empty()) {
            if (NUFFT_ESTIMATION_DONE) {
                for (auto t: trackers) {
                    if (auto shift = t->getShift(); shift.has_value()) {
                        const int x = std::get<0>(shift.value());
                        const int y = std::get<1>(shift.value());

                        // Draw predicted position
                        cv::circle(frame, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);

                        // Draw tracking rectangle
                        const int size = haste::HypothesisPatchTracker::kPatchSize;
                        const int half_size = size / 2;
                        const int c = t->color();
                        cv::rectangle(frame,
                                      cv::Point(x - half_size, y - half_size + 1),
                                      cv::Point(x + half_size, y + half_size + 1),
                                      cv::Scalar(c, 255 - c, 255 - int(0.2 * c)), 2);

                        // add text in the rectangle
//                        double ampl_x = t->getAmplitude().first;
                        std::ostringstream stream;
//                        stream << std::fixed << std::setprecision(1) << ampl_x;
//
//                        cv::putText(frame, stream.str(), cv::Point(x - half_size + 5, y - half_size + 20),
//                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                        // get fps
                        auto fps = frame_gen_undist.get_fps();
                        stream.clear();
                        stream.str("");
                        stream << std::fixed << std::setprecision(1) << fps;

                        std::string fps_text = "FPS: " + stream.str();
                        cv::putText(frame, fps_text, cv::Point(frame.rows - 20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                    cv::Scalar(255, 255, 255), 2);
                    }
                }
            }
        }
#endif
        visualizer.set_frame(frame);
    });

    std::vector<Metavision::EventCD> events_undistorted, events_compensated;

    // Main event processing callback
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        events_undistorted.resize(std::distance(begin, end));
        events_compensated.resize(std::distance(begin, end));

        std::cout << "ERC:" << (params.camera.erc_module().get_cd_event_rate() / 1000000) << "Mev/s";

        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            last_event_t = ev->t;
            // Undistort event coordinates
            auto undist_event = undistort(*ev);

            // Skip out-of-bounds events
            if (undist_event.x < 0 || undist_event.x >= width || undist_event.y < 0 || undist_event.y >= height) {
                continue;
            }


            {
                std::lock_guard<std::mutex> lock(processing_mutex);
                for (auto &tracker: trackers) {
                    // Feed event to each tracker
                    tracker->feed(undist_event);
                    if (!NUFFT_ESTIMATION_DONE) {
                        if (auto centroid = tracker->getCentroids(); nufft_estimator.feed(std::move(centroid))) {
//                                if (nufft_estimator.compute()) {
                            nufft_estimator.printResults();

                            // Extract harmonic parameters
                            extractHarmonicParameters(nufft_estimator, Ax, Ay, Bx, By, omegas, offsets);

                            if (Ax.empty()) {
                                std::cout << "No harmonics found, skipping tracker initialization."
                                          << std::endl;
                                continue;
                            }

                            trackers.back()->addFitters(std::make_unique<IEKFSinusoidFitter>(
                                                                createIEKFFitter(Ax[0], Bx[0], omegas[0], offsets[0],
                                                                                 params.params->iekf_iterations)),
                                                        std::make_unique<IEKFSinusoidFitter>(
                                                                createIEKFFitter(Ay[0], By[0], omegas[0],
                                                                                 offsets[1],
                                                                                 params.params->iekf_iterations)));

                            NUFFT_ESTIMATION_DONE = true;
                            visualizer.unblock();

                        }
                    }
                }
            }
            events_undistorted.push_back(undist_event);
        }

        // Process events through frame generators
        frame_gen_undist.process_events(events_undistorted.begin(), events_undistorted.end());
    });

    // Start camera
    params.camera.start();

    start_time = std::chrono::steady_clock::now();

    // Main processing loop
    while (params.camera.is_running()) {
        if (cv::waitKey(1) == ESC_KEY) {
            break;
        }
        Metavision::EventLoop::poll_and_dispatch(POLL_TIMEOUT_MS);
    }
    // time now
    end_time = std::chrono::steady_clock::now();

    // Stop camera
    if (params.camera.is_running()) {
        params.camera.stop();
    }


    std::cout << "Processing complete." << std::endl;
    return 0;
}