//
// Event-based Motion Tracking with Sinusoidal Fitting
// Created by viciopoli on 07/07/25.
//

#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/base_window.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <csignal>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/driver/hdf5_event_file_writer.h>

#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
#include "estimator/iekf_sinusoid_fitter.hpp"
#include "event_frontend/centroid.hpp"
#include "haste/app/command_parser.hpp"
#include "haste/tracking.hpp"
#include "event_frontend/undistort.hpp"
#include "slice_visualizer.hpp"

// Configuration flags
#define VISUALIZE
#define VISUALIZE_SLICES
//#define STORE_RESULTS
//#define STORE_HDF5

// Constants
namespace {
    constexpr double DEFAULT_FPS = 100.0;
    constexpr std::uint32_t DEFAULT_ACCUMULATION = 5000;
    constexpr double MIN_FREQUENCY = 5.0;   // Hz
    constexpr double MAX_FREQUENCY = 80.0;  // Hz
    constexpr int MAX_HARMONICS = 1;
    constexpr double TRACKER_RATE = 0.01;
    constexpr double DEFAULT_MEASUREMENT_NOISE = 0.5;
    constexpr int ESC_KEY = 27;
    constexpr int POLL_TIMEOUT_MS = 10;
    constexpr int SLEEP_DURATION_MS = 2;
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
 * Sets up output directories for storing results
 */
void setupOutputDirectories(const std::string &output_folder) {
    if (!std::filesystem::exists(output_folder)) {
        std::filesystem::create_directories(output_folder);
    }
    std::filesystem::create_directories(output_folder + "/undistorted");
    std::filesystem::create_directories(output_folder + "/compensated");
}

/**
 * Saves frames to disk with sequential naming
 */
void saveFrames(const std::vector<cv::Mat> &frames, const std::string &folder_path) {
    for (size_t i = 0; i < frames.size(); ++i) {
        std::ostringstream filename;
        filename << folder_path << "/" << std::setw(4) << std::setfill('0') << i << ".png";
        cv::imwrite(filename.str(), frames[i]);
    }
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

#ifdef VISUALIZE_SLICES

/**
 * Updates slice visualizers with prediction circles
 */
void updateSliceVisualizers(HARMEDA::SliceVisualizer &slice_x, HARMEDA::SliceVisualizer &slice_y,
                            const std::unique_ptr<IEKFSinusoidFitter> &x_fitter,
                            const std::unique_ptr<IEKFSinusoidFitter> &y_fitter,
                            double current_time) {
    if (y_fitter) {
        slice_y.editFrame([&](cv::Mat &frame) {
            auto y_pred = y_fitter->predict(current_time);
            cv::circle(frame, cv::Point(slice_x.time_value, y_pred), 1, cv::Scalar(255, 255, 0), -1);
        });
    }

    if (x_fitter) {
        slice_x.editFrame([&](cv::Mat &frame) {
            auto x_pred = x_fitter->predict(current_time);
            cv::circle(frame, cv::Point(slice_x.time_value, x_pred), 1, cv::Scalar(255, 255, 0), -1);
        });
    }
}


/**
 * Updates tracker visualization on slice frames
 */
void updateTrackerVisualization(HARMEDA::SliceVisualizer &slice_x, HARMEDA::SliceVisualizer &slice_y,
                                const std::shared_ptr<haste::HypothesisPatchTracker> &tracker) {

    slice_y.editFrame([&](cv::Mat &frame) {
        cv::circle(frame, cv::Point(tracker->t() * slice_x.time_scale, int(tracker->y())),
                   1, cv::Scalar(0, 255, 0), -1);
    });

    slice_x.editFrame([&](cv::Mat &frame) {
        cv::circle(frame, cv::Point(tracker->t() * slice_y.time_scale, int(tracker->x())),
                   1, cv::Scalar(0, 255, 255), -1);
    });
}

#endif


static std::unique_ptr<IEKFSinusoidFitter> x_fitter, y_fitter;
static std::chrono::steady_clock::time_point end_time;
static std::chrono::steady_clock::time_point start_time;
static Metavision::timestamp first_event_t = 0, last_event_t = 0;

void print_on_exit() {
    if (x_fitter) {
        std::cout << "X " << *x_fitter << std::endl;
    }
    if (y_fitter) {
        std::cout << "Y " << *y_fitter << std::endl;
    }

    // Calculate elapsed time in ms
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
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
    auto frame_gen_comp = Metavision::PeriodicFrameGenerationAlgorithm(
            width, height, DEFAULT_ACCUMULATION, DEFAULT_FPS);

    // Initialize fitters and estimator
    NUFFTHelixEstimator nufft_estimator(MIN_FREQUENCY, MAX_FREQUENCY, MAX_HARMONICS);

    // Initialize tracker
    std::shared_ptr<haste::HypothesisPatchTracker> tracker;

    Metavision::Pipeline p(true);

    // Tracker positioning
    int shift_x = params.params->tracker_x;
    int shift_y = params.params->tracker_y;

#ifdef STORE_HDF5
    Metavision::HDF5EventFileWriter hdf5_writer(params.params->output_folder + "/compensated.hdf5");
    hdf5_writer.add_metadata_map_from_camera(params.camera);
#endif

#ifdef VISUALIZE
    // Initialize visualization windows
    Metavision::Window window("Frames", width, height, Metavision::BaseWindow::RenderMode::BGR);
    Metavision::Window window_compensated("Frames compensated", width, height,
                                          Metavision::BaseWindow::RenderMode::BGR);
    double x_square_center = width / 2.0 + shift_x;
    double y_square_center = height / 2.0 + shift_y;
#endif

#ifdef VISUALIZE_SLICES
    // Initialize slice visualizers
    HARMEDA::SliceVisualizer slice_visualizer_x(height, width, 0, HARMEDA::X_AXIS, 3);
    HARMEDA::SliceVisualizer slice_visualizer_y(height, width, 0, HARMEDA::Y_AXIS, 3);
    cv::namedWindow("Slice X Visualizer", cv::WINDOW_NORMAL);
    cv::namedWindow("Slice Y Visualizer", cv::WINDOW_NORMAL);
#endif

#ifdef STORE_RESULTS
    // Setup output storage
    std::string output_folder = params.params->output_folder;
    setupOutputDirectories(output_folder);

    std::vector<cv::Mat> undistorted_frames, compensated_frames;
    std::fstream file_event, file_comp;
    file_event.open(output_folder + "/event.txt", std::ios::out);
    file_comp.open(output_folder + "/compensation_data.txt", std::ios::out);
#endif

    std::mutex processing_mutex;
    double tracker_latency = 0.0;

    // Setup frame generation callbacks
    frame_gen_undist.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
#ifdef STORE_RESULTS
        undistorted_frames.emplace_back(frame.clone());
#endif

#ifdef VISUALIZE
        if (x_fitter && y_fitter) {
            // Draw predicted position
            cv::circle(frame, cv::Point(x_fitter->getShift(), y_fitter->getShift()),
                       5, cv::Scalar(0, 255, 0), -1);
            x_square_center = x_fitter->getShift();
            y_square_center = y_fitter->getShift();
        }

        // Draw tracking rectangle
        int size = haste::HypothesisPatchTracker::kPatchSize;
        int half_size = size / 2;
        cv::rectangle(frame,
                      cv::Point(x_square_center - half_size, y_square_center - half_size + 1),
                      cv::Point(x_square_center + half_size, y_square_center + half_size + 1),
                      cv::Scalar(0, 0, 255), 2);

        window.show(frame);
#endif
    });

    frame_gen_comp.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
#ifdef STORE_RESULTS
        compensated_frames.emplace_back(frame.clone());
#endif

#ifdef VISUALIZE
        window_compensated.show(frame);
#endif
    });

    bool not_init = true;
    bool nufft_done = false;

    float x_undistorted, y_undistorted;
    // Main event processing callback
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::vector<Metavision::EventCD> events_undistorted, events_compensated;
        events_undistorted.reserve(std::distance(begin, end));
        events_compensated.reserve(std::distance(begin, end));

        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            last_event_t = ev->t;

            const Metavision::EventCD &event = *ev;

            std::tie(x_undistorted, y_undistorted) = undistort(event.x, event.y);
//             make sure the undistorted coordinates are within the image bounds
            if (x_undistorted < 0 || x_undistorted >= width || y_undistorted < 0 || y_undistorted >= height) {
                continue; // Skip events that are out of bounds
            }
            Metavision::EventCD undist_event(
                    static_cast<unsigned short>(x_undistorted),
                    static_cast<unsigned short>(y_undistorted),
                    event.p,
                    event.t
            );

            // Undistort event coordinates
//            auto undist_event = undistort(*ev);
//
//            // Skip out-of-bounds events
//            if (undist_event.x < 0 || undist_event.x >= width || undist_event.y < 0 || undist_event.y >= height) {
//                continue;
//            }

#ifdef STORE_RESULTS
            file_event << undist_event.t << " " << undist_event.x << " " << undist_event.y << "\n";
#endif

            events_undistorted.push_back(undist_event);

            // Initialize tracker on first event
            if (not_init) {
                // measure time now
                start_time = std::chrono::steady_clock::now();
                first_event_t = ev->t;
                tracker = std::make_shared<haste::HasteDifferenceStarTracker>(
                        TRACKER_RATE, width / 2.0 + shift_x, height / 2.0 + shift_y, 0.0);
                not_init = false;

                continue;
            }

            const float current_t_sec = static_cast<float>(ev->t - first_event_t) / 1e6;

            // Apply motion compensation if fitters are available
            if (x_fitter && y_fitter) {
                auto x_pred = x_fitter->predict_rel(current_t_sec);
                auto y_pred = y_fitter->predict_rel(current_t_sec);

#ifdef STORE_RESULTS
                file_comp << current_t_sec << " " << x_pred << " " << y_pred << "\n";
#endif

                events_compensated.emplace_back(static_cast<unsigned short>(x_undistorted - x_pred),
                                                static_cast<unsigned short>(y_undistorted - y_pred),
                                                0,
                                                event.t);
            }

#ifdef VISUALIZE_SLICES
            slice_visualizer_x.feed(undist_event);
            slice_visualizer_y.feed(undist_event);
            updateSliceVisualizers(slice_visualizer_x, slice_visualizer_y,
                                   x_fitter, y_fitter, current_t_sec);
#endif

            // Update tracker
            const auto update_type = tracker->pushEvent(current_t_sec,
                                                        x_undistorted,
                                                        y_undistorted);

            if (update_type == haste::HypothesisPatchTracker::EventUpdate::kStateEvent) {
                // Process NUFFT estimation
                if (!nufft_done && nufft_estimator.feed(Centroid(tracker->t(), tracker->x(), tracker->y()))) {

                    std::lock_guard<std::mutex> lock(processing_mutex);

//                    if (nufft_estimator.compute()) {
                    nufft_estimator.printResults();

                    // Extract harmonic parameters
                    std::vector<double> Ax, Ay, Bx, By, omegas, offsets;
                    extractHarmonicParameters(nufft_estimator, Ax, Ay, Bx, By, omegas, offsets);

                    if (!Ax.empty()) {
                        // Create new fitters with estimated parameters
                        x_fitter = std::make_unique<IEKFSinusoidFitter>(
                                createIEKFFitter(Ax[0], Bx[0], omegas[0], offsets[0],
                                                 params.params->iekf_iterations));

                        y_fitter = std::make_unique<IEKFSinusoidFitter>(
                                createIEKFFitter(Ay[0], By[0], omegas[0], offsets[1],
                                                 params.params->iekf_iterations));
                    }
                    nufft_done = true;
                }

#ifdef VISUALIZE_SLICES
                updateTrackerVisualization(slice_visualizer_x, slice_visualizer_y, tracker);
#endif

                // Update fitters with new tracker state
                if (x_fitter && y_fitter) {
                    x_fitter->update(tracker->t(), tracker->x());
                    y_fitter->update(tracker->t(), tracker->y());
                }
            }
        }

        // Process events through frame generators
        frame_gen_undist.process_events(events_undistorted.begin(), events_undistorted.end());
        frame_gen_comp.process_events(events_compensated.begin(), events_compensated.end());
#ifdef STORE_HDF5
        hdf5_writer.add_events(events_compensated.data(), events_compensated.data() + events_compensated.size());
#endif
    });

    // Start camera
    params.camera.start();


    // Main processing loop
    while (params.camera.is_running()) {
#ifdef VISUALIZE
        if (tracker) {
            haste::ImshowEigenArrayNormalized(
                    "Feature Event Window Projection",
                    tracker->eventWindowToModel(tracker->event_window(), tracker->state()).transpose());
            haste::ImshowEigenArrayNormalized("Feature Template", tracker->tracker_template().transpose());
        }
#endif

#ifdef VISUALIZE_SLICES
        cv::imshow("Slice X Visualizer", slice_visualizer_x.getFrameSide());
        cv::imshow("Slice Y Visualizer", slice_visualizer_y.getFrameSide());
#endif

#if defined(VISUALIZE_SLICES) || defined(VISUALIZE)
        if (cv::waitKey(1) == ESC_KEY) {
            break;
        }
        Metavision::EventLoop::poll_and_dispatch(POLL_TIMEOUT_MS);
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_DURATION_MS));
#endif
    }
    // time now
    end_time = std::chrono::steady_clock::now();

    // Stop camera
    if (params.camera.is_running()) {
        params.camera.stop();
    }

#ifdef STORE_HDF5
    hdf5_writer.close();
#endif

#ifdef STORE_RESULTS
    // Save all frames
    saveFrames(undistorted_frames, output_folder + "/undistorted");
    saveFrames(compensated_frames, output_folder + "/compensated");

    // Close output files
    file_event.close();
    file_comp.close();
#endif

    std::cout << "Processing complete." << std::endl;
    return 0;

}