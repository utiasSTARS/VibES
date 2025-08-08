//
// Enhanced Event-based Motion Tracking with Proper Visualization
// Based on Metavision SDK patterns
//

//#define STORE
//#define BINARY

#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/core/utils/rate_estimator.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/core/pipeline/stage.h>
#include <metavision/sdk/core/utils/misc.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <mutex>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <csignal>
#include <thread>

#ifdef STORE

#include <metavision/sdk/driver/hdf5_event_file_writer.h>

#endif

#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
//#include "estimator/iekf_sinusoid_fitter.hpp"
#include "estimator/iekf_sinusoid_fitter_multi_harmonic.hpp"
#include "event_frontend/undistort.hpp"
#include "haste_wrapper.hpp"
#include "utils.hpp"

#include "visualizer/ev2image.hpp"

// Constants
namespace {
    constexpr double DEFAULT_FPS = 1000.0;
    constexpr std::uint32_t DEFAULT_ACCUMULATION = 5000;
    constexpr int ESC_KEY = 27;
    constexpr int POLL_TIMEOUT_MS = 1;
    bool NUFFT_ESTIMATION_DONE = false;
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


    // create folder if not exists
#ifdef BINARY
    std::string output_images = params.params->output_folder + "/img_bin/";
#else
    std::string output_images = params.params->output_folder + "/img_gray/";
#endif
    if (!std::filesystem::exists(output_images)) {
        std::filesystem::create_directories(output_images);
    }

    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    const int size = haste::HypothesisPatchTracker::kPatchSize;
    const int half_size = size / 2;

    const cv::Scalar color_tracker(0, 255, 0); // Green color for tracker visualization
    unsigned short t_centre_x = 0, t_centre_y = 0;

    // Initialize undistortion
    Undistort undistort(params.params->calib_file);

    // Setup CD frame generator (similar to original)
    std::mutex cd_frame_mutex;
    cv::Mat cd_frame;
    Metavision::timestamp cd_frame_ts{0};

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
    std::shared_ptr<HasteWrapper<Metavision::EventCD>> tracker;



    std::vector<double> Ax, Ay, Bx, By, omegas, offsets;
    std::mutex processing_mutex;

#ifdef STORE
    std::filesystem::path out_hdf5_file_path = params.params->output_folder + "/events.hdf5";
    if (!out_hdf5_file_path.parent_path().empty() && !std::filesystem::exists(out_hdf5_file_path.parent_path())) {
        std::filesystem::create_directories(out_hdf5_file_path.parent_path());
    }
    Metavision::HDF5EventFileWriter hdf5_writer(out_hdf5_file_path);
    hdf5_writer.add_metadata_map_from_camera(params.camera);
#endif


    // Mouse callback for tracker initialization
    std::function<void(int, int)> mouse_callback = [&](const int x, const int y) {
        std::lock_guard<std::mutex> lock(processing_mutex);
        if (tracker || params.params->nocompensation) {
            return;
        }
        tracker = std::make_shared<HasteWrapper<Metavision::EventCD>>(x, y, TRACKER_RATE, first_event_t);
        t_centre_x = x;
        t_centre_y = y;
        std::cout << "Tracker initialized at (" << x << ", " << y << ")" << std::endl;
    };



    bool osd = false; // On-screen display toggle
    bool in_tracker = true, tracker_enable = true;

    double x_pred = 0, y_pred = 0;

    Metavision::Stage::EventBuffer compensated_events, frames_events;
    Metavision::timestamp duration_for_amiev = 0;
    unsigned short x_undistorted, y_undistorted;

#ifdef BINARY
    cv::Mat output_image = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
#else
    cv::Mat output_image = cv::Mat::zeros(cv::Size(width, height), CV_16UC1);
#endif

    long long counter = 0;
    std::once_flag init_flag;
    long long slice_initial_time = 0;
    // Main event processing callback
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::call_once(init_flag, [&]() {
            start_time = std::chrono::steady_clock::now();
            first_event_t = begin->t;
            slice_initial_time = first_event_t;

            if (!params.params->nocompensation && params.params->tracker_x != 0 && params.params->tracker_y != 0) {
                tracker = std::make_shared<HasteWrapper<Metavision::EventCD>>(params.params->tracker_x,
                                                                              params.params->tracker_y,
                                                                              TRACKER_RATE, first_event_t);
                t_centre_x = params.params->tracker_x;
                t_centre_y = params.params->tracker_y;
                std::cout << "Tracker initialized at (" << t_centre_x << ", " << t_centre_y << ")" << std::endl;
            }
        });

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
            if (tracker) {
                in_tracker = tracker->feed(event_to_build);
                if (NUFFT_ESTIMATION_DONE) [[likely]] {
//                    if(event_to_build.x > 320) {
                        if (tracker->getRelEstimate(event_to_build.t, current_t_sec, x_pred, y_pred)) {
                            auto x_new = static_cast<unsigned short>(x_undistorted - x_pred);
                            if (x_new < 0 || x_new >= width) {
                                continue; // Skip if out of bounds
                            }
                            auto y_new = static_cast<unsigned short>(y_undistorted - y_pred);
                            if (y_new < 0 || y_new >= height) {
                                continue; // Skip if out of bounds
                            }
                            event_to_build.x = x_new;
                            event_to_build.y = y_new;
//                        }
                    }
                } else {
                    if (in_tracker && nufft_estimator.feed(std::move(tracker->getCentroids()))) {
                        nufft_estimator.printResults();
                        NUFFTHelixEstimator::extractHarmonicParameters(nufft_estimator.getHarmonics(), Ax, Ay, Bx, By,
                                                                       omegas, offsets);

                        if (!Ax.empty()) {
                            std::cout << "\033[1;34mTracker initialized with parameters:\033[0m" << std::endl;
                            std::cout << "Ax: " << Ax[0] << ", Ay: " << Ay[0]
                                      << ", Bx: " << Bx[0] << ", By: " << By[0]
                                      << ", omega: " << omegas[0] << ", offset_x: " << offsets[0]
                                      << ", offset_y: " << offsets[1] << std::endl;
                            tracker->addFitters(
                                    std::make_unique<IEKFSinusoidFitter>(
                                            IEKFSinusoidFitter::createFromHarmonicEstimates(Ax, Bx, omegas, offsets)),
                                    std::make_unique<IEKFSinusoidFitter>(
                                            IEKFSinusoidFitter::createFromHarmonicEstimates(Ay, By, omegas, offsets)));
                        }
                        NUFFT_ESTIMATION_DONE = nufft_estimator.done();
                    }
                }
            }

            // Feed events to frame generator and rate estimator
            if (ev->t - slice_initial_time > 10000) { // 10 ms
#ifndef BINARY
                output_image.convertTo(output_image, CV_8UC1);
                cv::normalize(output_image, output_image, 0, 255, cv::NORM_MINMAX);
                // invert the image for better visualization
                output_image = 255 - output_image;
                cv::applyColorMap(output_image, output_image, cv::COLORMAP_BONE);
#endif
                cv::imwrite(output_images + std::to_string(counter) + ".png",
                            output_image);
#ifdef BINARY
                cv::imshow("img bin", output_image);
#else
                cv::imshow("img gray", output_image);
#endif
                slice_initial_time = ev->t;
                cv::waitKey(1);
                counter++;
#ifdef BINARY
                output_image = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
                output_image.at<uchar>(event_to_build.y, event_to_build.x) = 255;
#else
                output_image = cv::Mat::zeros(cv::Size(width, height), CV_16UC1);
                uint16_t& count_ref = output_image.at<uint16_t>(event_to_build.y, event_to_build.x);
                if (count_ref < 65535) count_ref++;
#endif
            } else {
#ifdef BINARY
                output_image.at<uchar>(event_to_build.y, event_to_build.x) = 255;
#else
                uint16_t& count_ref = output_image.at<uint16_t>(event_to_build.y, event_to_build.x);
                if (count_ref < 65535) count_ref++;
#endif
            }
        }
    });

    // Start camera
    params.camera.start();
    start_time = std::chrono::steady_clock::now();

    // Main processing loop (similar to original)
    while (params.camera.is_running()) {
        // Poll Metavision events
        Metavision::EventLoop::poll_and_dispatch(1);
    }

    end_time = std::chrono::steady_clock::now();
    // Print final results
    cv::imwrite(output_images + std::to_string(counter) + ".png",
                output_image);

    // Cleanup
    if (params.camera.is_running()) {
        params.camera.stop();
    }

    // print tracker status
    if (tracker) {
        tracker->printFittersStatus();
    } else {
        std::cout << "No tracker initialized." << std::endl;
    }

    return 0;
}