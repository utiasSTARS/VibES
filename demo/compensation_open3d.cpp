//
// Enhanced Event-based Motion Tracking with Proper Visualization
// Based on Metavision SDK patterns
//

//#define FANCY_VISUALIZATION


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

#ifdef STORE

#include <metavision/sdk/driver/hdf5_event_file_writer.h>

#endif

#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
#include "estimator/iekf_sinusoid_fitter_multi_harmonic.hpp"
#include "event_frontend/undistort.hpp"
#include "haste_wrapper.hpp"
#include "utils.hpp"

#include "visualizer/open3d_visualizer.hpp"

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

    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    Open3DVisualizer visualizer(width, height);

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
        if (tracker || params.params->nocompensation) {
            return;
        }
        tracker = std::make_shared<HasteWrapper<Metavision::EventCD>>(x, y, TRACKER_RATE, first_event_t);
        std::cout << "Tracker initialized at (" << x << ", " << y << ")" << std::endl;
    };

    cv::setMouseCallback(window_name, receiveMouseEvent, &mouse_callback);

    bool osd = false; // On-screen display toggle
    bool in_tracker = true, tracker_enable = true;

    double x_pred = 0, y_pred = 0;

    Metavision::Stage::EventBuffer compensated_events;
    unsigned short x_undistorted, y_undistorted;
    unsigned short x_pose, y_pose;
    double start_time_vis = 0;

    std::once_flag init_flag;
    // Main event processing callback
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::call_once(init_flag, [&]() {
            start_time = std::chrono::steady_clock::now();
            first_event_t = begin->t;
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
#ifdef FANCY_VISUALIZATION
                    // if the event is in the left half of the image skip
                    if (event_to_build.x < visualization_cut_off) {
                        continue;
                    }
#endif
                    tracker->getCurrentPosition(x_pose, y_pose);
                    visualizer.addLine(x_pose, y_pose, (current_t_sec - start_time_vis) * 100, 0, 0, 1);
                    auto q = tracker->getCentroids();
                    while (!q.empty()) {
                        auto c = q.front();
                        visualizer.addLine2(c.x, c.y, (c.t - start_time_vis) * 100, 0, 1, 0);
                        q.pop();
                    }
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
                        event_to_build.p = 0;
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
                        start_time_vis = current_t_sec;
                        NUFFT_ESTIMATION_DONE = nufft_estimator.done();
                    }
                }
            }
        }
        // Feed events to frame generator and rate estimator
        const auto *begin_comp = compensated_events.data();
        const auto *end_comp = begin_comp + compensated_events.size();
#ifdef STORE
        hdf5_writer.add_events(begin_comp, end_comp);
#endif
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

        visualizer.update();
        // Poll Metavision events
        Metavision::EventLoop::poll_and_dispatch(1);
    }

    end_time = std::chrono::steady_clock::now();


    // Cleanup
    cd_frame_generator.stop();
    if (params.camera.is_running()) {
        params.camera.stop();
    }

    visualizer.loop();

    return 0;
}