//
// Enhanced Event-based Motion Tracking with Proper Visualization
// Based on Metavision SDK patterns
//

#define FANCY_VISUALIZATION
//#define STORE

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
//#include "estimator/iekf_sinusoid_fitter.hpp"
#include "estimator/iekf_sinusoid_fitter_multi_harmonic.hpp"
#include "event_frontend/undistort.hpp"
#include "haste_wrapper.hpp"
#include "profiler.hpp"


// Constants
namespace {
    constexpr double DEFAULT_FPS = 100.0;
    constexpr std::uint32_t DEFAULT_ACCUMULATION = 5000;
    constexpr int ESC_KEY = 27;
    constexpr int POLL_TIMEOUT_MS = 10;
}


class MultipleNUFFT {
public:
    MultipleNUFFT(int x, int y, double first_event_t, unsigned short width, unsigned short height, bool is_front) :
            width(width), height(height), is_front(is_front) {
        nufft_estimator = std::make_unique<NUFFTHelixEstimator>(MIN_FREQUENCY, MAX_FREQUENCY, MAX_HARMONICS);
        tracker = std::make_unique<HasteWrapper<Metavision::EventCD>>(x, y, TRACKER_RATE, first_event_t);
        // output in red
        std::cout << "\033[1;31m" << "INIT: " << (is_front ? "FRONT" : "BACK") << "\033[0m" << std::endl;
    }

    bool isFront() { return is_front; }

    void feed(Metavision::EventCD &event_to_build, double current_t_sec) {
        if (!tracker->feed(event_to_build)) {
            return;
        }
        if(time_<0){time_ = current_t_sec;}

        if (NUFFT_ESTIMATION_DONE) [[likely]] {

            if (tracker->getRelEstimate(event_to_build.t, current_t_sec, x_pred, y_pred)) {
                auto x_new = static_cast<unsigned short>(event_to_build.x - x_pred);
                if (x_new < 0 || x_new >= width) {
                    return; // Skip if out of bounds
                }
                auto y_new = static_cast<unsigned short>(event_to_build.y - y_pred);
                if (y_new < 0 || y_new >= height) {
                    return; // Skip if out of bounds
                }
                event_to_build.x = x_new;
                event_to_build.y = y_new;
                event_to_build.p = 0;
                amplitudes.emplace_back(tracker->getAmplitude());
                return;
            }

        } else {
            if (nufft_estimator->feed(std::move(tracker->getCentroids()))) {
                nufft_estimator->printResults();
                NUFFTHelixEstimator::extractHarmonicParameters(nufft_estimator->getHarmonics(), Ax, Ay, Bx, By,
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
                NUFFT_ESTIMATION_DONE = nufft_estimator->done();
            }
        }
    }

    std::vector<double> medianFilterWithPadding(const std::vector<double> &data, int window_size) {
        if (data.empty()) {
            return {};
        }

        if (window_size <= 0 || window_size % 2 == 0) {
            throw std::invalid_argument("Window size must be positive and odd");
        }

        std::vector<double> result(data.size());
        int half_window = window_size / 2;

        for (size_t i = 0; i < data.size(); ++i) {
            std::vector<double> window;

            // Fill window with padding for edge cases
            for (int j = -half_window; j <= half_window; ++j) {
                int idx = static_cast<int>(i) + j;
                if (idx < 0) {
                    window.push_back(data[0]); // Pad with first value
                } else if (idx >= static_cast<int>(data.size())) {
                    window.push_back(data.back()); // Pad with last value
                } else {
                    window.push_back(data[idx]);
                }
            }

            // Sort and find median
            std::sort(window.begin(), window.end());
            result[i] = window[window.size() / 2];
        }

        return result;
    }


    [[nodiscard]] double avgAmplitude() {
        // return the meadian filter of the amplitudes
        if (amplitudes.empty()) return 0.0;
        auto res = medianFilterWithPadding(amplitudes, 5);
        double sum = std::accumulate(res.begin(), res.end(), 0.0);
        return sum / res.size();
    }

    [[nodiscard]] double stdAmplitude() {
        if (amplitudes.empty()) return 0.0;
        double mean = avgAmplitude();
        double accum = 0.0;
        for (const auto &a: medianFilterWithPadding(amplitudes, 5)) {
            accum += (a - mean) * (a - mean);
        }
        return std::sqrt(accum / amplitudes.size());
    }

    std::unique_ptr<HasteWrapper<Metavision::EventCD>> tracker;
private:
    std::unique_ptr<NUFFTHelixEstimator> nufft_estimator;
    double x_pred = 0, y_pred = 0;
    std::vector<double> Ax, Ay, Bx, By, omegas, offsets, amplitudes;
    const unsigned short width, height;

    double time_ = -1;

    bool NUFFT_ESTIMATION_DONE = false, is_front = false;
};

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

//    params.camera.biases().set_from_file("/home/viciopoli/Documents/metavision/biases/biases_filtered.bias");

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

    std::vector<std::pair<unsigned short, unsigned short>> tracker_centers;

    std::vector<std::shared_ptr<MultipleNUFFT>> trackers;
    for (int i = 0; i < params.params->trackers_x.size(); i++) {
        std::cout << "Initializing tracker with params from command line: "
                  << params.params->trackers_x[i] << ", " << params.params->trackers_y[i] << std::endl;
        tracker_centers.emplace_back(params.params->trackers_x[i],
                                     params.params->trackers_y[i]);
        trackers.push_back(
                std::make_shared<MultipleNUFFT>(params.params->trackers_x[i],
                                                params.params->trackers_y[i],
                                                first_event_t, width, height,
                                                i < 1)); // the first 4 trackers are front
    }

    std::cout << "All trackers initialized." << std::endl;

    bool is_init_front = true;
    // Mouse callback for tracker initialization
    std::function<void(int, int)> mouse_callback = [&](const int x, const int y) {
        std::lock_guard<std::mutex> lock(processing_mutex);
        std::cout << "Mouse clicked at (" << x << ", " << y << "), no info on the front-back, adding back" << std::endl;
        tracker_centers.emplace_back(x, y);
        trackers.push_back(
                std::make_shared<MultipleNUFFT>(x, y, first_event_t, width, height, is_init_front));
    };

    cv::setMouseCallback(window_name, receiveMouseEvent, &mouse_callback);

    bool osd = false; // On-screen display toggle
    bool tracker_enable = true;


    Metavision::Stage::EventBuffer compensated_events;
    unsigned short x_undistorted, y_undistorted, t_centre_x, t_centre_y;


    std::once_flag init_flag;
    // Main event processing callback
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {

        std::lock_guard<std::mutex> lock(processing_mutex);
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
            for (auto &tracker: trackers) {
                tracker->feed(event_to_build, current_t_sec);
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

    std::string front_init_string = "Click to init front tracker";
    // Main processing loop (similar to original)
    while (params.camera.is_running()) {
        // Display frame with thread safety
        {
            std::unique_lock<std::mutex> lock(cd_frame_mutex);
            if (!cd_frame.empty()) {
                cv::Mat display_frame;
                cd_frame.copyTo(display_frame);

                if (osd) {
                    std::lock_guard<std::mutex> lock(processing_mutex);
                    // Add on-screen display info
                    std::string text = Metavision::getHumanReadableTime(cd_frame_ts);
                    text += "     ";
                    text += Metavision::getHumanReadableRate(avg_rate);

                    cv::putText(display_frame, text, cv::Point(10, 20),
                                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(108, 143, 255), 1, cv::LINE_AA);

                    // Add tracker info if available
                    if (!trackers.empty()) {
                        int i = 0;
                        int color_main = trackers.back()->tracker->color();
                        const cv::Scalar main_color(128, 200, 128);
                        for (const auto &tracker: trackers) {
                            i++;
                            tracker->tracker->getCurrentPosition(t_centre_x, t_centre_y);
//                            std::cout << i << " " << tracker->color() << std::endl;
                            const auto rel_color = tracker->tracker->color() / double(color_main);
                            // write the text with the tracker color
//                            cv::putText(display_frame, "Tracker " + std::to_string(i) + ": " +
//                                                std::to_string(t_centre_x) + ", " + std::to_string(t_centre_y),
//                                        cv::Point(10, 20 + i * 20),
//                                        cv::FONT_HERSHEY_PLAIN, 1,
//                                        cv::Scalar(
//                                                std::clamp(main_color[0] * rel_color, 0.0, 255.0),
//                                                std::clamp(main_color[1] * rel_color, 0.0, 255.0),
//                                                std::clamp(main_color[2] * rel_color, 0.0, 255.0)
//                                        ), 1, cv::LINE_AA);


                            cv::rectangle(display_frame,
                                          cv::Point(t_centre_x - half_size, t_centre_y - half_size + 1),
                                          cv::Point(t_centre_x + half_size, t_centre_y + half_size + 1),
                                          cv::Scalar(
                                                  std::clamp(main_color[0] * rel_color, 0.0, 255.0),
                                                  std::clamp(main_color[1] * rel_color, 0.0, 255.0),
                                                  std::clamp(main_color[2] * rel_color, 0.0, 255.0)
                                          ), 2);
                        }

                        cv::putText(display_frame, "Tracker: Initialized", cv::Point(10, 40),
                                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                    } else {
                        cv::putText(display_frame, "Click to initialize tracker", cv::Point(10, 40),
                                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                    }
                }

                cv::putText(display_frame, front_init_string, cv::Point(10, height - 10),
                            cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);

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
            case 'i': {
                is_init_front = !is_init_front;
                front_init_string = is_init_front ? "Click to init front tracker" : "Click to init back tracker";
                std::cout << "Next tracker will be initialized as " << (is_init_front ? "front" : "back") << std::endl;
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

    for (auto tracker: trackers) {
        tracker->tracker->printFittersStatus();
    }

    std::vector<double> avg_amplitudes_front, avg_amplitudes_back, var_amplitudes_front, var_amplitudes_back;
    double var_sum_front = 0.0, var_sum_back = 0.0;
    for (const auto &tracker: trackers) {
        if (tracker->isFront()) {
            auto var = std::pow(tracker->stdAmplitude(), 2);
            avg_amplitudes_front.push_back(tracker->avgAmplitude() * 1. / var);
            var_sum_front += var;
            var_amplitudes_front.push_back(var);
            std::cout << "Tracker front amplitude avg (front): " << tracker->avgAmplitude() << std::endl;
            std::cout << "Tracker front amplitude std (front): " << tracker->stdAmplitude() << std::endl;
        } else {
            auto var = std::pow(tracker->stdAmplitude(), 2);
            avg_amplitudes_back.push_back(tracker->avgAmplitude() * 1. / var);
            var_sum_back += var;
            var_amplitudes_back.push_back(var);
            std::cout << "Tracker front amplitude avg (back): " << tracker->avgAmplitude() << std::endl;
            std::cout << "Tracker front amplitude std (back): " << tracker->stdAmplitude() << std::endl;
        }
    }
    std::cout << "Back size: " << avg_amplitudes_back.size() << ", Front size: " << avg_amplitudes_front.size()
              << std::endl;
    // compute the average amplitude across all trackers
    double avg_front = std::accumulate(avg_amplitudes_front.begin(), avg_amplitudes_front.end(), 0.0) / var_sum_front;
    double avg_back = std::accumulate(avg_amplitudes_back.begin(), avg_amplitudes_back.end(), 0.0) / var_sum_back;
    std::cout << "Average amplitude front: " << avg_front << std::endl;
    std::cout << "Average amplitude back: " << avg_back << std::endl;
    std::cout << "Average amplitude ratio (front/back): " << avg_front / avg_back << std::endl;

    return 0;
}