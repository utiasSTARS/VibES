#include <metavision/sdk/core/utils/rate_estimator.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/core/pipeline/stage.h>
#include <metavision/sdk/core/utils/misc.h>

#include <opencv2/imgproc.hpp>

#include <mutex>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <csignal>

#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
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
        if (time_ < 0) { time_ = current_t_sec; }

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
                if (current_t_sec - time_ > 0.001) {
                    amplitudes.emplace_back(tracker->getAmplitude());
                    time_ = current_t_sec;
                }
                return;
            }

        } else {
            if (nufft_estimator->feed(std::move(tracker->getCentroids()))) {
                nufft_estimator->printResults();
                NUFFTHelixEstimator::extractHarmonicParameters(nufft_estimator->getHarmonics(), Ax, Ay, Bx, By,
                                                               omegas, offsets);

                if (!Ax.empty()) {
                    auto a_x = nufft_estimator->getHarmonics()[0].amplitude_x;
                    auto a_y = nufft_estimator->getHarmonics()[0].amplitude_y;
                    amplitudes.emplace_back(std::sqrt(std::pow(a_x, 2) + std::pow(a_y, 2)));
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
//        auto res = medianFilterWithPadding(amplitudes, 5);
        double sum = std::accumulate(amplitudes.begin(), amplitudes.end(), 0.0);
        return sum / amplitudes.size();
    }

    [[nodiscard]] double stdAmplitude() {
        if (amplitudes.empty()) return 0.0;
        double mean = avgAmplitude();
        double accum = 0.0;
        for (const auto &a: amplitudes) {
            accum += (a - mean) * (a - mean);
        }
        return std::sqrt(accum / (amplitudes.size() + 1));
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


static std::pair<double, double> stats_fusion(std::vector<double> &means, std::vector<double> &stds) {
    if (means.size() != stds.size()) {
        throw std::invalid_argument("Means and standard deviations must have the same size");
    }

    double final_var_inv = 0.0;
    double final_mean = 0.0;
    for (size_t i = 0; i < means.size(); ++i) {
        if (stds[i] <= 0) {
            throw std::invalid_argument("Standard deviation must be positive");
        }
        auto var = stds[i] * stds[i];
        final_var_inv += 1.0 / var;
        final_mean += means[i] / var;
    }
    final_mean = final_mean / final_var_inv;
    return {final_mean, std::sqrt(1.0 / final_var_inv)};
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

double test_depth(int argc, char *argv[]) {
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);
    std::atexit(print_on_exit);

    // Initialize parameters and camera
    VibES::ParamsLoader params(argc, argv);
    std::cout << params;

    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    // Initialize undistortion
    Undistort undistort(params.params->calib_file);

    // Setup CD frame generator
    cv::Mat cd_frame;

    // Initialize fitters and estimator
    std::mutex processing_mutex;

    std::vector<std::pair<unsigned short, unsigned short>> tracker_centers;

    std::vector<std::shared_ptr<MultipleNUFFT>> trackers;
    for (int i = 0; i < params.params->trackers_x.size(); i++) {
        tracker_centers.emplace_back(params.params->trackers_x[i],
                                     params.params->trackers_y[i]);
        trackers.push_back(
                std::make_shared<MultipleNUFFT>(params.params->trackers_x[i],
                                                params.params->trackers_y[i],
                                                first_event_t, width, height,
                                                i < params.params->front_trackers));
    }

    Metavision::Stage::EventBuffer compensated_events;
    unsigned short x_undistorted, y_undistorted;


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
            // Make sure the undistorted coordinates are within the image bounds
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
    });

    // Start camera
    params.camera.start();
    start_time = std::chrono::steady_clock::now();

    // Main processing loop (similar to original)
    while (params.camera.is_running()) {
        // Display frame with thread safety
        // Poll Metavision events
        Metavision::EventLoop::poll_and_dispatch(1);
    }

    end_time = std::chrono::steady_clock::now();

    // Cleanup
    if (params.camera.is_running()) {
        params.camera.stop();
    }

    std::vector<double> avg_amplitudes_front, avg_amplitudes_back, std_amplitudes_front, std_amplitudes_back;
    for (const auto &tracker: trackers) {
        if (tracker->isFront()) {
            avg_amplitudes_front.push_back(tracker->avgAmplitude());
            std_amplitudes_front.push_back(tracker->stdAmplitude());
            std::cout << "Tracker front amplitude std (front): " << tracker->stdAmplitude() << std::endl;
        } else {
            avg_amplitudes_back.push_back(tracker->avgAmplitude());
            std_amplitudes_back.push_back(tracker->stdAmplitude());
            std::cout << "Tracker front amplitude std (back): " << tracker->stdAmplitude() << std::endl;
        }
    }

    // compute the average amplitude across all trackers
    auto [avg_front, std_front] = stats_fusion(avg_amplitudes_front, std_amplitudes_front);
    auto [avg_back, std_back] = stats_fusion(avg_amplitudes_back, std_amplitudes_back);
    std::cout << "Average amplitude front: " << avg_front << " std: " << std_front << std::endl;
    std::cout << "Average amplitude back: " << avg_back << " std: " << std_back << std::endl;
    std::cout << "Amplitude ratio (back/front): " << avg_back / avg_front << std::endl;
    // compute the error on the ratio using proper error propagation
    double ratio_error = std::sqrt(std_front * std_front / (avg_front * avg_front) +
                                   std_back * std_back / (avg_back * avg_back)) *
                         (avg_back / avg_front);
    std::cout << "Amplitude ratio error: " << ratio_error << std::endl;

    return avg_back / avg_front;
}

int main(int argc, char *argv[]) {
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);
    std::atexit(print_on_exit);

    std::cout << "Starting depth estimation test..." << std::endl;

    std::vector<double> ratios;
    for (int i = 0; i < 10; ++i) {
        ratios.push_back(test_depth(argc, argv));
    }

    // Fixed statistics calculation with proper sample standard deviation
    double avg_ratio = std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size();
    double std_ratio = 0.0;
    for (const auto &ratio: ratios) {
        std_ratio += (ratio - avg_ratio) * (ratio - avg_ratio);
    }
    std_ratio = std::sqrt(std_ratio / (ratios.size() - 1));
    std::cout << " ------ Statistics of ratios ------ " << std::endl << std::endl;
    std::cout << "Average ratio (back/front): " << avg_ratio << "\\pm" << std_ratio << std::endl;

    return 0;
}