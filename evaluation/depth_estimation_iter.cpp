/**
 * @file depth_test.cpp
 * @brief Statistical Depth Estimation via Motion Parallax.
 *
 * This script processes an event stream to track multiple features on two distinct planes
 * (Front and Back). It calculates the amplitude of oscillation for each feature and
 * computes the relative depth ratio (Back/Front) using Inverse Variance Weighting
 * to fuse the noisy measurements from multiple trackers.
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

#include <metavision/sdk/core/utils/cd_frame_generator.h>
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
#include <numeric>

// Project Headers
#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
#include "estimator/iekf_sinusoid_fitter_multi_harmonic.hpp"
#include "event_frontend/undistort.hpp"
#include "haste_wrapper.hpp"
#include "profiler.hpp"

// Constants
namespace {
    constexpr double DEFAULT_FPS = 100.0;
    constexpr int POLL_TIMEOUT_MS = 10;
}

/**
 * @class MultipleNUFFT
 * @brief Manages a single feature tracker and its associated frequency estimators.
 */
class MultipleNUFFT {
public:
    MultipleNUFFT(int x, int y, double first_event_t, unsigned short width, unsigned short height, bool is_front) :
            width(width), height(height), is_front(is_front) {

        nufft_estimator = std::make_unique<NUFFTHelixEstimator>(MIN_FREQUENCY, MAX_FREQUENCY, MAX_HARMONICS);
        tracker = std::make_unique<HasteWrapper<Metavision::EventCD>>(x, y, TRACKER_RATE, first_event_t);

        // Visual debug output
        std::cout << (is_front ? "\033[1;32m[INIT] FRONT Tracker" : "\033[1;31m[INIT] BACK Tracker")
                  << " @ (" << x << "," << y << ")\033[0m" << std::endl;
    }

    bool isFront() const { return is_front; }

    /**
     * @brief Feeds an event into the tracking/estimation pipeline.
     */
    void feed(Metavision::EventCD &event_to_build, double current_t_sec) {
        // 1. Update Spatial Tracker
        if (!tracker->feed(event_to_build)) {
            return;
        }

        if (time_ < 0) { time_ = current_t_sec; }

        // 2. Estimation Phase
        if (NUFFT_ESTIMATION_DONE) [[likely]] {
            // IEKF Running: Collect Amplitude Samples
            if (tracker->getRelEstimate(event_to_build.t, current_t_sec, x_pred, y_pred)) {

                // Sample amplitude at 1kHz to avoid over-sampling noise
                if (current_t_sec - time_ > 0.001) {
                    amplitudes.emplace_back(tracker->getAmplitude());
                    time_ = current_t_sec;
                }
                return;
            }
        }
            // 3. Initialization Phase
        else {
            // NUFFT Running: Accumulate Centroids
            if (nufft_estimator->feed(std::move(tracker->getCentroids()))) {
                nufft_estimator->printResults();
                NUFFTHelixEstimator::extractHarmonicParameters(nufft_estimator->getHarmonics(), Ax, Ay, Bx, By,
                                                               omegas, offsets);

                if (!Ax.empty()) {
                    // Seed initial amplitude guess
                    auto a_x = nufft_estimator->getHarmonics()[0].amplitude_x;
                    auto a_y = nufft_estimator->getHarmonics()[0].amplitude_y;
                    amplitudes.emplace_back(std::sqrt(std::pow(a_x, 2) + std::pow(a_y, 2)));

                    // Initialize IEKF
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

    /** @return Average amplitude (Simple Mean). Consider Trimmed Mean for robustness. */
    [[nodiscard]] double avgAmplitude() const {
        if (amplitudes.empty()) return 0.0;
        double sum = std::accumulate(amplitudes.begin(), amplitudes.end(), 0.0);
        return sum / amplitudes.size();
    }

    /** * @return Sample Standard Deviation of the amplitude.
     * Uses Bessel's correction (N-1) for unbiased estimation.
     */
    [[nodiscard]] double stdAmplitude() const {
        if (amplitudes.size() < 2) return 0.0; // Need at least 2 points for std dev

        double mean = avgAmplitude();
        double accum = 0.0;
        for (const auto &a: amplitudes) {
            accum += (a - mean) * (a - mean);
        }
        // FIX: Use size() - 1 for sample standard deviation
        return std::sqrt(accum / (amplitudes.size() - 1));
    }

private:
    std::unique_ptr<HasteWrapper<Metavision::EventCD>> tracker;
    std::unique_ptr<NUFFTHelixEstimator> nufft_estimator;
    double x_pred = 0, y_pred = 0;
    std::vector<double> Ax, Ay, Bx, By, omegas, offsets, amplitudes;
    const unsigned short width, height;

    double time_ = -1;
    bool NUFFT_ESTIMATION_DONE = false;
    const bool is_front;
};

/**
 * @brief Fuses multiple statistical measurements using Inverse Variance Weighting.
 * * This gives more weight to measurements with low variance (high certainty).
 * Formula:
 * Mean = Sum(Mean_i / Var_i) / Sum(1 / Var_i)
 * Var  = 1 / Sum(1 / Var_i)
 *
 * @param means Vector of mean values from different trackers.
 * @param stds Vector of standard deviations from different trackers.
 * @return pair<double, double> {Fused Mean, Fused Standard Deviation}
 */
static std::pair<double, double> stats_fusion(const std::vector<double> &means, const std::vector<double> &stds) {
    if (means.size() != stds.size()) {
        throw std::invalid_argument("Means and standard deviations must have the same size");
    }

    double sum_weights = 0.0;
    double weighted_mean_sum = 0.0;

    for (size_t i = 0; i < means.size(); ++i) {
        if (stds[i] <= 1e-9) continue; // Avoid division by zero

        double weight = 1.0 / (stds[i] * stds[i]); // Weight = 1/Variance

        sum_weights += weight;
        weighted_mean_sum += means[i] * weight;
    }

    if (sum_weights == 0) return {0.0, 0.0};

    double final_mean = weighted_mean_sum / sum_weights;
    double final_std = std::sqrt(1.0 / sum_weights);

    return {final_mean, final_std};
}

// Global Timing & Stats
static std::chrono::steady_clock::time_point end_time, start_time;
static Metavision::timestamp first_event_t = 0, last_event_t = 0;

void print_on_exit() {
    // Only print if we actually started processing
    if (start_time.time_since_epoch().count() == 0) return;

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "\033[1;33mProcessing complete.\033[0m" << std::endl;
}

void signal_handler(int signal) {
    const char *signal_name = (signal == SIGTERM) ? "SIGTERM" : "SIGINT";
    std::cout << "\n" << signal_name << " received..." << std::endl;
    std::exit(signal);
}

/**
 * @brief Runs a single pass of depth estimation on the input file/stream.
 * @return The estimated Back/Front amplitude ratio.
 */
double test_depth(int argc, char *argv[]) {
    // Initialize parameters (re-parses args each call, acceptable for batch testing)
    VibES::ParamsLoader params(argc, argv);

    // Note: We don't print params every time to keep console clean
    static bool first_run = true;
    if(first_run) { std::cout << params; first_run = false; }

    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    Undistort undistort(params.params->calib_file);

    std::mutex processing_mutex;
    std::vector<std::shared_ptr<MultipleNUFFT>> trackers;

    // Initialize Trackers
    for (size_t i = 0; i < params.params->trackers_x.size(); i++) {
        // Determine if this tracker belongs to the 'Front' or 'Back' group
        // based on the command line argument --front-trackers N
        bool is_front_plane = (i < params.params->front_trackers);

        trackers.push_back(std::make_shared<MultipleNUFFT>(
                params.params->trackers_x[i],
                params.params->trackers_y[i],
                first_event_t, width, height,
                is_front_plane
        ));
    }

    Metavision::Stage::EventBuffer compensated_events;
    unsigned short x_undistorted, y_undistorted;
    std::once_flag init_flag;

    // --- Main Callback ---
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::lock_guard<std::mutex> lock(processing_mutex);

        std::call_once(init_flag, [&]() {
            start_time = std::chrono::steady_clock::now();
            first_event_t = begin->t;
        });

        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            last_event_t = ev->t;

            undistort(ev->x, ev->y, x_undistorted, y_undistorted);

            if (x_undistorted < 0 || x_undistorted >= width || y_undistorted < 0 || y_undistorted >= height) {
                continue;
            }

            // Create a temporary event copy to feed into trackers
            // (We reconstruct it because HASTE might modify it)
            Metavision::EventCD event_temp = *ev;
            event_temp.x = x_undistorted;
            event_temp.y = y_undistorted;

            const float current_t_sec = static_cast<float>(ev->t - first_event_t) / 1e6f;

            for (auto &tracker: trackers) {
                tracker->feed(event_temp, current_t_sec);
            }
        }
    });

    // Run Processing
    params.camera.start();
    while (params.camera.is_running()) {
        // Minimal sleep to allow callback to run
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (params.camera.is_running()) {
        params.camera.stop();
    }
    end_time = std::chrono::steady_clock::now();


    // --- Post-Processing Statistics ---
    std::vector<double> avg_amps_front, avg_amps_back;
    std::vector<double> std_amps_front, std_amps_back;

    for (const auto &tracker: trackers) {
        double avg = tracker->avgAmplitude();
        double std = tracker->stdAmplitude();

        if (tracker->isFront()) {
            avg_amps_front.push_back(avg);
            std_amps_front.push_back(std);
        } else {
            avg_amps_back.push_back(avg);
            std_amps_back.push_back(std);
        }
    }

    // Fuse statistics
    auto [avg_front, std_front] = stats_fusion(avg_amps_front, std_amps_front);
    auto [avg_back, std_back] = stats_fusion(avg_amps_back, std_amps_back);

    // Calculate Ratio: Back / Front
    // Note: Error propagation for Ratio R = B/F
    // sigma_R = R * sqrt( (sigma_B/B)^2 + (sigma_F/F)^2 )
    double ratio = (avg_front > 0) ? (avg_back / avg_front) : 0.0;
    double ratio_error = 0.0;

    if (avg_front > 0 && avg_back > 0) {
        ratio_error = ratio * std::sqrt( std::pow(std_front/avg_front, 2) + std::pow(std_back/avg_back, 2) );
    }

    std::cout << "  Front Amp: " << std::fixed << std::setprecision(3) << avg_front << " +/- " << std_front << std::endl;
    std::cout << "  Back Amp:  " << avg_back << " +/- " << std_back << std::endl;
    std::cout << "  Ratio:     " << ratio << " +/- " << ratio_error << std::endl;

    return ratio;
}

int main(int argc, char *argv[]) {
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);
    std::atexit(print_on_exit);

    std::cout << "========================================" << std::endl;
    std::cout << "   VibES Statistical Depth Estimator    " << std::endl;
    std::cout << "========================================" << std::endl;

    // Run the estimation multiple times for statistical robustness on recorded files
    const int N_TRIALS = 10;
    std::vector<double> ratios;
    ratios.reserve(N_TRIALS);

    for (int i = 0; i < N_TRIALS; ++i) {
        std::cout << "\n[Run " << (i + 1) << "/" << N_TRIALS << "] Processing..." << std::endl;
        try {
            double r = test_depth(argc, argv);
            if (r > 0) ratios.push_back(r);
        } catch (const std::exception& e) {
            std::cerr << "Run failed: " << e.what() << std::endl;
        }
    }

    if (ratios.empty()) {
        std::cerr << "No valid ratios computed." << std::endl;
        return 1;
    }

    // Final Statistics across trials
    double avg_ratio = std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size();

    double std_ratio = 0.0;
    if (ratios.size() > 1) {
        for (const auto &r: ratios) {
            std_ratio += (r - avg_ratio) * (r - avg_ratio);
        }
        std_ratio = std::sqrt(std_ratio / (ratios.size() - 1));
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << " FINAL RESULTS (" << ratios.size() << " valid runs)" << std::endl;
    std::cout << " Average Ratio: " << avg_ratio << " +/- " << std_ratio << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}