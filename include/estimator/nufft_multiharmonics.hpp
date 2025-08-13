#ifndef PROJECT_NUFFT_HELIX_ESTIMATOR_H
#define PROJECT_NUFFT_HELIX_ESTIMATOR_H

#include <finufft.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <complex>
#include <opencv2/opencv.hpp>
#include <memory>
#include <tuple>
#include <algorithm>
#include <Eigen/Dense>
#include "utils.hpp"  // Assumes definition of rad2Hz<T>(...) and similar utilities

#ifdef TIMING
#include "profiler.hpp"
#endif

// Custom deleter for finufft_opts
struct FinufftOptsDeleter {
    void operator()(finufft_opts *opts) const {
        delete opts;
    }
};

// Structure to hold harmonic estimation results
struct HarmonicEstimate {
    double frequency;      // Angular frequency (rad/s)
    double amplitude_x;    // Amplitude in X dimension
    double amplitude_y;    // Amplitude in Y dimension
    double phase_x;        // Phase in X dimension
    double phase_y;        // Phase in Y dimension
    double magnitude;      // Overall magnitude for ranking
    double offset_x;      // Offset in X dimension
    double offset_y;      // Offset in Y dimension
};

class NUFFTHelixEstimator {
public:
    /**
    * Extracts harmonic parameters from NUFFT estimator results
    */
    static void extractHarmonicParameters(const std::vector<HarmonicEstimate> &harmonics,
                                          std::vector<double> &Ax, std::vector<double> &Ay,
                                          std::vector<double> &Bx, std::vector<double> &By,
                                          std::vector<double> &omegas, std::vector<double> &offsets) {

        for (const auto &harmonic: harmonics) {
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

    // Constructor
    NUFFTHelixEstimator(double f_min_hz, double f_max_hz, int max_harmonics = 3, int num_samples = 500)
            : N(num_samples), f_min(Hz2rad(f_min_hz)), f_max(Hz2rad(f_max_hz)),
              max_harmonics_(max_harmonics),
              t_data(N), x_data(N), y_data(N),
              n_modes(4 * static_cast<int>(Hz2rad(f_max_hz))), // Increased resolution
              F_out_x(n_modes), F_out_y(n_modes),
              opts(new finufft_opts, FinufftOptsDeleter()) {

        // Initialize the frequency grid properly
        freqs_grid.resize(n_modes);
        for (int k = 0; k < n_modes; ++k) {
            freqs_grid[k] = (k - n_modes / 2.0); // Use 2.0 instead of 2 for proper division
        }

        finufft_default_opts(opts.get());
        opts->debug = 0;  // Reduce verbosity

        reset();
    }

    ~NUFFTHelixEstimator() {
        // Signal shutdown and wait for thread to complete
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_requested_ = true;
        }
        cv_.notify_all();  // Wake up any waiting threads

        if (compute_thread_.joinable()) {
            compute_thread_.join();
        }

        if (!done_.load()) {
            std::cerr << "Warning: NUFFTHelixEstimator destroyed without completing computation." << std::endl;
        }
    }


    bool feed(std::queue<Centroid> &&events) {

        std::lock_guard<std::mutex> lock(mutex_);

        // If computation is already done, return true immediately
        if (done_.load()) {
            return true;
        }

        // If currently computing, return false (not done yet)
        if (computing_.load()) {
            return false;
        }

        while (!events.empty()) {
            auto event = events.front();
            events.pop();
            if (!(index_ > 0 && event.t < t_data[index_ - 1])) {

                // Add new data point
                t_data[index_] = event.t;
                x_data[index_] = event.x;
                y_data[index_] = event.y;

                sum_x_ += event.x;
                sum_y_ += event.y;

                ++index_;

                // Start computation if we have enough samples
                if (index_ >= N) {
                    std::cout << "Collected enough samples, starting computation..." << std::endl;
                    startComputation();
                    break;
                }
            }
        }
        return false;
    }

    bool feed(std::vector<Centroid> &&events) {

        std::lock_guard<std::mutex> lock(mutex_);

        // If computation is already done, return true immediately
        if (done_.load()) {
            return true;
        }

        // If currently computing, return false (not done yet)
        if (computing_.load()) {
            return false;
        }

        for (const auto event: events) {
            if (!(index_ > 0 && event.t < t_data[index_ - 1])) {

                // Add new data point
                t_data[index_] = event.t;
                x_data[index_] = event.x;
                y_data[index_] = event.y;

                sum_x_ += event.x;
                sum_y_ += event.y;

                ++index_;

                // Start computation if we have enough samples
                if (index_ >= N) {
                    std::cout << "Collected enough samples, starting computation..." << std::endl;
                    startComputation();
                    break;
                }
            }
        }
        return false;
    }

    // Feed new event data
    bool feed(const Centroid &event) {
        std::lock_guard<std::mutex> lock(mutex_);

        // If computation is already done, return true immediately
        if (done_.load()) {
            return true;
        }

        // If currently computing, return false (not done yet)
        if (computing_.load()) {
            return false;
        }

        // Check timestamp ordering
        if (index_ > 0 && event.t < t_data[index_ - 1]) {
            return false;
        }

        // Add new data point
        t_data[index_] = event.t;
        x_data[index_] = event.x;
        y_data[index_] = event.y;

        sum_x_ += event.x;
        sum_y_ += event.y;

        ++index_;

        // Start computation if we have enough samples
        if (index_ >= N) {
            std::cout << "Collected enough samples, starting computation..." << std::endl;
            startComputation();
        }

        return done_.load();
    }

    bool compute() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (done_.load()) {
            return true;
        }

        if (index_ < N) {
            return false; // Not enough samples yet
        }

        if (!computing_.load()) {
            startComputation();
        }

        return done_.load();
    }

    // Wait for computation to complete with timeout
    bool waitForCompletion(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return done_.load() || shutdown_requested_; });
    }

    // Get the estimated harmonics (thread-safe)
    std::vector<HarmonicEstimate> getHarmonics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return harmonics_;
    }

    // Get fundamental frequency
    double getFundamentalFreqHz() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (harmonics_.empty()) return 0.0;
        return rad2Hz(harmonics_[0].frequency);
    }

    double getFundamentalFreqRad() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (harmonics_.empty()) return 0.0;
        return harmonics_[0].frequency;
    }

    double getWindowEMA() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return _window_ema;
    }

    bool done() const {
        return done_.load();
    }

    // Get offsets
    std::tuple<double, double> getOffsets() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::make_tuple(mean_x_, mean_y_);
    }

    // Print results
    void printResults() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::cout << "\033[1;34m=== NUFFT Helix Harmonic Analysis ===" << std::endl;
        std::cout << "Offsets: X=" << mean_x_ << ", Y=" << mean_y_ << std::endl;
        std::cout << "Found " << harmonics_.size() << " dominant harmonics:" << std::endl;

        for (size_t i = 0; i < harmonics_.size(); ++i) {
            const auto &h = harmonics_[i];
            std::cout << "Harmonic " << (i + 1) << ":" << std::endl;
            std::cout << "  Frequency: " << rad2Hz(h.frequency) << " Hz (" << h.frequency << " rad/s)" << std::endl;
            std::cout << "  Amplitudes: X=" << h.amplitude_x << ", Y=" << h.amplitude_y << std::endl;
            std::cout << "  Phases: X=" << h.phase_x << ", Y=" << h.phase_y << std::endl;
            std::cout << "  Magnitude: " << h.magnitude << std::endl;
            std::cout << "  Offsets: X=" << h.offset_x << ", Y=" << h.offset_y << std::endl;
        }
        std::cout << "\033[0m" << std::endl;
    }

private:
    const int N;
    const double f_min, f_max;
    const int max_harmonics_;
    double _window_ema;

    // Thread synchronization - ADDED condition variable
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::atomic<bool> computing_{false};
    std::atomic<bool> done_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::thread compute_thread_;

    // Data storage (protected by mutex_)
    std::vector<double> x_data, y_data, t_data;
    std::vector<double> freqs_grid;

    int n_modes;
    std::vector<std::complex<double>> F_out_x, F_out_y;
    std::unique_ptr<finufft_opts, FinufftOptsDeleter> opts;

    // State variables (protected by mutex_)
    int index_ = 0;
    double sum_x_ = 0.0, sum_y_ = 0.0;
    double mean_x_ = 0.0, mean_y_ = 0.0;

    // Results (protected by mutex_)
    std::vector<HarmonicEstimate> harmonics_;

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        index_ = 0;
        sum_x_ = sum_y_ = 0.0;
        mean_x_ = mean_y_ = 0.0;
        harmonics_.clear();
        computing_ = false;
        done_ = false;
    }

    void startComputation() {
        // Must be called with mutex_ held
        if (!computing_.load()) {
            computing_ = true;
            if (compute_thread_.joinable()) {
                compute_thread_.join(); // Ensure previous thread is joined before starting a new one
            }
            compute_thread_ = std::thread(&NUFFTHelixEstimator::compute_thr, this);
        }
    }

    void compute_thr() {
#ifdef TIMING
        PROFILE_FUNCTION();
#endif
        try {
            // Create local copies of data to avoid holding mutex during computation
            std::vector<double> local_x_data, local_y_data, local_t_data;
            int local_index;
            double local_sum_x, local_sum_y;

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (shutdown_requested_.load()) {
                    computing_ = false;
                    return;
                }

                if (index_ < N) {
                    std::cerr << "Error: Not enough samples collected." << std::endl;
                    computing_ = false;
                    cv_.notify_all();
                    return;
                }

                local_x_data = x_data;
                local_y_data = y_data;
                local_t_data = t_data;
                local_sum_x = sum_x_;
                local_sum_y = sum_y_;
            }

            double t_max = *std::max_element(local_t_data.begin(), local_t_data.end());
            double t_min = *std::min_element(local_t_data.begin(), local_t_data.end());

            if (std::abs(t_max - t_min) < 1e-10) {
                std::cerr << "Error: Time range too small." << std::endl;
                std::lock_guard<std::mutex> lock(mutex_);
                computing_ = false;
                cv_.notify_all();
                return;
            }

            const double dt = t_max - t_min;
            const double scaling = 2 * M_PI / dt;

            // Rescale time points to [-pi, pi]
            std::vector<double> t_scaled(N);
            for (int i = 0; i < N; ++i) {
                t_scaled[i] = scaling * (local_t_data[i] - t_min) - M_PI;
            }

            // Compute means
            double local_mean_x = local_sum_x / N;
            double local_mean_y = local_sum_y / N;

            // Prepare complex data (remove DC component)
            std::vector<std::complex<double>> x_complex(N), y_complex(N);
            for (int i = 0; i < N; ++i) {
                x_complex[i] = local_x_data[i] - local_mean_x;
                y_complex[i] = local_y_data[i] - local_mean_y;
            }

            // Create local NUFFT output vectors
            std::vector<std::complex<double>> local_F_out_x(n_modes), local_F_out_y(n_modes);

            // Perform NUFFT for all dimensions
            const int iflag = 1;
            const double eps = 1e-9;

            int ier_x = finufft1d1(N, t_scaled.data(), x_complex.data(), iflag, eps, n_modes, local_F_out_x.data(),
                                   opts.get());
            int ier_y = finufft1d1(N, t_scaled.data(), y_complex.data(), iflag, eps, n_modes, local_F_out_y.data(),
                                   opts.get());

            if (ier_x != 0 || ier_y != 0) {
                std::cerr << "FINUFFT error - X: " << ier_x << ", Y: " << ier_y << std::endl;
                std::lock_guard<std::mutex> lock(mutex_);
                computing_ = false;
                cv_.notify_all();
                return;
            }

            // Find dominant harmonics
            std::vector<HarmonicEstimate> local_harmonics;
            if (!findDominantHarmonics(scaling, local_F_out_x, local_F_out_y, local_harmonics)) {
                std::lock_guard<std::mutex> lock(mutex_);
                computing_ = false;
                cv_.notify_all();
                return;
            }

            // FIXED: Update shared state with results and proper notification
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!shutdown_requested_.load()) {
                    refineEstimates(local_t_data, local_x_data, local_y_data,
                                    local_mean_x, local_mean_y, local_harmonics);
                    mean_x_ = local_mean_x;
                    mean_y_ = local_mean_y;
                    harmonics_ = std::move(local_harmonics);
                    done_ = true;  // Set done BEFORE computing = false
                }
                computing_ = false;
                cv_.notify_all();  // Notify waiting threads
            }

        } catch (const std::exception &e) {
            std::cerr << "Exception in compute thread: " << e.what() << std::endl;
            std::lock_guard<std::mutex> lock(mutex_);
            computing_ = false;
            cv_.notify_all();
        }
    }

    bool findDominantHarmonics(double scaling,
                               const std::vector<std::complex<double>> &F_out_x,
                               const std::vector<std::complex<double>> &F_out_y,
                               std::vector<HarmonicEstimate> &harmonics) {
        // Find peaks in the combined spectrum
        std::vector<std::pair<int, double>> peak_candidates;

        for (int k = 1; k < n_modes - 1; ++k) {
            double freq = freqs_grid[k] * scaling;
            if (freq < f_min || freq > f_max) continue;

            // Combined magnitude across all dimensions
            double mag_x = std::abs(F_out_x[k]);
            double mag_y = std::abs(F_out_y[k]);
            double combined_mag = std::sqrt(mag_x * mag_x + mag_y * mag_y);

            // Simple peak detection
            double mag_prev = std::sqrt(std::norm(F_out_x[k - 1]) + std::norm(F_out_y[k - 1]));
            double mag_next = std::sqrt(std::norm(F_out_x[k + 1]) + std::norm(F_out_y[k + 1]));

            if (combined_mag > mag_prev && combined_mag > mag_next && combined_mag > 0.01) {
                peak_candidates.push_back({k, combined_mag});
            }
        }

        if (peak_candidates.empty()) {
            std::cerr << "No significant peaks found in the spectrum." << std::endl;
            return false;
        }

        // Sort by magnitude
        std::sort(peak_candidates.begin(), peak_candidates.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        // Extract top harmonics
        int num_harmonics = std::min(max_harmonics_, static_cast<int>(peak_candidates.size()));
        harmonics.reserve(num_harmonics);

        for (int i = 0; i < num_harmonics; ++i) {
            int k = peak_candidates[i].first;
            double freq = std::abs(freqs_grid[k] * scaling);

            HarmonicEstimate harmonic;
            harmonic.frequency = freq;
            harmonic.magnitude = peak_candidates[i].second;

            // Extract amplitude and phase for each dimension
            harmonic.amplitude_x = 2.0 * std::abs(F_out_x[k]) / N;
            harmonic.amplitude_y = 2.0 * std::abs(F_out_y[k]) / N;

            harmonic.phase_x = std::arg(F_out_x[k]);
            harmonic.phase_y = std::arg(F_out_y[k]);

            harmonics.push_back(harmonic);
        }

        // Sort harmonics by frequency
        std::sort(harmonics.begin(), harmonics.end(),
                  [](const auto &a, const auto &b) { return a.magnitude > b.magnitude; });

        return !harmonics.empty();
    }

    static void refineEstimates(const std::vector<double> &t_data,
                         const std::vector<double> &x_data,
                         const std::vector<double> &y_data,
                         double mean_x, double mean_y,
                         std::vector<HarmonicEstimate> &harmonics) {
        // Refine each harmonic estimate using least squares
        for (auto &harmonic: harmonics) {
            double omega = harmonic.frequency;

            // Refine X dimension
            auto [A_x, B_x, offset_x] = leastSquaresHarmonic(omega, x_data, mean_x, t_data);
            harmonic.amplitude_x = std::sqrt(A_x * A_x + B_x * B_x);
            harmonic.phase_x = std::atan2(B_x, A_x);
            harmonic.offset_x = offset_x;

            // Refine Y dimension
            auto [A_y, B_y, offset_y] = leastSquaresHarmonic(omega, y_data, mean_y, t_data);
            harmonic.amplitude_y = std::sqrt(A_y * A_y + B_y * B_y);
            harmonic.phase_y = std::atan2(B_y, A_y);
            harmonic.offset_y = offset_y;
        }
    }

    static std::tuple<double, double, double> leastSquaresHarmonic(double omega,
                                                            const std::vector<double> &data,
                                                            double mean_val,
                                                            const std::vector<double> &t_data) {
        // Fit: y = A*sin(ωt) + B*cos(ωt) + C
        double S11 = 0.0, S12 = 0.0, S13 = 0.0;  // sin-sin, sin-cos, sin-1
        double S22 = 0.0, S23 = 0.0;              // cos-cos, cos-1
        auto n = data.size();
        auto S33 = static_cast<double>(n);    // 1-1
        double S1y = 0.0, S2y = 0.0, S3y = 0.0;   // sin-y, cos-y, 1-y

        for (int i = 0; i < n; ++i) {
            double s = std::sin(omega * t_data[i]);
            double c = std::cos(omega * t_data[i]);
            double y = data[i] - mean_val;

            S11 += s * s;
            S12 += s * c;
            S13 += s;
            S22 += c * c;
            S23 += c;
            S1y += s * y;
            S2y += c * y;
            S3y += y;
        }

        // Solve 3x3 system: [S11 S12 S13; S12 S22 S23; S13 S23 S33] * [A; B; C] = [S1y; S2y; S3y]
        Eigen::Matrix3d S;
        S << S11, S12, S13,
                S12, S22, S23,
                S13, S23, S33;

        Eigen::Vector3d rhs(S1y, S2y, S3y);

        if (S.determinant() < 1e-10) {
            return {0.0, 0.0, mean_val};
        }

        Eigen::Vector3d solution = S.ldlt().solve(rhs);
        return {solution(0), solution(1), solution(2)};
    }
};

#endif // PROJECT_NUFFT_HELIX_ESTIMATOR_H