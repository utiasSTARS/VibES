/**
 * @file nufft_helix_estimator.hpp
 * @brief Non-Uniform Fast Fourier Transform (NUFFT) estimator for multi-harmonic trajectory analysis.
 *
 * This file implements a robust spectral estimator that processes non-uniformly sampled
 * event-based data (timestamps and 2D positions) to identify dominant sinusoidal components.
 * It uses the FINUFFT library to perform the transform and a least-squares refinement step
 * to precisely extract amplitude, phase, and frequency parameters.
 *
 * Key features:
 * - Handles non-uniform sampling inherent to event cameras.
 * - Estimates multiple harmonics simultaneously.
 * - Refines estimates using linear least squares for high precision.
 * - Threaded computation to avoid blocking the main processing loop.
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

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
#include "utils.hpp"

#ifdef TIMING
#include "profiler.hpp"
#endif

/**
 * @brief Custom deleter for finufft_opts to ensure proper memory management with std::unique_ptr.
 */
struct FinufftOptsDeleter {
    void operator()(finufft_opts *opts) const {
        delete opts;
    }
};

/**
 * @brief Container for the parameters of a single detected harmonic component.
 */
struct HarmonicEstimate {
    double frequency;      ///< Angular frequency in rad/s.
    double amplitude_x;    ///< Amplitude of the sinusoidal component in the X dimension.
    double amplitude_y;    ///< Amplitude of the sinusoidal component in the Y dimension.
    double phase_x;        ///< Phase shift in the X dimension (radians).
    double phase_y;        ///< Phase shift in the Y dimension (radians).
    double magnitude;      ///< Combined spectral magnitude, used for ranking harmonic dominance.
    double offset_x;       ///< DC offset (bias) in the X dimension associated with this harmonic fit.
    double offset_y;       ///< DC offset (bias) in the Y dimension associated with this harmonic fit.
};

/**
 * @class NUFFTHelixEstimator
 * @brief Estimates orbital parameters (frequency, amplitude, phase) from 2D event streams.
 *
 * This class collects a batch of event centroids, performs a Non-Uniform FFT to find
 * dominant frequencies, and then refines the trajectory parameters using a linear
 * least-squares fit. It is designed to initialize tracking filters (like IEKF)
 * by providing accurate starting parameters without prior knowledge.
 */
class NUFFTHelixEstimator {
public:
    /**
     * @brief Extracts standard sinusoidal parameters from harmonic estimates.
     * * Converts the internal magnitude/phase representation into quadrature components
     * (A*sin + B*cos) suitable for Kalman Filter initialization.
     *
     * @param harmonics [in] Vector of harmonic estimates produced by the estimator.
     * @param Ax [out] Output vector for X-dimension sine amplitudes.
     * @param Ay [out] Output vector for Y-dimension sine amplitudes.
     * @param Bx [out] Output vector for X-dimension cosine amplitudes.
     * @param By [out] Output vector for Y-dimension cosine amplitudes.
     * @param omegas [out] Output vector for angular frequencies.
     * @param offsets [out] Output vector for DC offsets.
     */
    static void extractHarmonicParameters(const std::vector<HarmonicEstimate> &harmonics,
                                          std::vector<double> &Ax, std::vector<double> &Ay,
                                          std::vector<double> &Bx, std::vector<double> &By,
                                          std::vector<double> &omegas, std::vector<double> &offsets) {

        for (const auto &harmonic: harmonics) {
            // Reconstruct quadrature components from Amplitude/Phase
            // x(t) = Amp * cos(wt + phase) = A*sin(wt) + B*cos(wt)
            // Note: The mapping depends on the specific definition of phase in the NUFFT output.
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

    /**
     * @brief Constructor.
     *
     * @param f_min_hz Minimum frequency to search for (Hz).
     * @param f_max_hz Maximum frequency to search for (Hz).
     * @param max_harmonics Maximum number of dominant harmonics to extract.
     * @param num_samples Number of data points to collect before triggering computation.
     */
    NUFFTHelixEstimator(double f_min_hz, double f_max_hz, int max_harmonics = 3, int num_samples = 500)
            : N(num_samples), f_min(Hz2rad(f_min_hz)), f_max(Hz2rad(f_max_hz)),
              max_harmonics_(max_harmonics),
              t_data(N), x_data(N), y_data(N),
              n_modes(4 * static_cast<int>(Hz2rad(f_max_hz))), // Oversample modes for better resolution
              F_out_x(n_modes), F_out_y(n_modes),
              opts(new finufft_opts, FinufftOptsDeleter()) {

        // Initialize the frequency grid centered around DC
        freqs_grid.resize(n_modes);
        for (int k = 0; k < n_modes; ++k) {
            freqs_grid[k] = (k - n_modes / 2.0);
        }

        finufft_default_opts(opts.get());
        opts->debug = 0;  // Disable verbose debug output from FINUFFT

        reset();
    }

    /**
     * @brief Destructor. Ensures the background computation thread is joined properly.
     */
    ~NUFFTHelixEstimator() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_requested_ = true;
        }
        cv_.notify_all();

        if (compute_thread_.joinable()) {
            compute_thread_.join();
        }

        if (!done_.load()) {
            std::cerr << "[NUFFTHelixEstimator] Warning: Destroyed without completing computation." << std::endl;
        }
    }

    /**
     * @brief Feeds a queue of centroid data into the estimator buffer.
     *
     * @param events Queue of centroid measurements (moved).
     * @return true if computation is complete, false otherwise.
     */
    bool feed(std::queue<Centroid> &&events) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (done_.load()) return true;
        if (computing_.load()) return false;

        while (!events.empty()) {
            auto event = events.front();
            events.pop();
            // Ensure monotonic time ordering to prevent NUFFT artifacts
            if (!(index_ > 0 && event.t < t_data[index_ - 1])) {
                t_data[index_] = event.t;
                x_data[index_] = event.x;
                y_data[index_] = event.y;
                sum_x_ += event.x;
                sum_y_ += event.y;
                ++index_;

                if (index_ >= N) {
                    std::cout << "[NUFFT] Collected " << N << " samples, starting computation..." << std::endl;
                    startComputation();
                    break;
                }
            }
        }
        return false;
    }

    /**
     * @brief Feeds a vector of centroid data into the estimator buffer.
     *
     * @param events Vector of centroid measurements (moved).
     * @return true if computation is complete, false otherwise.
     */
    bool feed(std::vector<Centroid> &&events) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (done_.load()) return true;
        if (computing_.load()) return false;

        for (const auto event: events) {
            if (!(index_ > 0 && event.t < t_data[index_ - 1])) {
                t_data[index_] = event.t;
                x_data[index_] = event.x;
                y_data[index_] = event.y;
                sum_x_ += event.x;
                sum_y_ += event.y;
                ++index_;

                if (index_ >= N) {
                    std::cout << "[NUFFT] Collected " << N << " samples, starting computation..." << std::endl;
                    startComputation();
                    break;
                }
            }
        }
        return false;
    }

    /**
     * @brief Feeds a single centroid measurement into the estimator buffer.
     *
     * @param event The centroid measurement.
     * @return true if computation is complete, false otherwise.
     */
    bool feed(const Centroid &event) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (done_.load()) return true;
        if (computing_.load()) return false;

        // Check timestamp ordering
        if (index_ > 0 && event.t < t_data[index_ - 1]) {
            return false;
        }

        t_data[index_] = event.t;
        x_data[index_] = event.x;
        y_data[index_] = event.y;
        sum_x_ += event.x;
        sum_y_ += event.y;
        ++index_;

        if (index_ >= N) {
            std::cout << "[NUFFT] Collected " << N << " samples, starting computation..." << std::endl;
            startComputation();
        }

        return done_.load();
    }

    /**
     * @brief Manually triggers computation if enough data is present.
     * @return true if computation is complete.
     */
    bool compute() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (done_.load()) return true;
        if (index_ < N) return false;

        if (!computing_.load()) {
            startComputation();
        }

        return done_.load();
    }

    /**
     * @brief Blocks execution until the computation thread finishes or times out.
     *
     * @param timeout Maximum duration to wait.
     * @return true if computation finished, false if timed out.
     */
    bool waitForCompletion(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return done_.load() || shutdown_requested_; });
    }

    // --- Getters ---

    /** @return Vector of detected harmonic estimates (Thread-safe). */
    std::vector<HarmonicEstimate> getHarmonics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return harmonics_;
    }

    /** @return The frequency of the strongest harmonic in Hz. */
    double getFundamentalFreqHz() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (harmonics_.empty()) return 0.0;
        return rad2Hz(harmonics_[0].frequency);
    }

    /** @return The frequency of the strongest harmonic in rad/s. */
    double getFundamentalFreqRad() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (harmonics_.empty()) return 0.0;
        return harmonics_[0].frequency;
    }

    /** @return true if computation has finished successfully. */
    bool done() const {
        return done_.load();
    }

    /** @return Tuple containing the mean X and Y offsets. */
    std::tuple<double, double> getOffsets() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::make_tuple(mean_x_, mean_y_);
    }

    /**
     * @brief Prints the estimation results to standard output.
     */
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

    // Concurrency control
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::atomic<bool> computing_{false};
    std::atomic<bool> done_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::thread compute_thread_;

    // Data buffers
    std::vector<double> x_data, y_data, t_data;
    std::vector<double> freqs_grid;

    // NUFFT internals
    int n_modes;
    std::vector<std::complex<double>> F_out_x, F_out_y;
    std::unique_ptr<finufft_opts, FinufftOptsDeleter> opts;

    // Accumulators for mean calculation
    int index_ = 0;
    double sum_x_ = 0.0, sum_y_ = 0.0;
    double mean_x_ = 0.0, mean_y_ = 0.0;

    // Output results
    std::vector<HarmonicEstimate> harmonics_;

    /**
     * @brief Resets the internal state to accept new data.
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        index_ = 0;
        sum_x_ = sum_y_ = 0.0;
        mean_x_ = mean_y_ = 0.0;
        harmonics_.clear();
        computing_ = false;
        done_ = false;
    }

    /**
     * @brief Launches the computation thread.
     */
    void startComputation() {
        if (!computing_.load()) {
            computing_ = true;
            if (compute_thread_.joinable()) {
                compute_thread_.join();
            }
            compute_thread_ = std::thread(&NUFFTHelixEstimator::compute_thr, this);
        }
    }

    /**
     * @brief Main worker function for the computation thread.
     *
     * 1. Copies data locally to minimize mutex contention.
     * 2. Rescales time domain to [-pi, pi].
     * 3. Executes NUFFT (1D Type 1) on X and Y data.
     * 4. Identifies peaks in the frequency spectrum.
     * 5. Refines parameters using least-squares.
     */
    void compute_thr() {
#ifdef TIMING
        PROFILE_FUNCTION();
#endif
        try {
            // --- 1. Data Snapshot ---
            std::vector<double> local_x_data, local_y_data, local_t_data;
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

            // --- 2. Time Scaling ---
            const double dt = t_max - t_min;
            const double scaling = 2 * M_PI / dt;

            std::vector<double> t_scaled(N);
            for (int i = 0; i < N; ++i) {
                t_scaled[i] = scaling * (local_t_data[i] - t_min) - M_PI;
            }

            double local_mean_x = local_sum_x / N;
            double local_mean_y = local_sum_y / N;

            // Zero-center the data for FFT
            std::vector<std::complex<double>> x_complex(N), y_complex(N);
            for (int i = 0; i < N; ++i) {
                x_complex[i] = local_x_data[i] - local_mean_x;
                y_complex[i] = local_y_data[i] - local_mean_y;
            }

            std::vector<std::complex<double>> local_F_out_x(n_modes), local_F_out_y(n_modes);

            // --- 3. Execute FINUFFT ---
            const int iflag = 1; // Type 1: nonuniform to uniform
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

            // --- 4. Peak Detection ---
            std::vector<HarmonicEstimate> local_harmonics;
            if (!findDominantHarmonics(scaling, local_F_out_x, local_F_out_y, local_harmonics)) {
                std::lock_guard<std::mutex> lock(mutex_);
                computing_ = false;
                cv_.notify_all();
                return;
            }

            // --- 5. Refine & Publish Results ---
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!shutdown_requested_.load()) {
                    refineEstimates(local_t_data, local_x_data, local_y_data,
                                    local_mean_x, local_mean_y, local_harmonics);
                    mean_x_ = local_mean_x;
                    mean_y_ = local_mean_y;
                    harmonics_ = std::move(local_harmonics);
                    done_ = true;
                }
                computing_ = false;
                cv_.notify_all();
            }

        } catch (const std::exception &e) {
            std::cerr << "Exception in compute thread: " << e.what() << std::endl;
            std::lock_guard<std::mutex> lock(mutex_);
            computing_ = false;
            cv_.notify_all();
        }
    }

    /**
     * @brief Identifies spectral peaks in the NUFFT output.
     * @param scaling Time scaling factor used during NUFFT.
     * @param F_out_x NUFFT output for X dimension.
     * @param F_out_y NUFFT output for Y dimension.
     * @param harmonics Output vector for initial harmonic estimates.
     * @return true if peaks were found.
     */
    bool findDominantHarmonics(double scaling,
                               const std::vector<std::complex<double>> &F_out_x,
                               const std::vector<std::complex<double>> &F_out_y,
                               std::vector<HarmonicEstimate> &harmonics) {

        std::vector<std::pair<int, double>> peak_candidates;

        // Iterate through frequency bins (excluding boundaries)
        for (int k = 1; k < n_modes - 1; ++k) {
            double freq = freqs_grid[k] * scaling;
            if (freq < f_min || freq > f_max) continue;

            // Compute combined magnitude (L2 norm of X and Y components)
            double mag_x = std::abs(F_out_x[k]);
            double mag_y = std::abs(F_out_y[k]);
            double combined_mag = std::sqrt(mag_x * mag_x + mag_y * mag_y);

            // Check for local maximum
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

        // Sort candidates by magnitude (descending)
        std::sort(peak_candidates.begin(), peak_candidates.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        // Select top N harmonics
        int num_harmonics = std::min(max_harmonics_, static_cast<int>(peak_candidates.size()));
        harmonics.reserve(num_harmonics);

        for (int i = 0; i < num_harmonics; ++i) {
            int k = peak_candidates[i].first;
            double freq = std::abs(freqs_grid[k] * scaling);

            HarmonicEstimate harmonic;
            harmonic.frequency = freq;
            harmonic.magnitude = peak_candidates[i].second;

            // Initial amplitude/phase guesses from FFT coefficients
            harmonic.amplitude_x = 2.0 * std::abs(F_out_x[k]) / N;
            harmonic.amplitude_y = 2.0 * std::abs(F_out_y[k]) / N;
            harmonic.phase_x = std::arg(F_out_x[k]);
            harmonic.phase_y = std::arg(F_out_y[k]);

            harmonics.push_back(harmonic);
        }

        return !harmonics.empty();
    }

    /**
     * @brief Refines the initial spectral estimates using a precise Least Squares fit.
     */
    static void refineEstimates(const std::vector<double> &t_data,
                                const std::vector<double> &x_data,
                                const std::vector<double> &y_data,
                                double mean_x, double mean_y,
                                std::vector<HarmonicEstimate> &harmonics) {

        for (auto &harmonic: harmonics) {
            double omega = harmonic.frequency;

            // Fit: x(t) = A*sin(wt) + B*cos(wt) + C
            auto [A_x, B_x, offset_x] = leastSquaresHarmonic(omega, x_data, mean_x, t_data);
            harmonic.amplitude_x = std::sqrt(A_x * A_x + B_x * B_x);
            harmonic.phase_x = std::atan2(B_x, A_x);
            harmonic.offset_x = offset_x;

            auto [A_y, B_y, offset_y] = leastSquaresHarmonic(omega, y_data, mean_y, t_data);
            harmonic.amplitude_y = std::sqrt(A_y * A_y + B_y * B_y);
            harmonic.phase_y = std::atan2(B_y, A_y);
            harmonic.offset_y = offset_y;
        }
    }

    /**
     * @brief Solves the linear least squares problem for a single harmonic component.
     * * Solves for [A, B, C] in: data[i] = A*sin(w*t[i]) + B*cos(w*t[i]) + C
     *
     * @return Tuple containing {A, B, Offset}.
     */
    static std::tuple<double, double, double> leastSquaresHarmonic(double omega,
                                                                   const std::vector<double> &data,
                                                                   double mean_val,
                                                                   const std::vector<double> &t_data) {
        // Build the Normal Equations: (M^T * M) * X = M^T * Y
        // M = [sin(wt), cos(wt), 1]

        double S11 = 0.0, S12 = 0.0, S13 = 0.0;
        double S22 = 0.0, S23 = 0.0;
        auto n = data.size();
        auto S33 = static_cast<double>(n);
        double S1y = 0.0, S2y = 0.0, S3y = 0.0;

        for (int i = 0; i < n; ++i) {
            double s = std::sin(omega * t_data[i]);
            double c = std::cos(omega * t_data[i]);
            // Work with centered data to improve numerical stability, add mean back later
            double y = data[i] - mean_val;

            S11 += s * s; S12 += s * c; S13 += s;
            S22 += c * c; S23 += c;
            S1y += s * y; S2y += c * y; S3y += y;
        }

        Eigen::Matrix3d S;
        S << S11, S12, S13,
                S12, S22, S23,
                S13, S23, S33;

        Eigen::Vector3d rhs(S1y, S2y, S3y);

        if (std::abs(S.determinant()) < 1e-10) {
            return {0.0, 0.0, mean_val};
        }

        // Solve using LDLT decomposition (faster/stable for symmetric positive definite matrices)
        Eigen::Vector3d solution = S.ldlt().solve(rhs);

        // Return A, B, and the total offset (residual offset + original mean)
        return {solution(0), solution(1), solution(2) + mean_val};
    }
};

#endif // PROJECT_NUFFT_HELIX_ESTIMATOR_H