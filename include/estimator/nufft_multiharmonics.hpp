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
#include "event_frontend/centroid_base.h"

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
    // Constructor
    NUFFTHelixEstimator(double f_min_hz, double f_max_hz, int max_harmonics = 3)
            : N(500), f_min(Hz2rad(f_min_hz)), f_max(Hz2rad(f_max_hz)),
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
        delete opts.release(); // Clean up finufft_opts

        if (!done_) {
            std::cerr << "Warning: NUFFTHelixEstimator destroyed without calling compute()." << std::endl;
        }
    }


    bool feed(std::vector<Centroid> &&events) {
        std::ranges::for_each(events, [this](Centroid &event) {
            if (feed(event)) {
                return true;
            }
        });
        return false;
    }

    // Feed new event data
    bool feed(const Centroid &event) {
        if (index > 0 && event.t <= t_data[index - 1]) {
//            std::cerr << "Current event time: " << event.t
//                      << ", Previous event time: " << t_data[index - 1] << std::endl;
//            std::cerr << "Warning: Non-increasing time samples detected." << std::endl;
            return false;
        }

        t_data[index] = event.t;
        x_data[index] = event.x;
        y_data[index] = event.y;

        sum_x += event.x;
        sum_y += event.y;

        ++index;
        if (index >= N) {
//            std::cout << "Collected enough samples, starting computation..." << std::endl;
            return true; // compute();
        }
        return false;
    }

    // Main computation function
    bool compute() {
        if (index < N) {
            std::cerr << "Error: Not enough samples collected." << std::endl;
            return false;
        }

        double t_max = *std::max_element(t_data.begin(), t_data.end());
        double t_min = *std::min_element(t_data.begin(), t_data.end());

        if (std::abs(t_max - t_min) < 1e-10) {
            std::cerr << "Error: Time range too small." << std::endl;
            return false;
        }

        const double dt = t_max - t_min;
        const double scaling = 2 * M_PI / dt;

        // Rescale time points to [-pi, pi]
        std::vector<double> t_scaled(N);
        for (int i = 0; i < N; ++i) {
            t_scaled[i] = scaling * (t_data[i] - t_min) - M_PI;
        }

        // Compute means
        mean_x = sum_x / N;
        mean_y = sum_y / N;

        // Prepare complex data (remove DC component)
        std::vector<std::complex<double>> x_complex(N), y_complex(N);
        for (int i = 0; i < N; ++i) {
            x_complex[i] = x_data[i] - mean_x;
            y_complex[i] = y_data[i] - mean_y;
        }

        // Perform NUFFT for all three dimensions
        const int iflag = 1;
        const double eps = 1e-9;

        int ier_x = finufft1d1(N, t_scaled.data(), x_complex.data(), iflag, eps, n_modes, F_out_x.data(), opts.get());
        int ier_y = finufft1d1(N, t_scaled.data(), y_complex.data(), iflag, eps, n_modes, F_out_y.data(), opts.get());

        if (ier_x != 0 || ier_y != 0) {
            std::cerr << "FINUFFT error - X: " << ier_x << ", Y: " << ier_y << std::endl;
            return false;
        }

        // Find dominant harmonics
        if (!findDominantHarmonics(scaling)) {
            return false;
        }

        // Refine estimates using least squares
//        refineEstimates();

//        reset();
        done_ = true;
        return done_;
    }

    // Get the estimated harmonics
    const std::vector<HarmonicEstimate> &getHarmonics() const {
        return harmonics_;
    }

    // Get fundamental frequency
    double getFundamentalFreqHz() const {
        if (harmonics_.empty()) return 0.0;
        return rad2Hz(harmonics_[0].frequency);
    }

    double getFundamentalFreqRad() const {
        if (harmonics_.empty()) return 0.0;
        return harmonics_[0].frequency;
    }

    double getWindowEMA() const {
        return _window_ema;
    }

    bool done() { return done_; }

    // Get offsets
    std::tuple<double, double> getOffsets() const {
        return std::make_tuple(mean_x, mean_y);
    }

    // Initialize IEKF with estimated harmonics
    Eigen::VectorXd getIEKFInitialState(int num_harmonics_requested) const {
        if (harmonics_.empty()) {
            throw std::runtime_error("No harmonics estimated yet.");
        }

        int actual_harmonics = std::min(num_harmonics_requested, static_cast<int>(harmonics_.size()));
        int state_size = 6 * actual_harmonics + 4;

        Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_size);

        // Set harmonic coefficients
        for (int h = 0; h < actual_harmonics; ++h) {
            const auto &harmonic = harmonics_[h];

            // For each dimension (X, Y, T)
            for (int dim = 0; dim < 3; ++dim) {
                int base_idx = h * 6 + dim * 2;

                double amplitude, phase;
                if (dim == 0) {
                    amplitude = harmonic.amplitude_x;
                    phase = harmonic.phase_x;
                } else if (dim == 1) {
                    amplitude = harmonic.amplitude_y;
                    phase = harmonic.phase_y;
                }

                // Convert A*sin(wt + phi) to A_sin*sin(wt) + A_cos*cos(wt)
                // A*sin(wt + phi) = A*sin(phi)*cos(wt) + A*cos(phi)*sin(wt)
                initial_state(base_idx) = amplitude * std::cos(phase);     // A coefficient (sin term)
                initial_state(base_idx + 1) = amplitude * std::sin(phase); // B coefficient (cos term)
            }
        }

        // Set omega (fundamental frequency)
        initial_state(state_size - 4) = harmonics_[0].frequency / (actual_harmonics > 0 ? 1.0 : 1.0);

        // Set offsets
        initial_state(state_size - 2) = mean_x;  // Cx
        initial_state(state_size - 1) = mean_y;  // Cy

        return initial_state;
    }

    // Print results
    void printResults() const {
        std::cout << "\033[1;34m=== NUFFT Helix Harmonic Analysis ===" << std::endl;
        std::cout << "Offsets: X=" << mean_x << ", Y=" << mean_y << std::endl;
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

    // Data storage
    std::vector<double> x_data, y_data, t_data;
    std::vector<double> freqs_grid;

    int n_modes;
    std::vector<std::complex<double>> F_out_x, F_out_y;
    std::unique_ptr<finufft_opts, FinufftOptsDeleter> opts;

    // State variables
    int index = 0;
    double sum_x = 0.0, sum_y = 0.0;
    double mean_x = 0.0, mean_y = 0.0;

    bool done_ = false;

    // Results
    std::vector<HarmonicEstimate> harmonics_;

    void reset() {
        index = 0;
        sum_x = sum_y = 0.0;
        mean_x = mean_y = 0.0;
        harmonics_.clear();
    }

    bool findDominantHarmonics(double scaling) {
        // Find peaks in the combined spectrum
        std::vector<std::pair<int, double>> peak_candidates;

        for (int k = 1; k < n_modes - 1; ++k) {
            double freq = freqs_grid[k] * scaling; // * scaling;
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
        harmonics_.reserve(num_harmonics);

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

            harmonics_.push_back(harmonic);
        }

        // Sort harmonics by frequency
        std::sort(harmonics_.begin(), harmonics_.end(),
                  [](const auto &a, const auto &b) { return a.magnitude > b.magnitude; });

        return !harmonics_.empty();
    }

    void refineEstimates() {
        // Refine each harmonic estimate using least squares
        for (auto &harmonic: harmonics_) {
            double omega = harmonic.frequency;

            // Refine X dimension
            auto [A_x, B_x, offset_x] = leastSquaresHarmonic(omega, x_data, mean_x);
            harmonic.amplitude_x = std::sqrt(A_x * A_x + B_x * B_x);
            harmonic.phase_x = std::atan2(B_x, A_x);
            harmonic.offset_x = offset_x;

            // Refine Y dimension
            auto [A_y, B_y, offset_y] = leastSquaresHarmonic(omega, y_data, mean_y);
            harmonic.amplitude_y = std::sqrt(A_y * A_y + B_y * B_y);
            harmonic.phase_y = std::atan2(B_y, A_y);
            harmonic.offset_y = offset_y;
        }
    }

    std::tuple<double, double, double> leastSquaresHarmonic(double omega,
                                                            const std::vector<double> &data,
                                                            double mean_val) {
        // Fit: y = A*sin(ωt) + B*cos(ωt) + C
        double S11 = 0.0, S12 = 0.0, S13 = 0.0;  // sin-sin, sin-cos, sin-1
        double S22 = 0.0, S23 = 0.0;              // cos-cos, cos-1
        double S33 = static_cast<double>(N);       // 1-1
        double S1y = 0.0, S2y = 0.0, S3y = 0.0;   // sin-y, cos-y, 1-y

        for (int i = 0; i < N; ++i) {
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