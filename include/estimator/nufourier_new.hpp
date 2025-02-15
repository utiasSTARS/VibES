//
// Created by viciopoli on 18/11/24.
//

#ifndef PROJECT_NUFOURIER_H
#define PROJECT_NUFOURIER_H

#include <finufft.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <complex>
#include <opencv2/opencv.hpp>
#include <memory>
#include <tuple>
#include <algorithm>
#include "utils.hpp"  // Assumes definition of rad2Hz<T>(...) and similar utilities

// Custom deleter for finufft_opts
struct FinufftOptsDeleter {
    void operator()(finufft_opts *opts) const {
        delete opts;
    }
};

class FourierFreqEst {
public:
    // Delete default constructor – parameters must be specified.
    FourierFreqEst() = delete;

    // Constructor: n_samples is required. f_min and f_max are preserved for legacy or visualization.
    FourierFreqEst(int n_samples, double f_min_hz, double f_max_hz)
            : N(n_samples), f_min(Hz2rad(f_min_hz)), f_max(Hz2rad(f_max_hz)),
              t_data(n_samples), sx(n_samples), sy(n_samples),
              n_modes(2 * static_cast<int>(Hz2rad(f_max_hz))), F_out(n_modes),
              opts(new finufft_opts, FinufftOptsDeleter()) {
        // Initialize the frequency grid for visualization.
        freqs_t.resize(n_modes);
        for (int k = 0; k < n_modes; ++k) {
            freqs_t[k] = k - n_modes / 2;
        }
        finufft_default_opts(opts.get());
    }

    ~FourierFreqEst() = default;

    // Feed new event data. Returns true if a complete batch is reached and computed.
    bool feed(double x, double y, Time time) {
        auto t = double(time);
        // make sure the time is increasing
        if (t < t_data[index]) {
            std::cerr << "Error: Time samples are not increasing." << std::endl;
            return false;
        }

        t_data[index] = t;
        sx[index] = x;
        sy[index] = y;
        mean_x += x;
        mean_y += y;

        ++index;
        if (index >= N) {
            index = 0;
            return compute();
        }
        return false;
    }

    // Compute the dominant frequency and other parameters using 1D NUFFT.
    bool compute() {
        double t_max = t_data.back();
        double t_min = t_data.front();

        if (t_max == t_min) {
            std::cerr << "Error: All time samples are identical." << std::endl;
            return false;
        }

        const double dt = t_max - t_min;

        // Rescale time points to [-pi, pi].
        const double scaling = 2 * M_PI / dt;
        std::vector<double> t_scaled(N);
        for (int i = 0; i < N; ++i) {
            t_scaled[i] = scaling * (t_data[i] - t_min) - M_PI;
        }

        // Compute means.
        mean_x /= N;
        mean_y /= N;

        // Remove DC component: build complex vectors for sx and sy.
        std::vector<std::complex<double>> Sx_complex(N), Sy_complex(N);
        for (int i = 0; i < N; ++i) {
            Sx_complex[i] = sx[i] - mean_x;
            Sy_complex[i] = sy[i] - mean_y;
        }

        // --- Step 3: Perform 1D NUFFT on Sx ---
        const int iflag = 1;
        const double eps = 1e-6;
        int ier = finufft1d1(N, t_scaled.data(), Sx_complex.data(), iflag, eps, n_modes, F_out.data(), opts.get());
        if (ier != 0) {
            std::cerr << "FINUFFT error (Sx transform): " << ier << std::endl;
            return false;
        }

        // --- Step 4: Find the dominant Fourier mode in Sx within frequency bounds ---
        double omega_est = 0.0;
        double max_magnitude = 0.0;
        for (int k = 0; k < n_modes; ++k) {
            double freq = freqs_t[k] * scaling;  // Convert index to frequency
            if (freq >= f_min && freq <= f_max) {
                double magnitude = std::abs(F_out[k]);
                if (magnitude > max_magnitude) {
                    max_magnitude = magnitude;
                    omega_est = freq;
                }
            }
        }
        // --- Step 5: Estimate amplitude (A) and offset (c_x) using least squares ---
        double S11 = 0.0, S12 = 0.0, S22 = static_cast<double>(N);
        double S1y = 0.0, S2y = 0.0;
        for (int i = 0; i < N; i++) {
            double s = sin(omega_est * t_data[i]);
            S11 += s * s;
            S12 += s;
            S1y += s * sx[i];
            S2y += sx[i];
        }
        double det = S11 * S22 - S12 * S12;
        if (std::abs(det) < 1e-10) {
            std::cerr << "Error: Singular matrix in least squares fitting." << std::endl;
            return false;
        }
        double A_est = (S22 * S1y - S12 * S2y) / det;
        double c_x_est = (S11 * S2y - S12 * S1y) / det;

        // --- Step 6: Perform 1D NUFFT on Sy to extract phase information ---
        std::vector<std::complex<double>> F_out_y(n_modes);
        ier = finufft1d1(N, t_scaled.data(), Sy_complex.data(), iflag, eps, n_modes, F_out_y.data(), opts.get());
        if (ier != 0) {
            std::cerr << "FINUFFT error (Sy transform): " << ier << std::endl;
            return false;
        }

        // Use the Fourier peak from Sx to get phase information.
        // (Assumes the same index gives the dominant contribution in both transforms.)
        double phase_x = std::arg(F_out[findPeakIndex(F_out)]);
        double phase_y = std::arg(F_out_y[findPeakIndex(F_out)]);
        double phi_est = std::fmod(phase_y - phase_x, 2 * M_PI);
        if (phi_est < 0)
            phi_est += 2 * M_PI;

        // --- Step 7: Resolve ambiguities ---
        if (omega_est < 0) {
            omega_est = -omega_est;
            phi_est = std::fmod(phi_est + M_PI, 2 * M_PI);
        }
        if (A_est < 0) {
            A_est = -A_est;
            phi_est = std::fmod(phi_est + M_PI, 2 * M_PI);
        }

        // --- Step 8: Save the estimated parameters ---
        main_freq_t = omega_est;
        phase_shift = phi_est;
        amplitude = A_est;
        offset_x = c_x_est;
        offset_y = mean_y;

        std::cout << "Estimated ω: " << main_freq_t << " rad/s, " << rad2Hz(main_freq_t) << " Hz" << std::endl;
        std::cout << "Estimated Amplitude: " << amplitude << ", Offset (Sx): " << offset_x << std::endl;
        std::cout << "Estimated Phase Shift: " << phase_shift << " rad" << std::endl;
        std::cout << "Estimated Offset (Sy): " << offset_y << std::endl;

        // Reset data for next batch.
        std::fill(t_data.begin(), t_data.end(), 0.0);
        std::fill(sx.begin(), sx.end(), 0.0);
        std::fill(sy.begin(), sy.end(), 0.0);
        mean_x = 0.0;
        mean_y = 0.0;

        return true;
    }

    // Accessor methods.
    double getMainFreqHz() const {
        return rad2Hz(main_freq_t);
    }

    double getMainFreqRad() const {
        return main_freq_t;
    }

    double getPhaseShift() const {
        return phase_shift;
    }

    double getAmplitude() const {
        return amplitude;
    }

    std::tuple<double, double> getOffset() const {
        return std::make_tuple(offset_x, offset_y);
    }

    // A simple visualization function for debugging.
    void visualize_data(const std::vector<double> &Sx, const std::vector<double> &t_vals) {
        const int width = 800, height = 400;
        cv::Mat plot = cv::Mat::zeros(height, width, CV_8UC3);

        // Compute min/max for normalization.
        auto [t_min_it, t_max_it] = std::minmax_element(t_vals.begin(), t_vals.end());
        auto [Sx_min_it, Sx_max_it] = std::minmax_element(Sx.begin(), Sx.end());
        double t_min_val = *t_min_it, t_max_val = *t_max_it;
        double Sx_min_val = *Sx_min_it, Sx_max_val = *Sx_max_it;

        for (size_t i = 0; i < t_vals.size(); i++) {
            int x = static_cast<int>(((t_vals[i] - t_min_val) / (t_max_val - t_min_val)) * (width - 1));
            int y = height - 1 - static_cast<int>(((Sx[i] - Sx_min_val) / (Sx_max_val - Sx_min_val)) * (height - 1));
            cv::circle(plot, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
        }
        cv::imshow("Sx vs Time", plot);
        cv::waitKey(1);
    }

private:
    const int N;  // Number of samples.

    // Data storage.
    std::vector<double> t_data;              // Time stamps.
    std::vector<int> freqs_t;                // Frequency grid for visualization.
    std::vector<double> sx, sy;              // Signal data.
    int n_modes;                           // Number of Fourier modes.
    std::vector<std::complex<double>> F_out; // NUFFT output for Sx.

    // NUFFT options wrapped in a unique_ptr.
    std::unique_ptr<finufft_opts, FinufftOptsDeleter> opts;

    // Counters and indices.
    int index = 0;

    // Estimated parameters.
    double main_freq_t = 0.0;    // Angular frequency (rad/s) from Sx.
    double phase_shift = 0.0;    // Phase difference between Sx and Sy.
    double amplitude = 0.0;      // Amplitude from least squares.
    double offset_x = 0.0;       // Offset for Sx from LS fit.
    double offset_y = 0.0;       // Offset (mean) for Sy.

    // Accumulation variables.
    double mean_x = 0.0;
    double mean_y = 0.0;

    // Legacy frequency bounds.
    const double f_min = 0.0, f_max = 0.0;

    // Helper: find the index of the peak in a Fourier spectrum.
    int findPeakIndex(const std::vector<std::complex<double>> &F) const {
        int peak_idx = 0;
        double max_val = 0.0;
        for (int k = 0; k < static_cast<int>(F.size()); ++k) {
            double mag = std::abs(F[k]);
            if (mag > max_val) {
                max_val = mag;
                peak_idx = k;
            }
        }
        return peak_idx;
    }
};

#endif // PROJECT_NUFOURIER_H
