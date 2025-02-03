//
// Created by viciopoli on 03/02/25.
//
#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <random>
#include <algorithm>

// Include FINUFFT header. Adjust the include path as needed.
#include "finufft.h"

TEST(NUFFT_STANDALONE, Estimation) {
    // -------------------------------
    // 1. Generate non-uniform time samples and signals
    // -------------------------------
    const int N = 1000;         // Number of samples
    const double t_min = 0.0;
    const double t_max = 6.0;

    // Random number generators for uniform sampling and noise.
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> uniform_dist(t_min, t_max);
    std::normal_distribution<double> noise_dist(0.0, 0.2);

    // Generate non-uniform time samples in [0,10]
    std::vector<double> t(N);
    for (int i = 0; i < N; i++) {
        t[i] = uniform_dist(rng);
    }
    std::sort(t.begin(), t.end());

    // True signal parameters
    const double omega_true = 2 * M_PI * 1000.0;  // 1 Hz angular frequency
    const double A_true = 2.0;               // Amplitude
    const double phi_true = M_PI / 2;           // Phase shift for S_y
    const double c_x_true = 1.0;               // Offset for S_x
    const double c_y_true = 0.5;               // Offset for S_y

    // Generate signals S_x and S_y with added noise.
    std::vector<double> Sx(N), Sy(N);
    for (int i = 0; i < N; i++) {
        Sx[i] = A_true * sin(omega_true * t[i]) + c_x_true + noise_dist(rng);
        Sy[i] = A_true * sin(omega_true * t[i] + phi_true) + c_y_true + noise_dist(rng);
    }

    // -------------------------------
    // 2. Rescale time to [-π, π] for FINUFFT
    // -------------------------------
    std::vector<double> t_scaled(N);
    for (int i = 0; i < N; i++) {
        t_scaled[i] = 2 * M_PI * (t[i] - t_min) / (t_max - t_min) - M_PI;
    }

    // -------------------------------
    // 3. Estimate ω using FINUFFT (nufft1d1 on Sx)
    // -------------------------------
    // We'll use 2*N Fourier modes.
    const int n_modes = 2 * 10000;
    // Build the frequency grid corresponding to FINUFFT output.
    // FINUFFT returns modes ordered from -n_modes/2 to n_modes/2-1.
    std::vector<int> freqs(n_modes);
    for (int k = 0; k < n_modes; k++) {
        freqs[k] = k - n_modes / 2;
    }

    auto opts = new finufft_opts;
    finufft_default_opts(opts);

    // Remove the DC offset from Sx and prepare a complex array for FINUFFT.
    double Sx_mean = 0.0;
    for (int i = 0; i < N; i++) {
        Sx_mean += Sx[i];
    }
    Sx_mean /= N;

    std::vector<std::complex<double>> Sx_complex(N);
    for (int i = 0; i < N; i++) {
        Sx_complex[i] = Sx[i] - Sx_mean;
    }

    // Allocate output array for Fourier coefficients.
    std::vector<std::complex<double>> Fx(n_modes);

    // FINUFFT parameters:
    int iflag = 1;         // sign in the exponential
    double eps = 1e-6;     // desired accuracy

    // Call the 1D type-1 NUFFT:
    int ier = finufft1d1(N, t_scaled.data(), Sx_complex.data(), iflag, eps, n_modes, Fx.data(), opts);

    EXPECT_TRUE(ier == 0);

    // Find the peak frequency (largest magnitude coefficient)
    int peak_idx = 0;
    double max_val = 0.0;
    for (int k = 0; k < n_modes; k++) {
        double mag = std::abs(Fx[k]);
        if (mag > max_val) {
            max_val = mag;
            peak_idx = k;
        }
    }
    int k_peak = freqs[peak_idx];

    // Convert Fourier mode to angular frequency in the original time domain.
    double a_scale = 2 * M_PI / (t_max - t_min);
    double omega_est = k_peak * a_scale;

    // -------------------------------
    // 4. Estimate A and c_x using linear least-squares fitting on Sx.
    // Model: f(t) = A*sin(omega_est*t) + c.
    // -------------------------------
    double S11 = 0.0, S12 = 0.0, S22 = N;
    double S1y = 0.0, S2y = 0.0;
    for (int i = 0; i < N; i++) {
        double s = sin(omega_est * t[i]);
        S11 += s * s;
        S12 += s;
        S1y += s * Sx[i];
        S2y += Sx[i];
    }
    double det = S11 * S22 - S12 * S12;
    double A_est = (S22 * S1y - S12 * S2y) / det;
    double c_x_est = (S11 * S2y - S12 * S1y) / det;

    // -------------------------------
    // 5. Estimate φ by computing the NUFFT of Sy and comparing phases.
    // -------------------------------
    double Sy_mean = 0.0;
    for (int i = 0; i < N; i++) {
        Sy_mean += Sy[i];
    }
    Sy_mean /= N;

    std::vector<std::complex<double>> Sy_complex(N);
    for (int i = 0; i < N; i++) {
        Sy_complex[i] = Sy[i] - Sy_mean;
    }
    std::vector<std::complex<double>> Fy(n_modes);
    ier = finufft1d1(N, t_scaled.data(), Sy_complex.data(), iflag, eps, n_modes, Fy.data(), opts);

    EXPECT_TRUE(ier == 0);

    double phase_x = std::arg(Fx[peak_idx]);
    double phase_y = std::arg(Fy[peak_idx]);
    double phi_est = fmod(phase_y - phase_x, 2 * M_PI);
    if (phi_est < 0)
        phi_est += 2 * M_PI;

    // Estimate c_y as the mean of Sy.
    double c_y_est = Sy_mean;

    // -------------------------------
    // 6. Resolve amplitude/phase ambiguities.
    // -------------------------------
    // If omega_est or A_est is negative, flip the sign and adjust the phase by π.
    if (omega_est < 0) {
        omega_est = -omega_est;
        phi_est = fmod(phi_est + M_PI, 2 * M_PI);
    }
    if (A_est < 0) {
        A_est = -A_est;
        phi_est = fmod(phi_est + M_PI, 2 * M_PI);
    }

    // -------------------------------
    // 7. Print results
    // -------------------------------
    std::cout << "True ω: " << omega_true << ", Estimated ω: " << omega_est << "\n";
    std::cout << "True A: " << A_true << ", Estimated A: " << A_est << "\n";
    std::cout << "True φ: " << phi_true << ", Estimated φ: " << phi_est << "\n";
    std::cout << "True c_x: " << c_x_true << ", Estimated c_x: " << c_x_est << "\n";
    std::cout << "True c_y: " << c_y_true << ", Estimated c_y: " << c_y_est << "\n";

    // ------------------------------- TEST
    // Check if the estimated parameters are close to the true values.
    const double tol = 1e-1;
    EXPECT_NEAR(omega_true, omega_est, tol);
    EXPECT_NEAR(A_true, A_est, tol);
    EXPECT_NEAR(phi_true, phi_est, tol);
    EXPECT_NEAR(c_x_true, c_x_est, tol);
    EXPECT_NEAR(c_y_true, c_y_est, tol);

}
