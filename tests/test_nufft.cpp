//
// Created by viciopoli on 08/07/25.
//

#include <gtest/gtest.h>
#include <random>
#include <algorithm>
#include <vector>

#include "estimator/nufft_multiharmonics.hpp"

const int SAMPLES = 10000;
const double time_granularity= 0.001;

double wrapToPi(double angle) {
    while (angle < 0) angle += 2 * M_PI;
    while (angle >= 2 * M_PI) angle -= 2 * M_PI;
    return angle;
}

// Helper function to generate non-uniform time samples
std::vector<double> generateNonUniformTimes(int n_samples, double t_start, double t_end, double jitter_factor = 0.5) {
    std::vector<double> times;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);

    double dt_nominal = (t_end - t_start) / (n_samples - 1);

    for (int i = 0; i < n_samples; ++i) {
        double t_nominal = t_start + i * dt_nominal;
        double jitter = jitter_factor * dt_nominal * uniform_dist(gen);
        times.push_back(t_nominal + jitter);
    }

    // Sort to ensure monotonic time
    std::sort(times.begin(), times.end());
    return times;
}

TEST(NUFFT, SingleHarmonicUniformSampling) {
    double f_min = 10.0;
    double f_max = 100.0;
    int max_harmonics = 4;
    NUFFTHelixEstimator nufft_estimator(f_min, f_max, max_harmonics);

    // Single harmonic parameters
    double amp = 4.0;
    double freq_hz = 80.0;
    double freq = Hz2rad(freq_hz);
    double phase = 1.0;
    double offset_x = 3.0;
    double offset_y = 2.0;

    for (int i = 0; i < SAMPLES;
         ++i) {
        double t = i * time_granularity;
        double x = amp * std::cos(freq * t + phase) + offset_x;
        double y = amp * std::sin(freq * t + phase) + offset_y;

        if (nufft_estimator.feed(Centroid(t, x, y))) {
            if (nufft_estimator.compute()) {
                auto harmonics = nufft_estimator.getHarmonics();

                EXPECT_EQ(harmonics.size(), 1);
                EXPECT_NEAR(harmonics[0].frequency, freq, 1.0);
                EXPECT_NEAR(harmonics[0].amplitude_x, amp, 1e-1);
                EXPECT_NEAR(harmonics[0].amplitude_y, amp, 1e-1);
                EXPECT_NEAR(wrapToPi(harmonics[0].phase_x), phase, 1.);
                EXPECT_NEAR(wrapToPi(harmonics[0].phase_y), phase, 1.);
                break;
            }
        }
    }
}

TEST(NUFFT, MultipleHarmonicsUniformSampling) {
    double f_min = 10.0;
    double f_max = 200.0;
    int max_harmonics = 4;
    NUFFTHelixEstimator nufft_estimator(f_min, f_max, max_harmonics);

    // Multiple harmonic parameters
    std::vector<double> amplitudes = {4.0, 2.0, 1.5};
    std::vector<double> frequencies_hz = {30.0, 60.0, 120.0};
    std::vector<double> phases = {0.0, M_PI / 4, M_PI / 2};
    double offset_x = 1.0;
    double offset_y = 0.5;

    for (int i = 0; i < SAMPLES; ++i) {
        double t = i * time_granularity; // Higher sampling rate for multiple harmonics
        double x = offset_x;
        double y = offset_y;

        // Sum multiple harmonics
        for (size_t h = 0; h < amplitudes.size(); ++h) {
            double freq = Hz2rad(frequencies_hz[h]);
            x += amplitudes[h] * std::cos(freq * t + phases[h]);
            y += amplitudes[h] * std::sin(freq * t + phases[h]);
        }

        if (nufft_estimator.feed(Centroid(t, x, y))) {
            if (nufft_estimator.compute()) {
                auto harmonics = nufft_estimator.getHarmonics();

                EXPECT_GE(harmonics.size(), 2); // Should detect at least 2 harmonics

                // Check that detected frequencies match expected ones (within tolerance)
                for (const auto &expected_freq_hz: frequencies_hz) {
                    double expected_freq = Hz2rad(expected_freq_hz);
                    bool found = false;
                    for (const auto &h: harmonics) {
                        if (std::abs(h.frequency - expected_freq) < 1e-1) {
                            found = true;
                            EXPECT_GT(h.amplitude_x, 0.5); // Should have reasonable amplitude
                            EXPECT_GT(h.amplitude_y, 0.5);
                            break;
                        }
                    }
                    EXPECT_TRUE(found) << "Expected frequency " << expected_freq_hz << " Hz not found";
                }
                break;
            }
        }
    }
}

TEST(NUFFT, NonUniformSamplingSingleHarmonic) {
    double f_min = 10.0;
    double f_max = 100.0;
    int max_harmonics = 4;
    NUFFTHelixEstimator nufft_estimator(f_min, f_max, max_harmonics);

    // Single harmonic parameters
    double amp = 3.0;
    double freq_hz = 45.0;
    double freq = Hz2rad(freq_hz);
    double phase = 0.7;
    double offset_x = 2.0;
    double offset_y = 1.5;

    // Generate non-uniform time samples
    auto times = generateNonUniformTimes(SAMPLES, 0.0, 10.0, 0.3);

    for (double t: times) {
        double x = amp * std::cos(freq * t + phase) + offset_x;
        double y = amp * std::sin(freq * t + phase) + offset_y;

        if (nufft_estimator.feed(Centroid(t, x, y))) {
            if (nufft_estimator.compute()) {
                auto harmonics = nufft_estimator.getHarmonics();

                EXPECT_GE(harmonics.size(), 1);
                EXPECT_NEAR(harmonics[0].frequency, freq, 1e-1); // Slightly relaxed tolerance for non-uniform
                EXPECT_NEAR(harmonics[0].amplitude_x, amp, 1e-1);
                EXPECT_NEAR(harmonics[0].amplitude_y, amp, 1e-1);
                break;
            }
        }
    }
}

TEST(NUFFT, NonUniformSamplingMultipleHarmonics) {
    double f_min = 5.0;
    double f_max = 150.0;
    int max_harmonics = 5;
    NUFFTHelixEstimator nufft_estimator(f_min, f_max, max_harmonics);

    // Multiple harmonic parameters
    std::vector<double> amplitudes = {3.0, 1.8, 1.2, 0.8};
    std::vector<double> frequencies_hz = {20.0, 40.0, 80.0, 100.0};
    std::vector<double> phases = {0.0, M_PI / 3, M_PI / 2, M_PI};
    double offset_x = 0.5;
    double offset_y = -0.3;

    // Generate non-uniform time samples with moderate jitter
    auto times = generateNonUniformTimes(SAMPLES, 0.0, 15.0, 0.4);

    for (double t: times) {
        double x = offset_x;
        double y = offset_y;

        // Sum multiple harmonics
        for (size_t h = 0; h < amplitudes.size(); ++h) {
            double freq = Hz2rad(frequencies_hz[h]);
            x += amplitudes[h] * std::cos(freq * t + phases[h]);
            y += amplitudes[h] * std::sin(freq * t + phases[h]);
        }

        if (nufft_estimator.feed(Centroid(t, x, y))) {
            if (nufft_estimator.compute()) {
                auto harmonics = nufft_estimator.getHarmonics();

                EXPECT_GE(harmonics.size(), 2); // Should detect at least 2 harmonics

                // Check for the strongest harmonics
                for (size_t i = 0; i < std::min(size_t(2), amplitudes.size()); ++i) {
                    double expected_freq = Hz2rad(frequencies_hz[i]);
                    bool found = false;
                    for (const auto &h: harmonics) {
                        if (std::abs(h.frequency - expected_freq) < 2e-1) {
                            found = true;
                            EXPECT_GT(h.amplitude_x, amplitudes[i] * 0.7); // Allow some tolerance
                            EXPECT_GT(h.amplitude_y, amplitudes[i] * 0.7);
                            break;
                        }
                    }
                    EXPECT_TRUE(found) << "Expected frequency " << frequencies_hz[i] << " Hz not found";
                }
                break;
            }
        }
    }
}

TEST(NUFFT, HighlyNonUniformSampling) {
    double f_min = 10.0;
    double f_max = 80.0;
    int max_harmonics = 3;
    NUFFTHelixEstimator nufft_estimator(f_min, f_max, max_harmonics);

    // Single harmonic with high non-uniformity
    double amp = 2.5;
    double freq_hz = 35.0;
    double freq = Hz2rad(freq_hz);
    double phase = 1.2;
    double offset_x = 1.0;
    double offset_y = 0.0;

    // Generate highly non-uniform time samples (high jitter)
    auto times = generateNonUniformTimes(SAMPLES, 0.0, 12.0, 0.8);

    for (double t: times) {
        double x = amp * std::cos(freq * t + phase) + offset_x;
        double y = amp * std::sin(freq * t + phase) + offset_y;

        if (nufft_estimator.feed(Centroid(t, x, y))) {
            if (nufft_estimator.compute()) {
                auto harmonics = nufft_estimator.getHarmonics();

                EXPECT_GE(harmonics.size(), 1);

                // With high non-uniformity, we expect less precision
                bool found_main_freq = false;
                for (const auto &h: harmonics) {
                    if (std::abs(h.frequency - freq) < 1e-1) {
                        found_main_freq = true;
                        EXPECT_NEAR(h.amplitude_x, amp, 1e-1);
                        EXPECT_NEAR(h.amplitude_y, amp, 1e-1);
                        break;
                    }
                }
                EXPECT_TRUE(found_main_freq) << "Main frequency not detected with high non-uniformity";
                break;
            }
        }
    }
}

TEST(NUFFT, HarmonicSeries) {
    double f_min = 10.0;
    double f_max = 200.0;
    int max_harmonics = 6;
    NUFFTHelixEstimator nufft_estimator(f_min, f_max, max_harmonics);

    // Harmonic series: fundamental + harmonics
    double fundamental_hz = 25.0;
    double fundamental_freq = Hz2rad(fundamental_hz);
    std::vector<double> amplitudes = {2.0, 1.0, 0.5, 0.25}; // Decreasing amplitudes
    std::vector<int> harmonic_numbers = {1, 2, 3, 4}; // 1st, 2nd, 3rd, 4th harmonic

    double offset_x = 0.0;
    double offset_y = 0.0;

    for (int i = 0; i < SAMPLES; ++i) {
        double t = i * time_granularity;
        double x = offset_x;
        double y = offset_y;

        // Sum harmonic series
        for (size_t h = 0; h < amplitudes.size(); ++h) {
            double freq = fundamental_freq * harmonic_numbers[h];
            x += amplitudes[h] * std::cos(freq * t);
            y += amplitudes[h] * std::sin(freq * t);
        }

        if (nufft_estimator.feed(Centroid(t, x, y))) {
            if (nufft_estimator.compute()) {
                auto harmonics = nufft_estimator.getHarmonics();

                EXPECT_GE(harmonics.size(), 2); // Should detect at least fundamental + 2nd harmonic

                // Check for fundamental frequency
                bool found_fundamental = false;
                for (const auto &h: harmonics) {
                    if (std::abs(h.frequency - fundamental_freq) < 1e-2) {
                        found_fundamental = true;
                        EXPECT_NEAR(h.amplitude_x, amplitudes[0], 1e-2);
                        EXPECT_NEAR(h.amplitude_y, amplitudes[0], 1e-2);
                        break;
                    }
                }
                EXPECT_TRUE(found_fundamental) << "Fundamental frequency not detected";

                // Check for 2nd harmonic
                bool found_second = false;
                for (const auto &h: harmonics) {
                    if (std::abs(h.frequency - 2 * fundamental_freq) < 2e-2) {
                        found_second = true;
                        EXPECT_NEAR(h.amplitude_x, amplitudes[1], 2e-2);
                        EXPECT_NEAR(h.amplitude_y, amplitudes[1], 2e-2);
                        break;
                    }
                }
                EXPECT_TRUE(found_second) << "Second harmonic not detected";
                break;
            }
        }
    }
}

TEST(NUFFT, NoiseRobustness) {
    double f_min = 10.0;
    double f_max = 100.0;
    int max_harmonics = 4;
    NUFFTHelixEstimator nufft_estimator(f_min, f_max, max_harmonics);

    // Signal with noise
    double amp = 3.0;
    double freq_hz = 55.0;
    double freq = Hz2rad(freq_hz);
    double phase = 0.5;
    double offset_x = 1.0;
    double offset_y = 0.5;
    double noise_level = 0.1; // 10% noise

    std::mt19937 gen(123);
    std::normal_distribution<double> noise(0.0, noise_level);

    for (int i = 0; i < SAMPLES; ++i) {
        double t = i * time_granularity;
        double x = amp * std::cos(freq * t + phase) + offset_x + noise(gen);
        double y = amp * std::sin(freq * t + phase) + offset_y + noise(gen);

        if (nufft_estimator.feed(Centroid(t, x, y))) {
            if (nufft_estimator.compute()) {
                auto harmonics = nufft_estimator.getHarmonics();

                EXPECT_GE(harmonics.size(), 1);

                bool found_main_freq = false;
                for (const auto &h: harmonics) {
                    if (std::abs(h.frequency - freq) < 3e-2) {
                        found_main_freq = true;
                        EXPECT_NEAR(h.amplitude_x, amp, 3e-2); // Relaxed tolerance due to noise
                        EXPECT_NEAR(h.amplitude_y, amp, 3e-2);
                        break;
                    }
                }
                EXPECT_TRUE(found_main_freq) << "Main frequency not detected with noise";
                break;
            }
        }
    }
}