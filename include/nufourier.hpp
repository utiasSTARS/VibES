//
// Created by viciopoli on 18/11/24.
//

#ifndef PROJECT_NUFOURIER_H

#include <finufft.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

class FourierFreqEst {
public:
    FourierFreqEst() = delete;

    FourierFreqEst(int n_samples, int sampling_rate, double f_min, double f_max) : N(n_samples),
                                                                                   samplingRate(sampling_rate) {
        t_data.resize(N);
        cj.resize(N);
        freqs_t.resize(N);

        outXY.resize(N);  // For 2D Fourier coefficients

        opts = new finufft_opts;
        finufft_default_opts(opts);

        initialize_freqs_t(f_min, f_max);
    }

    ~FourierFreqEst() {
        delete opts;
    }

    void initialize_freqs_t(double f_min, double f_max) {
        double delta_f = (f_max - f_min) / (N - 1);
        for (int i = 0; i < N; ++i) {
            freqs_t[i] = f_min + i * delta_f;
        }
    }


    bool feed_sim(double x, double y, double t) {
        t_data[index] = t;
        cj[index] = std::complex<double>(x, y);
        index++;
        if (index >= N) {
            index = 0;
            return compute();
        }
        return false;
    }

    bool feed(double x, double y, double t) {
        if (prev_time == 0) {
            prev_time = t;
        }

        if (t - prev_time < 0.00001) {
            mean_x += x;
            mean_y += y;
            mean_t += t;
            counter += 1;
            prev_time = t;
            return false;
        }
        if (counter > 100) {
            t_data[index] = t;
            cj[index] = std::complex<double>(mean_x / counter, mean_y / counter);
            index++;
            if (index >= N) {

                index = 0;
                return compute();
            }
        }
        mean_x = x;
        mean_y = y;
        mean_t = t;
        counter = 1;

        prev_time = t;

        return false;
    }

    bool compute() {

        int ier = finufft1d3(
                N,
                t_data.data(),    // Non-uniform sampling positions in z
                cj.data(),          // Complex input signal
                1,                  // Forward transform
                1e-6,                // Desired accuracy
                N,            // Number of desired frequencies
                freqs_t.data(),     // Output frequencies
                outXY.data(),          // Output Fourier coefficients
                opts
        );
        // Check for errors
        if (ier != 0) {
            std::cerr << "FINUFFT type 3 failed with error code: " << ier << std::endl;
            return false;
        }

        // Find dominant frequencies in the x and y directions
        maxMagnitude = -1.0;
        int peakIndex = -1;

        magnitudes = std::vector<double>(N);
        for (int i = 0; i < N; ++i) {
            double magnitude = std::abs(outXY[i]);
            magnitudes[i] = magnitude;

            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
                peakIndex = i;
            }
        }

        // Calculate frequencies based on peak indices
        main_freq_t = freqs_t[peakIndex];

        std::cout << "Estimated Frequency for X: " << main_freq_t << " Hz" << std::endl;

        // Reset data
        t_data.assign(N, 0.0);
        cj.assign(N, std::complex<double>(0.0, 0.0));

        return true;
    }

    double getMainFreq() const {
        return main_freq_t;
    }

    void visualize() {
        int height = 400;
        cv::Mat peak = cv::Mat::zeros(height, N, CV_8UC1);
        for (int i = 0; i < N; i++) {
            cv::line(peak, cv::Point(i, 0), cv::Point(i, int(height * magnitudes[i] / maxMagnitude)),
                     cv::Scalar(255));
        }
        cv::imshow("2D Peak Magnitudes", peak);
        cv::waitKey(0);
    }

private:
    const int N = 10'000;
    const int samplingRate;

    std::vector<double> t_data;
    std::vector<double> freqs_t;
    std::vector<std::complex<double>> cj;
    std::vector<std::complex<double>> outXY;
    finufft_opts *opts;

    int index = 0;
    double main_freq_t = 0.0;
    double maxMagnitude = 0.0;

    std::vector<double> magnitudes;

    double prev_time = 0;
    double mean_x = 0;
    double mean_y = 0;
    double mean_t = 0;
    int64_t counter = 0;
};

#endif //PROJECT_NUFOURIER_H
