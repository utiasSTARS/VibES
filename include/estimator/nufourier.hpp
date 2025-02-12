//
// Created by viciopoli on 18/11/24.
//

#ifndef PROJECT_NUFOURIER_H

#include <finufft.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "utils.hpp"

class FourierFreqEst {
public:
    FourierFreqEst() = delete;

    FourierFreqEst(int n_samples, double f_min, double f_max) : N(n_samples) {
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
        if (prev_time == -1) {
            prev_time = t;
        }

        if (t - prev_time < 0.000'1) { // if the events are in the same time
            mean_x += x;
            mean_y += y;
            mean_t += t;
            counter += 1;
            // prev_time = t;
            return false;
        }
        if (counter > 100) {
            mean_x_out = mean_x / counter;
            mean_y_out = mean_y / counter;
            mean_t_out = mean_t / counter;

            t_data[index] = t;
            cj[index] = std::complex<double>(mean_x_out, mean_y_out);
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
        int M = N;  // Number of nonuniform points
        int N1 = N, N2 = N;  // Fourier grid dimensions

        // Extract x and y coordinates from complex input data
        std::vector<double> x_data(M), y_data(M);
        mean_x = 0;
        mean_y = 0;
        for (int i = 0; i < M; ++i) {
            x_data[i] = cj[i].real();
            mean_x += x_data[i];
            y_data[i] = cj[i].imag();
            mean_y += y_data[i];
        }
        mean_x /= M;
        mean_y /= M;

        // print on an image the x and y coordinates
        cv::namedWindow("2D Peak Magnitudes", cv::WINDOW_NORMAL);
        cv::Mat image = cv::Mat::zeros(500, 1500, CV_8UC1);
        for (int i = 0; i < 100; i++) {
            image.at<uchar>(cv::Point(int(x_data[i]) + 15 * i, int(y_data[i]))) = 255;
        }
        cv::imshow("2D Peak Magnitudes", image);
        cv::waitKey(0);

        // Resize output storage for Fourier coefficients
        outXY.resize(N1 * N2);

        // Perform 2D NUFFT Type 1
        int ier = finufft2d1(
                M,                         // Number of nonuniform points
                x_data.data(),              // Nonuniform X coordinates
                y_data.data(),              // Nonuniform Y coordinates
                cj.data(),                  // Complex input strengths
                1,                          // Forward transform (+i convention)
                1e-4,                       // Accuracy
                N1, N2,                     // Fourier grid dimensions (output size)
                outXY.data(),               // Output Fourier coefficients
                opts                        // NUFFT options
        );

        // Check for errors
        if (ier != 0) {
            std::cerr << "FINUFFT 2D failed with error code: " << ier << std::endl;
            return false;
        }

        // Find dominant frequency by searching max magnitude
        maxMagnitude = -1.0;
        int peakIndex = -1;

        magnitudes.resize(N1 * N2);
        for (int i = 0; i < N1 * N2; ++i) {
            double magnitude = std::abs(outXY[i]);
            magnitudes[i] = magnitude;

            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
                peakIndex = i;
            }
        }

        // Extract dominant frequency
        if (peakIndex != -1) {
            int freq_x = peakIndex % N1;  // Get X frequency index
            int freq_y = peakIndex / N1;  // Get Y frequency index

            main_freq_t = freqs_t[freq_x];  // Frequency in X
            phase_shift = std::arg(outXY[peakIndex]);

            std::cout << "Estimated Frequency: " << main_freq_t << " rad/s, "
                      << rad2Hz<double>(main_freq_t) << " Hz" << std::endl;
            std::cout << "Amplitude: " << maxMagnitude / (N1 * N2)
                      << ", Phase Shift: " << phase_shift << " rad" << std::endl;
        }

        // Reset data for next batch
        std::fill(t_data.begin(), t_data.end(), 0.0);
        std::fill(cj.begin(), cj.end(), std::complex<double>(0.0, 0.0));

        return true;
    }

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
        return maxMagnitude / N;
    }

    std::tuple<double, double, double> getMean() {
        return std::make_tuple(mean_x_out, mean_y_out, mean_t_out);
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

    std::vector<double> t_data;
    std::vector<double> freqs_t;
    std::vector<std::complex<double>> cj;
    std::vector<std::complex<double>> outXY;
    finufft_opts *opts;

    int index = 0;
    double main_freq_t = 0.0;
    double maxMagnitude = 0.0;
    double phase_shift = 0.0;

    std::vector<double> magnitudes;

    double prev_time = -1;
    double mean_x = 0;
    double mean_y = 0;
    double mean_t = 0;
    int64_t counter = 0;

    double mean_x_out = 0;
    double mean_y_out = 0;
    double mean_t_out = 0;
};

#endif //PROJECT_NUFOURIER_H
