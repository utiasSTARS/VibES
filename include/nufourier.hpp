//
// Created by viciopoli on 18/11/24.
//

#ifndef PROJECT_NUFOURIER_H
#define PROJECT_NUFOURIER_H
#ifndef PROJECT_FOURIER_H
#define PROJECT_FOURIER_H

#include <finufft.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

class FourierFreqEst {
public:
    FourierFreqEst() = delete;

    FourierFreqEst(int n_samples, int sampling_rate) : N(n_samples), samplingRate(sampling_rate) {
        x_data.resize(N);
        y_data.resize(N);
        cj.resize(N);
        freqs_x.resize(N);
        freqs_y.resize(N);

        outXY.resize(N);  // For 2D Fourier coefficients

        opts = new finufft_opts;
        finufft_default_opts(opts);
    }

    ~FourierFreqEst() {
        delete opts;
    }

    bool feed(double x, double y) {
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;

        x_data[index] = x;
        y_data[index] = y;
        index++;
        if (index >= N) {

            double mean_x = (max_x + min_x) / 2;
            double mean_y = (max_y + min_y) / 2;

            // Remove the mean max min from data
            for (int i = 0; i < N; i++) {
                x_data[i] -= mean_x;
                y_data[i] -= mean_y;
            }

            index = 0;
            return compute();
        }
        return false;
    }

    bool compute() {

        // Perform the 2D NUFFT (type 3)
        int ier = finufft2d3(
                N,                      // Number of input points
                x_data.data(),        // x-coordinates of input data
                y_data.data(),        // y-coordinates of input data
                cj.data(),          // Input strengths (signal)
                1,                       // Forward transform (+1)
                1e-6,                    // Desired accuracy
                N,                      // Number of output frequencies (same size as input)
                freqs_x.data(),         // x-frequency locations
                freqs_y.data(),         // y-frequency locations
                outXY.data(),           // 2D output Fourier coefficients
                opts                     // NUFFT options
        );

        // Check for errors
        if (ier != 0) {
            std::cerr << "FINUFFT type 3 failed with error code: " << ier << std::endl;
            return false;
        }

        // Find dominant frequencies in the x and y directions
        maxMagnitude = 0.0;
        int peakIndexX = -1, peakIndexY = -1;

        magnitudes = std::vector<double>(N);
        for (int i = 0; i < N; ++i) {
            double magnitude = std::abs(outXY[i]);
            magnitudes[i] = magnitude;

            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
                peakIndexX = i % N;  // X frequency index
                peakIndexY = i / N;  // Y frequency index
            }
        }

        // Calculate frequencies based on peak indices
        main_freq_x = freqs_x[peakIndexX];
        main_freq_y = freqs_y[peakIndexY];

        std::cout << "Estimated Frequency for X: " << main_freq_x << " Hz" << std::endl;
        std::cout << "Estimated Frequency for Y: " << main_freq_y << " Hz" << std::endl;

        // Reset data
        x_data.assign(N, 0.0);
        y_data.assign(N, 0.0);

        return true;
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

    std::vector<double> x_data, y_data;
    std::vector<double> freqs_x, freqs_y;
    std::vector<std::complex<double>> cj;
    std::vector<std::complex<double>> outXY;
    finufft_opts *opts;

    int index = 0;
    double main_freq_x = 0, main_freq_y = 0;
    double maxMagnitude = 0.0;

    std::vector<double> magnitudes;

    double max_x = std::numeric_limits<double>::min(), max_y = std::numeric_limits<double>::min();
    double min_x = std::numeric_limits<double>::max(), min_y = std::numeric_limits<double>::max();
};

#endif //PROJECT_FOURIER_H


#endif //PROJECT_NUFOURIER_H
