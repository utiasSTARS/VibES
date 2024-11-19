//
// Created by viciopoli on 18/11/24.
//

#ifndef PROJECT_FOURIER_H
#define PROJECT_FOURIER_H

#include <fftw3.h>
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

        outX = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
        outY = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));

        planX = fftw_plan_dft_r2c_1d(N, x_data.data(), outX, FFTW_ESTIMATE);
        planY = fftw_plan_dft_r2c_1d(N, y_data.data(), outY, FFTW_ESTIMATE);
    }

    ~FourierFreqEst() {
        fftw_destroy_plan(planX);
        fftw_destroy_plan(planY);
        fftw_free(outX);
        fftw_free(outY);
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

            // remove the mean max min from data
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

        fftw_execute(planX);
        fftw_execute(planY);

        // Analyze FFT results to find peak frequency for both X and Y
        maxMagnitudeX = 0.0, maxMagnitudeY = 0.0;
        int peakIndexX = -1, peakIndexY = -1;

        magnitudesX = std::vector<double>(N / 2);
        magnitudesY = std::vector<double>(N / 2);

        for (int i = 0; i < N / 2; ++i) {
            double magnitudeX = std::sqrt(outX[i][0] * outX[i][0] + outX[i][1] * outX[i][1]);
            double magnitudeY = std::sqrt(outY[i][0] * outY[i][0] + outY[i][1] * outY[i][1]);
            magnitudesX[i] = magnitudeX;
            magnitudesY[i] = magnitudeY;

            if (magnitudeX > maxMagnitudeX) {
                maxMagnitudeX = magnitudeX;
                peakIndexX = i;
            }
            if (magnitudeY > maxMagnitudeY) {
                maxMagnitudeY = magnitudeY;
                peakIndexY = i;
            }
        }

        main_freq_x = (peakIndexX) * samplingRate / N;
        main_freq_y = (peakIndexY) * samplingRate / N;

        std::cout << "Estimated " << peakIndexX << " Frequency for X: " << main_freq_x << " Hz" << std::endl;
        std::cout << "Estimated " << peakIndexY << " Frequency for Y: " << main_freq_y << " Hz" << std::endl;

        x_data.assign(N, 0.0);
        y_data.assign(N, 0.0);

        return true;
    }

    void visualize() {
        int height = 400;
        cv::Mat peak_x = cv::Mat::zeros(height, int(N / 2), CV_8UC1);
        for (int i = 0; i < N / 2; i++) {
            cv::line(peak_x, cv::Point(i, 0), cv::Point(i, int(height * magnitudesX[i] / maxMagnitudeX)),
                     cv::Scalar(255));
        }

        cv::Mat peak_y = cv::Mat::zeros(height, int(N / 2), CV_8UC1);
        for (int i = 0; i < N / 2; i++) {
            cv::line(peak_y, cv::Point(i, 0), cv::Point(i, int(height * magnitudesY[i] / maxMagnitudeY)),
                     cv::Scalar(255));
        }
        cv::imshow("Peak X", peak_x);
        cv::imshow("Peak Y", peak_y);
        cv::waitKey(0);
    }

private:
    const int N = 10'000;

    const int samplingRate;

    std::vector<double> x_data, y_data;

    fftw_complex *outX;
    fftw_complex *outY;

    fftw_plan planX;
    fftw_plan planY;

    int index = 0;
    double main_freq_x = 0, main_freq_y = 0;

    std::vector<double> magnitudesX, magnitudesY;
    double maxMagnitudeX = 0.0, maxMagnitudeY = 0.0;


    double max_x = std::numeric_limits<double>::min(), max_y = std::numeric_limits<double>::min();
    double min_x = std::numeric_limits<double>::max(), min_y = std::numeric_limits<double>::max();
};


#endif //PROJECT_FOURIER_H
