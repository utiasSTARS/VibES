//
// Created by viciopoli on 20/11/24.
//
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "centroid.hpp"
#include "sim/ini_sim.hpp"
#include "nufourier_new.hpp"
#include "filter/ekf.hpp"

using namespace harmeda;

TEST(CENTROID_NUFFT, Estimation) {
    double amp = 10.0;
    double freq = 551.0; // Hz

    CentroidCalculation centroid;

    auto reader = std::make_unique<EventsFreqCalibPattern>(0, cv::Size(640, 480), freq, amp, amp, 0., true);

    int counter = 0;
    while (reader->isRunning()) {
        if (const auto events_dist = reader->getNextEventBatch(); events_dist.has_value()) {
            for (const auto &event: events_dist.value()) {
                centroid.feed(event.x(), event.y(), event.timestamp() / 1e6);
            }
        }
        counter++;
        if (counter > 1000) {
            break;
        }
    }

    std::cout << "N. of samples: " << centroid.size() << std::endl;
    auto samples = centroid.getCentroids();

    ////////////////////// NUFFT

    double estimated_freq = 0;
    double phase_shift = 0;
    double amplitude = 0;

    FourierFreqEst fourier(1'000, 100, 1000);

    counter = 0;
    for (const auto [x, y, t]: samples) { // time in seconds
        counter++;
        if (fourier.feed(x, y, t)) {
            estimated_freq = fourier.getMainFreqRad();
            phase_shift = fourier.getPhaseShift();
            amplitude = fourier.getAmplitude();
            break;
        }
    }

    auto [c_x, c_y] = fourier.getOffset();

    // std::cout << "Estimated freq: " << estimated_freq << " Phase shift: " << phase_shift << " Amplitude: " << amplitude
    //           << std::endl;

    // ground truth values
    const double gt_shift_x = 640 / 2;
    const double gt_shift_y = 480 / 2;
    std::cout << "Ground truth freq: " << freq << " rad/s " << rad2Hz(freq) << " Hz, Phase shift: " << M_PI / 2.
              << " Amplitude: " << amp << " Shift x: " << gt_shift_x << " Shift y: " << gt_shift_y << std::endl;

    // create EKF and track the centroid
    EKF ekf(1, 0.1, 0.1);
    ekf.initialize(estimated_freq, amplitude, phase_shift, c_x, c_y);

    for (const auto [x, y, t]: samples) { // time in seconds
        ekf.update(x, y, t);
    }

    auto residuals = ekf.getResiduals();
    // plot residuals
    cv::Mat plot = cv::Mat::zeros(480, 1000, CV_8UC3);
    for (size_t i = 0; i < residuals.size(); i++) {
        if (i >= 1000) {
            break;
        }
        std::cout << "Residuals: " << get<0>(residuals[i]) << " " << get<1>(residuals[i]) << std::endl;
        plot.at<cv::Vec3b>(240 - static_cast<int>(get<0>(residuals[i]) * 10), i) = cv::Vec3b(0, 0, 255);
        plot.at<cv::Vec3b>(240 - static_cast<int>(get<1>(residuals[i]) * 10), i) = cv::Vec3b(255, 0, 0);
    }
    // draw a line at the middle
    cv::line(plot, cv::Point(0, 240), cv::Point(1000, 240), cv::Scalar(0, 255, 0), 1);

    cv::namedWindow("Residuals", cv::WINDOW_NORMAL);
    cv::imshow("Residuals", plot);
    cv::waitKey(0);



    // print all the values
    std::cout << "EKF estimate: " << ekf << std::endl;


    // check the estimated results
    EXPECT_NEAR(ekf.getRadS(), freq, 0.1);
    EXPECT_NEAR(ekf.getAmplX(), amp, 5);
    EXPECT_NEAR(ekf.getAmplY(), amp, 5);
    EXPECT_NEAR(ekf.getPhaseX(), phase_shift, 0.1);
    EXPECT_NEAR(ekf.getPhaseY(), phase_shift, 0.1);
    EXPECT_NEAR(ekf.getShiftX(), c_x, 0.1);
    EXPECT_NEAR(ekf.getShiftY(), c_y, 0.1);

}
