//
// Created by viciopoli on 20/11/24.
//
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "event_frontend/centroid.hpp"
#include "sim/ini_sim.hpp"
#include "estimator/nufourier_new.hpp"


TEST(CENTROID_NUFFT, Estimation) {
    double amp = 10.0;
    double freq = Hz2rad(10.0); // Hz

    CMassCalculation centroid;

    auto reader = std::make_unique<EventsFreqCalibPattern>(0, cv::Size(640, 480), freq, amp, amp, 0., true);

    int counter = 0;
    std::vector<std::tuple<double, double, Time>> samples;
    while (reader->isRunning()) {
        if (const auto events_dist = reader->getNextEventBatch(); events_dist.has_value()) {
            for (const auto &event: events_dist.value()) {
                auto centr = centroid.feed(event.x(), event.y(), event.timestamp() / 1e6);
                if (centr.has_value()) {
                    samples.push_back(centr.value());
                }
            }
        }
        counter++;
        if (counter > 10'000) {
            break;
        }
    }

    std::cout << "N. of samples: " << samples.size() << std::endl;

    ////////////////////// NUFFT

    double estimated_freq = 0;
    double phase_shift = 0;
    double amplitude = 0;

    FourierFreqEst fourier(10'000, 1, 1000);

    for (const auto [x, y, t]: samples) { // time in seconds
        if (fourier.feed(x, y, t)) {
            estimated_freq = fourier.getMainFreqRad();
            phase_shift = fourier.getPhaseShift();
            amplitude = fourier.getAmplitude();
            break;
        }
    }

    std::cout << "Estimated freq: " << estimated_freq << " Phase shift: " << phase_shift << " Amplitude: " << amplitude
              << std::endl;

    // ground truth values
    std::cout << "Ground truth freq: " << freq << " rad/s " << rad2Hz(freq) << " Hz, Phase shift: " << M_PI / 2.
              << " Amplitude: " << amp << std::endl;


    // check the estimated results
    EXPECT_NEAR(estimated_freq, freq, 0.1);
    EXPECT_NEAR(phase_shift, M_PI / 2., 0.1);
    EXPECT_NEAR(amplitude, amp, 5);

}
