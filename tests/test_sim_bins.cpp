//
// Created by viciopoli on 20/11/24.
//

#include <gtest/gtest.h>
#include <iostream>

#include "../include/bin.hpp"
#include "../include/sim/ini_sim.hpp"

TEST(IniBin, Estimation) {
    double target_freq = 700.;
    // double omega, double amplitude_x, double amplitude_y, double phi, int delta_time, bool noise = true
    EventsFreqCalibPattern reader(0, cv::Size(640, 480), target_freq, 3, 3, 0.01, 10, false);

    Bin bin(2, 0.1, 0.1, target_freq - 80, 0, 0);

    int i = 0;
    while (reader.isRunning() && i < 20) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {
            for (const auto &event: events.value()) {
                bin.update(static_cast<double>(event.x()), static_cast<double>(event.y()), event.timestamp() / 1e6);
            }
            std::cout << bin << std::endl;
        } else {
            std::cerr << "Failed to generate events.\n";
        }
        i++;
    }

    EXPECT_NEAR(bin.getEKF()->getRadS(), target_freq, 1.);

}