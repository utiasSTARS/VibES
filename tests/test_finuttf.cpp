//
// Created by viciopoli on 20/11/24.
//

#include <gtest/gtest.h>
#include <random>

#include "../include/nufourier.hpp"
#include "../include/sim/sinusoid_sim.hpp"
#include "../include/sim/ini_sim.hpp"

TEST(FourierFreqEst, SinusoidSim) {
    FourierFreqEst fourierFreqEst(1000, 500, 700);

    // generate some data
    double amp = 4.0;
    double freq = 512.0;
    double phase = 1.0;
    SinusoidSim<double> sim_x(amp, freq, phase);
    SinusoidSim<double> sim_y(amp, freq, phase + M_PI / 2);

    // sample in uniform time
    std::vector<double> uniform_time;
    // generate a random number between 0 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);
    for (int i = 0; i < 1000; i++) {
        double dt = dis(gen) / 250;
        if (uniform_time.empty()) {
            uniform_time.push_back(dt);
            continue;
        }
        uniform_time.push_back(uniform_time.back() + dt);
    }

    // sample the sinusoid
    for (const auto &t: uniform_time) {
        if (fourierFreqEst.feed_sim(sim_x(t), sim_y(t), t)) {
            EXPECT_NEAR(fourierFreqEst.getMainFreqRad(), freq, 1.);
        }
    }

}


TEST(FourierFreqEst, DVSSim) {
    double target_freq = 568.;
    // double omega, double amplitude_x, double amplitude_y, double phi, int delta_time, bool noise = true
    EventsFreqCalibPattern reader(0, cv::Size(640, 480), target_freq, 3, 3, 0.01, false);

    FourierFreqEst fourierFreqEst(1000, 500, 800);


    int i = 0;
    while (reader.isRunning() && i < 2000) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {
            for (const auto &event: events.value()) {
                if (fourierFreqEst.feed(event.x(), event.y(), event.timestamp() / 1e6)) {
                    EXPECT_NEAR(fourierFreqEst.getMainFreqRad(), target_freq, 1.);
                }
            }
        } else {
            std::cerr << "Failed to generate events.\n";
        }
        i++;
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}