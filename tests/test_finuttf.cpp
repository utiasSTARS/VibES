//
// Created by viciopoli on 20/11/24.
//

#include <gtest/gtest.h>
#include <random>

#include "../include/nufourier.hpp"
#include "../include/sim/sinusoid_sim.hpp"

TEST(FourierFreqEst, Constructor) {
    FourierFreqEst fourierFreqEst(1000, 10, 500, 700);

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
        fourierFreqEst.feed(sim_x(t), sim_y(t), t);
    }


}