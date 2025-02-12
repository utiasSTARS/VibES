//
// Created by viciopoli on 20/11/24.
//
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "sim/sinusoid_sim.hpp"
#include "tracker/bin.hpp"


bool testBin(double freq_target, double freq) {
    Bin bin(10, 0.1, 0.1, freq_target, 1., 1.);

    SinusoidSim<double> sim_x(10, freq, 1., 10);
    SinusoidSim<double> sim_y(10, freq, 1., 30);

    double delta = 1. / (2. * Hz2rad(freq) + 1.);

    for (int i = 0; i < 10'000; ++i) {
        double t = i * delta;
        double x = sim_x(t);
        double y = sim_y(t);
        bin.update(x, y, t);
    }

    std::cout << bin << std::endl;
    // return the estimated parameters
    return bin.is_stable();
}

TEST(Bin, Estimation) {
    EXPECT_TRUE(testBin(100., 100.));
    EXPECT_FALSE(testBin(10., 15.));
}
