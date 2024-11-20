//
// Created by viciopoli on 20/11/24.
//
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "../include/sim/sinusoid_sim.hpp"
#include "../include/filter/ekf.hpp"

std::tuple<double, double, double, double, double> testEKF(double amp, double freq, double phase) {
    EKF ekf(10, 0.1, 0.1);
    ekf.initialize(1., 1., 1., 1., 1.);

    SinusoidSim<double> sim_x(amp, freq, phase);
    SinusoidSim<double> sim_y(amp + 3, freq, phase);

    double delta = 0.01;

    for (int i = 0; i < 10'000; ++i) {
        double t = i * delta;
        double x = sim_x(t);
        double y = sim_y(t);
        ekf.update(x, y, t);
    }

    std::cout << ekf << std::endl;
    // return the estimated parameters
    return std::make_tuple(ekf.getRadS(), ekf.getAmplX(), ekf.getAmplY(), ekf.getPhaseX(), ekf.getPhaseY());
}

TEST(EKF, Estimation) {
    double amp = 4.0;
    double freq = 100.0;
    double phase = 1.0;

    auto [omega, A_x, A_y, phi_x, phi_y] = testEKF(amp, freq, phase);

    EXPECT_NEAR(omega, freq, 1.);
    EXPECT_NEAR(A_x, amp, 1.);
    EXPECT_NEAR(A_y, amp + 3, 1.);

    EXPECT_NEAR(phi_x, phase, 1.);
    EXPECT_NEAR(phi_y, phase, 1.);
}

