//
// Created by viciopoli on 20/11/24.
//

#include <gtest/gtest.h>
#include <iostream>

#include "../include/bin.hpp"
#include <dv-processing/core/core.hpp>


bool testBin(double freq_target, double freq) {
    Bin bin(10, 0.1, 0.1, freq_target, 1., 1.);

    double delta = 1. / (2. * Hz2rad(freq) + 1.);

    for (int i = 0; i < 10'000; ++i) {
        double t = i * delta;
        double x = 0;
        double y = 0;
        bin.update(x, y, t);
    }

    std::cout << bin << std::endl;
    // return the estimated parameters
    return bin.is_stable();
}

TEST(DVSBin, Estimation) {
    EXPECT_TRUE(testBin(100., 100.));
    EXPECT_FALSE(testBin(10., 15.));
}