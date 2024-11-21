//
// Created by viciopoli on 20/11/24.
//

#ifndef PROJECT_BIN_H
#define PROJECT_BIN_H

#include "filter/ekf.hpp"
#include "utils.h"
#include <memory>

class Bin {

public:
    Bin(int n_samples, double process_noise, double measurement_noise, double target_omega, double c_x, double c_y)
            : bin_id(bin_counter++), target_omega(target_omega) {
        ekf = std::make_shared<EKF>(n_samples, process_noise, measurement_noise);
        ekf->initialize(target_omega, 1., 1., c_x, c_y);
    }

    void update(double x, double y, double t) {
        if (prev_time == 0) {
            prev_time = t;
        }

        if (t - prev_time < 0.00001) {
            mean_x += x;
            mean_y += y;
            mean_t += t;
            counter += 1;
            prev_time = t;
            return;
        }
        if (counter > 100) {
            ekf->update(mean_x / counter, mean_y / counter, mean_t / counter);
        }
        mean_x = x;
        mean_y = y;
        mean_t = t;
        counter = 1;

        prev_time = t;
    }

    [[nodiscard]] double getCenterX() {
        return ekf->getShiftX();
    }

    [[nodiscard]] double getCenterY() {
        return ekf->getShiftY();
    }

    [[nodiscard]] bool is_stable() const {
        return std::abs(ekf->getRadS() - target_omega) < 1.;
    }

    [[nodiscard]] Colors getColor() const {
        return is_stable() ? GREEN : RED;
    }

    [[nodiscard]] bool is(double f) const {
        return std::abs(ekf->getRadS() - f) < 1.;
    }

    std::shared_ptr<EKF> getEKF() {
        return ekf;
    }

    friend std::ostream &operator<<(std::ostream &os, const Bin &bin) {
        os << "Bin id: " << bin.bin_id << ", " << *bin.ekf;
        return os;
    }

private:
    std::shared_ptr<EKF> ekf;
    const int64_t bin_id;
    static int64_t bin_counter;

    double prev_time = 0;
    double mean_x = 0;
    double mean_y = 0;
    double mean_t = 0;
    int64_t counter = 0;

    double target_omega;
    Colors colors = BLUE;

};

int64_t Bin::bin_counter = 0;

#endif //PROJECT_BIN_H
