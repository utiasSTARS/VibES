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
        ekf->initialize(1., 1., 1., c_x, c_y);
    }

    void update(double x, double y, double t) {
        ekf->update(x, y, t);
    }

    bool is_stable() {
        return std::abs(ekf->getRadS() - target_omega) < 1.;
    }

    Colors get_colors() {
        return is_stable() ? GREEN : RED;
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

    double target_omega;
    Colors colors = BLUE;

};

int64_t Bin::bin_counter = 0;

#endif //PROJECT_BIN_H
