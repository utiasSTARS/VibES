//
// Created by viciopoli on 20/11/24.
//

#ifndef PROJECT_BIN_H
#define PROJECT_BIN_H

#include "filter/ekf.hpp"
#include "utils.hpp"
#include <memory>

class Bin {

public:
    Bin(int n_samples, double process_noise, double measurement_noise, double target_omega, double c_x, double c_y,
        double phase_shift = M_PI / 2, double amplitude = 1.)
            : bin_id(bin_counter++), target_omega(target_omega) {
        ekf = std::make_shared<EKF>(n_samples, process_noise, measurement_noise);
        // double omega, double A, double phi, double C_x, double C_y
        ekf->initialize(target_omega, amplitude, phase_shift, c_x, c_y);
    }

    std::optional<std::tuple<int, int, int64_t>> update(double x, double y, double t) {
        std::optional<std::tuple<int, int, int64_t>> comp;
        if (prev_time == 0) {
            prev_time = t;
        }

        if (t - prev_time < 0.00001) {
            mean_x += x;
            mean_y += y;
            mean_t += t;
            counter += 1;
            prev_time = t;
            return comp;
        }
        if (counter > 100) {
            mean_x_out = mean_x / counter;
            mean_y_out = mean_y / counter;
            mean_t_out = mean_t / counter;
            comp = ekf->update(mean_x_out, mean_y_out, mean_t_out);
            updated = true;
        }
        mean_x = x;
        mean_y = y;
        mean_t = t;
        counter = 1;

        prev_time = t;
        return comp;
    }

    std::tuple<double, double, double> getMean() {
        return std::make_tuple(mean_x_out, mean_y_out, mean_t_out);
    }

    [[nodiscard]] std::optional<std::tuple<int, int>> compensate(int x, int y, double t) const {
        if (!updated) {
            return std::nullopt;
        }
        // according to out camera projection model
        double u = x - ekf->getAmplX() * std::sin(ekf->getRadS() * (t - ekf->getPrevT()) + ekf->getPhaseX());
        double v = y - ekf->getAmplY() * std::sin(ekf->getRadS() * (t - ekf->getPrevT()) + ekf->getPhaseY());

        return std::make_tuple(static_cast<int>(u), static_cast<int>(v));
    }

    [[nodiscard]] double getOmega() const {
        return ekf->getRadS();
    }

    [[nodiscard]] double getAmplitudeX() const {
        return ekf->getAmplX();
    }

    [[nodiscard]] double getAmplitudeY() const {
        return ekf->getAmplY();
    }

    [[nodiscard]] double getHz() const {
        return ekf->getHz();
    }

    [[nodiscard]] int64_t getBinId() const {
        return bin_id;
    }

    [[nodiscard]] double getCenterX() {
        return ekf->getShiftX();
    }

    [[nodiscard]] double getCenterY() {
        return ekf->getShiftY();
    }

    [[nodiscard]] bool is_stable() const {
        return updated && std::abs(ekf->getRadS() - target_omega) < 5.;
    }

    [[nodiscard]] Colors getColor() const {
        return is_stable() ? GREEN : RED;
    }

    [[nodiscard]] bool is(double f) const {
        return updated && std::abs(ekf->getRadS() - f) < 5.;
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

    double mean_x_out = 0;
    double mean_y_out = 0;
    double mean_t_out = 0;

    double target_omega;
    Colors colors = BLUE;
    bool updated = false;

};

int64_t Bin::bin_counter = 0;

#endif //PROJECT_BIN_H
