//
// Created by viciopoli on 20/11/24.
//

#ifndef PROJECT_EKF_H
#define PROJECT_EKF_H

#include <Eigen/Dense>
#include <deque>
#include "../utils.h"

class EKF {
public:
    EKF() = delete;

    EKF(int n_samples, double process_noise, double measurement_noise) : n_samples(n_samples) {
        // the state is a 8x1 vector [theta, omega, A_x, B_x, C_x, A_y, B_y, C_y]
        int state_dim = 8;
        I = Eigen::MatrixXd::Identity(state_dim, state_dim);
        F = Eigen::MatrixXd::Identity(state_dim, state_dim);
        P = Eigen::MatrixXd::Identity(state_dim, state_dim) * 1000.; // Initial covariance matrix
        Q = Eigen::MatrixXd::Identity(state_dim, state_dim) * process_noise;
        R = Eigen::MatrixXd::Identity(2 * n_samples, 2 * n_samples) * measurement_noise;

        residuals = Eigen::VectorXd(2 * n_samples);
        H = Eigen::MatrixXd::Zero(2 * n_samples, state_dim);
    }

    void initialize(double omega, double A, double phi, double C_x, double C_y) {
        A_x = A * std::cos(phi);
        B_x = A * std::sin(phi);
        B_x = A * std::cos(phi);
        B_y = A * std::sin(phi);
        this->C_x = C_x;
        this->C_y = C_y;
        this->omega = omega;
    }

    void update(double x, double y, double t) {
        if (prev_t == 0) {
            prev_t = t;
        }
        window_data.emplace_back(x, y, t);
        if (window_data.size() > n_samples) {
            windowedEKF();
            window_data.pop_front();
        }
    }

    void windowedEKF() {
        for (int i = 0; i < n_samples; i++) {
            auto [x, y, t] = window_data[i];
            double delta_t = t - prev_t;
            F(0, 1) = delta_t; // omega update
            theta = wrap_phase(theta + omega * delta_t); // Phase update
            prev_t = t;

            double x_pred = A_x * std::sin(theta) + B_x * std::cos(theta) + C_x;
            double y_pred = A_y * std::sin(theta) + B_y * std::cos(theta) + C_y;

            residuals(2 * i) = x - x_pred;
            residuals(2 * i + 1) = y - y_pred;

            // Jacobian computation
            H(2 * i, 0) = A_x * std::cos(theta) - B_x * std::sin(theta); // dx/dtheta
            H(2 * i, 1) = delta_t * (A_x * std::cos(theta) - B_x * std::sin(theta)); // dx/domega
            H(2 * i, 2) = std::sin(theta); // dx/dA_x
            H(2 * i, 3) = std::cos(theta); // dx/dB_x
            H(2 * i, 4) = 1.; // dx/dC_x

            H(2 * i + 1, 0) = A_y * std::cos(theta) - B_y * std::sin(theta); // dy/dtheta
            H(2 * i + 1, 1) = delta_t * (A_y * std::cos(theta) - B_y * std::sin(theta)); // dy/domega
            H(2 * i + 1, 5) = std::sin(theta); // dy/dA_y
            H(2 * i + 1, 6) = std::cos(theta); // dy/dB_y
            H(2 * i + 1, 7) = 1.; // dy/dC_y
        }

        // Measurement update
        Eigen::MatrixXd H_transpose = H.transpose();
        P = F * P * F.transpose() + Q;
        Eigen::MatrixXd S = H * P * H_transpose + R;
        Eigen::MatrixXd K = P * H_transpose * S.inverse();

        // Update
        Eigen::VectorXd state = K * residuals;
        theta += state(0);
        omega += state(1);
        A_x += state(2);
        B_x += state(3);
        C_x += state(4);
        A_y += state(5);
        B_y += state(6);
        C_y += state(7);

        P = (I - K * H) * P;
        P = 0.5 * (P + P.transpose()); // Ensure symmetry
    }

    [[nodiscard]] double getHz() const {
        return std::abs(omega) * 2 * M_PI;
    }

    [[nodiscard]] double getRadS() const {
        return std::abs(omega);
    }

    [[nodiscard]] double getAmplX() const {
        return std::sqrt(A_x * A_x + B_x * B_x);
    }

    [[nodiscard]] double getAmplY() const {
        return std::sqrt(A_y * A_y + B_y * B_y);
    }

    [[nodiscard]] double getShiftX() const {
        return C_x;
    }

    [[nodiscard]] double getShiftY() const {
        return C_y;
    }

    [[nodiscard]] double getPhaseX() const {
        // switch A and B signs, must be both positive
        return wrap_phase(std::atan2(std::abs(B_x), std::abs(A_x)));
    }

    [[nodiscard]] double getPhaseY() const {
        return wrap_phase(std::atan2(std::abs(B_y), std::abs(A_y)));
    }

    friend std::ostream &operator<<(std::ostream &os, const EKF &ekf) {
        os << "omega: " << ekf.getRadS() << " rad/s, " << ekf.getHz() << " Hz, ";
        os << "Amplitude X: " << ekf.getAmplX() << " px, ";
        os << "Amplitude Y: " << ekf.getAmplY() << " px, ";
        os << "Phase X: " << ekf.getPhaseX() << " rad, ";
        os << "Phase Y: " << ekf.getPhaseY() << " rad, ";
        os << "Shift X: " << ekf.getShiftX() << " px, ";
        os << "Shift Y: " << ekf.getShiftY() << " px";
        return os;
    }


private:
    // sinusoids of the form A * sin(omega * t) + B * cos(omega * t) + C
    // where amplitude = sqrt(A^2 + B^2) and phase = atan2(B, A)
    double omega;
    double A_x = 0, A_y = 0, B_x = 0, B_y = 0;
    double C_x = 0, C_y = 0;
    double theta = 0;
    double prev_t = 0;
    const int n_samples;

    std::deque<std::tuple<double, double, double>> window_data;

    Eigen::MatrixXd P;
    Eigen::MatrixXd F;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    Eigen::MatrixXd I;

    Eigen::VectorXd residuals;
    Eigen::MatrixXd H;

    double wrap_phase(double phase) const {
        while (phase > 2 * M_PI) {
            phase -= 2 * M_PI;
        }
        while (phase < 0) {
            phase += 2 * M_PI;
        }
        return phase;
    }
};

// double EKF::omega = 1.0;

#endif //PROJECT_EKF_H
