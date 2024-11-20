//
// Created by viciopoli on 20/11/24.
//

#ifndef PROJECT_EKF_H
#define PROJECT_EKF_H

#include <Eigen/Dense>
#include <deque>

class EKF {
public:
    EKF() = delete;

    EKF(int n_samples, double process_noise, double measurement_noise) : n_samples(n_samples) {
        // the state is a 7x1 vector [omega, A_x, B_x, C_x, A_y, B_y, C_y]
        P = Eigen::MatrixXd::Identity(7, 7) * 1000.; // Initial covariance matrix
        Q = Eigen::MatrixXd::Identity(7, 7) * process_noise;
        R = Eigen::MatrixXd::Identity(2 * n_samples, 2 * n_samples) * measurement_noise;

        residuals = Eigen::VectorXd(2 * n_samples);
        H = Eigen::MatrixXd::Zero(2 * n_samples, 7);

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
        window_data.emplace_back(x, y, t);
        if (window_data.size() > n_samples) {
            windowedEKF();
            window_data.pop_front();
        }
    }

    void windowedEKF() {
        // Compute the Jacobian matrix for each data point and residuals
        for (int i = 0; i < n_samples; ++i) {
            auto [x, y, t] = window_data[i];
            double x_pred = A_x * std::sin(omega * t) + B_x * std::cos(omega * t) + C_x;
            double y_pred = A_y * std::sin(omega * t) + B_y * std::cos(omega * t) + C_y;

            residuals(2 * i) = x - x_pred;
            residuals(2 * i + 1) = y - y_pred;

            H(2 * i, 0) = A_x * t * std::cos(omega * t) - B_x * t * std::sin(omega * t);
            H(2 * i, 1) = std::sin(omega * t);
            H(2 * i, 2) = std::cos(omega * t);
            H(2 * i, 3) = 1;

            H(2 * i + 1, 0) = A_y * t * std::cos(omega * t) - B_y * t * std::sin(omega * t);
            H(2 * i + 1, 4) = std::sin(omega * t);
            H(2 * i + 1, 5) = std::cos(omega * t);
            H(2 * i + 1, 6) = 1;
        }

        // Measurement update
        Eigen::MatrixXd H_transpose = H.transpose();
        Eigen::MatrixXd S = H * P * H_transpose + R;
        Eigen::MatrixXd K = P * H_transpose * S.inverse();

        // Update
        Eigen::VectorXd state = K * residuals;
        omega += state(0);
        A_x += state(1);
        B_x += state(2);
        C_x += state(3);
        A_y += state(4);
        B_y += state(5);
        C_y += state(6);

        P = (Eigen::MatrixXd::Identity(7, 7) - K * H) * P;
        P = 0.5 * (P + P.transpose());
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
    double omega;
    // sinusoids of the form A * sin(omega * t) + B * cos(omega * t) + C
    // where amplitude = sqrt(A^2 + B^2) and phase = atan2(B, A)
    double A_x = 0, A_y = 0, B_x = 0, B_y = 0;
    double C_x = 0, C_y = 0;
    const int n_samples;

    std::deque<std::tuple<double, double, double>> window_data;

    Eigen::MatrixXd P;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

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
