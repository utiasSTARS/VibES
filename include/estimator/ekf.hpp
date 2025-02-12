//
// Created by viciopoli on 20/11/24.
//

#ifndef PROJECT_EKF_H
#define PROJECT_EKF_H

#include <Eigen/Dense>
#include <deque>
#include "utils.hpp"

class EKF {
public:
    EKF() = delete;

    EKF(int n_samples, double process_noise, double measurement_noise) : n_samples(n_samples) {
        // the state is a 8x1 vector [theta, omega, A_x, B_x, C_x, A_y, B_y, C_y]
        int state_dim = 8;
        I = Eigen::MatrixXd::Identity(state_dim, state_dim);
        F = Eigen::MatrixXd::Identity(state_dim, state_dim);
        P = Eigen::MatrixXd::Identity(state_dim, state_dim) * 10.; // Initial covariance matrix
        // we are more certain about omega, respect to the other parameters
        P(1, 1) = 1.0;
        Q = Eigen::MatrixXd::Identity(state_dim, state_dim) * process_noise;
        R = Eigen::MatrixXd::Identity(2 * n_samples, 2 * n_samples) * measurement_noise;

        residuals = Eigen::VectorXd(2 * n_samples);
        H = Eigen::MatrixXd::Zero(2 * n_samples, state_dim);
    }

    void initialize(double omega, double A, double phi, double C_x, double C_y) {
        A_x = A * std::sin(0.);
        B_x = A * std::cos(0.);
        A_y = A * std::sin((phi == 0.0 ? M_PI / 2. : phi));
        B_y = A * std::cos((phi == 0.0 ? M_PI / 2. : phi));
        this->C_x = C_x;
        this->C_y = C_y;
        this->omega = omega;
    }

    std::optional<std::tuple<int, int, int64_t>> update(double x, double y, double t) {
        if (prev_t != -1) {
            computeEKF(x, y, t);
            return compensate(x, y, t);
        } else {
            prev_t = t;
            return std::nullopt;
        }
    }

    [[nodiscard]] std::optional<std::tuple<int, int, int64_t>> compensate(int x, int y, double t) const {
        // according to out camera projection model
        // double u = x - A_x * std::sin(theta) + B_x * std::cos(theta);
        // double v = y - A_y * std::sin(theta) + B_y * std::cos(theta);
        // return std::make_tuple(static_cast<int>(u), static_cast<int>(v), static_cast<int64_t>(t * 1e6));
        return std::make_tuple(static_cast<int>(C_x), static_cast<int>(C_y), static_cast<int64_t>(t * 1e6));
    }

    [[nodiscard]] std::tuple<double, double> getPred() const {
        return std::make_tuple(x_pred, y_pred);
    }

    void computeEKF(double x_meas, double y_meas, double t) {
        double delta_t = t - prev_t;

        // ========== PREDICT STEP ==========
        // 1. Reinitialize F as identity
        F(0, 1) = delta_t; // d(theta)/d(omega)

        // 2. Predict state (theta is updated here)
        theta = wrap_phase(theta + omega * delta_t);

        // 3. Propagate covariance
        P = F * P * F.transpose() + Q;

        // ========== UPDATE STEP ==========
        // 1. Compute predicted measurements
        x_pred = A_x * std::sin(theta) + B_x * std::cos(theta) + C_x;
        y_pred = A_y * std::sin(theta) + B_y * std::cos(theta) + C_y;

        // 2. Compute residual
        Eigen::Vector2d residual(x_meas - x_pred, y_meas - y_pred);

        // 3. Compute Jacobian H (2x8 for this measurement)
        double dxdtheta = A_x * std::cos(theta) - B_x * std::sin(theta);
        double dydtheta = A_y * std::cos(theta) - B_y * std::sin(theta);

        H.setZero();
        // dx/dtheta, dx/dA_x, dx/dB_x, dx/dC_x
        H(0, 0) = dxdtheta;
        H(0, 1) = delta_t * dxdtheta;
        H(0, 2) = std::sin(theta);
        H(0, 3) = std::cos(theta);
        H(0, 4) = 1.0;
        // dy/dtheta, dy/dA_y, dy/dB_y, dy/dC_y
        H(1, 0) = dydtheta;
        H(1, 1) = delta_t * dydtheta;
        H(1, 5) = std::sin(theta);
        H(1, 6) = std::cos(theta);
        H(1, 7) = 1.0;

        // 4. Kalman gain and covariance update
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();

        // 5. Update state
        Eigen::VectorXd state_update = K * residual;
        theta = wrap_phase(theta + state_update(0)); // Wrap phase
        omega += state_update(1);
        A_x += state_update(2);
        B_x += state_update(3);
        C_x += state_update(4);
        A_y += state_update(5);
        B_y += state_update(6);
        C_y += state_update(7);

        // 6. Update covariance (Joseph form)
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P.rows(), P.cols());
        P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();
        P = 0.5 * (P + P.transpose()); // Ensure symmetry

        prev_t = t;
    }

    void batchUpdate(
            double prev_t,
            const std::vector<double> &meas_times,
            const std::vector<Eigen::Vector2d> &meas,
            double &theta,
            double &omega,
            double &A_x, double &B_x, double &C_x,
            double &A_y, double &B_y, double &C_y,
            Eigen::MatrixXd &P,
            const Eigen::MatrixXd &Q,
            const Eigen::MatrixXd &R) {
        const int N = static_cast<int>(meas_times.size());
        if (N == 0) return; // no measurements to update with

        Eigen::MatrixXd H_batch(2 * N, 8);
        Eigen::VectorXd r_batch(2 * N);

        // For each measurement, compute predicted measurement and its Jacobian row.
        for (int i = 0; i < N; i++) {
            double t_i = meas_times[i];
            double delta_t = t_i - prev_t; // time difference from previous update time

            // Propagate the phase: note that only theta is time-varying.
            double theta_i = wrap_phase(theta + omega * delta_t);

            // Predicted measurements at time t_i:
            double x_pred = A_x * std::sin(theta_i) + B_x * std::cos(theta_i) + C_x;
            double y_pred = A_y * std::sin(theta_i) + B_y * std::cos(theta_i) + C_y;

            // Residual (measurement - prediction):
            r_batch(2 * i) = meas[i](0) - x_pred;
            r_batch(2 * i + 1) = meas[i](1) - y_pred;

            // Common derivatives:
            double dxdtheta = A_x * std::cos(theta_i) - B_x * std::sin(theta_i);
            double dydtheta = A_y * std::cos(theta_i) - B_y * std::sin(theta_i);

            // Fill in Jacobian rows for x measurement:
            H_batch(2 * i, 0) = dxdtheta;                    // d(x_pred)/dθ
            H_batch(2 * i, 1) = delta_t * dxdtheta;            // d(x_pred)/dω
            H_batch(2 * i, 2) = std::sin(theta_i);             // d(x_pred)/dAₓ
            H_batch(2 * i, 3) = std::cos(theta_i);             // d(x_pred)/dBₓ
            H_batch(2 * i, 4) = 1.0;                           // d(x_pred)/dCₓ
            H_batch(2 * i, 5) = 0.0;                           // x_pred does not depend on Aᵧ
            H_batch(2 * i, 6) = 0.0;                           // x_pred does not depend on Bᵧ
            H_batch(2 * i, 7) = 0.0;                           // x_pred does not depend on Cᵧ

            // Fill in Jacobian rows for y measurement:
            H_batch(2 * i + 1, 0) = dydtheta;                  // d(y_pred)/dθ
            H_batch(2 * i + 1, 1) = delta_t * dydtheta;          // d(y_pred)/dω
            H_batch(2 * i + 1, 2) = 0.0;                       // y_pred does not depend on Aₓ
            H_batch(2 * i + 1, 3) = 0.0;                       // y_pred does not depend on Bₓ
            H_batch(2 * i + 1, 4) = 0.0;                       // y_pred does not depend on Cₓ
            H_batch(2 * i + 1, 5) = std::sin(theta_i);         // d(y_pred)/dAᵧ
            H_batch(2 * i + 1, 6) = std::cos(theta_i);         // d(y_pred)/dBᵧ
            H_batch(2 * i + 1, 7) = 1.0;                       // d(y_pred)/dCᵧ
        }

        // For the covariance propagation, we assume a state transition from the previous update time to the time of the last measurement.
        double t_last = meas_times.back();
        double delta_last = t_last - prev_t;
        Eigen::MatrixXd F_batch = Eigen::MatrixXd::Identity(8, 8);
        F_batch(0, 1) = delta_last; // Only theta depends on omega over delta_last.
        Eigen::MatrixXd P_pred = F_batch * P * F_batch.transpose() + Q;

        // Form the block-diagonal measurement noise covariance for the batch.
        Eigen::MatrixXd R_batch = Eigen::MatrixXd::Zero(2 * N, 2 * N);
        for (int i = 0; i < N; i++) {
            R_batch.block(2 * i, 2 * i, 2, 2) = R;
        }

        // Innovation covariance for the batch update.
        Eigen::MatrixXd S = H_batch * P_pred * H_batch.transpose() + R_batch;

        // Batch Kalman gain.
        Eigen::MatrixXd K = P_pred * H_batch.transpose() * S.inverse();

        // Compute the state update from the stacked residual.
        Eigen::VectorXd state_update = K * r_batch;

        // Update the state:
        theta = wrap_phase(theta + state_update(0)); // Wrap phase after update.
        omega += state_update(1);
        A_x += state_update(2);
        B_x += state_update(3);
        C_x += state_update(4);
        A_y += state_update(5);
        B_y += state_update(6);
        C_y += state_update(7);

        // Update covariance using the Joseph form.
        I = Eigen::MatrixXd::Identity(8, 8);
        P = (I - K * H_batch) * P_pred * (I - K * H_batch).transpose() + K * R_batch * K.transpose();
        P = 0.5 * (P + P.transpose());  // Ensure symmetry.

        prev_t = meas_times.back();
    }


    [[nodiscard]] double getPrevT() const {
        return prev_t;
    }

    [[nodiscard]] double getHz() const {
        return rad2Hz(std::abs(omega));
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

    std::vector<std::tuple<double, double>> getResiduals() {
        return residuals_vec;
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
    double prev_t = -1;
    const int n_samples;

    Eigen::MatrixXd P;
    Eigen::MatrixXd F;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    Eigen::MatrixXd I;

    Eigen::VectorXd residuals;
    Eigen::MatrixXd H;

    std::vector<std::tuple<double, double>> residuals_vec;

    [[nodiscard]] double wrap_phase(double phase) const {
        while (phase > 2 * M_PI) {
            phase -= 2 * M_PI;
        }
        while (phase < 0) {
            phase += 2 * M_PI;
        }
        return phase;
    }

    double x_pred, y_pred;
};

// double EKF::omega = 1.0;

#endif //PROJECT_EKF_H
