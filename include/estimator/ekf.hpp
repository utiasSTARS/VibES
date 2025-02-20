//
// Created by viciopoli on 20/11/24.
//

#ifndef PROJECT_EKF_H
#define PROJECT_EKF_H

#include <Eigen/Dense>
#include <deque>
#include "utils.hpp"
#include <cmath>
#include <optional>

class EKF {
public:
    EKF() = delete;

    EKF(double dt, double a_noise, double c_noise, double omega_noise, double measurement_noise) : _dt(dt),
                                                                                                   _a_noise(a_noise),
                                                                                                   _c_noise(c_noise),
                                                                                                   _omega_noise(
                                                                                                           omega_noise) {
        // the state is a 7x1 vector [omega, phi_x, a_x, c_x, phi_y, a_y, c_y]
        int state_dim = 7;
        I = Eigen::MatrixXd::Identity(state_dim, state_dim);

        F = Eigen::MatrixXd::Identity(state_dim, state_dim);

        P = Eigen::MatrixXd::Identity(state_dim, state_dim) * 100.; // Initial covariance matrix
        P(0, 0) = 1.;

        Q = Eigen::MatrixXd::Identity(state_dim, state_dim);

        R = Eigen::MatrixXd::Identity(2, 2) * measurement_noise;

        residuals = Eigen::VectorXd(2);

        H = Eigen::MatrixXd::Zero(2, state_dim);
    }

    inline void computeF(double dt) {
        F(1, 0) = dt;
        F(4, 0) = dt;
    }

    void computeQ(double dt) {
        const auto dt2 = dt * dt;
        const auto dt3 = dt2 * dt;

        // [omega, phi_x, a_x, c_x, phi_y, a_y, c_y]
        // Omega noise (for x and y)
        Q(0, 0) = dt * _omega_noise;  // X-axis
        Q(1, 1) = dt3 * _omega_noise / 3.;
        Q(0, 1) = Q(1, 0) = dt2 * _omega_noise / 2.;

        Q(2, 2) = dt * _a_noise;
        Q(3, 3) = dt * _c_noise;

        Q(4, 4) = dt3 * _omega_noise / 3.;
        Q(0, 4) = Q(4, 0) = dt2 * _omega_noise / 2.;

        // Amplitude noise
        Q(5, 5) = dt * _a_noise;
        Q(6, 6) = dt * _c_noise;
    }

    void f() {
        _phi_x = wrap_phase(_phi_x + _omega * _dt);
        _phi_y = wrap_phase(_phi_y + _omega * _dt);
    }

    inline void h() {
        x_pred = _a_x * sin(_phi_x) + _c_x;
        y_pred = _a_y * sin(_phi_y) + _c_y;
    }

    void initialize(double omega, double a_x, double a_y, double c_x, double c_y, double phase_shift) {
        _a_x = a_x;
        _a_y = a_y;
        _phi_x = 0.;
        _phi_y = phase_shift;
        _omega = omega;
        _c_x = c_x;
        _c_y = c_y;
        x_pred = _a_x * sin(_phi_x) + _c_x;
        y_pred = _a_y * sin(_phi_y) + _c_y;
    }

    bool update(double x, double y, double t) {
        if (prev_t != -1) {
            computeEKF(x, y, t);
            return true;
        } else {
            prev_t = t;
            return false;
        }
    }

    void computeH(double dt) {
        // Jacobian of measurement function h(x) with respect to state variables
        H.setZero();

        // Partial derivatives for x = a_x * sin(phi_x) + c_x
        // H(0, 0) = _a_x * cos(_phi_x) * dt;                      // d(h_x)/d(omega)
        H(0, 1) = _a_x * cos(_phi_x);       // d(h_x)/d(phi_x)
        H(0, 2) = sin(_phi_x);              // d(h_x)/d(a_x)
        H(0, 3) = 1.;                       // d(h_x)/d(c_x)

        // Partial derivatives for y = a_y * sin(phi_y) + c_y
        // H(1, 0) = _a_y * cos(_phi_y) * dt;                      // d(h_y)/d(omega)
        H(1, 4) = _a_y * cos(_phi_y);      // d(h_y)/d(phi_y)
        H(1, 5) = sin(_phi_y);             // d(h_y)/d(a_y)
        H(1, 6) = 1.;                      // d(h_y)/d(c_y)
    }

    void updateState(Eigen::VectorXd &state_update) {
        _omega += state_update(0);

        _phi_x = wrap_phase(_phi_x + state_update(1));
        _a_x += state_update(2);
        _c_x += state_update(3);

        _phi_y = wrap_phase(_phi_y + state_update(4));
        _a_y += state_update(5);
        _c_y += state_update(6);
    }

    void computeEKF(double x_meas, double y_meas, double t) {
        double delta_t = t - prev_t;

        // ========== PREDICTION STEP ==========
        computeF(delta_t);
        f();
        computeQ(delta_t);
        P = F * P * F.transpose() + Q;

        // ========== UPDATE STEP ==========
        // prediction
        h();

        residuals(0) = x_meas - x_pred;
        residuals(1) = y_meas - y_pred;
        residuals_vec.push_back(residuals(0));
        residuals_vec.push_back(residuals(1));

        computeH(delta_t);

        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();

        Eigen::VectorXd state_update = K * residuals;
        updateState(state_update);

        Eigen::MatrixXd IKH = I - K * H;
        P = IKH * P * IKH.transpose() + K * R * K.transpose();
        P = 0.5 * (P + P.transpose());

        _trajectory.emplace_back(_c_x, _c_y);
        prev_t = t;

        // print state
        std::cout << "\romega: " << rad2Hz(_omega) << " a_x: " << _a_x << " phi_x: " << _phi_x << " c_x: " << _c_x
                  << " a_y: " << _a_y << " phi_y: "
                  << _phi_y
                  << " c_y: " << _c_y;
        std::cout.flush();
    }


    [[nodiscard]] inline std::tuple<double, double> getComp(double delta_t) const {
        // according to out camera projection model
        const double theta_hat = _omega * delta_t;
        return std::make_tuple(_a_x * std::sin(_phi_x - theta_hat), _a_y * std::sin(_phi_y - theta_hat));
    }

    [[nodiscard]] std::tuple<double, double> getPred() const {
        return std::make_tuple(x_pred, y_pred);
    }

    [[nodiscard]] std::vector<std::pair<double, double>> getTrajectory() const {
        return _trajectory;
    }

    [[nodiscard]] std::optional<std::pair<double, double>> getCenter() const {
        if (_trajectory.empty()) return std::nullopt;
        return std::make_pair(_trajectory.back().first, _trajectory.back().second);
    }

    [[nodiscard]] double getPrevT() const {
        return prev_t;
    }

    [[nodiscard]] double getHz() const {
        return rad2Hz(std::abs(_omega));
    }

    [[nodiscard]] double getRadS() const {
        return std::abs(_omega);
    }

    [[nodiscard]] double getAmplX() const {
        return _a_x;
    }

    [[nodiscard]] double getAmplY() const {
        return _a_y;
    }

    [[nodiscard]] double getShiftX() const {
        return _c_x;
    }

    [[nodiscard]] double getShiftY() const {
        return _c_y;
    }


    [[nodiscard]] std::tuple<double, double, double, double> getCov() const {
        return std::make_tuple(P(0, 0), P(1, 1), P(2, 2), P(3, 3));
    }

    [[nodiscard]] double getOmegaCov() const {
        return P(1, 1);
    }

    [[nodiscard]] std::vector<double> getResiduals() {
        return residuals_vec;
    }

    friend std::ostream &operator<<(std::ostream &os, const EKF &ekf) {
        os << "omega: " << ekf.getRadS() << " rad/s, " << ekf.getHz() << " Hz, ";
        os << "Amplitude X: " << ekf.getAmplX() << " px, ";
        os << "Amplitude Y: " << ekf.getAmplY() << " px, ";
        os << "Shift X: " << ekf.getShiftX() << " px, ";
        os << "Shift Y: " << ekf.getShiftY() << " px";
        return os;
    }


private:
    double _omega = 0., _a_x = 0., _phi_x = 0., _c_x = 0., _a_y = 0., _phi_y = 0., _c_y = 0.;
    double prev_t = -1;

    const double _a_noise = 0, _c_noise = 0, _omega_noise = 0;

    double x_pred = 0., y_pred = 0.;

    const double _dt = 0.;

    Eigen::MatrixXd P;
    Eigen::MatrixXd F;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    Eigen::MatrixXd I;

    Eigen::VectorXd residuals;
    Eigen::MatrixXd H;

    std::vector<double> residuals_vec;
    std::vector<std::pair<double, double>> _trajectory;

    [[nodiscard]] inline double wrap_phase(double phase) const {
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
