#ifndef PROJECT_IEKF_SINUSOID_FITTER_HPP
#define PROJECT_IEKF_SINUSOID_FITTER_HPP

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class IEKFSinusoidFitter {
public:
    // State: [A, B, D, E, omega, C]
    // A, B: amplitudes for sin(wt), cos(wt)
    // D, E: amplitudes for sin(2wt), cos(2wt)
    // omega: angular frequency
    // C: offset
    using StateVector = Eigen::Matrix<double, 6, 1>;
    using StateCovariance = Eigen::Matrix<double, 6, 6>;

    IEKFSinusoidFitter(const StateVector& initial_state,
                         const StateCovariance& initial_covariance,
                         const StateCovariance& process_noise,
                         double measurement_noise_variance,
                         int iterations = 5)
        : state_(initial_state),
          covariance_(initial_covariance),
          process_noise_q_(process_noise),
          measurement_noise_r_(measurement_noise_variance),
          iterations_(iterations),
          is_initialized_(false) {}

    void update(double t, double y) {
        if (!is_initialized_) {
            state_(5) = y; // Set offset C to the first measurement
            is_initialized_ = true;
            return;
        }

        // --- PREDICTION STEP ---
        // State is assumed constant, so x_k|k-1 = x_k-1|k-1
        StateVector predicted_state = state_;
        // P_k|k-1 = P_k-1|k-1 + Q
        StateCovariance predicted_covariance = covariance_ + process_noise_q_;

        // --- UPDATE STEP (ITERATIVE) ---
        StateVector eta = predicted_state;

        Eigen::Matrix<double, 1, 6> H;

        for (int i = 0; i < iterations_; ++i) {
            double A = eta(0);
            double B = eta(1);
            double D = eta(2);
            double E = eta(3);
            double omega = eta(4);
            double C = eta(5);

            // Calculate measurement prediction h(eta)
            double sin_wt = std::sin(omega * t);
            double cos_wt = std::cos(omega * t);
            double sin_2wt = std::sin(2 * omega * t);
            double cos_2wt = std::cos(2 * omega * t);
            double y_pred = A * sin_wt + B * cos_wt + D * sin_2wt + E * cos_2wt + C;

            // Calculate Jacobian H
            H(0, 0) = sin_wt;
            H(0, 1) = cos_wt;
            H(0, 2) = sin_2wt;
            H(0, 3) = cos_2wt;
            H(0, 4) = t * (A * cos_wt - B * sin_wt) + 2 * t * (D * cos_2wt - E * sin_2wt);
            H(0, 5) = 1.0;

            // Calculate Kalman Gain K
            double innovation_covariance_inv = 1.0 / (H * predicted_covariance * H.transpose() + measurement_noise_r_);
            Eigen::Matrix<double, 6, 1> K = predicted_covariance * H.transpose() * innovation_covariance_inv;

            // Update state estimate for this iteration
            eta = predicted_state + K * (y - y_pred - H * (predicted_state - eta));
        }

        // Finalize update
        state_ = eta;

        // Recalculate H at the final state estimate to update covariance
        double A_final = state_(0), B_final = state_(1), D_final = state_(2), E_final = state_(3), omega_final = state_(4);
        double sin_wt_f = std::sin(omega_final * t);
        double cos_wt_f = std::cos(omega_final * t);
        double sin_2wt_f = std::sin(2 * omega_final * t);
        double cos_2wt_f = std::cos(2 * omega_final * t);

        H(0, 0) = sin_wt_f;
        H(0, 1) = cos_wt_f;
        H(0, 2) = sin_2wt_f;
        H(0, 3) = cos_2wt_f;
        H(0, 4) = t * (A_final * cos_wt_f - B_final * sin_wt_f) + 2 * t * (D_final * cos_2wt_f - E_final * sin_2wt_f);
        H(0, 5) = 1.0;

        double innovation_cov_inv_final = 1.0 / (H * predicted_covariance * H.transpose() + measurement_noise_r_);
        Eigen::Matrix<double, 6, 1> K_final = predicted_covariance * H.transpose() * innovation_cov_inv_final;

        covariance_ = (StateCovariance::Identity() - K_final * H) * predicted_covariance;
    }

    double predict(double t) const {
        double A = state_(0);
        double B = state_(1);
        double D = state_(2);
        double E = state_(3);
        double omega = state_(4);
        double C = state_(5);
        return A * std::sin(omega * t) + B * std::cos(omega * t) + D * std::sin(2 * omega * t) + E * std::cos(2 * omega * t) + C;
    }

    double predict_rel(double t) const {
        double A = state_(0);
        double B = state_(1);
        double omega = state_(4);
        return A * std::sin(omega * t) + B * std::cos(omega * t) ;
    }

    const StateVector& get_state() const { return state_; }
    bool is_initialized() const { return is_initialized_; }

    void print_frequencies() const {
        if (!is_initialized_) {
            std::cout << "  Fitter not initialized." << std::endl;
            return;
        }
        double omega = state_(4);
        double freq_hz_1 = omega / (2 * M_PI);
        double freq_hz_2 = (2 * omega) / (2 * M_PI);
        std::cout << "  Harmonic 1 (Fundamental): " << std::fixed << std::setprecision(2) << freq_hz_1 << " Hz" << std::endl;
        std::cout << "  Harmonic 2: " << std::fixed << std::setprecision(2) << freq_hz_2 << " Hz" << std::endl;
    }

private:
    StateVector state_;
    StateCovariance covariance_;
    StateCovariance process_noise_q_;
    double measurement_noise_r_;
    int iterations_;
    bool is_initialized_;
};

#endif //PROJECT_IEKF_SINUSOID_FITTER_HPP 