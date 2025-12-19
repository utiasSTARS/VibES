#ifndef PROJECT_IEKF_SINUSOID_FITTER_HPP
#define PROJECT_IEKF_SINUSOID_FITTER_HPP

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

// Constants
namespace {
    constexpr double DEFAULT_MEASUREMENT_NOISE = 0.5;
}

class IEKFSinusoidFitter {
public:
    /**
 * Creates and configures an IEKF sinusoid fitter with given parameters
 */
    static IEKFSinusoidFitter createIEKFFitter(double A, double B, double omega, double C, int iterations = 1) {
        IEKFSinusoidFitter::StateVector initial_state;
        initial_state << A, B, omega, C;

        IEKFSinusoidFitter::StateCovariance initial_covariance;
        initial_covariance.setIdentity();
        initial_covariance(0, 0) = 1e2;  // A amplitude
        initial_covariance(1, 1) = 1e2;  // B amplitude
        initial_covariance(2, 2) = 1e1;  // omega frequency
        initial_covariance(3, 3) = 1e3;  // C DC offset

        IEKFSinusoidFitter::StateCovariance process_noise;
        process_noise.setIdentity();
        process_noise(0, 0) = 1e0;
        process_noise(1, 1) = 1e0;
        process_noise(2, 2) = 1e-3; // 1e-2;
        process_noise(3, 3) = 1e0; // 1e1;

        return {initial_state, initial_covariance, process_noise,
                DEFAULT_MEASUREMENT_NOISE, iterations};
    }

    // State: [A, B, omega, C]
    // A, B: amplitudes for sin(wt), cos(wt)
    // omega: angular frequency
    // C: offset
    using StateVector = Eigen::Matrix<double, 4, 1>;
    using StateCovariance = Eigen::Matrix<double, 4, 4>;

    IEKFSinusoidFitter(const StateVector &initial_state,
                       const StateCovariance &initial_covariance,
                       const StateCovariance &process_noise,
                       double measurement_noise_variance,
                       int max_iterations = 5,
                       double convergence_threshold = 1e-6)
            : state_(initial_state),
              covariance_(initial_covariance),
              process_noise_q_(process_noise),
              measurement_noise_r_(measurement_noise_variance),
              max_iterations_(max_iterations),
              convergence_threshold_(convergence_threshold),
              is_initialized_(false),
              measurement_count_(0) {

        // Pre-allocate temporary matrices to avoid repeated allocations
        H_temp_.setZero();
        K_temp_.setZero();
        eta_prev_.setZero();

        // Initialize identity matrix once
        identity_matrix_.setIdentity();
    }

    void update(double t, double y) {
        measurement_count_++;

        // Use first few measurements to get better initial estimate
        if (measurement_count_ <= 2) {
            if (measurement_count_ == 1) {
                state_(3) = y; // Set offset C to the first measurement
                first_measurement_time_ = t;
                first_measurement_value_ = y;
            } else if (measurement_count_ == 2) {
                // Use second measurement to refine initial estimate
                // This is a simple heuristic - could be improved with more sophisticated initialization
                double delta_y = y - first_measurement_value_;
                double delta_t = t - first_measurement_time_;
                if (std::abs(delta_t) > 1e-10) {
                    // Rough estimate of amplitude based on change
                    double rough_amplitude = std::abs(delta_y) / 2.0;
                    if (rough_amplitude > 0) {
                        state_(0) = rough_amplitude; // A
                        state_(1) = rough_amplitude; // B
                    }
                }
                is_initialized_ = true;
            }
            return;
        }

        if (!is_initialized_) {
            is_initialized_ = true;
        }

        // --- PREDICTION STEP ---
        // State is assumed constant, so x_k|k-1 = x_k-1|k-1
        StateVector predicted_state = state_;

        // P_k|k-1 = P_k-1|k-1 + Q
        StateCovariance predicted_covariance = covariance_ + process_noise_q_;

        // --- UPDATE STEP (ITERATIVE) ---
        StateVector eta = predicted_state;

        for (int i = 0; i < max_iterations_; ++i) {
            eta_prev_ = eta;

            double A = eta(0);
            double B = eta(1);
            double omega = eta(2);
            double C = eta(3);

            // Calculate trigonometric functions once per iteration
            double sin_wt = std::sin(omega * t);
            double cos_wt = std::cos(omega * t);

            // Calculate measurement prediction h(eta)
            double y_pred = A * sin_wt + B * cos_wt + C;

            // Calculate Jacobian H
            H_temp_(0, 0) = sin_wt;
            H_temp_(0, 1) = cos_wt;
            H_temp_(0, 2) = t * (A * cos_wt - B * sin_wt);
            H_temp_(0, 3) = 1.0;

            // Calculate innovation covariance more efficiently
            // S = H * P * H^T + R
            double innovation_covariance =
                    (H_temp_ * predicted_covariance * H_temp_.transpose())(0, 0) + measurement_noise_r_;

            // Check for numerical stability
            if (std::abs(innovation_covariance) < 1e-12) {
                break; // Avoid division by very small numbers
            }

            double innovation_covariance_inv = 1.0 / innovation_covariance;

            // Calculate Kalman Gain K more efficiently
            K_temp_.noalias() = predicted_covariance * H_temp_.transpose() * innovation_covariance_inv;

            // Innovation
            double innovation = y - y_pred - (H_temp_ * (predicted_state - eta))(0, 0);

            // Update state estimate for this iteration
            eta = predicted_state + K_temp_ * innovation;

            // Check for convergence
            if ((eta - eta_prev_).norm() < convergence_threshold_) {
                break;
            }
        }

        // Finalize update
        state_ = eta;

        // Recalculate H at the final state estimate to update covariance
        double A_final = state_(0);
        double B_final = state_(1);
        double omega_final = state_(2);

        // Reuse trigonometric calculations
        double sin_wt_final = std::sin(omega_final * t);
        double cos_wt_final = std::cos(omega_final * t);

        H_temp_(0, 0) = sin_wt_final;
        H_temp_(0, 1) = cos_wt_final;
        H_temp_(0, 2) = t * (A_final * cos_wt_final - B_final * sin_wt_final);
        H_temp_(0, 3) = 1.0;

        double innovation_cov_final =
                (H_temp_ * predicted_covariance * H_temp_.transpose())(0, 0) + measurement_noise_r_;

        if (std::abs(innovation_cov_final) > 1e-12) {
            double innovation_cov_inv_final = 1.0 / innovation_cov_final;
            K_temp_.noalias() = predicted_covariance * H_temp_.transpose() * innovation_cov_inv_final;

            // Update covariance using Joseph form for numerical stability
            StateCovariance I_KH = identity_matrix_ - K_temp_ * H_temp_;
            covariance_.noalias() = I_KH * predicted_covariance * I_KH.transpose() +
                                    K_temp_ * measurement_noise_r_ * K_temp_.transpose();
        }
    }

    double predict(double t) const {
        if (!is_initialized_) {
            return state_(3); // Return offset if not initialized
        }

        double A = state_(0);
        double B = state_(1);
        double omega = state_(2);
        double C = state_(3);

        return A * std::sin(omega * t) + B * std::cos(omega * t) + C;
    }

    double predict_rel(double t) const {
        if (!is_initialized_) {
            return 0.0;
        }

        double A = state_(0);
        double B = state_(1);
        double omega = state_(2);

        return A * std::sin(omega * t) + B * std::cos(omega * t);
    }

    const StateVector &get_state() const { return state_; }

    double getShift() const { return state_(3); }

    double getFrequency() const { return state_(2) / (2 * M_PI); }

    double getAmplitude() const {
        return std::sqrt(state_(0) * state_(0) + state_(1) * state_(1));
    }

    double getPhase() const {
        return std::atan2(state_(1), state_(0));
    }

    bool is_initialized() const { return is_initialized_; }

    int getMeasurementCount() const { return measurement_count_; }

    void print_frequencies() const {
        if (!is_initialized_) {
            std::cout << "  Fitter not initialized." << std::endl;
            return;
        }
        double freq_hz = getFrequency();
        std::cout << "  Harmonic 1 (Fundamental): " << std::fixed << std::setprecision(2)
                  << freq_hz << " Hz" << std::endl;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "IEKFSinusoidFitter State: [A=" << state_(0) << ", B=" << state_(1)
            << ", omega=" << state_(2) << ", C=" << state_(3) << "]" << std::endl;

        if (is_initialized_) {
            double amplitude = getAmplitude();
            double phase = getPhase();
            double frequency = getFrequency();

            oss << " Amplitude=" << amplitude << ", Phase=" << phase << " rad, C=" << state_(3)
                << ", Frequency=" << frequency << " Hz" << std::endl;
            oss << " Measurements processed: " << measurement_count_ << std::endl;
        } else {
            oss << " Not initialized (need at least 3 measurements)" << std::endl;
        }

        return oss.str();
    }

    friend std::ostream &operator<<(std::ostream &os, const IEKFSinusoidFitter &fitter) {
        os << fitter.to_string();
        return os;
    }

private:
    StateVector state_;
    StateCovariance covariance_;
    StateCovariance process_noise_q_;
    double measurement_noise_r_;
    int max_iterations_;
    double convergence_threshold_;
    bool is_initialized_;
    int measurement_count_;

    // Store first measurement for better initialization
    double first_measurement_time_;
    double first_measurement_value_;

    // Pre-allocated temporary matrices to avoid repeated allocations
    Eigen::Matrix<double, 1, 4> H_temp_;
    Eigen::Matrix<double, 4, 1> K_temp_;
    StateVector eta_prev_;
    StateCovariance identity_matrix_;
};

#endif //PROJECT_IEKF_SINUSOID_FITTER_HPP