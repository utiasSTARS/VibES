#ifndef PROJECT_IEKF_SINUSOID_FITTER_HPP
#define PROJECT_IEKF_SINUSOID_FITTER_HPP

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

#include "estimator/nufft_multiharmonics.hpp"

class IEKFSinusoidFitter {
public:
    static IEKFSinusoidFitter createFromHarmonicEstimates(
            std::vector<double> &A,
            std::vector<double> &B,
            std::vector<double> &omega,
            std::vector<double> &C,
            double process_noise_scale = 0.01,
            double measurement_noise_variance = 0.1,
            int max_iterations = 1,
            double convergence_threshold = 1e-6
    ) {
        if (A.empty()) {
            throw std::invalid_argument("HarmonicEstimate vector cannot be empty");
        }

        if (A.size() > 1) {
            throw std::invalid_argument("Maximum of 1 harmonic supported");
        }

        StateVector5 initial_state;
        // We initialize theta to 0.0 here.
        // The update() function will fast-forward this to (omega * start_time)
        // upon the first valid measurement.
        initial_state << A[0], B[0], omega[0], 0.0, C[0];

        // output the initial state
        std::cout << "Initial state: A=" << initial_state(0)
                  << ", B=" << initial_state(1)
                  << ", omega=" << initial_state(2)
                  << ", theta=" << initial_state(3)
                  << ", C=" << initial_state(4) << std::endl;


        // Create covariance matrices
        StateCovariance5 initial_covariance = StateCovariance5::Identity();
        initial_covariance(0, 0) = 1e2;  // A amplitude
        initial_covariance(1, 1) = 1e2;  // B amplitude
        initial_covariance(2, 2) = 1e1;  // omega frequency
        initial_covariance(3, 3) = 1e1;  // theta angular position
        initial_covariance(4, 4) = 1e3;  // C DC offset

        StateCovariance5 process_noise = StateCovariance5::Identity();
        process_noise(0, 0) = 1e0;  // A process noise
        process_noise(1, 1) = 1e0;  // B process noise
        process_noise(2, 2) = 1e-3; // omega process noise
        process_noise(3, 3) = 1e-3; // theta process noise
        process_noise(4, 4) = 1e0;  // C process noise

        return {initial_state, initial_covariance, process_noise,
                measurement_noise_variance, max_iterations, convergence_threshold};
    }

    using StateVector5 = Eigen::Matrix<double, 5, 1>;
    using StateCovariance5 = Eigen::Matrix<double, 5, 5>;
    using StateTransition5 = Eigen::Matrix<double, 5, 5>;

    IEKFSinusoidFitter(const StateVector5 &initial_state,
                       const StateCovariance5 &initial_covariance,
                       const StateCovariance5 &process_noise,
                       double measurement_noise_variance,
                       int max_iterations = 5,
                       double convergence_threshold = 1e-6)
            : use_dual_sinusoid_(false),
              state5_(initial_state),
              covariance5_(initial_covariance),
              process_noise_q5_(process_noise),
              measurement_noise_r_(measurement_noise_variance),
              max_iterations_(max_iterations),
              convergence_threshold_(convergence_threshold),
              is_initialized_(false),
              measurement_count_(0),
              last_time_(-1.0) {
        initialize_common();
    }

    void update(double t, double y) {
        measurement_count_++;

        if (last_time_ < 0.0) {
            state5_(3) += state5_(2) * t;
            last_time_ = t;
            is_initialized_ = true;
            return; // Skip prediction/update on the priming step
        }

        double delta_t = t - last_time_;
        last_time_ = t;

        // REMOVED: The block that attempted to self-initialize theta from y
        // and caused the phase mismatch.

        if (delta_t < 0) {
            std::cerr << "[Warning] Negative delta_t encountered, skipping update." << std::endl;
            return;
        }

        update_single_sinusoid(delta_t, y);
    }

    double predict(double t) const {
        if (!is_initialized_) {
            return state5_(4);
        }

        const double theta = state5_(3) + state5_(2) * (t - last_time_);
        return state5_(0) * std::sin(theta) + state5_(1) * std::cos(theta) + state5_(4);
    }

    [[nodiscard]] double predict_rel(double t) const {
        if (!is_initialized_) {
            return 0.0;
        }

        const double theta = state5_(3) + state5_(2) * (t - last_time_);
        return state5_(0) * std::sin(theta) + state5_(1) * std::cos(theta);
    }

    // Getters
    double getShift() const { return state5_(4); }
    double getFrequency() const { return state5_(2) / (2 * M_PI); }
    double getAmplitude() const { return std::sqrt(state5_(0) * state5_(0) + state5_(1) * state5_(1)); }
    double getPhase() const { return std::atan2(state5_(1), state5_(0)); }
    double getTheta() const { return state5_(3); }
    bool is_initialized() const { return is_initialized_; }
    bool uses_dual_sinusoid() const { return use_dual_sinusoid_; }
    int getMeasurementCount() const { return measurement_count_; }

    void print_frequencies() const {
        if (!is_initialized_) {
            std::cout << "  Fitter not initialized." << std::endl;
            return;
        }
        double freq1_hz = getFrequency();
        std::cout << "  Sinusoid 1: " << std::fixed << std::setprecision(2)
                  << freq1_hz << " Hz, θ=" << getTheta() << " rad" << std::endl;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "IEKFSinusoidFitter (Single Sinusoid) State: [A=" << state5_(0)
            << ", B=" << state5_(1) << ", omega=" << state5_(2)
            << ", theta=" << state5_(3) << ", C=" << state5_(4) << "]" << std::endl;

        if (is_initialized_) {
            oss << " Sinusoid 1: Amplitude=" << getAmplitude()
                << ", Phase=" << getPhase() << " rad, Frequency=" << getFrequency()
                << " Hz, θ=" << getTheta() << " rad" << std::endl;

            oss << " DC Offset=" << getShift() << std::endl;
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
    bool use_dual_sinusoid_;
    StateVector5 state5_;
    StateCovariance5 covariance5_;
    StateCovariance5 process_noise_q5_;

    double measurement_noise_r_;
    int max_iterations_;
    double convergence_threshold_;
    bool is_initialized_;
    int measurement_count_;
    double last_time_;

    // Pre-allocated matrices
    Eigen::Matrix<double, 1, 5> H_temp5_;
    Eigen::Matrix<double, 5, 1> K_temp5_;
    StateVector5 eta_prev5_;
    StateCovariance5 identity_matrix5_;
    StateTransition5 F_temp5_;

    void initialize_common() {
        H_temp5_.setZero();
        K_temp5_.setZero();
        eta_prev5_.setZero();
        identity_matrix5_.setIdentity();
        F_temp5_.setIdentity();
    }


    void update_single_sinusoid(double delta_t, double y) {
        // === PREDICTION STEP with State Augmentation ===
        StateVector5 predicted_state = state5_;
        predicted_state(3) = state5_(3) + state5_(2) * delta_t;  // θ = θ + ωΔt

        // State transition Jacobian F
        F_temp5_.setIdentity();
        F_temp5_(3, 2) = delta_t;  // ∂θ/∂ω
        F_temp5_(3, 3) = 1.0;      // ∂θ/∂θ

        // Predicted covariance
        StateCovariance5 predicted_covariance =
                F_temp5_ * covariance5_ * F_temp5_.transpose() + process_noise_q5_;

        // === ITERATIVE UPDATE STEP ===
        StateVector5 eta = predicted_state;

        for (int i = 0; i < max_iterations_; ++i) {
            eta_prev5_ = eta;

            double A = eta(0);
            double B = eta(1);
            double theta = eta(3);
            double C = eta(4);

            double sin_theta = std::sin(theta);
            double cos_theta = std::cos(theta);

            // Measurement prediction
            double y_pred = A * sin_theta + B * cos_theta + C;

            // Measurement Jacobian H
            H_temp5_(0, 0) = sin_theta;
            H_temp5_(0, 1) = cos_theta;
            H_temp5_(0, 2) = 0.0;
            H_temp5_(0, 3) = A * cos_theta - B * sin_theta;
            H_temp5_(0, 4) = 1.0;

            // Innovation covariance
            double innovation_covariance =
                    (H_temp5_ * predicted_covariance * H_temp5_.transpose())(0, 0) +
                    measurement_noise_r_;
            if (innovation_covariance < 1e-12) {
                innovation_covariance = 1e-12;
            }

            double innovation_covariance_inv = 1.0 / innovation_covariance;

            // Kalman gain
            K_temp5_.noalias() =
                    predicted_covariance * H_temp5_.transpose() * innovation_covariance_inv;

            // Innovation with linearization correction
            double innovation =
                    y - y_pred - (H_temp5_ * (predicted_state - eta))(0, 0);

            // State update
            eta = predicted_state + K_temp5_ * innovation;

            // mean recentering of theta
            double k = std::round(predicted_state(3) / (2 * M_PI));
            predicted_state(3) -= k * (2 * M_PI);

            // Convergence check
            if ((eta - eta_prev5_).norm() < convergence_threshold_) {
                break;
            }
        }

        // Update state
        state5_ = eta;

        // === COVARIANCE UPDATE (Joseph Form) ===
        double A_final = state5_(0);
        double B_final = state5_(1);
        double theta_final = state5_(3);

        double sin_theta_final = std::sin(theta_final);
        double cos_theta_final = std::cos(theta_final);

        H_temp5_(0, 0) = sin_theta_final;
        H_temp5_(0, 1) = cos_theta_final;
        H_temp5_(0, 2) = 0.0;
        H_temp5_(0, 3) = A_final * cos_theta_final - B_final * sin_theta_final;
        H_temp5_(0, 4) = 1.0;

        double innovation_cov_final =
                (H_temp5_ * predicted_covariance * H_temp5_.transpose())(0, 0) +
                measurement_noise_r_;
        if (innovation_cov_final < 1e-12) {
            innovation_cov_final = 1e-12;
        }

        double innovation_cov_inv_final = 1.0 / innovation_cov_final;
        K_temp5_.noalias() =
                predicted_covariance * H_temp5_.transpose() * innovation_cov_inv_final;

        StateCovariance5 I_KH = identity_matrix5_ - K_temp5_ * H_temp5_;
        covariance5_.noalias() =
                I_KH * predicted_covariance * I_KH.transpose() +
                K_temp5_ * measurement_noise_r_ * K_temp5_.transpose();

        // Optional: ensure covariance remains symmetric (avoid numerical drift)
        covariance5_ = 0.5 * (covariance5_ + covariance5_.transpose());
    }

};

#endif //PROJECT_IEKF_SINUSOID_FITTER_HPP