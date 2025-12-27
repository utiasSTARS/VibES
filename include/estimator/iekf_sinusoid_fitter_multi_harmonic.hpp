/**
 * @file iekf_sinusoid_fitter.hpp
 * @brief Iterated Extended Kalman Filter (IEKF) for real-time sinusoid tracking.
 *
 * This file implements a 5-state IEKF designed to track a single sinusoidal component
 * of the form: y(t) = A*sin(theta) + B*cos(theta) + C.
 *
 * It robustly handles phase wrapping and frequency drift, making it suitable for
 * long-duration signal tracking where absolute time references might cause
 * numerical instability.
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

#ifndef PROJECT_IEKF_SINUSOID_FITTER_HPP
#define PROJECT_IEKF_SINUSOID_FITTER_HPP

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

#include "estimator/nufft_multiharmonics.hpp"

/**
 * @class IEKFSinusoidFitter
 * @brief A class for tracking sinusoidal signal parameters using an IEKF.
 *
 * The state vector is defined as x = [A, B, omega, theta, C]^T, where:
 * - A, B: Quadrature amplitudes. Amplitude = sqrt(A^2 + B^2), Phase = atan2(B, A).
 * - omega: Angular frequency (rad/s).
 * - theta: Instantaneous phase (rad). Tracks total accumulated phase.
 * - C: DC Offset / Bias.
 */
class IEKFSinusoidFitter {
public:
    // Typedefs for Eigen matrices for better readability
    using StateVector5 = Eigen::Matrix<double, 5, 1>;
    using StateCovariance5 = Eigen::Matrix<double, 5, 5>;
    using StateTransition5 = Eigen::Matrix<double, 5, 5>;

    /**
     * @brief Factory method to create a fitter initialized from external harmonic estimates (e.g., NUFFT).
     *
     * @param A Vector containing the initial sine amplitude estimate.
     * @param B Vector containing the initial cosine amplitude estimate.
     * @param omega Vector containing the initial angular frequency estimate.
     * @param C Vector containing the initial DC offset estimate.
     * @param process_noise_scale Scaling factor for the process noise covariance (Q).
     * @param measurement_noise_variance Variance of the measurement noise (R).
     * @param max_iterations Maximum number of iterations for the IEKF update step.
     * @param convergence_threshold Threshold for stopping IEKF iterations.
     * @return A configured instance of IEKFSinusoidFitter.
     * @throws std::invalid_argument If input vectors are empty or contain more than 1 harmonic.
     */
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
            throw std::invalid_argument("Maximum of 1 harmonic supported by this fitter.");
        }

        StateVector5 initial_state;
        // Initialize state: [A, B, omega, theta, C]
        // Note: theta is initialized to 0.0 relative to the parameter definition time.
        // It will be fast-forwarded to the actual start time upon the first update() call.
        initial_state << A[0], B[0], omega[0], 0.0, C[0];

        // Debug output
        std::cout << "[IEKFSinusoidFitter] Initializing state: A=" << initial_state(0)
                  << ", B=" << initial_state(1)
                  << ", omega=" << initial_state(2)
                  << ", theta=" << initial_state(3)
                  << ", C=" << initial_state(4) << std::endl;

        // Initialize Error Covariance Matrix (P)
        StateCovariance5 initial_covariance = StateCovariance5::Identity();
        initial_covariance(0, 0) = 1e2;  // A amplitude variance
        initial_covariance(1, 1) = 1e2;  // B amplitude variance
        initial_covariance(2, 2) = 1e1;  // Omega variance (relaxed)
        initial_covariance(3, 3) = 1e1;  // Theta variance
        initial_covariance(4, 4) = 1e3;  // DC Offset variance (high uncertainty)

        // Initialize Process Noise Matrix (Q)
        StateCovariance5 process_noise = StateCovariance5::Identity();
        process_noise(0, 0) = 1e0;  // A drift
        process_noise(1, 1) = 1e0;  // B drift
        process_noise(2, 2) = 1e-3; // Omega drift (small, frequency assumed stable)
        process_noise(3, 3) = 1e-3; // Theta phase jitter
        process_noise(4, 4) = 1e0;  // DC Offset drift

        return {initial_state, initial_covariance, process_noise,
                measurement_noise_variance, max_iterations, convergence_threshold};
    }

    /**
     * @brief Standard Constructor.
     * Use the factory method `createFromHarmonicEstimates` for easier initialization.
     */
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

    /**
     * @brief Process a new measurement.
     *
     * @param t Current timestamp (in seconds).
     * @param y Measured value.
     */
    void update(double t, double y) {
        measurement_count_++;

        // Initialization Step:
        // On the very first call, we do not filter. Instead, we "prime" the filter
        // by advancing the phase (theta) from t=0 to the current start time t.
        if (last_time_ < 0.0) {
            state5_(3) += state5_(2) * t; // theta_now = theta_0 + omega * t_start
            last_time_ = t;
            is_initialized_ = true;
            return;
        }

        double delta_t = t - last_time_;
        last_time_ = t;

        if (delta_t < 0) {
            std::cerr << "[IEKFSinusoidFitter] Warning: Negative delta_t encountered ("
                      << delta_t << "), skipping update." << std::endl;
            return;
        }

        update_single_sinusoid(delta_t, y);
    }

    /**
     * @brief Predicts the full signal value at a future time t.
     * @param t Target time.
     * @return Predicted value y = AC_Component + DC_Offset.
     */
    double predict(double t) const {
        if (!is_initialized_) {
            return state5_(4); // Return only DC offset if not initialized
        }
        const double theta = state5_(3) + state5_(2) * (t - last_time_);
        return state5_(0) * std::sin(theta) + state5_(1) * std::cos(theta) + state5_(4);
    }

    /**
     * @brief Predicts only the relative (AC) component of the signal.
     * Useful for motion compensation where the DC offset is the object's center.
     * @param t Target time.
     * @return Predicted AC component y_rel = A*sin(theta) + B*cos(theta).
     */
    [[nodiscard]] double predict_rel(double t) const {
        if (!is_initialized_) {
            return 0.0;
        }
        const double theta = state5_(3) + state5_(2) * (t - last_time_);
        return state5_(0) * std::sin(theta) + state5_(1) * std::cos(theta);
    }

    // --- Getters for State Properties ---

    /** @return The DC offset (C) of the signal. */
    double getShift() const { return state5_(4); }

    /** @return The frequency in Hertz. */
    double getFrequency() const { return state5_(2) / (2 * M_PI); }

    /** @return The magnitude of the sinusoid: sqrt(A^2 + B^2). */
    double getAmplitude() const { return std::sqrt(state5_(0) * state5_(0) + state5_(1) * state5_(1)); }

    /** @return The phase angle: atan2(B, A). */
    double getPhase() const { return std::atan2(state5_(1), state5_(0)); }

    /** @return The current instantaneous phase accumulator (theta). */
    double getTheta() const { return state5_(3); }

    bool is_initialized() const { return is_initialized_; }
    int getMeasurementCount() const { return measurement_count_; }

    /**
     * @brief Prints current frequency and phase info to stdout.
     */
    void print_frequencies() const {
        if (!is_initialized_) {
            std::cout << "  Fitter not initialized." << std::endl;
            return;
        }
        double freq1_hz = getFrequency();
        std::cout << "  Sinusoid 1: " << std::fixed << std::setprecision(2)
                  << freq1_hz << " Hz, theta=" << getTheta() << " rad" << std::endl;
    }

    /**
     * @brief Returns a string representation of the fitter state.
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "IEKFSinusoidFitter State: [A=" << state5_(0)
            << ", B=" << state5_(1) << ", omega=" << state5_(2)
            << ", theta=" << state5_(3) << ", C=" << state5_(4) << "]" << std::endl;

        if (is_initialized_) {
            oss << "  Derived: Amp=" << getAmplitude()
                << ", Phase=" << getPhase() << " rad"
                << ", Freq=" << getFrequency() << " Hz" << std::endl;
            oss << "  Processed Measurements: " << measurement_count_ << std::endl;
        } else {
            oss << "  Status: Not Initialized" << std::endl;
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

    // Pre-allocated temporary matrices to avoid dynamic allocation in the update loop
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

    /**
     * @brief Performs the core IEKF Prediction and Update steps.
     *
     * Steps:
     * 1. Predict state forward by delta_t (Theta evolves, others constant).
     * 2. Predict covariance using Jacobian F.
     * 3. Iteratively refine the measurement update (IEKF loop) to handle nonlinearity.
     * 4. Update final state and covariance.
     */
    void update_single_sinusoid(double delta_t, double y) {
        // === 1. PREDICTION STEP ===
        StateVector5 predicted_state = state5_;
        // Evolve phase: theta_new = theta_old + omega * delta_t
        predicted_state(3) = state5_(3) + state5_(2) * delta_t;

        // Build Jacobian F = d(f)/d(state)
        // Only theta depends on omega and previous theta.
        F_temp5_.setIdentity();
        F_temp5_(3, 2) = delta_t;  // d(theta)/d(omega)
        F_temp5_(3, 3) = 1.0;      // d(theta)/d(theta)

        // Propagate Covariance: P = F*P*F^T + Q
        StateCovariance5 predicted_covariance =
                F_temp5_ * covariance5_ * F_temp5_.transpose() + process_noise_q5_;

        // === 2. ITERATIVE UPDATE STEP (IEKF) ===
        // We iterate to find the best linearization point 'eta' for the measurement function h(x).
        StateVector5 eta = predicted_state;

        for (int i = 0; i < max_iterations_; ++i) {
            eta_prev5_ = eta;

            double A = eta(0);
            double B = eta(1);
            double theta = eta(3);
            double C = eta(4);

            double sin_theta = std::sin(theta);
            double cos_theta = std::cos(theta);

            // Measurement model: h(x) = A*sin(theta) + B*cos(theta) + C
            double y_pred = A * sin_theta + B * cos_theta + C;

            // Jacobian H = d(h)/d(state)
            H_temp5_(0, 0) = sin_theta;                           // d/dA
            H_temp5_(0, 1) = cos_theta;                           // d/dB
            H_temp5_(0, 2) = 0.0;                                 // d/dOmega (indirect via theta)
            H_temp5_(0, 3) = A * cos_theta - B * sin_theta;       // d/dTheta
            H_temp5_(0, 4) = 1.0;                                 // d/dC

            // Innovation Covariance: S = H*P*H^T + R
            double innovation_covariance =
                    (H_temp5_ * predicted_covariance * H_temp5_.transpose())(0, 0) +
                    measurement_noise_r_;

            // Numerical stability check
            if (innovation_covariance < 1e-12) {
                innovation_covariance = 1e-12;
            }

            double innovation_covariance_inv = 1.0 / innovation_covariance;

            // Kalman Gain: K = P * H^T * S^-1
            K_temp5_.noalias() =
                    predicted_covariance * H_temp5_.transpose() * innovation_covariance_inv;

            // Innovation: y - h(eta) - H*(predicted_state - eta)
            // Note: The term H*(predicted - eta) accounts for the shift in linearization point.
            double innovation =
                    y - y_pred - (H_temp5_ * (predicted_state - eta))(0, 0);

            // Update State estimate
            eta = predicted_state + K_temp5_ * innovation;

            // Wrap theta to keep numbers small: theta = theta - k*2pi
            double k = std::round(predicted_state(3) / (2 * M_PI));
            predicted_state(3) -= k * (2 * M_PI);

            // Convergence Check
            if ((eta - eta_prev5_).norm() < convergence_threshold_) {
                break;
            }
        }

        // Final State Update
        state5_ = eta;

        // === 3. COVARIANCE UPDATE (Joseph Form) ===
        // Recalculate H at the final estimate for the covariance update
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

        // Recompute S and K for final covariance update
        double innovation_cov_final =
                (H_temp5_ * predicted_covariance * H_temp5_.transpose())(0, 0) +
                measurement_noise_r_;
        if (innovation_cov_final < 1e-12) innovation_cov_final = 1e-12;

        double innovation_cov_inv_final = 1.0 / innovation_cov_final;
        K_temp5_.noalias() =
                predicted_covariance * H_temp5_.transpose() * innovation_cov_inv_final;

        // Joseph Form Update: P = (I - KH)P(I - KH)^T + KRK^T
        // Ensures P remains positive definite.
        StateCovariance5 I_KH = identity_matrix5_ - K_temp5_ * H_temp5_;
        covariance5_.noalias() =
                I_KH * predicted_covariance * I_KH.transpose() +
                K_temp5_ * measurement_noise_r_ * K_temp5_.transpose();

        // Enforce Symmetry
        covariance5_ = 0.5 * (covariance5_ + covariance5_.transpose());
    }
};

#endif //PROJECT_IEKF_SINUSOID_FITTER_HPP