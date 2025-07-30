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

        if (A.size() > 2) {
            throw std::invalid_argument("Maximum of 2 harmonics supported");
        }

        if (A.size() == 1) {

            StateVector4 initial_state;

            initial_state << A[0], B[0], omega[0], C[0];

            // Create covariance matrices
            StateCovariance4 initial_covariance = StateCovariance4::Identity();
            initial_covariance(0, 0) = 1e2;  // A amplitude
            initial_covariance(1, 1) = 1e2;  // B amplitude
            initial_covariance(2, 2) = 1e1;  // omega frequency
            initial_covariance(3, 3) = 1e4      ;  // C DC offset

            StateCovariance4 process_noise = StateCovariance4::Identity();
            process_noise(0, 0) = 1e0;
            process_noise(1, 1) = 1e0;
            process_noise(2, 2) = 1e-3;
            process_noise(3, 3) = 1e2;

            return IEKFSinusoidFitter(initial_state, initial_covariance, process_noise,
                                      measurement_noise_variance, max_iterations, convergence_threshold);
        } else {

            StateVector7 initial_state;

            initial_state<< A[0], B[0], A[1], B[1], omega[0], omega[1],
                    (C[0] + C[1]) / 2.0;  // Average offset for dual sinusoid


            // Create covariance matrices
            StateCovariance7 initial_covariance = StateCovariance7::Identity();
            initial_covariance(0, 0) = 1e2;  // A amplitude
            initial_covariance(1, 1) = 1e2;  // B amplitude
            initial_covariance(2, 2) = 1e2;  // D amplitude
            initial_covariance(3, 3) = 1e2;  // E amplitude
            initial_covariance(4, 4) = 1e1;  // omega frequency
            initial_covariance(5, 5) = 1e1;  // omega frequency
            initial_covariance(6, 6) = 1e3;  // C DC offset


            StateCovariance7 process_noise = StateCovariance7::Identity() * process_noise_scale;
            process_noise(0, 0) = 1e0;
            process_noise(1, 1) = 1e0;
            process_noise(2, 2) = 1e0;
            process_noise(3, 3) = 1e0;
            process_noise(4, 4) = 1e-3;
            process_noise(5, 5) = 1e-3;
            process_noise(6, 6) = 1e2;

            return IEKFSinusoidFitter(initial_state, initial_covariance, process_noise,
                                      measurement_noise_variance, max_iterations, convergence_threshold);
        }
    }

    // State vectors for different configurations
    // Single sinusoid: [A, B, omega, C] (4 states)
    // Dual sinusoid: [A, B, D, E, omega1, omega2, C] (7 states)
    // A, B: amplitudes for sin(omega1*t), cos(omega1*t)
    // D, E: amplitudes for sin(omega2*t), cos(omega2*t) (only in dual sinusoid mode)
    // omega1: angular frequency of first sinusoid
    // omega2: angular frequency of second sinusoid (only in dual sinusoid mode)
    // C: offset
    using StateVector4 = Eigen::Matrix<double, 4, 1>;
    using StateVector7 = Eigen::Matrix<double, 7, 1>;
    using StateCovariance4 = Eigen::Matrix<double, 4, 4>;
    using StateCovariance7 = Eigen::Matrix<double, 7, 7>;

    // Constructor for single sinusoid (maintains backward compatibility)
    IEKFSinusoidFitter(const StateVector4 &initial_state,
                       const StateCovariance4 &initial_covariance,
                       const StateCovariance4 &process_noise,
                       double measurement_noise_variance,
                       int max_iterations = 5,
                       double convergence_threshold = 1e-6)
            : use_dual_sinusoid_(false),
              state4_(initial_state),
              covariance4_(initial_covariance),
              process_noise_q4_(process_noise),
              measurement_noise_r_(measurement_noise_variance),
              max_iterations_(max_iterations),
              convergence_threshold_(convergence_threshold),
              is_initialized_(false),
              measurement_count_(0) {
        initialize_common();
    }

    // Constructor for dual sinusoid
    IEKFSinusoidFitter(const StateVector7 &initial_state,
                       const StateCovariance7 &initial_covariance,
                       const StateCovariance7 &process_noise,
                       double measurement_noise_variance,
                       int max_iterations = 1,
                       double convergence_threshold = 1e-6)
            : use_dual_sinusoid_(true),
              state7_(initial_state),
              covariance7_(initial_covariance),
              process_noise_q7_(process_noise),
              measurement_noise_r_(measurement_noise_variance),
              max_iterations_(max_iterations),
              convergence_threshold_(convergence_threshold),
              is_initialized_(false),
              measurement_count_(0) {
        initialize_common();
    }

    void update(double t, double y) {
        measurement_count_++;

        // Use first few measurements to get better initial estimate
        if (measurement_count_ <= 2) {
            if (measurement_count_ == 1) {
                if (use_dual_sinusoid_) {
                    state7_(6) = y; // Set offset C to the first measurement
                } else {
                    state4_(3) = y; // Set offset C to the first measurement
                }
                first_measurement_time_ = t;
                first_measurement_value_ = y;
            } else if (measurement_count_ == 2) {
                // Use second measurement to refine initial estimate
                double delta_y = y - first_measurement_value_;
                double delta_t = t - first_measurement_time_;
                if (std::abs(delta_t) > 1e-10) {
                    // Rough estimate of amplitude based on change
                    double rough_amplitude = std::abs(delta_y) / 2.0;
                    if (rough_amplitude > 0) {
                        if (use_dual_sinusoid_) {
                            state7_(0) = rough_amplitude; // A (first sinusoid)
                            state7_(1) = rough_amplitude; // B (first sinusoid)
                            // Keep second sinusoid amplitudes small initially
                            state7_(2) = rough_amplitude * 0.1; // D (second sinusoid)
                            state7_(3) = rough_amplitude * 0.1; // E (second sinusoid)
                        } else {
                            state4_(0) = rough_amplitude; // A
                            state4_(1) = rough_amplitude; // B
                        }
                    }
                }
                is_initialized_ = true;
            }
            return;
        }

        if (!is_initialized_) {
            is_initialized_ = true;
        }

        if (use_dual_sinusoid_) {
            update_dual_sinusoid(t, y);
        } else {
            update_single_sinusoid(t, y);
        }
    }

    double predict(double t) const {
        if (!is_initialized_) {
            return use_dual_sinusoid_ ? state7_(6) : state4_(3); // Return offset if not initialized
        }

        if (use_dual_sinusoid_) {
            double A = state7_(0);
            double B = state7_(1);
            double D = state7_(2);
            double E = state7_(3);
            double omega1 = state7_(4);
            double omega2 = state7_(5);
            double C = state7_(6);

            return A * std::sin(omega1 * t) + B * std::cos(omega1 * t) +
                   D * std::sin(omega2 * t) + E * std::cos(omega2 * t) + C;
        } else {
            double A = state4_(0);
            double B = state4_(1);
            double omega = state4_(2);
            double C = state4_(3);

            return A * std::sin(omega * t) + B * std::cos(omega * t) + C;
        }
    }

    double predict_rel(double t) const {
        if (!is_initialized_) {
            return 0.0;
        }

        if (use_dual_sinusoid_) {
            double A = state7_(0);
            double B = state7_(1);
            double D = state7_(2);
            double E = state7_(3);
            double omega1 = state7_(4);
            double omega2 = state7_(5);

            return A * std::sin(omega1 * t) + B * std::cos(omega1 * t) +
                   D * std::sin(omega2 * t) + E * std::cos(omega2 * t);
        } else {
            double A = state4_(0);
            double B = state4_(1);
            double omega = state4_(2);

            return A * std::sin(omega * t) + B * std::cos(omega * t);
        }
    }

    double getShift() const {
        return use_dual_sinusoid_ ? state7_(6) : state4_(3);
    }

    double getFrequency() const {
        double omega = use_dual_sinusoid_ ? state7_(4) : state4_(2);
        return omega / (2 * M_PI);
    }

    double getSecondFrequency() const {
        if (!use_dual_sinusoid_) return 0.0;
        return state7_(5) / (2 * M_PI);
    }

    double getAmplitude() const {
        if (use_dual_sinusoid_) {
            return std::sqrt(state7_(0) * state7_(0) + state7_(1) * state7_(1));
        } else {
            return std::sqrt(state4_(0) * state4_(0) + state4_(1) * state4_(1));
        }
    }

    double getSecondAmplitude() const {
        if (!use_dual_sinusoid_) return 0.0;
        return std::sqrt(state7_(2) * state7_(2) + state7_(3) * state7_(3));
    }

    double getPhase() const {
        if (use_dual_sinusoid_) {
            return std::atan2(state7_(1), state7_(0));
        } else {
            return std::atan2(state4_(1), state4_(0));
        }
    }

    double getSecondPhase() const {
        if (!use_dual_sinusoid_) return 0.0;
        return std::atan2(state7_(3), state7_(2));
    }

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
                  << freq1_hz << " Hz" << std::endl;

        if (use_dual_sinusoid_) {
            double freq2_hz = getSecondFrequency();
            std::cout << "  Sinusoid 2: " << std::fixed << std::setprecision(2)
                      << freq2_hz << " Hz" << std::endl;
        }
    }

    std::string to_string() const {
        std::ostringstream oss;

        if (use_dual_sinusoid_) {
            oss << "IEKFSinusoidFitter (Dual Sinusoid) State: [A=" << state7_(0)
                << ", B=" << state7_(1) << ", D=" << state7_(2) << ", E=" << state7_(3)
                << ", omega1=" << state7_(4) << ", omega2=" << state7_(5)
                << ", C=" << state7_(6) << "]" << std::endl;
        } else {
            oss << "IEKFSinusoidFitter (Single Sinusoid) State: [A=" << state4_(0)
                << ", B=" << state4_(1) << ", omega=" << state4_(2)
                << ", C=" << state4_(3) << "]" << std::endl;
        }

        if (is_initialized_) {
            double amplitude1 = getAmplitude();
            double phase1 = getPhase();
            double frequency1 = getFrequency();

            oss << " Sinusoid 1: Amplitude=" << amplitude1 << ", Phase=" << phase1
                << " rad, Frequency=" << frequency1 << " Hz" << std::endl;

            if (use_dual_sinusoid_) {
                double amplitude2 = getSecondAmplitude();
                double phase2 = getSecondPhase();
                double frequency2 = getSecondFrequency();
                oss << " Sinusoid 2: Amplitude=" << amplitude2 << ", Phase=" << phase2
                    << " rad, Frequency=" << frequency2 << " Hz" << std::endl;
            }

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

    // Single sinusoid state variables
    StateVector4 state4_;
    StateCovariance4 covariance4_;
    StateCovariance4 process_noise_q4_;

    // Dual sinusoid state variables
    StateVector7 state7_;
    StateCovariance7 covariance7_;
    StateCovariance7 process_noise_q7_;

    // Common variables
    double measurement_noise_r_;
    int max_iterations_;
    double convergence_threshold_;
    bool is_initialized_;
    int measurement_count_;

    // Store first measurement for better initialization
    double first_measurement_time_;
    double first_measurement_value_;

    // Pre-allocated temporary matrices (single sinusoid)
    Eigen::Matrix<double, 1, 4> H_temp4_;
    Eigen::Matrix<double, 4, 1> K_temp4_;
    StateVector4 eta_prev4_;
    StateCovariance4 identity_matrix4_;

    // Pre-allocated temporary matrices (dual sinusoid)
    Eigen::Matrix<double, 1, 7> H_temp7_;
    Eigen::Matrix<double, 7, 1> K_temp7_;
    StateVector7 eta_prev7_;
    StateCovariance7 identity_matrix7_;

    void initialize_common() {
        if (use_dual_sinusoid_) {
            H_temp7_.setZero();
            K_temp7_.setZero();
            eta_prev7_.setZero();
            identity_matrix7_.setIdentity();
        } else {
            H_temp4_.setZero();
            K_temp4_.setZero();
            eta_prev4_.setZero();
            identity_matrix4_.setIdentity();
        }
    }

    void update_single_sinusoid(double t, double y) {
        // --- PREDICTION STEP ---
        StateVector4 predicted_state = state4_;
        StateCovariance4 predicted_covariance = covariance4_ + process_noise_q4_;

        // --- UPDATE STEP (ITERATIVE) ---
        StateVector4 eta = predicted_state;

        for (int i = 0; i < max_iterations_; ++i) {
            eta_prev4_ = eta;

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
            H_temp4_(0, 0) = sin_wt;
            H_temp4_(0, 1) = cos_wt;
            H_temp4_(0, 2) = t * (A * cos_wt - B * sin_wt);
            H_temp4_(0, 3) = 1.0;

            // Calculate innovation covariance
            double innovation_covariance =
                    (H_temp4_ * predicted_covariance * H_temp4_.transpose())(0, 0) + measurement_noise_r_;

            // Check for numerical stability
            if (std::abs(innovation_covariance) < 1e-12) {
                break;
            }

            double innovation_covariance_inv = 1.0 / innovation_covariance;

            // Calculate Kalman Gain K
            K_temp4_.noalias() = predicted_covariance * H_temp4_.transpose() * innovation_covariance_inv;

            // Innovation
            double innovation = y - y_pred - (H_temp4_ * (predicted_state - eta))(0, 0);

            // Update state estimate for this iteration
            eta = predicted_state + K_temp4_ * innovation;

            // Check for convergence
            if ((eta - eta_prev4_).norm() < convergence_threshold_) {
                break;
            }
        }

        // Finalize update
        state4_ = eta;

        // Update covariance using Joseph form
        double A_final = state4_(0);
        double B_final = state4_(1);
        double omega_final = state4_(2);

        double sin_wt_final = std::sin(omega_final * t);
        double cos_wt_final = std::cos(omega_final * t);

        H_temp4_(0, 0) = sin_wt_final;
        H_temp4_(0, 1) = cos_wt_final;
        H_temp4_(0, 2) = t * (A_final * cos_wt_final - B_final * sin_wt_final);
        H_temp4_(0, 3) = 1.0;

        double innovation_cov_final =
                (H_temp4_ * predicted_covariance * H_temp4_.transpose())(0, 0) + measurement_noise_r_;

        if (std::abs(innovation_cov_final) > 1e-12) {
            double innovation_cov_inv_final = 1.0 / innovation_cov_final;
            K_temp4_.noalias() = predicted_covariance * H_temp4_.transpose() * innovation_cov_inv_final;

            // Update covariance using Joseph form for numerical stability
            StateCovariance4 I_KH = identity_matrix4_ - K_temp4_ * H_temp4_;
            covariance4_.noalias() = I_KH * predicted_covariance * I_KH.transpose() +
                                     K_temp4_ * measurement_noise_r_ * K_temp4_.transpose();
        }
    }

    void update_dual_sinusoid(double t, double y) {
        // --- PREDICTION STEP ---
        StateVector7 predicted_state = state7_;
        StateCovariance7 predicted_covariance = covariance7_ + process_noise_q7_;

        // --- UPDATE STEP (ITERATIVE) ---
        StateVector7 eta = predicted_state;

        for (int i = 0; i < max_iterations_; ++i) {
            eta_prev7_ = eta;

            double A = eta(0);
            double B = eta(1);
            double D = eta(2);
            double E = eta(3);
            double omega1 = eta(4);
            double omega2 = eta(5);
            double C = eta(6);

            // Calculate trigonometric functions once per iteration
            double sin_w1t = std::sin(omega1 * t);
            double cos_w1t = std::cos(omega1 * t);
            double sin_w2t = std::sin(omega2 * t);
            double cos_w2t = std::cos(omega2 * t);

            // Calculate measurement prediction h(eta)
            double y_pred = A * sin_w1t + B * cos_w1t + D * sin_w2t + E * cos_w2t + C;

            // Calculate Jacobian H
            H_temp7_(0, 0) = sin_w1t;
            H_temp7_(0, 1) = cos_w1t;
            H_temp7_(0, 2) = sin_w2t;
            H_temp7_(0, 3) = cos_w2t;
            H_temp7_(0, 4) = t * (A * cos_w1t - B * sin_w1t);  // derivative w.r.t. omega1
            H_temp7_(0, 5) = t * (D * cos_w2t - E * sin_w2t);  // derivative w.r.t. omega2
            H_temp7_(0, 6) = 1.0;

            // Calculate innovation covariance
            double innovation_covariance =
                    (H_temp7_ * predicted_covariance * H_temp7_.transpose())(0, 0) + measurement_noise_r_;

            // Check for numerical stability
            if (std::abs(innovation_covariance) < 1e-12) {
                break;
            }

            double innovation_covariance_inv = 1.0 / innovation_covariance;

            // Calculate Kalman Gain K
            K_temp7_.noalias() = predicted_covariance * H_temp7_.transpose() * innovation_covariance_inv;

            // Innovation
            double innovation = y - y_pred - (H_temp7_ * (predicted_state - eta))(0, 0);

            // Update state estimate for this iteration
            eta = predicted_state + K_temp7_ * innovation;

            // Check for convergence
            if ((eta - eta_prev7_).norm() < convergence_threshold_) {
                break;
            }
        }

        // Finalize update
        state7_ = eta;

        // Update covariance using Joseph form
        double A_final = state7_(0);
        double B_final = state7_(1);
        double D_final = state7_(2);
        double E_final = state7_(3);
        double omega1_final = state7_(4);
        double omega2_final = state7_(5);

        double sin_w1t_final = std::sin(omega1_final * t);
        double cos_w1t_final = std::cos(omega1_final * t);
        double sin_w2t_final = std::sin(omega2_final * t);
        double cos_w2t_final = std::cos(omega2_final * t);

        H_temp7_(0, 0) = sin_w1t_final;
        H_temp7_(0, 1) = cos_w1t_final;
        H_temp7_(0, 2) = sin_w2t_final;
        H_temp7_(0, 3) = cos_w2t_final;
        H_temp7_(0, 4) = t * (A_final * cos_w1t_final - B_final * sin_w1t_final);
        H_temp7_(0, 5) = t * (D_final * cos_w2t_final - E_final * sin_w2t_final);
        H_temp7_(0, 6) = 1.0;

        double innovation_cov_final =
                (H_temp7_ * predicted_covariance * H_temp7_.transpose())(0, 0) + measurement_noise_r_;

        if (std::abs(innovation_cov_final) > 1e-12) {
            double innovation_cov_inv_final = 1.0 / innovation_cov_final;
            K_temp7_.noalias() = predicted_covariance * H_temp7_.transpose() * innovation_cov_inv_final;

            // Update covariance using Joseph form for numerical stability
            StateCovariance7 I_KH = identity_matrix7_ - K_temp7_ * H_temp7_;
            covariance7_.noalias() = I_KH * predicted_covariance * I_KH.transpose() +
                                     K_temp7_ * measurement_noise_r_ * K_temp7_.transpose();
        }
    }
};

#endif //PROJECT_IEKF_SINUSOID_FITTER_HPP