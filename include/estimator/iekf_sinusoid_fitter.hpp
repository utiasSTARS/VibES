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
                       int iterations = 1)
            : state_(initial_state),
              covariance_(initial_covariance),
              process_noise_q_(process_noise),
              measurement_noise_r_(measurement_noise_variance),
              iterations_(iterations),
              is_initialized_(false) {
        // Initialize static omega from the first object created
        if (omega_ == 0.0) {
            omega_ = initial_state(2);
        }
        // Ensure this object uses the shared omega
        state_(2) = omega_;
    }

    void update(double t, double y) {
        if (!is_initialized_) {
            state_(3) = y; // Set offset C to the first measurement
            is_initialized_ = true;
            return;
        }

        // --- PREDICTION STEP ---
        // State is assumed constant, so x_k|k-1 = x_k-1|k-1
        StateVector predicted_state = state_;
        // Ensure we use the shared omega
        predicted_state(2) = omega_;

        // P_k|k-1 = P_k-1|k-1 + Q
        StateCovariance predicted_covariance = covariance_ + process_noise_q_;

        // --- UPDATE STEP (ITERATIVE) ---
        StateVector eta = predicted_state;

        Eigen::Matrix<double, 1, 4> H;

        for (int i = 0; i < iterations_; ++i) {
            double A = eta(0);
            double B = eta(1);
            double omega = eta(2);
            double C = eta(3);

            // Calculate measurement prediction h(eta)
            double sin_wt = std::sin(omega * t);
            double cos_wt = std::cos(omega * t);
            double y_pred = A * sin_wt + B * cos_wt + C;

            // Calculate Jacobian H
            H(0, 0) = sin_wt;
            H(0, 1) = cos_wt;
            H(0, 2) = t * (A * cos_wt - B * sin_wt);
            H(0, 3) = 1.0;

            // Calculate Kalman Gain K
            double innovation_covariance_inv = 1.0 / (H * predicted_covariance * H.transpose() + measurement_noise_r_);
            Eigen::Matrix<double, 4, 1> K = predicted_covariance * H.transpose() * innovation_covariance_inv;

            // Update state estimate for this iteration
            eta = predicted_state + K * (y - y_pred - H * (predicted_state - eta));

            // Keep omega fixed to the shared value
//            eta(2) = omega_;
        }

        // Finalize update
        state_ = eta;
        // Ensure state uses shared omega
        omega_ = state_(2);

        // Recalculate H at the final state estimate to update covariance
        double A_final = state_(0), B_final = state_(1);
        double sin_wt_f = std::sin(omega_ * t);
        double cos_wt_f = std::cos(omega_ * t);

        H(0, 0) = sin_wt_f;
        H(0, 1) = cos_wt_f;
        H(0, 2) = t * (A_final * cos_wt_f - B_final * sin_wt_f);
        H(0, 3) = 1.0;

        double innovation_cov_inv_final = 1.0 / (H * predicted_covariance * H.transpose() + measurement_noise_r_);
        Eigen::Matrix<double, 4, 1> K_final = predicted_covariance * H.transpose() * innovation_cov_inv_final;

        covariance_ = (StateCovariance::Identity() - K_final * H) * predicted_covariance;
    }

    double predict(double t) const {
        double A = state_(0);
        double B = state_(1);
        double C = state_(3);
        return A * std::sin(omega_ * t) + B * std::cos(omega_ * t) + C;
    }

    double predict_rel(double t) const {
        double A = state_(0);
        double B = state_(1);
        return A * std::sin(omega_ * t) + B * std::cos(omega_ * t);
    }

    const StateVector &get_state() const { return state_; }

    const double getShift() const { return state_(3); }

    bool is_initialized() const { return is_initialized_; }

    void print_frequencies() const {
        if (!is_initialized_) {
            std::cout << "  Fitter not initialized." << std::endl;
            return;
        }
        double freq_hz_1 = omega_ / (2 * M_PI);
        std::cout << "  Harmonic 1 (Fundamental): " << std::fixed << std::setprecision(2) << freq_hz_1 << " Hz"
                  << std::endl;
    }

    // print on sstream
    std::string to_string() const {
        std::ostringstream oss;
        oss << "IEKFSinusoidFitter State: [A=" << state_(0) << ", B=" << state_(1)
            << ", omega=" << omega_ << ", C=" << state_(3) << "]" << std::endl;
        // print the amplitude and shift as G sin(omega t + phi) + C
        double amplitude = std::sqrt(state_(0) * state_(0) + state_(1) * state_(1));
        double phase = std::atan2(state_(1), state_(0));
        oss << " Amplitude=" << amplitude << ", Phase=" << phase << " rad, C=" << state_(3)
            << ", Frequency=" << omega_ / (2 * M_PI) << " Hz" << std::endl;
        return oss.str();
    }

    friend std::ostream &operator<<(std::ostream &os, const IEKFSinusoidFitter &fitter) {
        os << fitter.to_string();
        return os;
    }

private:
    StateVector state_;
    static double omega_;
    StateCovariance covariance_;
    StateCovariance process_noise_q_;
    double measurement_noise_r_;
    int iterations_;
    bool is_initialized_;
};

inline double IEKFSinusoidFitter::omega_ = 0.0;

#endif //PROJECT_IEKF_SINUSOID_FITTER_HPP