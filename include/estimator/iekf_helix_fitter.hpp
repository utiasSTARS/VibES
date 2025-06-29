#ifndef PROJECT_IEKF_HELIX_FITTER_HPP
#define PROJECT_IEKF_HELIX_FITTER_HPP

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <metavision/sdk/base/events/event_cd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class IEKFHelixFitter {
public:
    // State vector for helix fitting with dynamic harmonics
    // For n harmonics and 3D helix (x, y, t):
    // State: [A1_x, B1_x, A1_y, B1_y, w1, ..., An_x, Bn_x, An_y, Bn_y, wn, Cx, Cy]
    // Ai_x, Bi_x, wi: amplitudes for sin(wi*t), cos(wi*t) in X dimension
    // Ai_y, Bi_y, wi: amplitudes for sin(wi*t), cos(wi*t) in Y dimension
    // Cx, Cy: offsets for each dimension
    struct HelixPoint {
        double x, y, t;

        HelixPoint(double x = 0, double y = 0, double t = 0) : x(x), y(y), t(t) {}

        // Constructor from Metavision::EventCD
        HelixPoint(const Metavision::EventCD &event, double time_scale = 1e-6)
                : x(static_cast<double>(event.x)),
                  y(static_cast<double>(event.y)),
                  t(static_cast<double>(event.t) * time_scale) {}
    };

    static Eigen::VectorXd
    build_state(const std::vector<double> &Ax,
                const std::vector<double> &Bx,
                const std::vector<double> &Ay,
                const std::vector<double> &By,
                const std::vector<double> &omegas,
                const std::vector<double> &offsets) {
        const int num_harmonics = Ax.size();
        if (Ax.size() != num_harmonics || Bx.size() != num_harmonics ||
            Ay.size() != num_harmonics || By.size() != num_harmonics ||
            omegas.size() != num_harmonics || offsets.size() != 2) {
            throw std::invalid_argument("Size of A, B, omegas, and offsets must match the number of harmonics");
        }

        Eigen::VectorXd state(5 * num_harmonics + 2);
        for (int i = 0; i < num_harmonics; ++i) {
            state(5 * i) = Ax[i];          // A coefficients for X
            state(5 * i + 1) = Bx[i];      // B coefficients for X
            state(5 * i + 2) = Ay[i];      // A coefficients for Y
            state(5 * i + 3) = By[i];      // B coefficients for Y
            state(5 * i + 4) = omegas[i];  // Omega for the harmonic
        }

        // Offsets for X and Y
        state(5 * num_harmonics) = offsets[0];     // Cx
        state(5 * num_harmonics + 1) = offsets[1]; // Cy

        return state;
    }

    IEKFHelixFitter(int num_harmonics,
                    const Eigen::VectorXd &initial_state,
                    const Eigen::MatrixXd &initial_covariance,
                    const Eigen::MatrixXd &process_noise,
                    double measurement_noise_variance,
                    int iterations = 5)
            : num_harmonics_(num_harmonics),
              state_size_(5 * num_harmonics + 2), // 5 coefficients per harmonic + 2 offsets
              state_(initial_state),
              covariance_(initial_covariance),
              process_noise_q_(process_noise),
              measurement_noise_r_(measurement_noise_variance),
              iterations_(iterations),
              is_initialized_(false),
              reference_time_(0.0),
              time_scale_(1e-6) { // Default: microseconds to seconds

        // Validate input dimensions
        if (initial_state.size() != state_size_) {
            throw std::invalid_argument("Initial state size mismatch");
        }
        if (initial_covariance.rows() != state_size_ || initial_covariance.cols() != state_size_) {
            throw std::invalid_argument("Initial covariance size mismatch");
        }
        if (process_noise.rows() != state_size_ || process_noise.cols() != state_size_) {
            throw std::invalid_argument("Process noise size mismatch");
        }
    }

    // Constructor with default initialization
    IEKFHelixFitter(int num_harmonics,
                    std::vector<double> initial_omegas, // Default initial omega
                    std::vector<double> offsets, // Offsets for X and Y
                    double initial_covariance_scale = 1.0,
                    double process_noise_scale = 0.01,
                    double measurement_noise_variance = 0.1,
                    int iterations = 5,
                    double time_scale = 1e-6) // Default: microseconds to seconds
            : num_harmonics_(num_harmonics),
              state_size_(5 * num_harmonics + 2),
              iterations_(iterations),
              is_initialized_(false),
              measurement_noise_r_(measurement_noise_variance),
              reference_time_(0.0),
              time_scale_(time_scale) {

        // Initialize state vector
        state_ = Eigen::VectorXd::Zero(state_size_);

        for (int i = 0; i < num_harmonics_; ++i) {
            state_(5 * i + 4) = initial_omegas[i]; // Set initial omega for each harmonic
        }
        // Set offsets for X and Y
        if (offsets.size() != 2) {
            throw std::invalid_argument("Offsets must contain exactly two elements for X and Y");
        }
        state_(5 * num_harmonics_ + 0) = offsets[0]; // Cx
        state_(5 * num_harmonics_ + 1) = offsets[1]; // Cy
        is_initialized_ = true; // Mark as initialized

        // Initialize covariance matrices
        covariance_ = Eigen::MatrixXd::Identity(state_size_, state_size_) * initial_covariance_scale;
        process_noise_q_ = Eigen::MatrixXd::Identity(state_size_, state_size_) * process_noise_scale;
    }

    // Direct update with Metavision::EventCD
    void update(const Metavision::EventCD &event, double reference_time_input) {
        double relative_time = (static_cast<double>(event.t) * time_scale_) - reference_time_input;
        HelixPoint measurement(event, time_scale_);
        measurement.t = relative_time; // Use relative time for the t dimension
        update(relative_time, measurement);
    }

    // Alternative update method using relative timestamp
    void update(const Metavision::EventCD &event) {
        if (!is_initialized_) {
            reference_time_ = static_cast<double>(event.t) * time_scale_;
        }
        double relative_time = (static_cast<double>(event.t) * time_scale_) - reference_time_;
        HelixPoint measurement(event, time_scale_);
        measurement.t = relative_time; // Use relative time for the t dimension
        update(relative_time, measurement);
    }

    void update(double t, const HelixPoint &measurement) {
        // --- PREDICTION STEP ---
        Eigen::VectorXd predicted_state = state_;
        Eigen::MatrixXd predicted_covariance = covariance_ + process_noise_q_;

        // --- UPDATE STEP (ITERATIVE) ---
        Eigen::VectorXd eta = predicted_state;

        for (int iter = 0; iter < iterations_; ++iter) {
            // Update for each dimension (x, y)
            for (int dim = 0; dim < 2; ++dim) {
                double measurement_value = (dim == 0) ? measurement.x : measurement.y;

                updateSingleDimension(t, measurement_value, dim, predicted_state,
                                      predicted_covariance, eta, iter);
            }
        }

        state_ = eta;

        // Update covariance using the final state estimate
        updateCovariance(t, predicted_covariance);
    }

    HelixPoint predict(double t) const {
        if (!is_initialized_) {
            return HelixPoint(0, 0, 0);
        }

        HelixPoint result;

        // Calculate prediction for each dimension
        double *coords[2] = {&result.x, &result.y};

        for (int dim = 0; dim < 2; ++dim) {
            double value = state_(state_size_ - 2 + dim); // offset

            for (int h = 0; h < num_harmonics_; ++h) {
                int base_idx = h * 5 + dim * 2;
                double A = state_(base_idx);
                double B = state_(base_idx + 1);
                double harmonic_freq = state_(5 * h + 4); // omega

                value += A * std::sin(harmonic_freq * t) + B * std::cos(harmonic_freq * t);
            }

            *coords[dim] = value;
        }

        return result;
    }

    // Predict only the oscillatory part (without offset)
    HelixPoint predict_oscillatory(double t) const {
        if (!is_initialized_) {
            return HelixPoint(0, 0, 0);
        }

        HelixPoint result(0, 0, 0);
        double *coords[3] = {&result.x, &result.y, &result.t};

        for (int dim = 0; dim < 2; ++dim) {
            double value = 0.0;

            for (int h = 0; h < num_harmonics_; ++h) {
                int base_idx = h * 5 + dim * 2;
                double A = state_(base_idx);
                double B = state_(base_idx + 1);
                double harmonic_freq = state_(5 * h + 4);

                value += A * std::sin(harmonic_freq * t) + B * std::cos(harmonic_freq * t);
            }

            *coords[dim] = value;
        }

        return result;
    }

    // Predict EventCD at a given time
    Metavision::EventCD predictEvent(double t, int64_t absolute_timestamp = 0) const {
        HelixPoint prediction = predict(t);
        Metavision::EventCD event;
        event.x = static_cast<int>(std::round(prediction.x));
        event.y = static_cast<int>(std::round(prediction.y));

        if (absolute_timestamp != 0) {
            event.t = absolute_timestamp;
        } else {
            // Convert relative time back to absolute timestamp
            event.t = static_cast<int64_t>((prediction.t + reference_time_) / time_scale_);
        }

        return event;
    }

    const Eigen::VectorXd &get_state() const { return state_; }

    bool is_initialized() const { return is_initialized_; }

    int get_num_harmonics() const { return num_harmonics_; }

    double get_time_scale() const { return time_scale_; }

    double get_reference_time() const { return reference_time_; }

    void print_frequencies() const {
        if (!is_initialized_) {
            std::cout << "  Helix fitter not initialized." << std::endl;
            return;
        }

        for(int h = 0; h < num_harmonics_; ++h) {
            int base_idx = h * 6;
            double A_x = state_(base_idx);
            double B_x = state_(base_idx + 1);
            double A_y = state_(base_idx + 2);
            double B_y = state_(base_idx + 3);
            double omega = state_(base_idx + 4);

            std::cout << "Harmonic " << (h + 1) << ": "
                      << "A_x: " << A_x << ", B_x: " << B_x
                      << ", A_y: " << A_y << ", B_y: " << B_y
                      << ", Omega: " << omega / (2 * M_PI) << " Hz" << std::endl;
        }
    }

    void print_state() const {
        if (!is_initialized_) {
            std::cout << "  Helix fitter not initialized." << std::endl;
            return;
        }

        std::cout << "Helix State Parameters:" << std::endl;
        std::cout << "  Omega: " << state_(state_size_ - 4) << std::endl;
        std::cout << "  Offsets - X: " << state_(state_size_ - 3)
                  << ", Y: " << state_(state_size_ - 2)
                  << ", T: " << state_(state_size_ - 1) << std::endl;

        for (int h = 0; h < num_harmonics_; ++h) {
            std::cout << "  Harmonic " << (h + 1) << ":" << std::endl;
            for (int dim = 0; dim < 2; ++dim) {
                int base_idx = h * 5 + dim * 2;
                char dim_name = (dim == 0) ? 'X' : (dim == 1) ? 'Y' : 'T';
                std::cout << "    " << dim_name << ": A=" << state_(base_idx)
                          << ", B=" << state_(base_idx + 1) << std::endl;
            }
        }
    }

private:
    void updateSingleDimension(double t, double measurement_value, int dim,
                               const Eigen::VectorXd &predicted_state,
                               const Eigen::MatrixXd &predicted_covariance,
                               Eigen::VectorXd &eta, int i) {

        // Calculate measurement prediction h(eta) for this dimension
        double y_pred = eta(state_size_ - 2 + dim); // offset

        // Calculate Jacobian H for this dimension
        Eigen::RowVectorXd H = Eigen::RowVectorXd::Zero(state_size_);
        H(state_size_ - 2 + dim) = 1.0; // derivative w.r.t. offset


        for (int h = 0; h < num_harmonics_; ++h) {
            int base_idx = h * 5 + dim * 2;
            double A = eta(base_idx);
            double B = eta(base_idx + 1);
            double harmonic_freq = eta(5 * h + 4);

            double sin_term = std::sin(harmonic_freq * t);
            double cos_term = std::cos(harmonic_freq * t);

            y_pred += A * sin_term + B * cos_term;

            // Jacobian elements
            H(base_idx) = sin_term;     // derivative w.r.t. A
            H(base_idx + 1) = cos_term; // derivative w.r.t. B

            // Derivative w.r.t. omega
            H(5 * h + 4) = t * (A * cos_term - B * sin_term); // derivative w.r.t. omega
        }

        // Calculate Kalman Gain K
        Eigen::MatrixXd S = H * predicted_covariance * H.transpose();
        S(0, 0) += measurement_noise_r_;

        Eigen::VectorXd K = predicted_covariance * H.transpose() * S.ldlt().solve(Eigen::VectorXd::Ones(1));

        // Update state estimate for this iteration
        eta = predicted_state + K * (measurement_value - y_pred - H * (predicted_state - eta));
    }

    void updateCovariance(double t, const Eigen::MatrixXd &predicted_covariance) {
        // Calculate final Jacobian matrix for all dimensions
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, state_size_);

        for (int dim = 0; dim < 2; ++dim) {
            H(dim, state_size_ - 2 + dim) = 1.0; // offset derivatives

            for (int h = 0; h < num_harmonics_; ++h) {
                int base_idx = h * 5 + dim * 2;
                double A = state_(base_idx);
                double B = state_(base_idx + 1);
                double harmonic_freq = state_(5 * h + 4); // omega for this harmonic

                double sin_term = std::sin(harmonic_freq * t);
                double cos_term = std::cos(harmonic_freq * t);

                H(dim, base_idx) = sin_term;     // derivative w.r.t. A
                H(dim, base_idx + 1) = cos_term; // derivative w.r.t. B

                H(dim, 5 * h + 4) = t * (A * cos_term - B * sin_term); // derivative w.r.t. omega
            }

        }

        Eigen::MatrixXd innovation_covariance = H * predicted_covariance * H.transpose() +
                                                Eigen::MatrixXd::Identity(2, 2) * measurement_noise_r_;
        Eigen::MatrixXd K = predicted_covariance * H.transpose() * innovation_covariance.inverse();

        covariance_ = (Eigen::MatrixXd::Identity(state_size_, state_size_) - K * H) * predicted_covariance;
    }

    int num_harmonics_;
    int state_size_;
    Eigen::VectorXd state_;
    Eigen::MatrixXd covariance_;
    Eigen::MatrixXd process_noise_q_;
    double measurement_noise_r_;
    int iterations_;
    bool is_initialized_;
    double reference_time_;  // Reference timestamp for relative time calculations
    double time_scale_;      // Scale factor for timestamp conversion (default: microseconds to seconds)
};

#endif //PROJECT_IEKF_HELIX_FITTER_HPP