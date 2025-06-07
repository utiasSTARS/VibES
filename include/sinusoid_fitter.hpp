#ifndef PROJECT_SINUSOID_FITTER_HPP
#define PROJECT_SINUSOID_FITTER_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <Eigen/Dense>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SinusoidFitter {
public:
    SinusoidFitter() = default;

    void add_point(double t, double val) {
        times_.push_back(t);
        values_.push_back(val);
    }

    bool fit() {
        size_t n_points = times_.size();
        if (n_points < 4) {
            std::cerr << "Not enough points to fit a sinusoid." << std::endl;
            return false;
        }

        // To handle noise, we first find the frequency using a smoothed version of the signal.
        // This makes the frequency search more robust.
        const int smoothing_window = 5; // A small, odd window size is best.
        std::vector<double> smoothed_values(n_points);
        for(size_t i = 0; i < n_points; ++i) {
            double sum = 0;
            int count = 0;
            for (int j = -(smoothing_window / 2); j <= smoothing_window / 2; ++j) {
                if (static_cast<int>(i) + j >= 0 && i + j < n_points) {
                    sum += values_[i + j];
                    count++;
                }
            }
            smoothed_values[i] = (count > 0) ? sum / count : values_[i];
        }
        
        double smoothed_offset = std::accumulate(smoothed_values.begin(), smoothed_values.end(), 0.0) / n_points;
        Eigen::VectorXd demeaned_smoothed_values(n_points);
        for (size_t i = 0; i < n_points; ++i) {
            demeaned_smoothed_values(i) = smoothed_values[i] - smoothed_offset;
        }

        double t_span = times_.back() - times_.front();
        if (t_span <= 0) return false;

        double min_omega = 2.0 * M_PI / t_span;
        double max_omega = 2.0 * M_PI * (n_points / 2.0) / t_span;
        const int n_freq_steps = 200;

        double best_omega = 0;
        double min_residual = std::numeric_limits<double>::max();

        Eigen::MatrixXd A(n_points, 2);

        for (int i = 0; i <= n_freq_steps; ++i) {
            double omega = min_omega + (max_omega - min_omega) * i / n_freq_steps;
            if (omega <= 0) continue;

            for (size_t j = 0; j < n_points; ++j) {
                A(j, 0) = std::sin(omega * times_[j]);
                A(j, 1) = std::cos(omega * times_[j]);
            }
            
            // Find residual against the SMOOTHED data to get a stable frequency estimate
            Eigen::Vector2d x = A.colPivHouseholderQr().solve(demeaned_smoothed_values);
            Eigen::VectorXd residuals = demeaned_smoothed_values - A * x;
            double current_residual = residuals.squaredNorm();
            
            if (current_residual < min_residual) {
                min_residual = current_residual;
                best_omega = omega;
            }
        }
        
        if (best_omega <= 0) return false;
        
        // Now that we have a robust frequency, fit the other parameters to the ORIGINAL, unfiltered data.
        frequency_ = best_omega;

        offset_ = std::accumulate(values_.begin(), values_.end(), 0.0) / n_points;
        Eigen::VectorXd demeaned_values(n_points);
        for (size_t i = 0; i < n_points; ++i) {
            demeaned_values(i) = values_[i] - offset_;
        }

        for (size_t j = 0; j < n_points; ++j) {
            A(j, 0) = std::sin(frequency_ * times_[j]);
            A(j, 1) = std::cos(frequency_ * times_[j]);
        }
        Eigen::Vector2d final_x = A.colPivHouseholderQr().solve(demeaned_values);
        double a = final_x(0);
        double b = final_x(1);

        amplitude_ = std::sqrt(a * a + b * b);
        phase_ = std::atan2(b, a);

        is_fitted_ = true;
        std::cout << "Sinusoid fit successful. Freq (Hz): " << frequency_ / (2*M_PI) << ", Amp: " << amplitude_ << std::endl;
        return true;
    }

    double predict(double t) const {
        if (!is_fitted_) return 0.0;
        return amplitude_ * std::sin(frequency_ * t + phase_) + offset_;
    }
    
    bool is_fitted() const { return is_fitted_; }
    double get_amplitude() const { return amplitude_; }
    double get_frequency() const { return frequency_; }
    double get_phase() const { return phase_; }
    double get_offset() const { return offset_; }

private:
    std::vector<double> times_;
    std::vector<double> values_;

    bool is_fitted_ = false;
    double amplitude_ = 0.0;
    double frequency_ = 0.0;
    double phase_ = 0.0;
    double offset_ = 0.0;
};

#endif //PROJECT_SINUSOID_FITTER_HPP 