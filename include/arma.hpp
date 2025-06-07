#ifndef PROJECT_ARMA_HPP
#define PROJECT_ARMA_HPP

#include <vector>
#include <deque>
#include <numeric>
#include <stdexcept>

/**
 * @brief Implements a simple Autoregressive Moving-Average (ARMA) model for a single time series.
 *
 * The ARMA(p, q) model is defined as:
 * X_t = c + epsilon_t + sum_{i=1 to p} (phi_i * X_{t-i}) + sum_{j=1 to q} (theta_j * epsilon_{t-j})
 */
class ARMA {
public:
    /**
     * @brief Constructs an ARMA(p, q) model.
     * @param ar_coeffs The autoregressive coefficients (phi_1, phi_2, ..., phi_p).
     *                  The size of this vector determines the order p.
     * @param ma_coeffs The moving-average coefficients (theta_1, theta_2, ..., theta_q).
     *                  The size of this vector determines the order q.
     * @param constant The constant term (c) in the model. Defaults to 0.0.
     */
    ARMA(std::vector<double> ar_coeffs, std::vector<double> ma_coeffs, double constant = 0.0)
        : p_(ar_coeffs.size()),
          q_(ma_coeffs.size()),
          ar_coeffs_(std::move(ar_coeffs)),
          ma_coeffs_(std::move(ma_coeffs)),
          constant_(constant) {
    }

    /**
     * @brief Predicts the next value in the time series based on the history of observations and errors.
     * @return The one-step-ahead forecasted value.
     */
    double predict() const {
        double prediction = constant_;

        // AR part: Sum of past observations weighted by AR coefficients
        for (size_t i = 0; i < p_ && i < observations_.size(); ++i) {
            prediction += ar_coeffs_[i] * observations_[observations_.size() - 1 - i];
        }

        // MA part: Sum of past errors weighted by MA coefficients
        for (size_t i = 0; i < q_ && i < errors_.size(); ++i) {
            prediction += ma_coeffs_[i] * errors_[errors_.size() - 1 - i];
        }

        return prediction;
    }

    /**
     * @brief Updates the model with a new observation.
     * This method first calculates the prediction error based on the new observation,
     * then adds the observation and error to the history.
     * @param observation The new observed value (X_t) in the time series.
     * @return The prediction error (epsilon_t = observation - prediction).
     */
    double update(double observation) {
        double one_step_prediction = predict();
        
        double error = observation - one_step_prediction;

        observations_.push_back(observation);
        errors_.push_back(error);

        if (observations_.size() > p_) {
            observations_.pop_front();
        }
        if (errors_.size() > q_) {
            errors_.pop_front();
        }
        
        return error;
    }

private:
    const size_t p_; // Order of AR part
    const size_t q_; // Order of MA part
    const std::vector<double> ar_coeffs_; // phi coefficients
    const std::vector<double> ma_coeffs_; // theta coefficients
    const double constant_;

    // History of observations (X) and errors (epsilon)
    std::deque<double> observations_;
    std::deque<double> errors_;
};

#endif //PROJECT_ARMA_HPP 