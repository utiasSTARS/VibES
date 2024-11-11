#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <Eigen/Dense>
#include <thread>
#include <chrono>

// Function to generate synthetic sinusoidal data
std::pair<double, double> genSinusoidal(double time) {
    double x = 2 * std::sin(500.0 * time + 0.2);
    return {time, x};
}

// Windowed Extended Kalman Filter (EKF) for continuous data stream
void windowedEKF(std::deque<std::pair<double, double>> &window_data, Eigen::VectorXd &x, Eigen::MatrixXd &P,
                 double process_noise, double measurement_noise) {
    int N = window_data.size(); // Number of data points in the window

    if (N == 0) return;

    // Initialize measurement residual vector and Jacobian matrix
    Eigen::VectorXd residuals(N);
    Eigen::MatrixXd H(N, 3);  // Jacobian matrix for the sinusoidal model

    // Compute the Jacobian matrix for each data point and the residuals
    for (int i = 0; i < N; ++i) {
        double t = window_data[i].first;
        double y = window_data[i].second;

        double A = x(0);
        double omega = x(1);
        double phi = x(2);

        // Prediction: y(t) = A * sin(omega * t + phi)
        double y_pred = A * std::sin(omega * t + phi);
        residuals(i) = y - y_pred;

        // Jacobian: H
        H(i, 0) = std::sin(omega * t + phi);         // ∂y/∂A
        H(i, 1) = A * t * std::cos(omega * t + phi); // ∂y/∂omega
        H(i, 2) = A * std::cos(omega * t + phi);     // ∂y/∂phi
    }

    // Measurement update
    Eigen::MatrixXd H_transpose = H.transpose();
    Eigen::MatrixXd S =
            H * P * H_transpose + Eigen::MatrixXd::Identity(N, N) * measurement_noise; // Innovation covariance
    Eigen::MatrixXd K = P * H_transpose * S.inverse(); // Kalman gain

    // State update
    Eigen::VectorXd delta_x = K * residuals; // Update in state
    x += delta_x;

    // Covariance update
    P = (Eigen::MatrixXd::Identity(3, 3) - K * H) * P;
    P = 0.5 * (P + P.transpose()); // Ensure symmetry
}

int main() {
    // Initial state: [A, omega, phi, C]
    Eigen::VectorXd x(3);
    x << 1.0, 1.0, 0.0; // Initial guess for amplitude, frequency, phase, offset

    // Initial covariance matrix P (uncertainty in the initial estimate)
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(3, 3) * 1000.0; // High uncertainty initially

    // Process noise (Q) and measurement noise (R)
    double process_noise = 0.01;
    double measurement_noise = 0.1;

    // Sliding window size
    const int window_size = 10000000;
    std::deque<std::pair<double, double>> window_data;

    // Simulate continuous data stream
    double current_time = 0.0;
    double delta_t = 0.00001;  // Time step for simulation

    while (true) {
        // Generate synthetic data point
        auto new_data = genSinusoidal(current_time);

        // Add new data to the sliding window
        window_data.push_back(new_data);

        // Maintain a sliding window of the last `window_size` data points
        if (window_data.size() > window_size) {
            window_data.pop_front();
        }

        // Apply the windowed EKF to update the model parameters
        windowedEKF(window_data, x, P, process_noise, measurement_noise);

        // Output current parameter estimates
        std::cout << "\rAmplitude (A): " << x(0) << ", ";
        std::cout << "Frequency (omega): " << x(1) << ", ";
        std::cout << "Phase shift (phi): " << x(2);
        std::cout.flush();

        // Increment time
        current_time += delta_t;

        // Simulate delay to mimic real-time data streaming
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}
