#include <iostream>
#include <vector>
#include <cmath>
#include <map>

// Define the RLS parameters
struct RLSModel {
    double A = 1.0;    // Amplitude
    double omega = 1.0; // Frequency
    double phi = 0.0;   // Phase
    double C = 0.0;     // Offset

    double forgetting_factor = 0.99;  // Lambda: controls memory of past data
    double estimation_error = 1.0;    // Initial error estimate
};

// RLS update function for sinusoidal parameters
void updateRLS(RLSModel &model, double x, double y_observed) {
    double A = model.A;
    double omega = model.omega;
    double phi = model.phi;
    double C = model.C;
    double lambda = model.forgetting_factor;

    // Predicted y value based on current parameters
    double y_pred = A * std::sin(omega * x + phi) + C;

    // Compute the residual error
    double error = y_observed - y_pred;

    // Gain vector calculation (each parameter update depends on the partial derivative)
    double gain_A = error * std::sin(omega * x + phi);
    double gain_omega = error * A * x * std::cos(omega * x + phi);
    double gain_phi = error * A * std::cos(omega * x + phi);
    double gain_C = error;

    // Update parameters using the gain values and forgetting factor
    model.A += lambda * gain_A / model.estimation_error;
    model.omega += lambda * gain_omega / model.estimation_error;
    model.phi += lambda * gain_phi / model.estimation_error;
    model.C += lambda * gain_C / model.estimation_error;

    // Update the estimation error
    model.estimation_error = (1.0 - lambda) * model.estimation_error + std::abs(error);
}

std::pair<double, double> genSinusolidal(double time) {
    double x = 2 * std::sin(0.5 * time + 0.2);
    return {time, x};
}

// Main example function
int main() {
    RLSModel model;

    double current_time = 0.0;
    double delta_t = 0.00001;  // Time step (example)

    // Simulate a continuous data stream
    while (true) {
        // Simulate incoming data point
        std::pair<double, double> data = genSinusolidal(current_time);
        double new_value = data.second/* insert function to retrieve or simulate stream data */;

        // Update the model with the new data point
        updateRLS(model, current_time, new_value);

        // Output current parameter estimates
        std::cout << "Amplitude (A): " << model.A << "\n";
        std::cout << "Frequency (omega): " << model.omega << "\n";
        std::cout << "Phase shift (phi): " << model.phi << "\n";
        std::cout << "Offset (C): " << model.C << "\n";

        // Increment time
        current_time += delta_t;

        // Sleep or wait until the next data point
    }

    return 0;
}
