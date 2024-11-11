#include <iostream>
#include <deque>

#include <dv-processing/io/mono_camera_recording.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
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
    std::deque<std::pair<double, double>> window_data;

    //////////////////////////////////////////////////

    dv::io::MonoCameraRecording reader(
            "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/file.aedat4");

    // Get and print the camera name that data from recorded from
    std::cout << "Opened an AEDAT4 file which contains data from [" << reader.getCameraName() << "] camera"
              << std::endl;
    const cv::Size resolution = *reader.getEventResolution();
    const cv::Size half_resolution = cv::Size(resolution.width / 2, resolution.height / 2);
    // dv::Accumulator accumulator(resolution);
    dv::EdgeMapAccumulator accumulator(resolution);

    // Apply configuration, these values can be modified to taste
    accumulator.setNeutralPotential(0.5f);
    accumulator.setEventContribution(0.25f);
    // accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
    // accumulator.setDecayParam(1e+6);
    accumulator.setIgnorePolarity(false);

    // Initialize a preview window
    cv::namedWindow("center_of_mass", cv::WINDOW_NORMAL);

    cv::Mat center_of_mass = cv::Mat::zeros(resolution.height, resolution.width, CV_8UC3);

    cv::Mat plot = cv::Mat::zeros(resolution.height * resolution.width / 1000, 1000, CV_8UC3);
    const int half_width = plot.rows / 2;

    int64_t freq_pixel_accumulator_prev = 0;
    int prev_time = 0;

    int64_t time_0 = 0;
    int64_t total_accumulation = 0;
    int64_t estimates = 0;

    const int square = 100;


    int64_t x_mass = 0, y_mass = 0;
    int counter = 0;
    bool continue_ = true;
    while (reader.isRunning() && continue_) {
        // Read batch of events, check whether received data is correct.
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {

            for (const auto e: *events) {
                // find the mass of the x and y
                x_mass += e.x();
                y_mass += e.y();

                if (++counter == 20'000) {
                    // std::cout << "x: " << e.x() << ", y: " << e.y() << std::endl;
                    // draw the center of the image
                    cv::circle(center_of_mass,
                               cv::Point(static_cast<int>(x_mass / counter), static_cast<int>(y_mass / counter)),
                               1, cv::Scalar(0, 255, static_cast<int>(estimates/3.2)), -1);
                    cv::imshow("center_of_mass", center_of_mass);
                    counter = 0;
                    x_mass = 0;
                    y_mass = 0;
                    estimates++;
                    if (estimates > 800) {
                        std::cout << "STOP" << std::endl;
                        cv::waitKey(15);
                        continue_ = false;
                        break;
                    }
                    cv::waitKey(1);
                }
            }


            continue;
            int64_t freq_pixel_accumulator = 0;
            dv::EventStore event_store;
            int64_t prev_time_ = 0;
            for (const auto &e: *events) {
                // process only the events at the center of the image
                if (e.x() < half_resolution.width - square || e.x() > half_resolution.width + square ||
                    e.y() < half_resolution.height - square || e.y() > half_resolution.height + square) {
                    continue;
                }

                if (total_accumulation == 0) {
                    if (time_0 == 0) {
                        time_0 = e.timestamp();
                    } else {
                        int64_t delta_t = e.timestamp() - time_0;
                        if (delta_t == 0) continue;

                        double freq = 1e6 / static_cast<double>(delta_t); // time is in microseconds
                        std::cout << "\restimates:" << estimates << ", delta_t: " << delta_t << ", freq: " << freq;
                        std::cout.flush();

                        time_0 = e.timestamp();
                        estimates++;
                        accumulator.accept(event_store);
                        dv::Frame frame = accumulator.generateFrame();
                        event_store = dv::EventStore();

                        // draw the center of the image
                        cv::rectangle(frame.image,
                                      cv::Point(half_resolution.width - square, half_resolution.height - square),
                                      cv::Point(half_resolution.width + square, half_resolution.height + square),
                                      cv::Scalar(0, 255, 0), 1);

                        // Show the accumulated image
                        cv::imshow("Preview", frame.image);
                        cv::waitKey(1);
                    }
                }
                total_accumulation += e.polarity() ? 1 : -1;
                freq_pixel_accumulator += e.polarity() ? 1 : -1;
                event_store.push_back(e);

                if (prev_time_ > e.timestamp()) {
                    throw std::runtime_error("Events are not sorted by time");
                }
                prev_time_ = e.timestamp();

                auto pair = std::make_pair<double, double>(static_cast<double>(e.timestamp()) / 1e6,
                                                           static_cast<double>(total_accumulation) / 1000);
                window_data.push_back(pair);
                if (window_data.size() > 2000) {

                    windowedEKF(window_data, x, P, process_noise, measurement_noise);
                    std::cout << "Amplitude (A): " << x(0) << ", ";
                    std::cout << "Frequency (omega): " << x(1) << ", ";
                    std::cout << "Phase shift (phi): " << x(2) << std::endl;
                    std::cout.flush();
                    window_data.pop_front();
                }

                cv::line(plot,
                         cv::Point(prev_time, half_width + static_cast<int>(freq_pixel_accumulator_prev / 1000)),
                         cv::Point(prev_time + static_cast<int>(e.timestamp() - prev_time_ / 1000),
                                   half_width + static_cast<int>(freq_pixel_accumulator / 1000)),
                         cv::Scalar(0, 255, 255), 1);
                freq_pixel_accumulator_prev += freq_pixel_accumulator;
                prev_time += static_cast<int>((e.timestamp() - prev_time_) / 1000);

                cv::imshow("plot", plot);
                cv::waitKey(1);
            }



            // draw plot on the image
            // std::cout << "\r" << freq_pixel_accumulator;





        }
    }


    return 0;
}
