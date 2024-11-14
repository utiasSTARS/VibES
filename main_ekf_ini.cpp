#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <Eigen/Dense>
#include <thread>
#include <chrono>

#include <dv-processing/io/mono_camera_recording.hpp>
#include <open3d/Open3D.h>
#include <dv-processing/io/camera_capture.hpp>

struct Point {
    int x;
    int y;
    int64_t timestamp;
};

class RingBuffer : public std::deque<Point> {
public:
    RingBuffer(size_t capacity) : capacity(capacity) {}

    void push_back(const Point &value) {
        if (this->size() == capacity) {
            this->pop_front();
        }
        std::deque<Point>::push_back(value);
    }

    size_t getCapacity() const {
        return capacity;
    }

    int64_t getHighestTime() const {
        return this->back().timestamp;
    }

    void writeOut(std::vector<Eigen::Vector3d> &data) {
        data.clear();
        auto last_time = this->back().timestamp;
        for (const auto &d: *this) {
            Eigen::Vector3d point(d.x, d.y, static_cast<double>(last_time - d.timestamp));
            data.push_back(point);
        }
    }

private:
    size_t capacity;
};


// Function to generate synthetic sinusoidal data
std::pair<double, double> genSinusoidal(double time) {
    double x = 2 * std::sin(500.0 * time + 0.2);
    return {time, x};
}

// Windowed Extended Kalman Filter (EKF) for continuous data stream
void windowedEKF(const dv::EventStore &events, Eigen::VectorXd &x, Eigen::MatrixXd &P, double measurement_noise) {
    int N = events.size(); // Number of data points in the window

    if (N == 0) return;

    // Initialize measurement residual vector and Jacobian matrix
    Eigen::VectorXd residuals(N);
    Eigen::MatrixXd H(N, 3);  // Jacobian matrix for the sinusoidal model

    // Compute the Jacobian matrix for each data point and the residuals
    for (int i = 0; i < N; ++i) {
        double t = events[i].timestamp();
        double y = events[i].y();

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
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N, N);
    Eigen::MatrixXd H_transpose = H.transpose();
    Eigen::MatrixXd S =
            H * P * H_transpose + I * measurement_noise; // Innovation covariance
    Eigen::MatrixXd K = P * H_transpose * S.inverse(); // Kalman gain

    // State update
    Eigen::VectorXd delta_x = K * residuals; // Update in state
    x += delta_x;

    // Covariance update
    P = (Eigen::MatrixXd::Identity(3, 3) - K * H) * P;
    P = 0.5 * (P + P.transpose()); // Ensure symmetry
}

int main() {
    dv::io::MonoCameraRecording reader(
            "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/file.aedat4");
    // dv::io::CameraCapture reader;
    // Get and print the camera name that data from recorded from
    std::cout << "Opened an AEDAT4 file which contains data from [" << reader.getCameraName() << "] camera"
              << std::endl;
    cv::Size resolution = *reader.getEventResolution();
    cv::Size resolution_half(resolution.width / 2, resolution.height / 2);

    // hashmap to track time over pixels
    std::unordered_map<int64_t, int64_t> time_over_pixels;
    time_over_pixels.reserve(static_cast<int>(resolution_half.width * resolution_half.height));
    // init to 0
    for (int i = 0; i < resolution_half.width * resolution_half.height; i++) {
        time_over_pixels[i] = 0;
    }

    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Event Camera Visualization", 800, 600);

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

    // Define resolution and bounding box parameters
    int x_min = 0, y_min = 0;
    int x_max = resolution.width, y_max = resolution.height;
    int64_t time_window_micro = 10 * 1e6;  // 10 seconds in microseconds

    // Define 3D bounding box for last 10 seconds with visible z-axis size
    auto bounding_box = std::make_shared<open3d::geometry::AxisAlignedBoundingBox>(
            Eigen::Vector3d(x_min, y_min, 0.0),
            Eigen::Vector3d(x_max, y_max, 10));

    // Set bounding box color for visibility
    bounding_box->color_ = Eigen::Vector3d(0.0, 1.0, 0.0);  // Green color

    // Add the box to the visualizer
    vis.AddGeometry(bounding_box);

    // Point cloud setup and sliding buffer for events
    auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();
    RingBuffer points_buffer(100'000);

    bool is_geometry_added = false;

    Eigen::Vector3d color_1(1, 0, 0);
    Eigen::Vector3d color_2(0, 0, 1);
    Eigen::Vector3d color_3(0, 1, 0);
    Eigen::Vector3d color_4(1, 0, 1);

    int i = 0;
    int64_t t0 = 0;

    while (reader.isRunning()) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {
            std::cout << "Received " << *events << " events" << std::endl;
            // Add or update point cloud in the visualizer
            if (!is_geometry_added) {
                vis.AddGeometry(point_cloud);
                is_geometry_added = true;

                // Set camera view to focus on bounding box area
                vis.GetViewControl().SetLookat(
                        Eigen::Vector3d((x_max - x_min) / 2, (y_max - y_min) / 2, time_window_micro / 2 / 1e6));
                vis.GetViewControl().SetZoom(0.8);
                vis.GetRenderOption().point_size_ = 2.0;
            }

            for (const auto &event: events.value()) {
                if (event.x() < 0 || event.y() < 0 || event.x() >= resolution_half.width ||
                    event.y() >= resolution_half.height) {
                    auto value = time_over_pixels[event.y() * resolution_half.width + event.x()];
                    time_over_pixels[event.y() * resolution_half.width + event.x()] =
                            value == 0 ? event.timestamp() : value-event.timestamp();
                    if (value != 0) {
                        std::cout << "time value: " << -time_over_pixels[event.y() * resolution_half.width + event.x()]
                                  << std::endl;
                        time_over_pixels[event.y() * resolution_half.width + event.x()] = 0;
                    }
                }

                points_buffer.push_back({event.x(), event.y(), event.timestamp()});
                points_buffer.writeOut(point_cloud->points_);

                vis.UpdateGeometry(point_cloud);
                vis.PollEvents();
                vis.UpdateRender();


                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }

            // std::this_thread::sleep_for(std::chrono::microseconds(100000));
            // vis.PollEvents();
            // vis.UpdateRender();
        }
        // if (i == 2) {
        //     break;
        // }
    }


    while (vis.PollEvents()) {
        vis.UpdateRender();
    }

    vis.DestroyVisualizerWindow();

    while (reader.isRunning()) {
        // Read batch of events, check whether received data is correct.
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {
            // Print received event packet information
            // events->front().x()
            // Apply the windowed EKF to update the model parameters
            windowedEKF(*events, x, P, measurement_noise);

            // Output current parameter estimates
            std::cout << "\rAmplitude (A): " << x(0) << ", ";
            std::cout << "Frequency (omega): " << x(1) << ", ";
            std::cout << "Phase shift (phi): " << x(2);
            std::cout.flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        }
    }


    return 0;
}
