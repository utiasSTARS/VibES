#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <Eigen/Dense>
#include <thread>
#include <chrono>

#include <dv-processing/io/mono_camera_recording.hpp>
#include <open3d/Open3D.h>


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

    // Get and print the camera name that data from recorded from
    std::cout << "Opened an AEDAT4 file which contains data from [" << reader.getCameraName() << "] camera"
              << std::endl;
    cv::Size resolution;
    if (reader.isEventStreamAvailable()) {
        // Check the resolution of event stream. Since the getEventResolution() method returns
        // a std::optional, we use *operator to get the value. The method returns std::nullopt
        // only in case the stream is unavailable, which is already checked.
        resolution = *reader.getEventResolution();

        // Print that the stream is present and its resolution
        std::cout << "  * Event stream with resolution " << resolution << std::endl;
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
            Eigen::Vector3d(x_max, y_max, time_window_micro / 1e6));

    // Set bounding box color for visibility
    bounding_box->color_ = Eigen::Vector3d(0.0, 1.0, 0.0);  // Green color

    // Add the box to the visualizer
    vis.AddGeometry(bounding_box);

    // Point cloud setup and sliding buffer for events
    auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();
    std::deque<Eigen::Vector3d> points_buffer;  // Sliding window buffer for points
    std::deque<Eigen::Vector3d> colors_buffer;  // Buffer for point colors
    bool is_geometry_added = false;

    Eigen::Vector3d color_1(1, 0, 0);
    Eigen::Vector3d color_2(0, 0, 1);
    Eigen::Vector3d color_3(0, 1, 0);
    Eigen::Vector3d color_4(1, 0, 1);

    int i = 0;
    int64_t t0 = 0;

    while (reader.isRunning()) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {

            auto latest_timestamp = events.value().back().timestamp();

            // Remove outdated points from buffer
            while (!points_buffer.empty() &&
                   static_cast<double>(latest_timestamp - points_buffer.front().z() * 1e6) > time_window_micro) {
                points_buffer.pop_front();
                colors_buffer.pop_front();
            }

            // Add new events to the point cloud within the sliding window
            for (const auto &event: events.value()) {
                if (t0 == 0) {
                    t0 = event.timestamp();
                }
                auto relative_time = static_cast<double>(event.timestamp() - t0);
                if (t0 == event.timestamp()) {
                    std::cout << "relative_time: " << relative_time << std::endl;
                }
                Eigen::Vector3d point(event.x(), event.y(), relative_time);  // Scale to seconds

                point_cloud->points_.push_back(point);
                if (i % 2 == 0) {
                    colors_buffer.push_back(event.polarity() ? color_3 : color_4);
                } else {
                    point_cloud->colors_.push_back(event.polarity() ? color_1 : color_2);
                }
            }

            // Update point cloud with current points in the sliding window
            // point_cloud->points_.assign(points_buffer.begin(), points_buffer.end());
            // point_cloud->colors_.assign(colors_buffer.begin(), colors_buffer.end());

            // Add or update point cloud in the visualizer
            if (!is_geometry_added) {
                vis.AddGeometry(point_cloud);
                is_geometry_added = true;

                // Set camera view to focus on bounding box area
                vis.GetViewControl().SetLookat(
                        Eigen::Vector3d((x_max - x_min) / 2, (y_max - y_min) / 2, -time_window_micro / 2 / 1e6));
                vis.GetViewControl().SetZoom(0.8);
                vis.GetRenderOption().point_size_ = 2.0;
            } else {
                vis.UpdateGeometry(point_cloud);
            }

            std::this_thread::sleep_for(std::chrono::microseconds(100000));
            vis.PollEvents();
            vis.UpdateRender();
            i++;
            if (i == 2) {
                while (vis.PollEvents()) {
                    vis.UpdateRender();
                }
            }
        }
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
