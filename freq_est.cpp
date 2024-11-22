#include <iostream>
#include <deque>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/camera_capture.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <thread>
#include <chrono>
#include "include/sim/ini_sim.hpp"
#include "include/open3d_visualizer.hpp"
#include <boost/math/distributions/chi_squared.hpp>


void visualizeSinusoid(std::deque<std::pair<double, double>> &data, cv::Mat &image) {
    // Set up variables
    int width = image.cols;
    int height = image.rows;
    double timeWindow = .01; // Time window to display in seconds
    double timeScale = width / timeWindow; // Scale time to fit within the defined window width

    // Clear the image to start fresh for each frame
    image.setTo(cv::Scalar(0, 0, 0));

    // Determine the minimum time value to display (oldest time that still fits within timeWindow)
    double latestTime = data.back().first;
    double minTime = latestTime - timeWindow;

    // Draw sinusoid points within the time window
    for (const auto &[time, y]: data) {
        if (time >= minTime) {
            int x = static_cast<int>((time - minTime) * timeScale); // Position x based on the scaled time
            int yPos = static_cast<int>(y); // Scale y to fit in the image height

            // Ensure (x, yPos) is within bounds and draw the point
            if (yPos >= 0 && yPos < height && x < width) {
                //image.at<cv::Vec3b>(yPos, x) = cv::Vec3b(0, 0, 255); // Red color for the sinusoid point
                cv::circle(image, cv::Point(x, yPos), 1, cv::Scalar(0, 255, 255),
                           -1); // White circle for visualization
            }
        }
        // Display the updated image
        cv::imshow("Sinusoid Visualization", image);
        cv::waitKey(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

}

void visualizeSinusoids(std::deque<std::pair<double, double>> &data1, std::deque<std::pair<double, double>> &data2,
                        cv::Mat &image) {
    // Set up variables
    int width = image.cols;
    int height = image.rows;
    double timeWindow = .0001; // Time window to display in seconds
    double timeScale = width / timeWindow; // Scale time to fit within the defined window width

    // Clear the image to start fresh for each frame
    image.setTo(cv::Scalar(0, 0, 0));

    // Determine the minimum time value to display (oldest time that still fits within timeWindow)
    double latestTime = data1.back().first;
    double minTime = latestTime - timeWindow;

    // Draw first sinusoid points within the time window
    // In `visualizeSinusoids`, ensure both datasets are of equal size:
    for (int i = 0; i < std::min(data1.size(), data2.size()); i++) {
        const auto &[time, y] = data1[i];
        const auto &[time2, y2] = data2[i];
        if (time >= minTime) {
            int x = static_cast<int>((time - minTime) * timeScale);
            int yPos = static_cast<int>(y);
            if (yPos >= 0 && yPos < height && x < width) {
                cv::circle(image, cv::Point(x, yPos), 2, cv::Scalar(0, 255, 0), -1);
            }

            int x2 = static_cast<int>((time2 - minTime) * timeScale);
            int yPos2 = static_cast<int>(y2);
            if (yPos2 >= 0 && yPos2 < height && x2 < width) {
                cv::circle(image, cv::Point(x2, yPos2), 2, cv::Scalar(0, 0, 255), -1);
            }
        }
    }


    // Display the updated image
    cv::imshow("Sinusoid Visualization", image);
    // cv::waitKey(1);
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Windowed Extended Kalman Filter (EKF) for continuous data stream
void windowedEKF(std::deque<std::pair<double, double>> &window_data, Eigen::VectorXd &x, Eigen::MatrixXd &P,
                 double process_noise, double measurement_noise, double mid) {
    int N = window_data.size();
    if (N == 0) return;

    Eigen::VectorXd residuals(N);
    Eigen::MatrixXd H(N, 4);

    // Compute the Jacobian matrix for each data point and residuals
    for (int i = 0; i < N; ++i) {
        double t = window_data[i].first;
        double y = window_data[i].second;
        double A = x(0), omega = x(1), phi = x(2), C = x(3);

        double y_pred = A * std::sin(omega * t + phi) + C;
        residuals(i) = y - y_pred;

        H(i, 0) = std::sin(omega * t + phi);
        H(i, 1) = A * t * std::cos(omega * t + phi);
        H(i, 2) = A * std::cos(omega * t + phi);
        H(i, 3) = 1;
    }

    // Measurement update
    Eigen::MatrixXd H_transpose = H.transpose();
    Eigen::MatrixXd S = H * P * H_transpose + Eigen::MatrixXd::Identity(N, N) * measurement_noise;
    Eigen::MatrixXd K = P * H_transpose * S.inverse();


    x += K * residuals;
    P = (Eigen::MatrixXd::Identity(4, 4) - K * H) * P;
    P = 0.5 * (P + P.transpose());
}

int main() {
    // Initialize state, covariance, and noise parameters
    Eigen::VectorXd x(4);
    x << 5.0, 2.0, 0.0, 227.0;
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(4, 4) * 1000.0;
    P(1, 1) = 1.0;
    double process_noise = 1., measurement_noise = 5.;

    std::deque<std::pair<double, double>> window_data;

    // Set up event reader
    // dv::io::MonoCameraRecording reader(
    //         "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/circle2.aedat4");
    EventsFreqCalibPattern reader(0, cv::Size(640, 480), 500, 10, 10, 0.01, false);
    // dv::io::MonoCameraRecording reader(
    //         "/home/viciopoli/Downloads/synth_data_slow.aedat4");

    // dv::io::CameraCapture reader;
    std::cout << "Opened AEDAT4 file from [" << reader.getCameraName() << "] camera\n";

    const cv::Size resolution = *reader.getEventResolution();
    const cv::Size half_resolution = cv::Size(resolution.width / 2, resolution.height / 2);

    // visualizer
    Open3DVisualizer vis;

    // Set up accumulators and display windows
    dv::EdgeMapAccumulator accumulator(resolution);
    accumulator.setNeutralPotential(0.5f);
    accumulator.setEventContribution(0.25f);
    accumulator.setIgnorePolarity(false);

    cv::namedWindow("Center of Mass", cv::WINDOW_NORMAL);
    cv::Mat center_of_mass = cv::Mat::zeros(resolution.height, resolution.width, CV_8UC3);
    cv::Mat plot = cv::Mat::zeros(resolution.height, resolution.width, CV_8UC3);
    const int half_width = plot.rows / 2;

    int64_t total_accumulation = 0, freq_pixel_accumulator_prev = 0, time_0 = 0;
    int estimates = 0, counter = 0;
    int64_t x_mass = 0, y_mass = 0;
    bool continue_running = true;

    int64_t initial_time = 0;
    int max = 0;
    int min = 1000;

    int square = 150;
    // EventsFreqCalibPattern pattern(0, resolution, 400, 10, 10, 0.01);

    while (reader.isRunning() && continue_running) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {
            accumulator.accumulate(events.value());
            auto acc_frame = accumulator.generateFrame();
            cv::imshow("Accumulator", acc_frame.image);

            for (const auto &e: *events) {
                // consider the events that are at the center only according to square
                // if (e.x() < half_resolution.width - square || e.x() > half_resolution.width + square ||
                //     e.y() < half_resolution.height - square || e.y() > half_resolution.height + square) {
                //     continue;
                // }

                x_mass += e.x();
                y_mass += e.y();
                if (initial_time == 0) {
                    initial_time = e.timestamp();
                }

                if (++counter == 10'000) {
                    // cv::circle(center_of_mass,
                    //            cv::Point(static_cast<int>(x_mass / counter), static_cast<int>(y_mass / counter)),
                    //            1, cv::Scalar(std::max(255, static_cast<int>(estimates / 6.2)), 255,
                    //                          std::max(255, static_cast<int>(estimates / 3.2))), -1);
                    cv::circle(center_of_mass,
                               cv::Point(static_cast<int>(x_mass / counter), static_cast<int>(y_mass / counter)),
                               1, cv::Scalar(0, 255,
                                             0), -1);
                    cv::imshow("Center of Mass", center_of_mass);

                    window_data.emplace_back(static_cast<double>(e.timestamp() - initial_time) / 1e6,
                                             static_cast<double>(y_mass) / counter);

                    vis.addPoint(static_cast<double>(x_mass) / counter, static_cast<double>(y_mass) / counter,
                                 static_cast<double>(e.timestamp() - initial_time) / counter, e.polarity());

                    if (max < y_mass / counter) {
                        max = y_mass / counter;
                    }
                    if (min > y_mass / counter) {
                        min = y_mass / counter;
                    }

                    if (window_data.size() > 100) {

                        // draw window_data on a plot
                        // for (int i = 0; i < window_data.size(); i++) {
                        //     cv::circle(plot, cv::Point(static_cast<int>(window_data[i].first * 5000),
                        //                                static_cast<int>(window_data[i].second)),
                        //                1, cv::Scalar(255, 255, 255), -1);
                        // }
                        // cv::imshow("Plot", plot);
                        // cv::waitKey(0);

                        windowedEKF(window_data, x, P, process_noise, measurement_noise, min + ((max - min) / 2.));

                        std::cout << "\rAmplitude (A): " << x(0) << ", Frequency (omega): " << x(1) << ", RPM: "
                                  << (x(1) * 9.5493) << ", Phase (phi): "
                                  << x(2) << ", Offset (C): " << x(3) << std::endl;
                        std::cout.flush();


                        // for (int i = 0; i < window_data.size(); i++) {
                        //     auto y = 10 * std::sin(x(1) * window_data[i].first + x(2)) + x(3);
                        //     cv::circle(plot,
                        //                cv::Point(static_cast<int>(window_data[i].first * 5000), static_cast<int>(y)),
                        //                1, cv::Scalar(255, 0, 0), -1);
                        // }

                        std::deque<std::pair<double, double>> window_data_syn;
                        for (int i = 0; i < window_data.size(); i++) {
                            auto y = x(0) * std::sin(x(1) * window_data[i].first + x(2)) + x(3);
                            window_data_syn.emplace_back(window_data[i].first, y);
                        }
                        visualizeSinusoids(window_data, window_data_syn, plot);

                        window_data.pop_front();
                    }
                    counter = 0;
                    x_mass = y_mass = 0;
                    estimates++;
                }
            }
            vis.update();
        }
        cv::waitKey(1);
        if (cv::waitKey(1) == 27) {
            break;
        } else if (cv::waitKey(1) == 32) {
            center_of_mass = cv::Mat::zeros(resolution.height, resolution.width, CV_8UC3);
        }
    }

    return 0;
}

//Amplitude (A): 0.369753, Frequency (omega): 700.008, RPM: 6684.59, Phase (phi): -0.0568086, Offset (C): 238.06