#include <iostream>
#include <deque>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/camera_capture.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <thread>
#include <chrono>

#include <boost/math/distributions/chi_squared.hpp>

#include "events_freq_calib_pattern.hpp"

enum Colors {
    RED = 0,
    GREEN = 1,
    BLUE = 2
};

std::map<Colors, cv::Scalar> color_map = {
        {RED,   cv::Scalar(0, 0, 255)},
        {GREEN, cv::Scalar(0, 255, 0)},
        {BLUE,  cv::Scalar(255, 0, 0)}
};


class BinSin {
public:
    BinSin(double process_noise, double measurement_noise, double center) : bin_id(bin_counter++),
                                                                            process_noise(process_noise),
                                                                            measurement_noise(measurement_noise) {
        // initial state guess
        A = 1.0;
        phi = 1.0;
        C = center;
        P = Eigen::MatrixXd::Identity(4, 4) * 1000.0;
    }

    void addData(int64_t time, double y) {
        if (prev_time == 0) {
            prev_time = time;
        }
        y_mean += y;
        time_mean += (time - prev_time);
        counter++;
        if (counter >= 3) {
            y_mean /= counter;
            window_data.emplace_back((static_cast<double>(time_mean) / counter) / 1e6, y_mean);
            if (window_data.size() > 10) {
                windowedEKF();
                window_data.pop_front();
                counter = 0;
                y_mean = 0;
            }
        }
    }

    double getY() {
        return y_mean;
    }


    friend std::ostream &operator<<(std::ostream &os, const BinSin &bin) {
        os << "Bin ID: " << bin.bin_id << ", Amplitude (A): " << bin.A << ", Frequency (omega): "
           << BinSin::omega << ", Phase (phi): " << bin.phi << ", Offset (C): " << bin.C << ", data size: "
           << bin.counter;
        return os;
    }

    cv::Scalar getColor() {
        return color_map[colors];
    }


private:
    static int64_t bin_counter;
    const int64_t bin_id;
    int counter = 0;
    double y_mean = 0;
    int64_t time_mean = 0;
    std::deque<std::pair<double, double>> window_data;
    double A = 0;
    static double omega;
    double phi = 0;
    double C = 0;
    Eigen::MatrixXd P;
    double process_noise;
    double measurement_noise;
    int64_t prev_time = 0;

    Colors colors = BLUE;

    // Windowed Extended Kalman Filter (EKF) for continuous data stream
    void windowedEKF() {
        int N = window_data.size();
        if (N == 0) return;

        Eigen::VectorXd residuals(N);
        Eigen::MatrixXd H(N, 4);

        // Compute the Jacobian matrix for each data point and residuals
        for (int i = 0; i < N; ++i) {
            double t = window_data[i].first;
            double y = window_data[i].second;

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

        // Mahalanobis distance
        double mahalanobis = residuals.transpose() * S.inverse() * residuals;
        // chi-squared test with N degrees of freedom
        const boost::math::chi_squared_distribution<> my_chisqr(N * 4);  // 2*Mj-3 DOFs
        const double chi = quantile(my_chisqr, 0.95);

        if (mahalanobis > chi) {
            colors = RED;
            return;
        }
        colors = GREEN;

        Eigen::Vector4d state = K * residuals;
        A += state(0);
        omega += state(1);
        phi += state(2);
        C += state(3);
        P = (Eigen::MatrixXd::Identity(4, 4) - K * H) * P;
        P = 0.5 * (P + P.transpose());
    }

};

// static member initialization
int64_t BinSin::bin_counter = 0;
double BinSin::omega = 700.0;


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
    double timeWindow = .01; // Time window to display in seconds
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

int main() {
    // Initialize state, covariance, and noise parameters
    Eigen::VectorXd x(4);
    x << 5.0, 700.0, 0.0, 227.0;
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(4, 4) * 1000.0;
    double process_noise = 1., measurement_noise = 1.5;

    std::deque<std::pair<double, double>> window_data;

    // Set up event reader
    dv::io::MonoCameraRecording reader(
            "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/dvSave-2024_11_11_15_36_53.aedat4");
    // dv::io::CameraCapture reader;
    std::cout << "Opened AEDAT4 file from [" << reader.getCameraName() << "] camera\n";

    const cv::Size resolution = *reader.getEventResolution();
    // output resolution
    std::cout << "Resolution: " << resolution << std::endl;

    // colored image with bins colors
    cv::Mat colored_image(resolution, CV_8UC3, cv::Scalar(0, 0, 0));

    dv::EdgeMapAccumulator accumulator(resolution);
    accumulator.setNeutralPotential(0.5f);
    accumulator.setEventContribution(0.25f);
    accumulator.setIgnorePolarity(false);

    std::vector<BinSin> bins;

    int win_h = 80, win_w = 80;

    // make sure we have an integer number of bins in each direction
    if (resolution.height % win_h != 0 || resolution.width % win_w != 0) {
        throw std::runtime_error("Window size must be a multiple of the resolution size.");
    }

    // Calculate the number of bins in each direction
    int num_bins_h = resolution.height / win_h;
    int num_bins_w = resolution.width / win_w;

    // Create bins for the entire camera plane
    for (int i = 0; i < num_bins_h; ++i) {
        for (int j = 0; j < num_bins_w; ++j) {
            bins.emplace_back(process_noise, measurement_noise, i * win_h + win_h / 2);
        }
    }


    /////// test event generation

    EventsFreqCalibPattern pattern(0, resolution, 700.0, 5, 5, 0.0);
    for (int i = 0; i < 10000; i++) {
        auto events = pattern.get_events();
        if (events.has_value()) {
            accumulator.accumulate(events.value());
            auto acc_frame = accumulator.generateFrame();
            //show bins grid on image
            for (int i = 0; i < num_bins_h; ++i) {
                cv::line(acc_frame.image, cv::Point(0, i * win_h), cv::Point(resolution.width, i * win_h),
                         cv::Scalar(0, 255, 255));
            }

            for (int i = 0; i < num_bins_w; ++i) {
                cv::line(acc_frame.image, cv::Point(i * win_w, 0), cv::Point(i * win_w, resolution.height),
                         cv::Scalar(0, 255, 255));
            }
            cv::imshow("Accumulator", acc_frame.image);

            for (const auto &event: *events) {
                // Determine the bin row and column
                int bin_row = static_cast<int>(event.y() / win_h);
                int bin_col = static_cast<int>(event.x() / win_w);

                // Find the corresponding bin index
                int bin_index = bin_row * num_bins_w + bin_col;

                // Add the event to the corresponding bin
                bins[bin_index].addData(event.timestamp(), event.y());

                // color image bins
                cv::rectangle(colored_image, cv::Rect(bin_col * win_w, bin_row * win_h, win_w, win_h),
                              bins[bin_index].getColor(), -1);
                // draw a circle with the mean y
                cv::circle(colored_image, cv::Point(int(bin_col * win_w + win_w / 2), bins[bin_index].getY()), 1,
                           cv::Scalar(255, 255, 255),
                           -1);
                cv::imshow("Colored Image", colored_image);
            }
        }
        cv::waitKey(1);
        if (cv::waitKey(1) == 27) {
            break;
        } else if (cv::waitKey(1) == 32) {
        }
    }


    for (const auto &bin: bins) {
        std::cout << bin << std::endl;
    }
    return 0;


    while (reader.isRunning()) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {
            // Loop through each event and add it to the corresponding bin

            accumulator.accumulate(events.value());
            auto acc_frame = accumulator.generateFrame();
            //show bins grid on image
            for (int i = 0; i < num_bins_h; ++i) {
                cv::line(acc_frame.image, cv::Point(0, i * win_h), cv::Point(resolution.width, i * win_h),
                         cv::Scalar(0, 255, 255));
            }

            for (int i = 0; i < num_bins_w; ++i) {
                cv::line(acc_frame.image, cv::Point(i * win_w, 0), cv::Point(i * win_w, resolution.height),
                         cv::Scalar(0, 255, 255));
            }
            cv::imshow("Accumulator", acc_frame.image);

            for (const auto &event: *events) {
                // Determine the bin row and column
                int bin_row = static_cast<int>(event.y() / win_h);
                int bin_col = static_cast<int>(event.x() / win_w);

                // Find the corresponding bin index
                int bin_index = bin_row * num_bins_w + bin_col;

                // Add the event to the corresponding bin
                bins[bin_index].addData(event.timestamp(), event.y());

                // color image bins
                cv::rectangle(colored_image, cv::Rect(bin_col * win_w, bin_row * win_h, win_w, win_h),
                              bins[bin_index].getColor(), -1);
                // draw a circle with the mean y
                cv::circle(colored_image, cv::Point(int(bin_col * win_w + win_w / 2), bins[bin_index].getY()), 1,
                           cv::Scalar(255, 255, 255),
                           -1);
                cv::imshow("Colored Image", colored_image);
            }
        }
        cv::waitKey(1);
        if (cv::waitKey(1) == 27) {
            break;
        } else if (cv::waitKey(1) == 32) {
        }
    }

    for (const auto &bin: bins) {
        std::cout << bin << std::endl;
    }

    return 0;
}

//Amplitude (A): 0.369753, Frequency (omega): 700.008, RPM: 6684.59, Phase (phi): -0.0568086, Offset (C): 238.06