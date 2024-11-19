#include <iostream>
#include <deque>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>


#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <thread>
#include <chrono>

#include <boost/math/distributions/chi_squared.hpp>

#include "include/events_freq_calib_pattern.hpp"

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
        // init window data with all 0
        for (int i = 0; i < N_samples; i++) {
            window_data.emplace_back(0, 0, 0);
        }

        // initial state guess
        A_y = 1.0;
        phi_y = 1.0;
        C_y = center;
        P = Eigen::MatrixXd::Identity(4, 4) * 1000.0;
        P(0, 0) = 1.0;
    }

    void addData(int64_t time, double y, double x) {
        if (prev_time == 0) {
            prev_time = time;
        }
        y_mean += y;
        x_mean += x;
        time_mean += (time - prev_time);
        counter++;
        if (counter >= 3) {
            events_buffer.emplace_back(x_mean / counter, y_mean / counter, time_mean / 1e6);
            x_mean = 0;
            y_mean = 0;
            time_mean = 0;
            counter = 0;
        }

        while (events_buffer.size() > 1 && index < N_samples) {
            auto [x1, y1, t1] = events_buffer.front();
            auto [x2, y2, t2] = *(++events_buffer.begin());

            if (lastTimestamp == 0.0) lastTimestamp = t1;

            // Interpolate to fill data at regular intervals
            while (lastTimestamp + deltaTime <= t2 && index < N_samples) {
                double ratio = (lastTimestamp + deltaTime - t1) / (t2 - t1);

                window_data[index] = {x1 + ratio * (x2 - x1), y1 + ratio * (y2 - y1), lastTimestamp + deltaTime};

                lastTimestamp += deltaTime;
                index++;
            }

            // Remove processed events
            events_buffer.pop_front();
        }


        if (index >= N_samples) {
            index = 0;
            windowedEKF();
        }
    }

    double getY() {
        return y_mean;
    }


    friend std::ostream &operator<<(std::ostream &os, const BinSin &bin) {
        os << "Bin ID: " << bin.bin_id << ", Amplitude (A): " << bin.A_y << ", Frequency (rad/sec): "
           << BinSin::omega << ", Phase (phi): " << bin.phi_y << ", Offset (C): " << bin.C_y << ", data size: "
           << bin.counter;
        return os;
    }

    cv::Scalar getColor() {
        return color_map[colors];
    }


private:
    const int N_samples = 1024;
    static int64_t bin_counter;
    const int64_t bin_id;
    int counter = 0, index = 0;
    double y_mean = 0, x_mean = 0;
    int64_t time_mean = 0;
    std::deque<std::tuple<double, double, double>> window_data;
    std::deque<std::tuple<double, double, double>> events_buffer;
    double A_y = 0, A_x = 0;
    static double omega;
    double phi_y = 0, phi_x = 0;
    double C_y = 0, C_x = 0;
    Eigen::MatrixXd P;
    double process_noise;
    double measurement_noise;
    int64_t prev_time = 0;
    double lastTimestamp = 0.0;
    double deltaTime = 1.0 / 100'000;

    Colors colors = BLUE;

    // Windowed Extended Kalman Filter (EKF) for continuous data stream
    void windowedEKF() {
        Eigen::VectorXd residuals(N_samples);
        Eigen::MatrixXd H(N_samples, 4);

        // Compute the Jacobian matrix for each data point and residuals
        for (int i = 0; i < N_samples; ++i) {
            auto [x, y, t] = window_data[i];
            // double y_pred = A_y * std::sin(omega * t + phi_y) + C_y; // NOTE: better a sin(w t) + b cos(w t) + c
            double y_pred = A_y * std::sin(omega * t) + phi_y * std::cos(omega * t) + C_y;
            residuals(i) = y - y_pred;

            H(i, 0) = std::sin(omega * t);
            H(i, 1) = A_y * t * std::cos(omega * t) - phi_y * t * std::sin(omega * t);
            H(i, 2) = std::cos(omega * t);
            H(i, 3) = 1;

            // H(i, 0) = std::sin(omega * t + phi_y);
            // H(i, 1) = A_y * t * std::cos(omega * t + phi_y);
            // H(i, 2) = A_y * std::cos(omega * t + phi_y);
            // H(i, 3) = 1;
        }

        // Measurement update
        Eigen::MatrixXd H_transpose = H.transpose();
        Eigen::MatrixXd S = H * P * H_transpose + Eigen::MatrixXd::Identity(N_samples, N_samples) * measurement_noise;
        Eigen::MatrixXd K = P * H_transpose * S.inverse();

        // Mahalanobis distance
        double mahalanobis = residuals.transpose() * S.inverse() * residuals;
        // chi-squared test with N degrees of freedom
        const boost::math::chi_squared_distribution<> my_chisqr(N_samples *
        4);  // 2*Mj-3 DOFs
        const double chi = quantile(my_chisqr, 0.95);

        if (mahalanobis > chi) {
            colors = RED;
            return;
        }
        colors = GREEN;

        Eigen::Vector4d state = K * residuals;
        A_y += state(0);
        omega += state(1);
        phi_y += state(2);
        C_y += state(3);
        P = (Eigen::MatrixXd::Identity(4, 4) - K * H) * P;
        P = 0.5 * (P + P.transpose());
    }

};

// static member initialization
int64_t BinSin::bin_counter = 0;
double BinSin::omega = 700.0;


int main(int argc, char *argv[]) {
    // Initialize state, covariance, and noise parameters
    Eigen::VectorXd x(4);
    x << 5.0, 700.0, 0.0, 227.0;
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(4, 4) * 1000.0;
    double process_noise = 1., measurement_noise = 1.5;

    std::deque<std::pair<double, double>> window_data;

    // Set up event reader
    Metavision::Camera cam; // create the camera

    if (argc >= 2) {
        // if we passed a file path, open it
        cam = Metavision::Camera::from_file(argv[1]);
    } else {
        // open the first available camera
        cam = Metavision::Camera::from_first_available();
    }

    std::vector<BinSin> bins;

    int win_h = 80, win_w = 80;

    int camera_width = cam.geometry().width();
    int camera_height = cam.geometry().height();
    cv::Size resolution(camera_width, camera_height);

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

    // colored image with bins colors
    cv::Mat colored_image(resolution, CV_8UC3, cv::Scalar(0, 0, 0));

    // event callback
    auto events_callback = [&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            // Determine the bin row and column
            int bin_row = static_cast<int>(ev->x / win_h);
            int bin_col = static_cast<int>(ev->y / win_w);

            // Find the corresponding bin index
            int bin_index = bin_row * num_bins_w + bin_col;

            // Add the event to the corresponding bin
            bins[bin_index].addData(ev->t, ev->x, ev->y);

            // color image bins
            cv::rectangle(colored_image, cv::Rect(bin_col * win_w, bin_row * win_h, win_w, win_h),
                          bins[bin_index].getColor(), -1);
            // draw a circle with the mean y
            // cv::circle(colored_image, cv::Point(int(bin_col * win_w + win_w / 2), bins[bin_index].getY()), 3,
            //            cv::Scalar(0, 255, 255),
            //            -1);

            // print each event
            std::cout << "Event received: coordinates (" << ev->x << ", " << ev->y << "), t: " << ev->t
                      << ", polarity: " << ev->p << std::endl;
        }
    };

    // to analyze the events, we add a callback that will be called periodically to give access to the latest events
    cam.cd().add_callback(events_callback);

    // start the camera
    cam.start();

    // keep running while the camera is on or the recording is not finished
    while (cam.is_running()) {}

    // the recording is finished, stop the camera.
    // Note: we will never get here with a live camera
    cam.stop();

    return 0;
}

//Amplitude (A): 0.369753, Frequency (omega): 700.008, RPM: 6684.59, Phase (phi): -0.0568086, Offset (C): 238.06