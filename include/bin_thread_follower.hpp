//
// Created by viciopoli on 20/11/24.
//

#ifndef PROJECT_BIN_H
#define PROJECT_BIN_H

#include "filter/ekf.hpp"
#include "utils.hpp"
#include "open3d_visualizer.hpp"
#include <memory>
#include <thread>

#include <boost/lockfree/spsc_queue.hpp>
#include <thread>

class BinThreadFollower {

public:
    ~BinThreadFollower() {
        stop();
    }

    BinThreadFollower(int n_samples, cv::Mat &_image, std::shared_ptr<Open3DVisualizer> &_vis,
                      int _bin_w, int _bin_h,
                      double process_noise, double measurement_noise, double _target_omega,
                      double c_x,
                      double c_y,
                      double phase_shift = M_PI / 2, double amplitude = 1.)
            : bin_id(bin_counter++), target_omega(_target_omega), image{_image}, vis(_vis),
              bin_w(_bin_w), bin_h(_bin_h), _bin_center_x(c_x), _bin_center_y(c_y) {
        ekf = std::make_shared<EKF>(n_samples, process_noise, measurement_noise);
        // double omega, double A, double phi, double C_x, double C_y
        ekf->initialize(target_omega, amplitude, phase_shift, c_x, c_y);

        estimated_sampling_time = 1. / (2 * rad2Hz(target_omega) + 10); // in s

        // if the vis is not initialized reaise an error
        if (!vis) {
            throw std::runtime_error("Open3DVisualizer is not initialized");
        }

        // start thread
        thread = std::thread([&]() {
            while (running) {
                events_queue.consume_all([&](auto &event) {
                    auto [x, y, t] = event;
                    auto comp = update(x, y, t);

                    if (!comp.has_value()) {
                        return;
                    }

                    // add mean vals
                    auto [m_x, m_y, m_t] = getMean();
                    // estimated curve
                    auto [est_x, est_y] = getEKF()->getPred();
                    //
                    auto [comp_x, comp_y, comp_t] = comp.value();


                    auto [B, G, R] = color_map[getColor()];
                    auto cv_color = cv::Scalar(B, G, R);
                    int amp_x = getAmplitudeX(); // + 1;
                    int amp_y = getAmplitudeY(); // + 1;


                    const auto left_x = _bin_center_x - bin_w / 2, top_y = _bin_center_y - bin_h / 2;

                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        // thread safe
                        vis->addPoint(m_x, m_y, m_t * 100, 0.1, 1.0);
                        // vis->addPoint(est_x, est_y, mean_t * 100, 0.1, 0.1, 1.0);
                        // vis->addPoint(getEKF()->getShiftX(), getEKF()->getShiftY(), mean_t * 100, 0.1, 0.1, 1.0);

                        // draw a rectangle indicating the bin, using bin center and bin width and height
                        cv::rectangle(image,
                                      cv::Rect(left_x, top_y, bin_w, bin_h),
                                      cv::Scalar(0, 0, 255), 1);

                        cv::putText(
                                image, "f: " + fp2str(getHz(), 1) + " Hz",
                                cv::Point(left_x, top_y + 10),
                                cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);

                        cv::circle(image, cv::Point(comp_x, comp_y),
                                   std::min(std::max(std::abs(amp_x), std::abs(amp_y)), int(bin_w / 4)),
                                   cv_color,
                                   -1);

                        // set pixel color at est_x, est_y, use directly pixel operation
                        // image.at<cv::Vec3b>(cv::Point(est_x, est_y)) = cv::Vec3b(255, 255, 255);
                        image.at<cv::Vec3b>(cv::Point(m_x, m_y)) = cv::Vec3b(255, 0, 255);
                    }

                    _bin_center_x = comp_x;
                    _bin_center_y = comp_y;

                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                });
            }
        });
    }

    void stop() {
        running = false;
        thread.join();
    }

    void add_event(double x, double y, Time t) {
        // check if event is in the bin
        if (check(x, y)) {
            events_queue.push(std::make_tuple(x, y, t));
        }
    }

    bool check(double x, double y) {
        const double x_max = _bin_center_x + bin_w / 2.0;
        const double x_min = _bin_center_x - bin_w / 2.0;
        const double y_max = _bin_center_y + bin_h / 2.0;
        const double y_min = _bin_center_y - bin_h / 2.0;
        return x >= x_min && x <= x_max && y >= y_min && y <= y_max;
    }

    std::optional<std::tuple<int, int, int64_t>> update(double x, double y, Time t) {
        std::optional<std::tuple<int, int, int64_t>> comp;
        // t += 1; // to avoid odd time mean
        if (first_time == -1) {
            first_time = double(t - 0.1);
        }
        // t -= first_time;
        if (prev_time == 0) {
            prev_time = double(t);
        }

        if (t - prev_time < estimated_sampling_time) {
            mean_x += x;
            mean_y += y;
            mean_t += double(t);
            counter += 1;
            // prev_time = t;
            return comp;
        }
        if (counter > 1) { // 100
            mean_x_out = mean_x / counter;
            mean_y_out = mean_y / counter;
            mean_t_out = mean_t / counter;
            comp = ekf->update(mean_x_out, mean_y_out, mean_t_out);
            updated = true;
        }
        mean_x = x;
        mean_y = y;
        mean_t = double(t);
        counter = 1;

        prev_time = double(t);
        return comp;
    }

    std::tuple<double, double, double> getMean() {
        return std::make_tuple(mean_x_out, mean_y_out, mean_t_out);
    }

    std::tuple<double, double> estimate(double t) {
        return std::make_tuple(
                ekf->getAmplX() * std::sin(ekf->getRadS() * (t - first_time) + ekf->getPhaseX()) + ekf->getShiftX(),
                ekf->getAmplY() * std::sin(ekf->getRadS() * (t - first_time) + ekf->getPhaseY()) + ekf->getShiftY());
    }

    [[nodiscard]] std::optional<std::tuple<int, int>> compensate(int x, int y, double t) const {
        if (!updated) {
            return std::nullopt;
        }
        // according to out camera projection model
        double u = x - ekf->getAmplX() * std::sin(ekf->getRadS() * (t - ekf->getPrevT()) + ekf->getPhaseX());
        double v = y - ekf->getAmplY() * std::sin(ekf->getRadS() * (t - ekf->getPrevT()) + ekf->getPhaseY());

        return std::make_tuple(static_cast<int>(u), static_cast<int>(v));
    }

    [[nodiscard]] double getOmega() const {
        return ekf->getRadS();
    }

    [[nodiscard]] double getAmplitudeX() const {
        return ekf->getAmplX();
    }

    [[nodiscard]] double getAmplitudeY() const {
        return ekf->getAmplY();
    }

    [[nodiscard]] double getHz() const {
        return ekf->getHz();
    }

    [[nodiscard]] double getRad() const {
        return ekf->getRadS();
    }

    [[nodiscard]] int64_t getBinId() const {
        return bin_id;
    }

    [[nodiscard]] double getCenterX() {
        return ekf->getShiftX();
    }

    [[nodiscard]] double getCenterY() {
        return ekf->getShiftY();
    }

    [[nodiscard]] bool is_stable() const {
        return updated && std::abs(ekf->getRadS() - target_omega) < Hz2rad(2);
    }

    [[nodiscard]] Colors getColor() const {
        return is_stable() ? GREEN : RED;
    }

    [[nodiscard]] bool is(double f) const {
        return updated && std::abs(ekf->getRadS() - f) < 5.;
    }

    std::shared_ptr<EKF> getEKF() {
        return ekf;
    }

    friend std::ostream &operator<<(std::ostream &os, const BinThreadFollower &bin) {
        os << "Bin id: " << bin.bin_id << ", " << *bin.ekf;
        return os;
    }

private:
    double _bin_center_x = 0, _bin_center_y = 0;
    std::shared_ptr<EKF> ekf;
    const int64_t bin_id;
    static int64_t bin_counter;
    double first_time = -1;

    double prev_time = 0;
    double mean_x = 0;
    double mean_y = 0;
    double mean_t = 0;
    int64_t counter = 0;

    double mean_x_out = 0;
    double mean_y_out = 0;
    double mean_t_out = 0;

    double target_omega;
    Colors colors = BLUE;
    bool updated = false;

    double estimated_sampling_time = 0;

    std::thread thread;
    std::atomic_bool running{true};
    boost::lockfree::spsc_queue<std::tuple<double, double, Time>> events_queue{10000};

    cv::Mat image;
    std::shared_ptr<Open3DVisualizer> vis;
    static std::mutex mtx;

    int bin_w = 0, bin_h = 0;

};

std::mutex BinThreadFollower::mtx;
int64_t BinThreadFollower::bin_counter = 0;

#endif //PROJECT_BIN_H
