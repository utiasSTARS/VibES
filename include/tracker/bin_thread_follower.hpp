//
// Created by viciopoli on 20/11/24.
//

#ifndef PROJECT__bin_h
#define PROJECT__bin_h

#include "estimator/ekf.hpp"
#include "utils.hpp"
#include "visualizer/open3d_visualizer.hpp"
#include "event_frontend/centroid.hpp"
#include <memory>
#include <thread>

#include <boost/lockfree/spsc_queue.hpp>
#include <utility>
#include <thread>


class BinThreadFollower {

public:
    ~BinThreadFollower() {
        stop();
    }

    BinThreadFollower(int n_samples,
                      double bin_size,
                      double process_noise,
                      double measurement_noise,
                      double target_omega,
                      double c_x,
                      double c_y,
                      double phase_shift = M_PI / 2,
                      double amplitude = 1.,
                      std::shared_ptr<Open3DVisualizer> vis = nullptr)
            : _bin_id(bin_counter++),
              _target_omega(target_omega),
              _vis(std::move(vis)),
              _bin_size(bin_size),
              _bin_size_half(bin_size / 2.0),
              _bin_center_x(c_x),
              _bin_center_y(c_y) {

        // create the image
        cv::namedWindow("Bin " + std::to_string(_bin_id), cv::WINDOW_NORMAL);
        image = cv::Mat::zeros(480, 640, CV_8UC3);

        ekf = std::make_shared<EKF>(n_samples, process_noise, measurement_noise);
        // double omega, double A, double phi, double C_x, double C_y
        ekf->initialize(_target_omega, amplitude, phase_shift, c_x, c_y);

        _centroid = std::make_shared<CentroidCalculation>();

        // if the _vis is not initialized reaise an error
        if (!_vis) {
            throw std::runtime_error("Open3DVisualizer is not initialized");
        }

        // start thread
        thread = std::thread([&]() {
            while (running) {
                events_queue.consume_all([&](auto &event) {
                    auto [x, y, t] = event;
                    if (auto sample = _centroid->feed(x, y, t); sample.has_value()) {
                        const auto &[s_x, s_y, s_t] = sample.value();
                        auto est = ekf->update(s_x, s_y, s_t);
                        if (est.has_value()) {
                            updated = true;
                            auto [comp_x, comp_y, comp_t] = est.value();

                            // estimated curve
                            auto [est_x, est_y] = ekf->getPred();

                            auto [B, G, R] = color_map[getColor()];
                            auto cv_color = cv::Scalar(B, G, R);
                            int amp_x = ceil(getAmplitudeX()); // + 1;
                            int amp_y = ceil(getAmplitudeY()); // + 1;


                            const auto left_x = _bin_center_x - _bin_size_half, top_y = _bin_center_y - _bin_size_half;

                            {
                                std::lock_guard<std::mutex> lock(mtx);
                                // thread safe
                                _vis->addPoint(x, y, double(s_t) * 100, 0.1, 1.0);
                                _vis->addPoint(est_x, est_y, double(s_t) * 100, 0.1, 0.1, 1.0);
                                // _vis->addPoint(getEKF()->getShiftX(), getEKF()->getShiftY(), mean_t * 100, 0.1, 0.1, 1.0);

                                // draw a rectangle indicating the bin, using bin center and bin width and height
                                cv::rectangle(image,
                                              cv::Rect(left_x, top_y, _bin_size, _bin_size),
                                              cv::Scalar(0, 0, 255), 1);

                                cv::putText(
                                        image, "f: " + fp2str(getHz(), 1) + " Hz",
                                        cv::Point(left_x, top_y + 10),
                                        cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);

                                cv::circle(image, cv::Point(comp_x, comp_y),
                                           std::min(std::max(std::abs(amp_x), std::abs(amp_y)), int(_bin_size / 4)),
                                           cv_color,
                                           -1);

                                // set pixel color at est_x, est_y, use directly pixel operation
                                // image.at<cv::Vec3b>(cv::Point(est_x, est_y)) = cv::Vec3b(255, 255, 255);
                            }

                            _bin_center_x = comp_x;
                            _bin_center_y = comp_y;
                        }
                    }
                    cv::imshow("Bin " + std::to_string(_bin_id), image);
                    // cv::waitKey(1);

                    std::this_thread::sleep_for(std::chrono::nanoseconds(10));
                });
            }
        });
    }

    void stop() {
        running = false;
        thread.join();
    }

    void feed(double x, double y, double t) {
        // check if event is in the bin
        if (check(x, y)) {
            events_queue.push(std::make_tuple(x, y, t));
        }
    }

    [[nodiscard]] bool check(double x, double y) const {
        const double x_max = _bin_center_x + _bin_size_half;
        const double x_min = _bin_center_x - _bin_size_half;
        const double y_max = _bin_center_y + _bin_size_half;
        const double y_min = _bin_center_y - _bin_size_half;
        return x >= x_min && x <= x_max && y >= y_min && y <= y_max;
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
        return _bin_id;
    }

    [[nodiscard]] double getCenterX() {
        return ekf->getShiftX();
    }

    [[nodiscard]] double getCenterY() {
        return ekf->getShiftY();
    }

    [[nodiscard]] bool is_stable() const {
        return updated && std::abs(ekf->getRadS() - _target_omega) < Hz2rad(2);
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
        os << "Bin id: " << bin._bin_id << ", " << *bin.ekf;
        return os;
    }

private:
    double _bin_center_x = 0, _bin_center_y = 0;
    std::shared_ptr<EKF> ekf;
    const int64_t _bin_id;
    static int64_t bin_counter;
    double first_time = -1;

    double mean_x_out = 0, mean_y_out = 0, mean_t_out = 0;

    double _target_omega;
    Colors colors = BLUE;
    bool updated = false;

    std::thread thread;
    std::atomic_bool running{true};
    boost::lockfree::spsc_queue<std::tuple<double, double, double>> events_queue{10000};

    cv::Mat image;
    std::shared_ptr<Open3DVisualizer> _vis;
    static std::mutex mtx;

    double _bin_size = 0, _bin_size_half = 0;

    CentroidPtr _centroid;

};

std::mutex BinThreadFollower::mtx;
int64_t BinThreadFollower::bin_counter = 0;


// alias
using BinPtr = std::shared_ptr<BinThreadFollower>;
using BinsVec = std::vector<std::shared_ptr<BinThreadFollower>>;

#endif //PROJECT__bin_h
