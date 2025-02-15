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

struct DataPoint {
    double x, y, t;
};

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

        ekf = std::make_shared<EKF>(n_samples, process_noise, measurement_noise);
        // double omega, double A, double phi, double C_x, double C_y
        ekf->initialize(_target_omega, amplitude, phase_shift, c_x, c_y);

        // std::cout << "\033[1;34m" << "Centroid initialization with delta t: " << 1. / (10. * rad2Hz(target_omega))
        //           << "\033[0m" << std::endl;
        _centroid = std::make_shared<CMassCalculation>(100, 1. / (10. * rad2Hz(target_omega)));

        // if the _vis is not initialized reaise an error
        if (!_vis) {
            throw std::runtime_error("Open3DVisualizer is not initialized");
        }
        // start thread
        thread = std::thread([&]() {
            while (running.load(std::memory_order::relaxed)) {
                events_queue.consume_all([&](auto &sample) {
                    const auto &[s_x, s_y, s_t] = sample;
                    if (ekf->update(s_x, s_y, s_t)) {
                        updated = true;

                        // estimated curve
                        const auto &[centre_x, centre_y] = ekf->getCenter();

                        const auto &[est_x, est_y] = ekf->getPred();


                        // std::lock_guard<std::mutex> lock(mtx);
                        // // thread safe
                        // _vis->addPoint(s_x, s_y, s_t * 1000, 0.1, 1.0);
                        // _vis->addPoint(est_x, est_y, s_t * 1000, 0.1, 0.1, 1.0);


                        _bin_center_x = centre_x;
                        _bin_center_y = centre_y;
                    }

                    // std::this_thread::sleep_for(std::chrono::nanoseconds(10));
                });
            }
        });
    }

    void stop() {
        running.store(false, std::memory_order_relaxed);
        thread.join();
    }

    void feed(double x, double y, double t) {
        // check if event is in the bin
        if (check(x, y)) {
            auto sample = _centroid->feed(x, y, t);
            if (sample) {
                events_queue.push(sample.value());
            }
        }
    }

    void draw(cv::Mat &frame) {
        std::lock_guard<std::mutex> lock(mtx);
        // draw a rectangle indicating the bin, using bin center and bin width and height

        auto [B, G, R] = color_map[getColor()];

        auto cv_color = cv::Scalar(B, G, R);

        cv::rectangle(frame,
                      cv::Rect(_bin_center_x - _bin_size_half, _bin_center_y - _bin_size_half, _bin_size, _bin_size),
                      cv::Scalar(0, 0, 255), 1);

        // draw the freq
        cv::putText(frame, "f:" + fp2str(ekf->getHz(), 1) + " Hz",
                    cv::Point(_bin_center_x - _bin_size_half, _bin_center_y - _bin_size_half - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv_color, 1, cv::LINE_AA);
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
