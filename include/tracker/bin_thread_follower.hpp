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

using Event = std::tuple<int, int, double, short>;
using EventVect = std::vector<Event>;
using EventVectPtr = std::shared_ptr<EventVect>;


class BinThreadFollower {

public:
    ~BinThreadFollower() {
        stop();
    }

    BinThreadFollower(
            int width,
            int height,
            double bin_size,
            double process_noise,
            double measurement_noise,
            double target_omega,
            double c_x,
            double c_y,
            double phase_shift = M_PI / 2,
            double amplitude_x = 1.,
            double amplitude_y = 1.,
            std::shared_ptr<Open3DVisualizer> vis = nullptr,
            bool fixed_bin = false,
            bool motion_compensation = false)
            : _bin_id(bin_counter++),
              _target_omega(target_omega),
              _vis(std::move(vis)),
              _bin_size(bin_size),
              _bin_size_half(bin_size / 2.0),
              _bin_center_x(c_x),
              _bin_center_y(c_y),
              _fixed_bin(fixed_bin),
              _motion_compensation(motion_compensation) {
        auto centroid_freq = 1. / (10. * rad2Hz(target_omega));
        _centroid = std::make_shared<CMassCalculation>(width, height, 100, centroid_freq); // 1e-3); //

        ekf = std::make_shared<EKF>(centroid_freq, 1., 1., .1, 3.);
        // double omega, double A, double phi, double C_x, double C_y
        ekf->initialize(_target_omega, amplitude_x, amplitude_y, c_x, c_y, phase_shift);

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
                        auto centre = ekf->getCenter();
                        const auto &[centre_x, centre_y] = centre.value();

                        if (!_fixed_bin) {
                            _bin_center_x = centre_x;
                            _bin_center_y = centre_y;
                        }

                        const auto &[est_x, est_y] = ekf->getPred();

                        omega_covariance = ekf->getOmegaCov();

                        c_m_x = s_x;
                        c_m_y = s_y;

                        auto cov = ekf->getCov();
                        // print the covariance, cov is a tuple
                        // std::cout << "\rCovariance: ";
                        // std::apply([&](auto... args) {
                        //     ((std::cout << args << " "), ...);
                        // }, cov);
                        // std::cout.flush();

                        const double scale = 100;
                        _vis->addPoint(centre_x, centre_y, s_t * scale, 1., 0., 0.);
                        _vis->addLine(s_x, s_y, s_t * scale, 0.1, 1.0);
                        _vis->addLine2(est_x, est_y, s_t * scale, 0.1, 0.1, 1.0);

                        if (_motion_compensation) {
                            compensate();
                            // compensate(ekf->getTheta());
                        }
                    } else {
                        _events_centre.pop();
                    }
                });
            }
        });
    }

    void compensate() {
        std::lock_guard<std::mutex> lock(mtx_out);
        auto &[events, t_hat] = _events_centre.front();
        // if (t_start == -1) {
        //     t_start = std::get<2>(events->front());
        // }
        // double C_x_est = 0;
        for (auto &event: *events) {
            const auto &[shift_x, shift_y] = ekf->getComp(get<2>(event) - t_hat);
            get<0>(event) -= ceil(shift_x);
            get<1>(event) -= ceil(shift_y);
            get<3>(event) = 1;
            // cv::circle(_frame, cv::Point((t - t_start) * 1000, 10 * shift_x + 240), 2, cv::Scalar(0, 255, 0), -1);
        }
        _events_out = events;
        // cv::imshow("Compensated", _frame);
        // cv::waitKey(0);
        // C_x_est /= events->size();
        // std::cout << "C_x_est: " << C_x_est << ", should be same as: " << ekf->getShiftX() << std::endl;
        // std::cout << "t_hat: " << t_hat << ", should be same as: " << ekf->getPrevT() << std::endl;
        // assert(C_x_est == ekf->getShiftX());
        // assert(t_hat == ekf->getPrevT());
        _events_centre.pop();
    }

    void getEventsOut(std::shared_ptr<EventVect> &events) {
        std::lock_guard<std::mutex> lock(mtx_out);
        events = _events_out;
        _events_out = std::make_shared<EventVect>();
    }


    void stop() {
        running.store(false, std::memory_order_relaxed);
        thread.join();
    }

    void feed(double x, double y, double t, short pol) {
        // check if event is in the bin
        if (check(x, y)) {
            auto sample = _centroid->feed(x, y, t);
            _events->emplace_back(x, y, t, pol);
            if (sample) {
                events_queue.push(sample.value());
                _events_centre.emplace(_events, std::get<2>(sample.value()));
                _events = std::make_shared<EventVect>();
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
                      cv_color, 2);

        // draw the freq
        cv::putText(frame, "f:" + fp2str(ekf->getHz(), 1) + " Hz",
                    cv::Point(_bin_center_x - _bin_size_half, _bin_center_y - _bin_size_half - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv_color, 2, cv::LINE_AA);

        // draw covariance as an ellipse
        cv::ellipse(frame, cv::Point(_bin_center_x, _bin_center_y),
                    cv::Size(omega_covariance, omega_covariance), 0, 0, 360, cv_color, 2);

        // draw c_m_x and c_m_y as a dot
        cv::circle(frame, cv::Point(c_m_x, c_m_y), 2, cv_color, -1);

    }

    [[nodiscard]] bool check(double x, double y) const {
        const double x_max = _bin_center_x + _bin_size_half;
        const double x_min = _bin_center_x - _bin_size_half;
        const double y_max = _bin_center_y + _bin_size_half;
        const double y_min = _bin_center_y - _bin_size_half;
        return x >= x_min && x <= x_max && y >= y_min && y <= y_max;
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
    static std::mutex mtx, mtx_out;

    double _bin_size = 0, _bin_size_half = 0;

    CentroidPtr _centroid;

    double omega_covariance = 0;
    double c_m_x = 0, c_m_y = 0;

    std::queue<std::tuple<EventVectPtr, double>> _events_centre;
    EventVectPtr _events = std::make_shared<EventVect>();
    EventVectPtr _events_out = std::make_shared<EventVect>();


    cv::Mat _frame = cv::Mat::zeros(480, 640, CV_8UC3);
    double t_start = -1;
    bool _fixed_bin = false, _motion_compensation = false;

};

std::mutex BinThreadFollower::mtx;
std::mutex BinThreadFollower::mtx_out;
int64_t BinThreadFollower::bin_counter = 0;


// alias
using BinPtr = std::shared_ptr<BinThreadFollower>;
using BinsVec = std::vector<std::shared_ptr<BinThreadFollower>>;

#endif //PROJECT__bin_h
