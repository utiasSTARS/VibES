//
// Created by viciopoli on 07/02/25.
//

#ifndef PROJECT_HARMEDA_H
#define PROJECT_HARMEDA_H

#include "tracker/bin_thread_follower.hpp"
#include "event_frontend/centroid.hpp"
#include "estimator/nufourier_new.hpp"
#include "estimator/nufourier_new.hpp"

#include <atomic>
#include <thread>

#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
#include <memory>

#include <boost/lockfree/spsc_queue.hpp>

class HARMEDA {
public:
    HARMEDA() = delete;

    HARMEDA(int n_samples, double f_min, double f_max)
            : _fourierFreqEst(n_samples, f_min, f_max) {
        _initializer = std::make_shared<CMassCalculation>(10, 1e-3);

        _loading_text.loading("Initializing HARMEDA");
    }

    ~HARMEDA() {
        stop();
    }

    [[nodiscard]] bool initialized() const {
        return _initialized;
    }

    void add_bin(double x, double y, double size = 40,
                 std::shared_ptr<Open3DVisualizer> vis = nullptr) {
        if (!_initialized) {
            throw std::runtime_error("HARMEDA is not initialized yet.");
        }
        std::lock_guard<std::mutex> lock(_mtx);
        _bins.emplace_back(
                std::make_shared<BinThreadFollower>(1, size, 0.1, 0.1,
                                                    _estimated_freq,
                                                    x, y,
                                                    _phase_shift,
                                                    _amplitude,
                                                    vis));
    }

    void feed(double x, double y, double time) {
        if (!_initialized) {
            if (_start_time.time_since_epoch().count() == 0) {
                _start_time = std::chrono::high_resolution_clock::now();
            }
            // add this here, because it is true that the following code will be run less than the estimation code above
            auto sample = _initializer->feed(x, y, time);
            if (!sample) {
                return;
            }

            const auto &[s_x, s_y, s_t] = sample.value();
            if (!_fourierFreqEst.feed(s_x, s_y, s_t)) {
                return;
            }

            _estimated_freq = _fourierFreqEst.getMainFreqRad();
            _phase_shift = _fourierFreqEst.getPhaseShift();
            _amplitude = _fourierFreqEst.getAmplitude();

            const auto offset_est = _fourierFreqEst.getOffset();
            _ini_events_centre_x = std::get<0>(offset_est);
            _ini_events_centre_y = std::get<1>(offset_est);

            _loading_text.stop();

            _initialized = true;

            // time measurement
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = end_time - _start_time;
            // text in red
            std::cout << "\033[1;31m";
            std::cout << "Initialization time: " << elapsed_time.count() << " s\n";
            std::cout << "\033[0m" << std::endl;
            return;
        }

        if (!_bins.empty()) {
            std::lock_guard<std::mutex> lock(_mtx);
            // bool pushed = _events_queue.push({x, y, time});
            _bins[0]->feed(x, y, time);
// #pragma omp parallel for
//                     for (std::size_t i = 0; i < _bins.size(); ++i) {
//                         _bins[i]->feed(x, y, time);
//                     }
        }
    }

    void draw(cv::Mat &image) {
        if (!_initialized) {
            return;
        }

        std::lock_guard<std::mutex> lock(_mtx);
        for (const auto &bin: _bins) {
            bin->draw(image);
        }
    }

    void stop() {
        _loading_text.stop();
    }

    [[nodiscard]] size_t size() const {
        return _bins.size();
    }

    std::pair<double, double> getInitialCenter() {
        return {_ini_events_centre_x, _ini_events_centre_y};
    }

    friend std::ostream &operator<<(std::ostream &os, const HARMEDA &harmeda) {
        os << "Estimated frequency: " << harmeda._estimated_freq << " rad/s\n";
        os << "Estimated frequency: " << rad2Hz(harmeda._estimated_freq) << " Hz\n";
        os << "Phase shift: " << harmeda._phase_shift << "\n";
        os << "Amplitude: " << harmeda._amplitude << "\n";
        os << "Initialized: " << harmeda._initialized << "\n";
        os << "Initial events centre x: " << harmeda._ini_events_centre_x << "\n";
        os << "Initial events centre y: " << harmeda._ini_events_centre_y << "\n";
        return os;
    }

private:
    LoadingText _loading_text;
    // Bins container (assumed to be a vector of shared pointers)
    BinsVec _bins;

    FourierFreqEst _fourierFreqEst;
    CentroidPtr _initializer;

    double _estimated_freq = 0;
    double _phase_shift = 0;
    double _amplitude = 0;

    bool _initialized = false;
    double _ini_events_centre_x = 0, _ini_events_centre_y = 0;

    // Use std::jthread for cooperative cancellation.
    std::jthread _thread;

    std::mutex _mtx;

    // Lock-free single-producer single-consumer queues.
    boost::lockfree::spsc_queue<std::tuple<double, double, Time>> _init_events_queue{10000};
    boost::lockfree::spsc_queue<std::tuple<double, double, Time>> _events_queue{10000};

    // chrono for time measurement
    std::chrono::high_resolution_clock::time_point _start_time;
};

#endif // PROJECT_HARMEDA_H
