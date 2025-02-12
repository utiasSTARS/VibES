//
// Created by viciopoli on 07/02/25.
//

#ifndef PROJECT_HARMEDA_H
#define PROJECT_HARMEDA_H

#include "tracker/bin_thread_follower.hpp"
#include "estimator/nufourier_new.hpp"
#include "event_frontend/centroid.hpp"
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
        _initializer = std::make_shared<CentroidCalculation>();

        // For the main event processing thread:
        _thread = std::jthread([this](std::stop_token st) {
            while (_running.load(std::memory_order_relaxed) && !st.stop_requested()) {
                // Process all available events using consume_all
                _events_queue.consume_all([this](const std::tuple<double, double, Time> &event) {
                    const auto &[x, y, time] = event;

                    // Process each bin in parallel
#pragma omp parallel for
                    for (std::size_t i = 0; i < _bins.size(); ++i) {
                        _bins[i]->feed(x, y, time);
                    }
                });

                // Optionally, sleep a bit to reduce CPU usage if there are no events
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });


    }

    [[nodiscard]] bool initialized() const {
        return _initialized.load(std::memory_order_relaxed);
    }

    void add_bin(double x, double y, double size = 40,
                 std::shared_ptr<Open3DVisualizer> vis = nullptr) {
        _bins.emplace_back(
                std::make_shared<BinThreadFollower>(1, size, 0.1, 0.1,
                                                    _estimated_freq,
                                                    x, y,
                                                    _phase_shift,
                                                    _amplitude,
                                                    vis));
    }

    bool feed(double x, double y, Time time) {
        if (_initialized.load(std::memory_order_relaxed)) {
            bool pushed = _events_queue.push({x, y, time});
            _cv.notify_one();
            return pushed;
        } else {
            // add this here, because it is true that the following code will be run less than the estimation code above
            auto sample = _initializer->feed(x, y, time);
            if (!sample) {
                return false;
            }

            const auto &[s_x, s_y, s_t] = sample.value();
            if (!_fourierFreqEst.feed(s_x, s_y, s_t)) {
                return false;
            }

            _estimated_freq = _fourierFreqEst.getMainFreqRad();
            _phase_shift = _fourierFreqEst.getPhaseShift();
            _amplitude = _fourierFreqEst.getAmplitude();

            _initialized.store(true, std::memory_order_relaxed);

            const auto offset_est = _fourierFreqEst.getOffset();
            _ini_events_centre_x = std::get<0>(offset_est);
            _ini_events_centre_y = std::get<1>(offset_est);

            return true;
        }
        return false;
    }

    void stop() {
        _running.store(false, std::memory_order_relaxed);
        _init_cv.notify_all();
        _cv.notify_all();
        // std::jthread will automatically join upon destruction.
    }

    std::pair<double, double> getInitialCenter() {
        return {_ini_events_centre_x, _ini_events_centre_y};
    }

    friend std::ostream &operator<<(std::ostream &os, const HARMEDA &harmeda) {
        os << "Estimated frequency: " << harmeda._estimated_freq << " rad/s\n";
        os << "Estimated frequency: " << rad2Hz(harmeda._estimated_freq) << " Hz\n";
        os << "Phase shift: " << harmeda._phase_shift << "\n";
        os << "Amplitude: " << harmeda._amplitude << "\n";
        os << "Initialized: " << harmeda._initialized.load(std::memory_order_relaxed) << "\n";
        os << "Initial events centre x: " << harmeda._ini_events_centre_x << "\n";
        os << "Initial events centre y: " << harmeda._ini_events_centre_y << "\n";
        return os;
    }

private:
    // Bins container (assumed to be a vector of shared pointers)
    BinsVec _bins;

    FourierFreqEst _fourierFreqEst;
    CentroidPtr _initializer;

    double _estimated_freq = 0;
    double _phase_shift = 0;
    double _amplitude = 0;

    std::atomic_bool _initialized{false};
    std::atomic_bool _running{true};

    double _ini_events_centre_x = 0, _ini_events_centre_y = 0;

    // Use std::jthread for cooperative cancellation.
    std::jthread _thread;

    // Condition variables and mutexes for waiting on new events.
    std::mutex _init_mutex;
    std::condition_variable _init_cv;

    std::mutex _mutex;
    std::condition_variable _cv;

    // Lock-free single-producer single-consumer queues.
    boost::lockfree::spsc_queue<std::tuple<double, double, Time>> _init_events_queue{10000};
    boost::lockfree::spsc_queue<std::tuple<double, double, Time>> _events_queue{10000};
};

#endif // PROJECT_HARMEDA_H
