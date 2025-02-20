//
// Created by viciopoli on 20/11/24.
//
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "event_frontend/centroid.hpp"
#include "estimator/nufourier_new.hpp"

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/ui/utils/event_loop.h>

TEST(CENTROID, Estimation) {
    CMassCalculation centroid(10, 1e-3);

    auto cam = Metavision::Camera::from_file(
            "/home/viciopoli/datasets/event_harmeda/april_2v.raw");

    std::vector<std::tuple<double, double, double>> samples;
    int initial_time = -1;

    FourierFreqEst fourier(100, 5, 1000);
    double estimated_freq = 0;
    double phase_shift = 0;
    double amplitude = 0;

    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        auto ev_prev = begin;
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            if (ev_prev->t > ev->t) {
                throw std::runtime_error("Events are not sorted by time");
            }
            if (initial_time == -1) {
                initial_time = ev->t;
            }
            auto centr = centroid.feed(ev->x, ev->y, double(ev->t - initial_time) / 1e6);
            if (centr.has_value()) {
                samples.push_back(centr.value());
                const auto &[x, y, t] = centr.value();
                if (fourier.feed(x, y, t)) {
                    estimated_freq = fourier.getMainFreqRad();
                    phase_shift = fourier.getPhaseShift();
                    amplitude = fourier.getAmplitude();
                    cam.stop();
                }
            }
        }
    });

    // start the camera
    cam.start();
    // keep running until the camera is off, the recording is finished or the escape key was pressed
    while (cam.is_running()) {
        // we poll events (keyboard, mouse etc.) from the system with a 20ms sleep to avoid using 100% of a CPU's core
        // and we push them into the window where the callback on the escape key will ask the windows to close
        static constexpr std::int64_t kSleepPeriodMs = 20;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
    }

}
