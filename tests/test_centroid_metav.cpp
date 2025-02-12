//
// Created by viciopoli on 20/11/24.
//
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "event_frontend/centroid.hpp"


#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/algorithms/flip_x_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

TEST(CENTROID, Estimation) {
    CentroidCalculation centroid;

    auto cam = Metavision::Camera::from_file(
            "/home/viciopoli/datasets/event_harmeda/recording_2025-02-07_18-51-27.raw");


    int counter = 0;
    std::vector<std::tuple<double, double, Time>> samples;
    int initial_time = -1;

    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            if (initial_time == -1) {
                initial_time = ev->t;
            }
            auto centr = centroid.feed(ev->x, ev->y, double(ev->t - initial_time) / 1e6);
            if (centr.has_value()) {
                samples.push_back(centr.value());
            }
        }

        counter++;
        if (counter > 50'000) {
            std::cout << "N. of samples: " << samples.size() << std::endl;

            // show samples on an image
            int scaling = 1000;
            cv::Mat img(600, 600, CV_8UC3, cv::Scalar(0, 0, 0));
            for (const auto [x, y, time]: samples) {
                auto t = double(time);
                if (t * scaling < 600 && x < 600 && y < 600) {
                    img.at<cv::Vec3b>(x, t * scaling) = cv::Vec3b(255, 255, 255);
                    img.at<cv::Vec3b>(t * scaling, y) = cv::Vec3b(255, 255, 255);
                }
            }

            cv::namedWindow("Centroids", cv::WINDOW_NORMAL);
            cv::imshow("Centroids", img);
            cv::waitKey(0);

            cam.stop();
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
