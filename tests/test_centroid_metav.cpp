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
#include <metavision/sdk/ui/utils/event_loop.h>
#include <fstream>

TEST(CENTROID, Estimation) {
    CMassCalculation centroid(100, 1e-4);

    auto cam = Metavision::Camera::from_file(
            "/home/viciopoli/datasets/event_harmeda/pattern.raw");


    int counter = 0;
    std::vector<std::tuple<double, double, double>> samples;
    int initial_time = -1;

    // create a file we want to store the centroid
    std::ofstream file_x(
            "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/project/scripts/centroid_data_real/centroids_x_pattern.txt");
    std::ofstream file_y(
            "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/project/scripts/centroid_data_real/centroids_y_pattern.txt");

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
                auto [x, y, time] = centr.value();
                file_x << x << "," << time << std::endl;
                file_y << y << "," << time << std::endl;
                samples.push_back(centr.value());
            }
        }

        counter++;
        if (counter > 50'000) {
            std::cout << "N. of samples: " << samples.size() << std::endl;

            // show samples on an image
            int scaling = 1e4;
            auto win_size = 1000;
            double initial_time = -1, final_time = 0;
            cv::Mat img(win_size, win_size, CV_8UC3, cv::Scalar(0, 0, 0));
            for (int i = 1; i < samples.size(); i++) {
                auto [x, y, time] = samples[i];
                if (initial_time == -1) {
                    initial_time = time;
                }
                if (time * scaling < win_size && x < win_size && y < win_size) {
                    auto [x_prev, y_prev, time_prev] = samples[i - 1];
                    cv::line(img, cv::Point(x_prev, time_prev * scaling), cv::Point(x, time * scaling),
                             cv::Scalar(255, 255, 255), 1);
                    cv::line(img, cv::Point(time_prev * scaling, y_prev), cv::Point(time * scaling, y),
                             cv::Scalar(255, 255, 255), 1);
                    final_time = time;
                }
            }
            auto delta_t = final_time - initial_time;
            std::cout << "Initial time: " << initial_time << " Final time: " << final_time << " delta: "
                      << delta_t << ", one px: " << delta_t / win_size << "s" << std::endl;


            cv::namedWindow("Centroids", cv::WINDOW_NORMAL);
            cv::imshow("Centroids", img);
            cv::waitKey(0);
            cam.stop();


            // close files
            file_x.close();
            file_y.close();
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
