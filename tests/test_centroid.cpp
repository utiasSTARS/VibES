//
// Created by viciopoli on 20/11/24.
//
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>

#include "event_frontend/centroid.hpp"
#include "sim/ini_sim.hpp"

TEST(CENTROID, Estimation) {
    CentroidCalculation centroid;

    // auto reader = std::make_unique<EventsFreqCalibPattern>(0, cv::Size(480, 640), Hz2rad(100), 5, 5, 0., true);
    auto reader = std::make_shared<dv::io::MonoCameraRecording>("/home/viciopoli/datasets/event_harmeda/harmeda_newpattern_hz100_3mm.aedat4");

    int counter = 0;
    std::vector<std::tuple<double, double, Time>> samples;
    int initial_time = -1;
    while (reader->isRunning()) {
        if (const auto events_dist = reader->getNextEventBatch(); events_dist.has_value()) {
            for (const auto &event: events_dist.value()) {
                if (initial_time == -1) {
                    initial_time = event.timestamp();
                }
                auto centr = centroid.feed(event.x(), event.y(), double(event.timestamp() - initial_time) / 1e6);
                if (centr.has_value()) {
                    samples.push_back(centr.value());
                }
            }
        }
        counter++;
        if (counter > 5'000) {
            break;
        }
    }

    std::cout << "N. of samples: " << samples.size() << std::endl;

    // show samples on an image
    int scaling = 10000;
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
}
