//
// Created by viciopoli on 20/11/24.
//
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "centroid.hpp"
#include "sim/ini_sim.hpp"

using namespace harmeda;

TEST(CENTROID, Estimation) {
    CentroidCalculation centroid;

    auto reader = std::make_unique<EventsFreqCalibPattern>(0, cv::Size(480, 640), 100, 5, 5, 0., true);

    int counter = 0;
    while (reader->isRunning()) {
        if (const auto events_dist = reader->getNextEventBatch(); events_dist.has_value()) {
            for (const auto &event: events_dist.value()) {
                centroid.feed(event.x(), event.y(), event.timestamp() / 1e6);
            }
        }
        counter++;
        if (counter > 1000) {
            break;
        }
    }

    std::cout << "N. of samples: " << centroid.size() << std::endl;
    auto samples = centroid.getCentroids();

    // show samples on an image
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    for (const auto [x, y, t]: samples) {
        if (t * 1000 < 480 && x < 480 && y < 640) {
            img.at<cv::Vec3b>(x, t * 1000) = cv::Vec3b(255, 255, 255);
            img.at<cv::Vec3b>(t * 1000, y) = cv::Vec3b(255, 255, 255);
        }
    }

    cv::namedWindow("Centroids", cv::WINDOW_NORMAL);
    cv::imshow("Centroids", img);
    cv::waitKey(0);
}
