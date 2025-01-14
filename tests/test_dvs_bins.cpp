//
// Created by viciopoli on 20/11/24.
//
#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "bin.hpp"
#include <dv-processing/io/mono_camera_recording.hpp>

TEST(DVSBin, OneBin) {
    // read aedat4 file
    dv::io::MonoCameraRecording reader(
            "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/file.aedat4");
    double target_freq = 1.;
    Bin bin(20, 0.1, 0.1, 690., 0, 0);

    int i = 0;
    while (reader.isRunning() && i < 30) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {
            for (const auto &event: events.value()) {
                bin.update(static_cast<double>(event.x()), static_cast<double>(event.y()), event.timestamp() / 1e6);
            }
            std::cout << bin << std::endl;
        } else {
            std::cerr << "Failed to generate events.\n";
        }
        i++;
    }

    EXPECT_NEAR(bin.getEKF()->getRadS(), target_freq, 1.);

}

TEST(DVSBin, MultiBin) {
    // read aedat4 file
    dv::io::MonoCameraRecording reader(
            "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/file.aedat4");
    double target_freq = 600.;

    cv::Size resolution = *reader.getEventResolution();

    int win_h = 80;
    int win_w = 80;

    int num_bins_h = resolution.height / win_h;
    int num_bins_w = resolution.width / win_w;

    // ensure bins are submultiple of the resolution
    assert(resolution.width % win_h == 0);
    assert(resolution.height % win_w == 0);

    std::vector<Bin> bins;
    for (int i = 0; i < num_bins_h; i++) {
        for (int j = 0; j < num_bins_w; j++) {
            // c_x and x_y are the center of the bin
            double c_x = j * win_w + win_w / 2;
            double c_y = i * win_h + win_h / 2;
            bins.emplace_back(10, 0.1, 0.1, target_freq, c_x, c_y);
        }
    }

    int i = 0;
    while (reader.isRunning() && i < 20) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {
            for (const auto &event: events.value()) {
                // Determine the bin row and column
                int bin_row = static_cast<int>(event.y() / win_h);
                int bin_col = static_cast<int>(event.x() / win_w);

                // Find the corresponding bin index
                int bin_index = bin_row * num_bins_w + bin_col;

                bins[bin_index].update(static_cast<double>(event.x()), static_cast<double>(event.y()),
                                       event.timestamp() / 1e6);
            }
        } else {
            std::cerr << "Failed to generate events.\n";
        }
        i++;
    }

    cv::Mat img(resolution, CV_8UC1, cv::Scalar(0));
    for (auto &bin: bins) {
        std::cout << bin << std::endl;
        if (bin.is(target_freq)) {
            std::cout << bin << std::endl;
            auto c_x = bin.getCenterX();
            auto c_y = bin.getCenterY();
            cv::rectangle(img, cv::Rect(c_x - win_w / 2, c_y - win_h / 2, win_w, win_h), cv::Scalar(255), 1);
        }
    }

    cv::imshow("Bins", img);
    cv::waitKey(0);

}


