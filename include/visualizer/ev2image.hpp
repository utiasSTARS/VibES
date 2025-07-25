//
// Created by viciopoli on 20/12/24.
//

#ifndef PROJECT_EV2IMAGE_HPP
#define PROJECT_EV2IMAGE_HPP

#include <metavision/sdk/core/pipeline/stage.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils.hpp"

struct ImageResults {
    cv::Mat img_bin;
    cv::Mat img_cnt_gray;
    cv::Mat img_ts;
    cv::Mat img_avgts;
    cv::Mat img_cnt_color;
    cv::Mat img_ts_color;
    cv::Mat img_avgts_color;
};

ImageResults ev2img_metavision(Metavision::Stage::EventBuffer &evs, int MAT_ROWS, int MAT_COLS) {
    ImageResults results;

    if (evs.empty()) {
        // Return empty matrices if no events
        results.img_bin = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC1);
        results.img_cnt_gray = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_16UC1);
        results.img_ts = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC1);
        results.img_avgts = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_32FC1);
        results.img_cnt_color = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC3);
        results.img_ts_color = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC3);
        results.img_avgts_color = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC3);
        return results;
    }

    // Get timestamp of first event (equivalent to t0)
    auto t_end = (evs.end() - 1)->t;

    // Initialize matrices exactly as in original
    cv::Mat eventBinary = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC1);
    cv::Mat eventCount = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_16UC1);
    cv::Mat eventTS = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_32FC1);
    cv::Mat eventSum = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_32FC1);

    // Process events exactly as in original
    for (auto &ev: evs) {
        // Calculate deltaT exactly as in original: time from current event to last event
        float deltaT = float(t_end - ev.t) / 1000000.0f; // Convert microseconds to seconds

        // Update event count with overflow protection (exactly as original)
        if (eventCount.at<uint16_t>(ev.y, ev.x) < 65535) {
            eventCount.at<uint16_t>(ev.y, ev.x) += 1;
        }

        // Set binary image pixel
        eventBinary.at<uchar>(ev.y, ev.x) = 255;

        // Calculate exponential time decay (exactly as original)
        eventTS.at<float>(ev.y, ev.x) = std::exp(-deltaT / 0.02f);

        // Accumulate time sum for average calculation
        eventSum.at<float>(ev.y, ev.x) += deltaT;
    }

    // Store results exactly as original
    results.img_bin = eventBinary;
    eventCount.copyTo(results.img_cnt_gray);
    results.img_ts = eventTS;

    // Calculate average timestamp (exactly as original)
    cv::divide(eventSum, eventCount, results.img_avgts, 1.0f, CV_32FC1);

    // Create normalized versions for color mapping (exactly as original)
    cv::Mat eventCount_norm, eventTS_norm, img_avgts_color_temp;

    // Normalize event count
    normalize(eventCount, eventCount_norm, 0, 255, cv::NORM_MINMAX);
    eventCount_norm.convertTo(eventCount_norm, CV_8UC1);

    // Normalize event timestamp
    normalize(eventTS, eventTS_norm, 0, 255, cv::NORM_MINMAX);
    eventTS_norm.convertTo(eventTS_norm, CV_8UC1);

    // Normalize average timestamp
    normalize(results.img_avgts, img_avgts_color_temp, 0, 255, cv::NORM_MINMAX);
    img_avgts_color_temp.convertTo(img_avgts_color_temp, CV_8UC1);

    // Apply median blur to average timestamp (exactly as original)
    cv::medianBlur(img_avgts_color_temp, img_avgts_color_temp, 3);

    // Apply color maps exactly as original
    cv::applyColorMap(eventCount_norm, results.img_cnt_color, cv::COLORMAP_JET);
    cv::applyColorMap(eventTS_norm, results.img_ts_color, cv::COLORMAP_JET);
    cv::applyColorMap(img_avgts_color_temp, results.img_avgts_color, cv::COLORMAP_BONE);

    return results;
}

#endif //PROJECT_EV2IMAGE_HPP
