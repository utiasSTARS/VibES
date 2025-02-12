//
// Created by viciopoli on 20/12/24.
//

#ifndef PROJECT_EV2IMAGE_HPP
#define PROJECT_EV2IMAGE_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "utils.hpp"

void ev2img(std::vector<EventStruct> &events, cv::Mat &output, EventRepresentation ev_rep) {
    // ev_rep: 0: binary, 1: count, 2: ts, 3: average_ts
    cv::Mat image;

    // create a lambda that takes as input a function and calls it for each event
    auto process_event = [&](auto f) {
        for (auto &ev: events) {
            f(ev, events, image);
        }
    };

    switch (ev_rep) {
        case BINARY:
            image = cv::Mat::zeros(output.rows, output.cols, CV_8UC1);
            process_event([](EventStruct &ev, std::vector<EventStruct> &events, cv::Mat &image) {
                image.at<uchar>(ev.y, ev.x) = 255;
            });
            break;
        case COUNT:
            image = cv::Mat::zeros(output.rows, output.cols, CV_8UC1);
            process_event([](EventStruct &ev, std::vector<EventStruct> &events, cv::Mat &image) {
                image.at<uchar>(ev.y, ev.x) += 1;
            });
            image.convertTo(image, CV_8UC1);
            cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
            break;
        case TS:
            image = cv::Mat::zeros(output.rows, output.cols, CV_8UC1);
            process_event([](EventStruct &ev, std::vector<EventStruct> &events, cv::Mat &image) {
                float deltaT = float((events.end() - 1)->timestamp - ev.timestamp);
                image.at<uchar>(ev.y, ev.x) = std::exp(-deltaT / 0.02);
            });
            break;
        case AVERAGE_TS:
            image = cv::Mat::zeros(output.rows, output.cols, CV_8UC1);
            process_event([](EventStruct &ev, std::vector<EventStruct> &events, cv::Mat &image) {
                float deltaT = float((events.end() - 1)->timestamp - ev.timestamp);
                image.at<uchar>(ev.y, ev.x) += deltaT;
            });
            break;
        default:
            throw std::runtime_error("Invalid Event Representation");
    }

    cv::applyColorMap(image, output, cv::COLORMAP_JET);
}

#endif //PROJECT_EV2IMAGE_HPP
