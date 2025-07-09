//
// Created by viciopoli on 30/06/25.
//

#ifndef PROJECT_SLICE_VISUALIZER_HPP
#define PROJECT_SLICE_VISUALIZER_HPP

#include <metavision/sdk/driver/camera.h>
#include <opencv2/opencv.hpp>

namespace HARMEDA {
    enum Axis {
        X_AXIS = 0,
        Y_AXIS = 1
    };

    class SliceVisualizer {
    public:
        SliceVisualizer(int h, int w, int axis_value, Axis axis = X_AXIS, int margin = 3) : _axis(axis),
                                                                                            _margin(margin) {
            // Initialize the visualizer with the specified axis
            _frame = cv::Mat::zeros(h, w, CV_8UC3);
            const int max_time_width = 1920;
            if (axis == X_AXIS) {
                _frame_side = cv::Mat::zeros(w, max_time_width, CV_8UC3);
            } else {
                _frame_side = cv::Mat::zeros(h, max_time_width, CV_8UC3);
            }

            // add line at 0.1 seconds
            const int second_line_x = 0.1 * time_scale; // 0.01 seconds in the visualization
            cv::line(_frame_side, cv::Point(second_line_x, 0), cv::Point(second_line_x, _frame_side.rows),
                     cv::Scalar(255, 255, 255), 1);

            _axis_value = axis_value;
            _lower_bound = axis_value - _margin;
            _upper_bound = axis_value + _margin;
        }

        void setAxis(int axis_value) {
            std::lock_guard<std::mutex> lock(_mtx);
            _axis_value = axis_value;
            _lower_bound = axis_value - _margin;
            _upper_bound = axis_value + _margin;
        }

        void feed(const Metavision::EventCD &event) {
            std::lock_guard<std::mutex> lock(_mtx);
            if (initial_time == 0) {
                initial_time = event.t; // Set the initial time on the first event
            }
            auto x = event.x;
            auto y = event.y;
            time_value = static_cast<int>((event.t - initial_time) * 1e-6 *
                                          time_scale); // Convert timestamp to seconds and scale
            if (time_value > _frame_side.cols) {
                auto times = int(time_value / _frame_side.cols);
                time_value = time_value % _frame_side.cols;
                if (time_to_clean < times) {
                    _frame_side.setTo(cv::Scalar(0, 0, 0));
                    time_to_clean = times; // Reset the flag after cleaning
                }
            }

            auto color = (event.p == 0) ? cv::Vec3b(255, 0, 0) : cv::Vec3b(0, 0, 255); // Blue for p=0, Red for p=1
            if (_axis == X_AXIS && x >= _lower_bound && x <= _upper_bound) {
                // Draw a vertical line for X_AXIS
                _frame.at<cv::Vec3b>(x, _axis_value) = color;
            } else if (_axis == Y_AXIS && y >= _lower_bound && y <= _upper_bound) {
                // Draw a horizontal line for Y_AXIS
                _frame.at<cv::Vec3b>(_axis_value, y) = color;
            }

            _frame_side.at<cv::Vec3b>(_axis == X_AXIS ? x : y, time_value) = color;
        }

        cv::Mat getFrame() {
            std::lock_guard<std::mutex> lock(_mtx);
            cv::Mat frame_copy;
            _frame.copyTo(frame_copy);
            // reset the frame to black
            _frame.setTo(cv::Scalar(0, 0, 0));
            return frame_copy;
        }

        cv::Mat getFrameSide() {
            std::lock_guard<std::mutex> lock(_mtx);
            return _frame_side.clone();
        }

        void editFrame(std::function<void(cv::Mat &)> edit_func) {
            std::lock_guard<std::mutex> lock(_mtx);
            edit_func(_frame_side);
        }


        int time_value = 0; // Time value in the visualization
        int time_scale = 10000; // Scale for time visualization, can be adjusted as needed
    private:
        int _margin{0}, _lower_bound, _upper_bound, _axis_value;
        Axis _axis{X_AXIS}; // Default to X_AXIS
        cv::Mat _frame;
        cv::Mat _frame_side;
        std::mutex _mtx; // Mutex for thread safety
        Metavision::timestamp initial_time = 0;
        int time_to_clean = 0; // Flag to indicate if the time value exceeds the frame width
    };

}

#endif //PROJECT_SLICE_VISUALIZER_HPP
