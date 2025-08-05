//
// Created by viciopoli on 30/06/25.
// Modified to include scrolling temporal visualization
//

#ifndef PROJECT_SLICE_VISUALIZER_HPP
#define PROJECT_SLICE_VISUALIZER_HPP

#include <metavision/sdk/driver/camera.h>
#include <opencv2/opencv.hpp>
#include <cstring> // For std::memmove

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

            // Add reference lines at regular intervals (every 0.1 seconds)
            _addTimeReferenceLines();

            _axis_value = axis_value;
            _lower_bound = axis_value - _margin;
            _upper_bound = axis_value + _margin;

            // Initialize scrolling parameters
            _current_time_column = 0;  // Start at leftmost column
            _last_scroll_time = 0;
            _scroll_interval_us = 100; // Scroll every 0.1 seconds (100ms)
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
                _last_scroll_time = event.t;
            }

            _processEvent(event.x, event.y, event.p, event.t);
        }

        void feed(const unsigned short x, const unsigned short y, short p, const Metavision::timestamp t) {
            std::lock_guard<std::mutex> lock(_mtx);
            if (initial_time == 0) {
                initial_time = t; // Set the initial time on the first event
                _last_scroll_time = t;
            }

            _processEvent(x, y, p, t);
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

        // Method to manually trigger scrolling (useful for testing or external control)
        void forceScroll() {
            std::lock_guard<std::mutex> lock(_mtx);
            _scrollFrameSide();
        }

        int time_value = 0; // Time value in the visualization
        int time_scale = 10000; // Scale for time visualization, can be adjusted as needed

    private:
        void _processEvent(unsigned short x, unsigned short y, short p, Metavision::timestamp t) {
            // Check if we need to scroll based on time
            if (t - _last_scroll_time >= _scroll_interval_us) {
                _scrollFrameSide();
                _last_scroll_time = t;
            }

            auto color = (p == 0) ? cv::Vec3b(255, 0, 0) : cv::Vec3b(0, 0, 255); // Blue for p=0, Red for p=1

            // Update the main frame (spatial view)
            if (_axis == X_AXIS && x >= _lower_bound && x <= _upper_bound) {
                _frame.at<cv::Vec3b>(x, _axis_value) = color;
            } else if (_axis == Y_AXIS && y >= _lower_bound && y <= _upper_bound) {
                _frame.at<cv::Vec3b>(_axis_value, y) = color;
            }

            // Update the temporal frame (always draw at the current time column)
            int spatial_coord = (_axis == X_AXIS) ? x : y;
            if (spatial_coord >= 0 && spatial_coord < _frame_side.rows &&
                _current_time_column >= 0 && _current_time_column < _frame_side.cols) {
                _frame_side.at<cv::Vec3b>(spatial_coord, _current_time_column) = color;
            }
        }

        void _scrollFrameSide() {
            if (_frame_side.cols <= 1) return;

            // Use memmove-like operation to shift data efficiently
            // This shifts all data one column to the right
            for (int row = 0; row < _frame_side.rows; ++row) {
                // Get pointer to the row
                cv::Vec3b* row_ptr = _frame_side.ptr<cv::Vec3b>(row);
                // Shift all pixels in this row one position to the right
                std::memmove(row_ptr + 1, row_ptr, (_frame_side.cols - 1) * sizeof(cv::Vec3b));
                // Clear the first pixel in the row (where new data will go)
                row_ptr[0] = cv::Vec3b(0, 0, 0);
            }

            // Update current time column to point to the leftmost column
            _current_time_column = 0;

            // Redraw time reference lines after scrolling
            _addTimeReferenceLines();
        }

        void _addTimeReferenceLines() {
            // Add subtle reference lines every 100 pixels (adjustable)
            const int line_interval = 100;
            const cv::Vec3b line_color(32, 32, 32); // Very dark gray lines, less intrusive

            for (int col = line_interval; col < _frame_side.cols; col += line_interval) {
                // Only draw on every 10th row to make lines less intrusive
                for (int row = 0; row < _frame_side.rows; row += 10) {
                    if (_frame_side.at<cv::Vec3b>(row, col) == cv::Vec3b(0, 0, 0)) {
                        _frame_side.at<cv::Vec3b>(row, col) = line_color;
                    }
                }
            }
        }

        int _margin{0}, _lower_bound, _upper_bound, _axis_value;
        Axis _axis{X_AXIS}; // Default to X_AXIS
        cv::Mat _frame;
        cv::Mat _frame_side;
        std::mutex _mtx; // Mutex for thread safety
        Metavision::timestamp initial_time = 0;

        // Scrolling-related members
        int _current_time_column;
        Metavision::timestamp _last_scroll_time;
        Metavision::timestamp _scroll_interval_us; // Scroll interval in microseconds
    };
}

#endif //PROJECT_SLICE_VISUALIZER_HPP