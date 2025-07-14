//
// Created by viciopoli on 13/07/25.
//

#ifndef PROJECT_INTERACTIVE_FRAME_VISUALIZER_HPP
#define PROJECT_INTERACTIVE_FRAME_VISUALIZER_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <functional>

class EventFrameVisualizer {
private:
    cv::Mat current_frame;
    std::string window_name;
    std::vector<cv::Point2i> tracker_points;
    static bool block_;

    // Callback function type for new tracker points
    using TrackerPointCallback = std::function<void(const int x, const int y)>;
    TrackerPointCallback point_callback;

    // Your existing tracker initialization function
    void initialize_tracker(int x, int y) {
        tracker_points.emplace_back(x, y);

        // Call the callback if it's set
        if (point_callback) {
            point_callback(x, y);
        }
    }

    // Mouse callback function (must be static)
    static void mouse_callback(int event, int x, int y, int flags, void *userdata) {
        if (block_) return;
        EventFrameVisualizer *visualizer = static_cast<EventFrameVisualizer *>(userdata);

        if (event == cv::EVENT_LBUTTONDOWN) {
            visualizer->initialize_tracker(x, y);
            visualizer->update_display();
        }
    }

    void update_display() {
        if (current_frame.empty()) return;

        cv::Mat display_frame = current_frame.clone();

        // Draw tracker initialization points
        for (const auto &point: tracker_points) {
            cv::circle(display_frame, point, 5, cv::Scalar(0, 255, 0), -1);
            cv::circle(display_frame, point, 10, cv::Scalar(0, 255, 0), 2);
        }

        // Add instructions
        cv::putText(display_frame, "Click to initialize tracker",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2);
        cv::putText(display_frame, "Press 'c' to clear, 'ESC' to exit",
                    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2);

        cv::imshow(window_name, display_frame);
    }

public:
    EventFrameVisualizer(const std::string &win_name = "Event Frame Visualizer")
            : window_name(win_name) {
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(window_name, mouse_callback, this);
    }

    ~EventFrameVisualizer() {
        cv::destroyWindow(window_name);
    }

    // Set callback for new tracker points
    void set_tracker_point_callback(TrackerPointCallback callback) {
        point_callback = callback;
    }

    void block() {
        block_ = true;
    }

    void unblock() {
        block_ = false;
    }

    // Remove callback
    void clear_tracker_point_callback() {
        point_callback = nullptr;
    }

    void set_frame(const cv::Mat &frame) {
        current_frame = frame.clone();
        update_display();
    }

    void run_interactive() {
        if (current_frame.empty()) {
            std::cerr << "No frame loaded. Call set_frame() first." << std::endl;
            return;
        }

        std::cout << "Interactive mode started. Click on the image to initialize trackers." << std::endl;
        std::cout << "Press 'c' to clear all tracker points, 'ESC' to exit." << std::endl;

        while (true) {
            int key = cv::waitKey(30) & 0xFF;

            if (key == 27) { // ESC key
                break;
            } else if (key == 'c' || key == 'C') {
                clear_trackers();
            }
        }
    }

    void clear_trackers() {
        tracker_points.clear();
        std::cout << "All tracker points cleared." << std::endl;
        update_display();
    }

    // Get all tracker points
    [[nodiscard]] std::vector<cv::Point2i> get_tracker_points() const {
        return tracker_points;
    }
};

bool EventFrameVisualizer::block_ = false;

#endif //PROJECT_INTERACTIVE_FRAME_VISUALIZER_HPP
