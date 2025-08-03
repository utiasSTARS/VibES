//
// Created by viciopoli on 01/08/25.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <string>

class Slider {
private:
    cv::Rect rect;
    double min_val, max_val, val;
    bool dragging;
    int handle_radius;
    cv::Rect track_rect;
    int handle_x, handle_y;

public:
    Slider(const int x, const int y, const int width, const int height, double min_val, double max_val,
           double initial_val)
            : min_val(min_val), max_val(max_val), val(initial_val), dragging(false) {

        rect = cv::Rect(x, y, width, height);
        handle_radius = height / 2;
        track_rect = cv::Rect(x + handle_radius, y + height / 4,
                              width - 2 * handle_radius, height / 2);
        updateHandlePos();
    }

    void updateHandlePos() {
        double ratio = (val - min_val) / (max_val - min_val);
        handle_x = track_rect.x + static_cast<int>(ratio * track_rect.width);
        handle_y = rect.y + rect.height / 2;
    }

    void handleMouseEvent(int event, int x, int y, int flags) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            cv::Rect handle_rect(handle_x - handle_radius, handle_y - handle_radius,
                                 handle_radius * 2, handle_radius * 2);
            if (handle_rect.contains(cv::Point(x, y))) {
                dragging = true;
            }
        } else if (event == cv::EVENT_LBUTTONUP) {
            dragging = false;
        } else if (event == cv::EVENT_MOUSEMOVE && dragging) {
            // Clamp to track bounds
            int clamped_x = std::max(track_rect.x, std::min(track_rect.x + track_rect.width, x));

            // Calculate new value
            double ratio = static_cast<double>(clamped_x - track_rect.x) / track_rect.width;
            val = min_val + ratio * (max_val - min_val);
            updateHandlePos();
        }
    }

    void draw(cv::Mat &img) {
        // Colors
        cv::Scalar gray(128, 128, 128);
        cv::Scalar blue(255, 150, 100);
        cv::Scalar white(255, 255, 255);

        // Draw track
        cv::rectangle(img, track_rect, gray, cv::FILLED);

        // Draw handle
        cv::circle(img, cv::Point(handle_x, handle_y), handle_radius, blue, cv::FILLED);
        cv::circle(img, cv::Point(handle_x, handle_y), handle_radius, white, 2);
    }

    double getValue() const { return val; }

    void updatePosition(int new_y) {
        rect.y = new_y;
        track_rect.y = new_y + rect.height / 4;
        updateHandlePos();
    }
};

// Global variables for mouse callback
Slider *g_frequency_slider = nullptr;

void onMouse(int event, int x, int y, int flags, void *userdata) {
    if (g_frequency_slider) {
        g_frequency_slider->handleMouseEvent(event, x, y, flags);
    }
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    double amplitude = 30.0; // Default amplitude

    if (argc == 2) {
        try {
            amplitude = std::stod(argv[1]);
            if (amplitude <= 0) {
                std::cerr << "Error: Amplitude must be positive" << std::endl;
                return 1;
            }
        } catch (const std::exception &e) {
            std::cerr << "Error: Amplitude must be a number" << std::endl;
            return 1;
        }
    } else if (argc > 2) {
        std::cout << "Usage: " << argv[0] << " [amplitude]" << std::endl;
        std::cout << "Example: " << argv[0] << " 30" << std::endl;
        std::cout << "  amplitude: vibration amplitude in pixels (positive number, default: 30)" << std::endl;
        return 1;
    }

    // Window settings
    int WIDTH = 800;
    int HEIGHT = 600;
    const std::string WINDOW_TITLE = "Vibrating Triangle - Interactive Frequency Control";

    // Colors
    cv::Scalar BLACK(0, 0, 0);
    cv::Scalar WHITE(255, 255, 255);
    cv::Scalar RED(0, 0, 255);  // BGR format in OpenCV
    cv::Scalar DARK_GRAY(50, 50, 50);

    // Triangle settings
    const int TRIANGLE_SIZE = 80;

    // Create window
    cv::namedWindow(WINDOW_TITLE, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_TITLE, WIDTH, HEIGHT);

    // Create frequency slider
    Slider frequency_slider(50, HEIGHT - 120, 300, 30, 0.1, 100.0, 2.5);
    g_frequency_slider = &frequency_slider;

    // Set mouse callback
    cv::setMouseCallback(WINDOW_TITLE, onMouse, nullptr);

    // Animation variables
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Vibrating triangle started with amplitude: " << amplitude << "px" << std::endl;
    std::cout << "Use the frequency slider to adjust vibration speed" << std::endl;
    std::cout << "Press ESC or close window to exit" << std::endl;
    std::cout << "Window is resizable - drag the corners or edges" << std::endl;

    while (true) {
        // Get current window size
//        cv::Rect window_rect = cv::getWindowImageRect(WINDOW_TITLE);
//        if (window_rect.width != WIDTH || window_rect.height != HEIGHT) {
//            WIDTH = window_rect.width > 0 ? window_rect.width : 800; // Ensure positive width
//            HEIGHT = window_rect.height > 0 ? window_rect.height : 600; // Ensure positive height
//            frequency_slider.updatePosition(HEIGHT - 120);
//        }

        // Create image
        cv::Mat img(HEIGHT, WIDTH, CV_8UC3, BLACK);

        // Get current frequency from slider
        double frequency = frequency_slider.getValue();

        // Calculate center position
        int CENTER_X = WIDTH / 2;
        int CENTER_Y = HEIGHT / 2;

        // Calculate elapsed time
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
        double elapsed_time = duration.count() / 1000.0;

        // Calculate vibration offset using sine wave
        double vibration_offset = amplitude * std::sin(2 * M_PI * frequency * elapsed_time);
        double vibration_offset_Y = amplitude * std::sin(2 * M_PI * frequency * elapsed_time+ M_PI / 2); // For vertical vibration

        // Calculate triangle position (vibrating horizontally)
        int triangle_x = CENTER_X + static_cast<int>(vibration_offset) - TRIANGLE_SIZE / 2;
        int triangle_y = CENTER_Y + static_cast<int>(vibration_offset_Y) - TRIANGLE_SIZE / 2;

        // Draw the vibrating triangle
        std::vector<cv::Point> triangle_points = {
                cv::Point(triangle_x + TRIANGLE_SIZE / 2, triangle_y),  // Top point
                cv::Point(triangle_x, triangle_y + TRIANGLE_SIZE),     // Bottom left
                cv::Point(triangle_x + TRIANGLE_SIZE, triangle_y + TRIANGLE_SIZE)  // Bottom right
        };

        cv::fillPoly(img, triangle_points, RED);

        // Draw control panel background
        cv::Rect panel_rect(0, HEIGHT - 150, WIDTH, 150);
        cv::rectangle(img, panel_rect, DARK_GRAY, cv::FILLED);
        cv::line(img, cv::Point(0, HEIGHT - 150), cv::Point(WIDTH, HEIGHT - 150), WHITE, 2);

        // Draw frequency slider
        frequency_slider.draw(img);

        // Draw info text
        std::string freq_text = "Frequency: " + std::to_string(frequency).substr(0, 4) + " Hz";
        std::string amp_text = "Amplitude: " + std::to_string(static_cast<int>(amplitude)) + "px";

        cv::putText(img, freq_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2);
        cv::putText(img, amp_text, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2);

        // Draw slider label
        cv::putText(img, "Frequency (Hz):", cv::Point(50, HEIGHT - 125),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1);

        // Draw frequency value near slider
        std::string freq_value = std::to_string(frequency).substr(0, 4);
        cv::putText(img, freq_value, cv::Point(360, HEIGHT - 95),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1);

        // Instructions
        cv::putText(img, "Drag slider to adjust frequency • Press ESC to exit",
                    cv::Point(10, HEIGHT - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1);

        // Display the image
        cv::imshow(WINDOW_TITLE, img);

        // Handle key events
        int key = cv::waitKey(1); // ~75 FPS
        if (key == 27 || key == 'q' || key == 'Q') { // ESC or Q
            break;
        }

        // Check if window was closed
//        if (cv::getWindowProperty(WINDOW_TITLE, cv::WND_PROP_VISIBLE) < 1) {
//            break;
//        }
    }

    // Cleanup
    cv::destroyAllWindows();
    std::cout << "Animation closed" << std::endl;

    return 0;
}