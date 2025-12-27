/**
 * @file utils.hpp
 * @brief Common utility structures and helper functions for UI and signal processing.
 *
 * This file contains shared data structures (like Centroid), OpenCV UI helpers,
 * and mathematical conversion templates used throughout the tracking pipeline.
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

#ifndef PROJECT_UTILS_HPP
#define PROJECT_UTILS_HPP

#include <algorithm>
#include <sstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <optional>
#include <chrono>
#include <thread>
#include <functional>
#include <opencv2/highgui.hpp> // Required for cv::waitKey and Mouse events

/**
 * @struct Centroid
 * @brief Lightweight representation of a cluster center or tracked object.
 *
 * Used to pass simplified event data between the tracker (HASTE) and the
 * estimator (NUFFT).
 */
struct Centroid {
    float t;           ///< Timestamp in seconds.
    unsigned short x;  ///< X coordinate (pixel).
    unsigned short y;  ///< Y coordinate (pixel).

    /**
     * @brief Constructor for easy emplacement.
     */
    Centroid(double t_sec, double x_pos, double y_pos)
        : t(static_cast<float>(t_sec)),
          x(static_cast<unsigned short>(x_pos)),
          y(static_cast<unsigned short>(y_pos)) {}

    // Default constructor
    Centroid() : t(0), x(0), y(0) {}
};

/**
 * @brief Handles OpenCV UI events with consistent timing.
 *
 * Wraps `cv::waitKey` but ensures that the function takes *at least*
 * `delay_ms` to execute. This helps maintain a maximum frame rate limit
 * when the processing loop is faster than the desired display rate.
 *
 * @param delay_ms Minimum duration to wait in milliseconds.
 * @return int The ASCII code of the key pressed, or -1 if no key was pressed.
 */
inline int processUI(int delay_ms) {
    auto then = std::chrono::high_resolution_clock::now();

    // OpenCV waitKey handles window events (redraw, resize, input)
    int key = cv::waitKey(delay_ms);

    auto now = std::chrono::high_resolution_clock::now();

    // Calculate how much time we actually spent in waitKey
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();

    // If waitKey returned early (or was faster than expected), sleep the difference
    // to enforce a stable frame pacing.
    if (elapsed < delay_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms - elapsed));
    }

    return key;
}

/**
 * @brief Bridge function to connect OpenCV C-style mouse callbacks to C++ std::functions.
 *
 * Usage:
 * @code
 * std::function<void(int, int)> my_callback = [&](int x, int y) { ... };
 * cv::setMouseCallback("Window", receiveMouseEvent, &my_callback);
 * @endcode
 *
 * @param event The OpenCV mouse event type (e.g., EVENT_LBUTTONDOWN).
 * @param x Mouse X coordinate.
 * @param y Mouse Y coordinate.
 * @param flags Event flags (Ctrl, Shift, etc.).
 * @param userdata Pointer to the `std::function<void(int,int)>` object.
 */
inline void receiveMouseEvent(int event, int x, int y, int flags, void *userdata) {
    auto *callback = reinterpret_cast<std::function<void(int, int)> *>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN && callback) {
        (*callback)(x, y);
    }
}

/**
 * @brief Converts Angular Frequency (rad/s) to Hertz (Hz).
 * @tparam T Floating point type (float, double).
 */
template<typename T>
inline T rad2Hz(T rad) {
    return rad / (2 * M_PI);
}

/**
 * @brief Converts Hertz (Hz) to Angular Frequency (rad/s).
 * @tparam T Floating point type (float, double).
 */
template<typename T>
inline T Hz2rad(T hz) {
    return hz * (2 * M_PI);
}

#endif  // PROJECT_UTILS_HPP