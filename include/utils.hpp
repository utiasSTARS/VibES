//
// Created by viciopoli on 18/11/24.
//
#ifndef PROJECT_UTILS_HPP
#define PROJECT_UTILS_HPP

#include <algorithm>
#include <sstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <optional>


struct Centroid {
    float t;
    unsigned short x;
    unsigned short y;
};


// UI processing function similar to original
int processUI(int delay_ms) {
    auto then = std::chrono::high_resolution_clock::now();
    int key = cv::waitKey(delay_ms);
    auto now = std::chrono::high_resolution_clock::now();

    // Ensure consistent timing
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
    if (elapsed < delay_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms - elapsed));
    }

    return key;
}

// Mouse callback for tracker selection
void receiveMouseEvent(int event, int x, int y, int flags, void *userdata) {
    auto *callback = reinterpret_cast<std::function<void(int, int)> *>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN && callback) {
        (*callback)(x, y);
    }
}


template<typename T>
inline T rad2Hz(T rad) {
    return rad / (2 * M_PI);
}

template<typename T>
inline T Hz2rad(T hz) {
    return hz * (2 * M_PI);
}


#endif  // PROJECT_UTILS_HPP
