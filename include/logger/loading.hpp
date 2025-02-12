//
// Created by viciopoli on 06/02/25.
//

#ifndef PROJECT_LOADING_H
#define PROJECT_LOADING_H

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>

class LoadingText {
public:
    LoadingText() : _running(false) {}

    // Start displaying the loading text with an animated spinner.
    void loading(const std::string &text) {
        _loadingText = text;
        _running = true;
        // Launch a thread that runs the display method.
        _thread = std::thread(&LoadingText::display, this);
    }

    // Stop displaying the loading text.
    void stop() {
        if (!_running) {
            return;
        }
        _running = false;
        if (_thread.joinable()) {
            _thread.join();
        }
        // Optionally, clear the line or print a "Done" message.
        std::cout << "\r" << _loadingText << "... Done!" << std::endl;
    }

    // Destructor ensures the thread is stopped properly.
    ~LoadingText() {
        stop();
    }

private:
    // This function runs in a separate thread to animate the spinner.
    void display() {
        const char spinner[] = {'|', '/', '-', '\\'};
        int i = 0;
        while (_running) {
            std::cout << "\r" << _loadingText << "... " << spinner[i % 4] << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            ++i;
        }
    }

    std::atomic<bool> _running;  // Flag to control the spinner thread.
    std::string _loadingText;    // The loading message.
    std::thread _thread;         // Thread for the spinner animation.
};


#endif //PROJECT_LOADING_H
