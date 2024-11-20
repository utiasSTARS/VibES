//
// Created by viciopoli on 18/11/24.
//

#include <iostream>
#include "include/sim/ini_sim.hpp"
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/camera_capture.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void print_event(const dv::Event &event) {
    std::cout << "Timestamp: " << event.timestamp() << ", "
              << "X: " << event.x() << ", "
              << "Y: " << event.y() << ", "
              << "Polarity: " << event.polarity() << '\n';
}

int main() {
    // Initialize test parameters
    int64_t initial_timestamp = 0; // Start at 0 microseconds
    cv::Size size(640, 480);       // Event frame size: 640x480
    double omega = 100;             // 700 Hz sinusoidal motion
    double amplitude_x = 10.0;     // X amplitude of 10 pixels
    double amplitude_y = 5.0;      // Y amplitude of 5 pixels
    double phi = 0.0;              // No phase shift


    // Set up accumulators and display windows
    dv::EdgeMapAccumulator accumulator(size);
    accumulator.setNeutralPotential(0.5f);
    accumulator.setEventContribution(0.25f);
    accumulator.setIgnorePolarity(false);


    // Create an instance of the class
    EventsFreqCalibPattern reader(initial_timestamp, size, omega, amplitude_x, amplitude_y, phi, 5);


    while (reader.isRunning()) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {

            std::cout << "Generated " << events->size() << " events over 10 ms.\n";
            std::cout << "Timestamp: " << events->getLowestTime() << " to " << events->getHighestTime() << '\n';

            accumulator.accumulate(events.value());
            auto acc_frame = accumulator.generateFrame();
            cv::imshow("Accumulator", acc_frame.image);
            cv::waitKey(1);
        } else {
            std::cerr << "Failed to generate events.\n";
        }
    }

    return 0;
}
