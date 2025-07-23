#include <iostream>
#include <vector>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <opencv2/imgproc.hpp>

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include "sim/ini_sim.hpp"
#include "event_frontend/undistort.hpp"

// Main example function
int main(int argc, char *argv[]) {

    Metavision::Camera cam;       // create the camera
    if (argc >= 2) {
        // if we passed a file path, open it
        cam = Metavision::Camera::from_file(argv[1]);
    } else {
        // open the first available camera
        cam = Metavision::Camera::from_first_available();
    }

    // Recording state
    bool is_recording = false;

    // Get camera resolution for frame generation
    int camera_width = cam.geometry().width();
    int camera_height = cam.geometry().height();

    // Frame generation parameters
    const std::uint32_t acc = 20000;  // accumulation time in microseconds
    double fps = 20;

    // Create frame generator
    auto frame_gen = Metavision::PeriodicFrameGenerationAlgorithm(camera_width, camera_height, acc, fps);

    // Add callback that processes events for frame generation
    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        // Process the actual events from begin to end
        frame_gen.process_events(begin, end);
    });

    // Create window for visualization
    Metavision::Window window("Metavision SDK Get Started", camera_width, camera_height,
                              Metavision::BaseWindow::RenderMode::BGR);

    // Set keyboard callback for window controls and recording toggle
    window.set_keyboard_callback(
            [&](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE) {
                    if (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q) {
                        window.set_close_flag();
                    } else if (key == Metavision::UIKeyEvent::KEY_R) {
                        // Toggle recording
                        if (is_recording) {
                            cam.stop_recording();
                            is_recording = false;
                            std::cout << "Recording stopped." << std::endl;
                        } else {
                            // Generate new timestamp for recording filename
                            auto t = std::time(nullptr);
                            auto tm = *std::localtime(&t);
                            std::ostringstream oss;
                            oss << std::put_time(&tm, "%Y%m%d_%H%M%S");

                            cam.start_recording(
                                    "/home/viciopoli/datasets/event_harmeda/harmeda_dataset/log_" + oss.str() + ".raw");
                            is_recording = true;
                            std::cout << "Recording started: log_" << oss.str() << ".raw" << std::endl;
                        }
                    }
                }
            });

    // Set frame generator output callback to display frames with recording indicator
    frame_gen.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
        // Draw red recording dot if recording is active
        if (is_recording) {
            int dot_radius = 10;
            cv::Point center(frame.cols - 20, 20);  // Top right corner
            cv::circle(frame, center, dot_radius, cv::Scalar(0, 0, 255), -1);  // Red filled circle
        }
        window.show(frame);
    });

    // Start the camera
    try {
        cam.start();
        std::cout << "Camera started successfully." << std::endl;
        std::cout << "Press 'R' to start/stop recording, ESC or Q to quit." << std::endl;
    } catch (const Metavision::CameraException &e) {
        std::cerr << "Error starting camera: " << e.what() << std::endl;
        return 1;
    }

    // Main loop: keep running until camera stops or user quits
    while (cam.is_running() && !window.should_close()) {
        static constexpr std::int64_t kSleepPeriodMs = 5;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
    }

    // Stop camera and any active recording
    std::cout << "Stopping camera";
    if (is_recording) {
        std::cout << " and recording";
        cam.stop_recording();
    }
    std::cout << "..." << std::endl;
    cam.stop();

    return 0;
}