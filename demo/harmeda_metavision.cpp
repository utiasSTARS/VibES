#include <iostream>
#include <vector>
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <csignal>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>


#include "utils.hpp"
#include "sim/ini_sim.hpp"
#include "visualizer/open3d_visualizer.hpp"
#include "visualizer/ev2image.hpp"

#include "logger/loading.hpp"
#include "harmeda.h"

std::atomic<bool> signal_caught{false};

[[maybe_unused]] void signalHandler(int s) {
    std::cout << "Caught signal " << s << std::endl;
    signal_caught = true;
}

// Main example function
int main(int argc, char *argv[]) {
    std::cout << "\033[1;31mThis is a demo for the HARMEDA project.\nPress ESC to close the windows.\033[0m"
              << std::endl;

    // register the signal handler
    signal(SIGINT, signalHandler);

    Metavision::Camera cam;       // create the camera

    if (argc >= 2) {
        // if we passed a file path, open it
        cam = Metavision::Camera::from_file(argv[1]);
    } else {
        // open the first available camera
        cam = Metavision::Camera::from_first_available();
    }

    // to visualize the events, we will need to build frames and render them.
    // building frame will be done with a frame generator that will accumulate the events over time.
    // we need to provide it the camera resolution that we can retrieve from the camera instance
    int camera_width = cam.geometry().width();
    int camera_height = cam.geometry().height();


    HARMEDA harmeda(100, 5, 1000);
    int64_t initial_timestamp = -1;
    auto vis = std::make_shared<Open3DVisualizer>(camera_width, camera_height);

    // we add the callback that will pass the events to the algo and then the frame generator
    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (initial_timestamp == -1) {
            initial_timestamp = begin->t;
        }
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            harmeda.feed(ev->x, ev->y,
                         static_cast<double>(ev->t - initial_timestamp) / 1e6);
        }

        if (harmeda.initialized() && harmeda.size() == 0) {
            // we want to track only one patter in the screen
            const auto &[x_centre, y_centre] = harmeda.getInitialCenter();
            harmeda.add_bin(x_centre, y_centre, std::max(camera_width, camera_height) / 6., vis);
        }
    });

    // visualization

    const std::uint32_t acc = 20000;
    double fps = 100;

    auto frame_gen = Metavision::PeriodicFrameGenerationAlgorithm(camera_width, camera_height, acc, fps);

    // we add the callback that will pass the events to the frame generator
    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        frame_gen.process_events(begin, end);
    });

    // to render the frames, we create a window using the Window class of the UI module
    Metavision::Window window("Metavision SDK Get Started", camera_width, camera_height,
                              Metavision::BaseWindow::RenderMode::BGR);

    // we set a callback on the windows to close it when the Escape or Q key is pressed
    window.set_keyboard_callback(
            [&window](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE &&
                    (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                    window.set_close_flag();
                }
            });

    // we set a callback on the frame generator so that it calls the window object to display the generated frames
    frame_gen.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
        if (harmeda.initialized()) {
            // add a text to the frame
            cv::putText(frame, "HARMEDA: ON", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0),
                        2);
        } else {
            cv::putText(frame, "HARMEDA: OFF", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255),
                        2);
        }
        harmeda.draw(frame);
        window.show(frame);
    });



    // start the camera
    try {
        cam.start();
    } catch (const Metavision::CameraException &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // keep running until the camera is off, the recording is finished or the escape key was pressed
    while (cam.is_running() && !signal_caught.load(std::memory_order::relaxed) && !window.should_close()) {
        // we need to update the visualizer
        vis->update();

        static constexpr std::int64_t kSleepPeriodMs = 10;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
    }

    // the recording is finished or the user wants to quit, stop the camera.
    cam.stop();

    vis->loop();

    return 0;
}