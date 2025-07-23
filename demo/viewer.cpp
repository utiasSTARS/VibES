#include <iostream>
#include <vector>
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

    // to visualize the events, we will need to build frames and render them.
    // building frame will be done with a frame generator that will accumulate the events over time.
    // we need to provide it the camera resolution that we can retrieve from the camera instance
    int camera_width = cam.geometry().width();
    int camera_height = cam.geometry().height();

    // unidistortion
    auto filepath = "/home/viciopoli/datasets/event_harmeda/intrinsics.json";
    Undistort undistort(filepath);

    // HARMEDA stuffs

    // visualization
    const std::uint32_t acc = 20000;
    double fps = 20;

    auto frame_gen = Metavision::PeriodicFrameGenerationAlgorithm(camera_width, camera_height, acc, fps);

    // we add the callback that will pass the events to the frame generator
    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::vector<Metavision::EventCD> events;
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            // check x and y are within the camera resolution
            const auto &[x, y] = undistort(ev->x, ev->y);
            if (x < 0 || x >= camera_width || y < 0 || y >= camera_height) {
                continue;
            }
            events.emplace_back(x, y, ev->p, ev->t);
        }
        frame_gen.process_events(events.begin(), events.end());
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
    while (cam.is_running() && !window.should_close()) {
        static constexpr std::int64_t kSleepPeriodMs = 5;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
    }


    // the recording is finished or the user wants to quit, stop the camera.
    cam.stop();

    return 0;
}