#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/algorithms/flip_x_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include "utils.hpp"
#include "tracker/bin_thread_follower.hpp"
#include "sim/ini_sim.hpp"
#include "visualizer/open3d_visualizer.hpp"
#include "visualizer/ev2image.hpp"

#include "event_frontend/centroid.hpp"
#include "estimator/nufourier_new.hpp"
#include "logger/loading.hpp"
#include "harmeda.h"


// Main example function
int main(int argc, char *argv[]) {
    std::cout << "\033[1;31mThis is a demo for the HARMEDA project.\nPress ESC to close the windows.\033[0m"
              << std::endl;

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

    // we also need to choose an accumulation time and a frame rate (here of 20ms and 50 fps)
    const std::uint32_t acc = 20000;
    double fps = 50;

    // now we can create our frame generator using previous variables
    auto frame_gen = Metavision::PeriodicFrameGenerationAlgorithm(camera_width, camera_height, acc, fps);

    HARMEDA harmeda(10'000, 2, 150);
    int64_t initial_timestamp = -1;
    auto vis = std::make_shared<Open3DVisualizer>();

    // we add the callback that will pass the events to the algo and then the frame generator
    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        // we call the frame generator on the processed events
        frame_gen.process_events(begin, end);

        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            if (initial_timestamp == -1) {
                initial_timestamp = ev->t;
            }
            harmeda.feed(ev->x, ev->y,
                         static_cast<double>(ev->t - initial_timestamp));
            if (harmeda.initialized() && harmeda.size() == 0) {
                // we want to track only one patter in the screen
                const auto &[x_centre, y_centre] = harmeda.getInitialCenter();
                harmeda.add_bin(x_centre, y_centre, std::max(camera_width, camera_height) / 4., vis);
            }
        }
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
    frame_gen.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) { window.show(frame); });

    // start the camera
    cam.start();

    // keep running until the camera is off, the recording is finished or the escape key was pressed
    while (cam.is_running() && !window.should_close()) {
        // we poll events (keyboard, mouse etc.) from the system with a 20ms sleep to avoid using 100% of a CPU's core
        // and we push them into the window where the callback on the escape key will ask the windows to close
        static constexpr std::int64_t kSleepPeriodMs = 20;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
    }

    // the recording is finished or the user wants to quit, stop the camera.
    cam.stop();


    /*

    cv::Size resolution = reader->getEventResolution().value();
    std::cout << "Resolution: " << resolution << std::endl;

    //
    // visualization stuffs
    //
    // open3d vis
    auto vis = std::make_shared<Open3DVisualizer>();
    cv::namedWindow("Accumulated", cv::WINDOW_NORMAL);
    cv::resizeWindow("Accumulated", resolution.width, resolution.height);

    // create accumulator
    dv::EdgeMapAccumulator accumulator(resolution);
    accumulator.setNeutralPotential(0.5f);
    accumulator.setEventContribution(0.25f);
    accumulator.setIgnorePolarity(false);


    // HARMEDA stuffs
    HARMEDA harmeda(10'000, 2, 150);
    int64_t initial_timestamp = -1;

    dv::EventStore events;
    auto start = std::chrono::high_resolution_clock::now();
    while (reader->isRunning()) {
        if (const auto events_dist = reader->getNextEventBatch(); events_dist.has_value()) {
            if (geometry != nullptr) {
                events = geometry->undistortEvents(events_dist.value());
            } else {
                events = events_dist.value();
            }

            auto start = std::chrono::high_resolution_clock::now();
            for (auto &event: events) {
                if (initial_timestamp == -1) {
                    initial_timestamp = event.timestamp();
                }
                harmeda.feed(event.x(), event.y(),
                             Time(event.timestamp() - initial_timestamp));
                if (harmeda.initialized() && harmeda.size() == 0) {
                    // we want to track only one patter in the screen
                    const auto &[x_centre, y_centre] = harmeda.getInitialCenter();
                    harmeda.add_bin(x_centre, y_centre, std::max(resolution.width, resolution.height) / 4., vis);
                }
            }

            accumulator.accumulate(events);
            cv::imshow("Standard", accumulator.generateFrame().image);
            cv::waitKey(1);
            vis->update();
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;

    harmeda.stop();
    std::cout << harmeda << std::endl;

    vis->loop();*/

    return 0;
}