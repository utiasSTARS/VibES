#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/camera_capture.hpp>
#include <dv-processing/camera/calibration_set.hpp>

#include "utils.hpp"
#include "tracker/bin_thread_follower.hpp"
#include "sim/ini_sim.hpp"
#include "visualizer/open3d_visualizer.hpp"
#include "visualizer/ev2image.hpp"

#include "event_frontend/centroid.hpp"
#include "estimator/nufourier_new.hpp"
#include "data_loaders/dvs_loader.h"
#include "logger/loading.hpp"
#include "harmeda.h"


// Main example function
int main(int argc, char *argv[]) {
    std::cout << "\033[1;31mThis is a demo for the HARMEDA project.\nPress ESC to close the windows.\033[0m"
              << std::endl;

    auto data_loader = DVSLoader(argc, argv);
    auto reader = data_loader.getReader();
    auto geometry = data_loader.getGeometry();

    cv::Size resolution = reader->getEventResolution().value();
    std::cout << "Resolution: " << resolution << std::endl;

    //
    // visualization stuffs
    //
    // open3d vis
    // auto vis = std::make_shared<Open3DVisualizer>();
    cv::namedWindow("Accumulated", cv::WINDOW_NORMAL);
    cv::resizeWindow("Accumulated", resolution.width, resolution.height);

    // create accumulator
    dv::EdgeMapAccumulator accumulator(resolution);
    accumulator.setNeutralPotential(0.5f);
    accumulator.setEventContribution(0.25f);
    accumulator.setIgnorePolarity(false);


    // HARMEDA stuffs
    dv::EventStore events;

    HARMEDA harmeda(10'000, 2, 150);
    int64_t initial_timestamp = -1;

    auto start = std::chrono::high_resolution_clock::now();
    while (reader->isRunning()) {
        if (const auto events_dist = reader->getNextEventBatch(); events_dist.has_value()) {
            if (geometry != nullptr) {
                events = geometry->undistortEvents(events_dist.value());
            } else {
                events = events_dist.value();
            }
            // accumulator.accumulate(events);
            // cv::imshow("Standard", accumulator.generateFrame().image);
            // cv::waitKey(1);

            auto start = std::chrono::high_resolution_clock::now();
            for (auto &event: events) {
                if (initial_timestamp == -1) {
                    initial_timestamp = event.timestamp();
                }
                // time this function in ms
                // auto start = std::chrono::high_resolution_clock::now()
                harmeda.feed(event.x(), event.y(),
                             Time(event.timestamp() - initial_timestamp));



                // vis->update();
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;

    harmeda.stop();
    std::cout << harmeda << std::endl;

    // vis->loop();

    return 0;
}