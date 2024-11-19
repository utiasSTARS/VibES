#include <unordered_map>
#include <deque>
#include <iostream>
#include <cmath>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/camera_capture.hpp>
#include <opencv2/highgui.hpp>
#include "include/fourier.hpp"
#include "include/events_freq_calib_pattern.hpp"

#include <open3d/Open3D.h>


int main() {
    // dv::io::MonoCameraRecording reader(
    //         "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/file.aedat4");
    EventsFreqCalibPattern reader(0, cv::Size(640, 480), 500, 40, 40, 0.01, 100);

    std::cout << "Opened an AEDAT4 file which contains data from [" << reader.getCameraName() << "] camera"
              << std::endl;

    cv::Size resolution = *reader.getEventResolution();

    // visualizer

    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Event Camera Visualization", 800, 600);
    auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();

    vis.AddGeometry(point_cloud);

    auto bounding_box = std::make_shared<open3d::geometry::AxisAlignedBoundingBox>(
            Eigen::Vector3d(0, 0, 0.0),
            Eigen::Vector3d(640, 480, 10));

    // Set bounding box color for visibility
    bounding_box->color_ = Eigen::Vector3d(0.0, 1.0, 0.0);  // Green color

    // Add the box to the visualizer
    vis.AddGeometry(bounding_box);


    int64_t x_mean = 0, y_mean = 0, timestamp_mean = 0, timestamp = 0;
    int counter = 0;

    int N = 1e6;  // Number of samples to accumulate for FFT
    double samplingRate = 1e6;   // Fixed sampling rate (Hz)

    FourierFreqEst freqEst(N, samplingRate);

    int iters = 0;
    while (reader.isRunning()) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {

            // Collect events in a buffer with timestamp normalization to seconds
            for (const auto &event: events.value()) {
                if (timestamp == 0) {
                    timestamp = event.timestamp();
                }
                // if (counter > 1000) {
                //         freqEst.feed(x_mean / counter, y_mean / counter);
                //         x_mean = 0;
                //         y_mean = 0;
                //         counter = 0;
                // }
                // x_mean += event.x();
                // y_mean += event.y();

                point_cloud->points_.emplace_back(event.x(), event.y(), (event.timestamp() - timestamp)/10);
                point_cloud->colors_.emplace_back(Eigen::Vector3d(0.1, 0.1, event.polarity()));

                counter++;
            }
        }

        vis.UpdateGeometry(point_cloud);
        vis.PollEvents();
        vis.UpdateRender();
        iters++;
        if (iters > 1000) { break; }
    }

    while (vis.PollEvents()) {
        vis.UpdateRender();
    }

    vis.DestroyVisualizerWindow();

    return 0;
}