#include <unordered_map>
#include <deque>
#include <iostream>
#include <cmath>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/camera_capture.hpp>
#include <opencv2/highgui.hpp>
#include "include/fourier.hpp"
#include "include/events_freq_calib_pattern.hpp"
#include "include/helpers.h"

#include <open3d/Open3D.h>


int main() {
    // dv::io::MonoCameraRecording reader(
    //         "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/file.aedat4");
    EventsFreqCalibPattern reader(0, cv::Size(640, 480), 700, 40, 40, 0.01, 100, false);

    std::cout << "Opened an AEDAT4 file which contains data from [" << reader.getCameraName() << "] camera"
              << std::endl;

    cv::Size resolution = *reader.getEventResolution();

    // visualizer
    Open3DVisualizer vis;

    std::vector<std::tuple<double, double, int64_t>> events_buffer;
    int64_t x_mean = 0, y_mean = 0, timestamp_mean = 0, timestamp = 0;
    int counter = 0;

    int N = 10'000;  // Number of samples to accumulate for FFT
    int samplingRate = 1e5;   // Fixed sampling rate (Hz)

    FourierFreqEst freqEst(N, samplingRate);

    int iters = 0;
    int d_t = 0;
    int64_t prev_time = 0;
    while (reader.isRunning()) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {

            // Collect events in a buffer with timestamp normalization to seconds
            for (const auto &event: events.value()) {
                if (timestamp == 0) {
                    timestamp = event.timestamp();
                }
                if (prev_time != 0 && prev_time != event.timestamp()) {
                    events_buffer.emplace_back(x_mean / counter, y_mean / counter, d_t / (counter));

                    // interpolate events_buffer at sampling rate
                    if (events_buffer.size() > 1) {
                        auto interpolated = interpolate_events(events_buffer, 1e6 / samplingRate);
                        if (interpolated.has_value()) {
                            for (const auto &e: *interpolated) {
                                freqEst.feed(std::get<0>(e), std::get<1>(e));


                                if (iters < 100) {
                                    vis.addPoint(std::get<0>(e), std::get<1>(e), std::get<2>(e), event.polarity());
                                }
                            }
                        }
                    }

                    // if (iters < 100) {
                    //     point_cloud->points_.emplace_back(x_mean / counter, y_mean / counter, d_t / (counter));
                    //     point_cloud->colors_.emplace_back(Eigen::Vector3d(0.1, 0.1, event.polarity()));
                    // }
                    x_mean = 0;
                    y_mean = 0;
                    counter = 0;
                }
                x_mean += event.x();
                y_mean += event.y();
                d_t = event.timestamp() - timestamp;
                if (d_t < 0)throw std::runtime_error("Negative time difference");
                prev_time = event.timestamp();

                counter++;
            }
        }
        vis.update();
        iters++;
    }

    vis.loop();


    return 0;
}