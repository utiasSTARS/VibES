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

    // create windows
    cv::namedWindow("Standard", cv::WINDOW_NORMAL);
    cv::namedWindow("Compensated", cv::WINDOW_NORMAL);

    cv::resizeWindow("Standard", resolution.width, resolution.height);
    cv::resizeWindow("Compensated", resolution.width, resolution.height);

    cv::Mat compensated_frame = cv::Mat::zeros(resolution, CV_8UC3);

    // create accumulator
    dv::EdgeMapAccumulator accumulator(resolution);
    accumulator.setNeutralPotential(0.5f);
    accumulator.setEventContribution(0.25f);
    accumulator.setIgnorePolarity(false);

    // open3d vis
    auto vis = std::make_shared<Open3DVisualizer>();


    // HARMEDA stuffs

    // create the bins for tracking regions in the image plane
    std::vector<std::shared_ptr<BinThreadFollower>> bins;

    // initialize the NUFFT to estimate the frequency
    FourierFreqEst fourier(1'000, 200, 800);

    double estimated_freq = 0;
    double phase_shift = 0;
    double amplitude = 0;

    CentroidCalculation centroid;

    // read the events
    bool initialized = false;
    dv::EventStore events;

    double c_x_estimate = 0, c_y_estimate = 0;

    int bin_h = 2 * 40, bin_w = 2 * 40;

    std::shared_ptr<BinThreadFollower> bin_follower;

    // track the bins
    auto last_time = std::chrono::high_resolution_clock::now();

    int64_t initial_timestamp = 0;

    LoadingText loading;
    loading.loading("Initializing niuFFT...");

    while (reader->isRunning()) {
        if (const auto events_dist = reader->getNextEventBatch(); events_dist.has_value()) {
            if (geometry != nullptr) {
                events = geometry->undistortEvents(events_dist.value());
            } else {
                events = events_dist.value();
            }

            if (initial_timestamp == 0) {
                initial_timestamp = events.getLowestTime();
            }

            if (initialized) { accumulator.accumulate(events); }

            for (const auto &event: events) {
                if (auto sample = centroid.feed(event.x(), event.y(),
                                                Time(event.timestamp() - initial_timestamp)); sample.has_value()) {
                    auto [s_x, s_y, s_t] = sample.value();

                    if (!initialized && fourier.feed(s_x, s_y, s_t)) {
                        estimated_freq = fourier.getMainFreqRad();
                        phase_shift = fourier.getPhaseShift();
                        amplitude = fourier.getAmplitude();

                        initialized = true;

                        auto offset_est = fourier.getOffset();
                        c_x_estimate = get<0>(offset_est);
                        c_y_estimate = get<1>(offset_est);

                        bin_follower = std::make_shared<BinThreadFollower>(3, compensated_frame, vis, bin_w, bin_h,
                                                                           0.1,
                                                                           0.1,
                                                                           estimated_freq,
                                                                           c_x_estimate, c_y_estimate,
                                                                           phase_shift,
                                                                           amplitude);
                        loading.stop();
                    }
                    if (!initialized) { continue; }

                    auto highest_time = events.getHighestTime();
                    auto lowest_time = events.getLowestTime();
                    auto time_diff = (highest_time - lowest_time) / 1e6;

                    bin_follower->add_event(s_x, s_y, s_t);


                    cv::imshow("Standard", accumulator.generateFrame().image);
                    cv::imshow("Compensated", compensated_frame);
                    // std::cout << *bin_follower << std::endl;

                    if (cv::waitKey(1) == 27) {
                        // stop and exit
                        for (auto &bin: bins) {
                            bin->stop();
                        }
                        std::cout << bin_follower.get() << std::endl;
                        return 0;
                    }

                    // Measure the time elapsed for this loop iteration
                    auto end_time = std::chrono::high_resolution_clock::now();
                    const auto loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - last_time);
                    last_time = end_time; // Update for next iteration

                    // Calculate and print frequency in Hz
                    const auto loop_time = loop_duration.count() / 1e6;
                    const double frequency = 1.0 / loop_time; // Convert microseconds to seconds
                    // std::cout << "\rLoop frequency: " << frequency << " Hz, time diff: " << (time_diff - loop_time);
                    // print ekf estimate
                    std::cout << "\rEKF: " << *bin_follower->getEKF();
                    std::cout.flush();

                    vis->update();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));

                }
            }
        }
    }

    for (auto &bin: bins) {
        bin->stop();
    }
    vis->loop();

    return 0;
}