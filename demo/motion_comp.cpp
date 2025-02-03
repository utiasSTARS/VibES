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
#include "nufourier.hpp"
#include "bin_thread_follower.hpp"
#include "sim/ini_sim.hpp"
#include "open3d_visualizer.hpp"
#include "ev2image.hpp"

// Main example function
int main(int argc, char *argv[]) {
    std::cout << "\033[1;31mThis is a demo for the HARMEDA project.\nPress ESC to close the windows.\033[0m"
              << std::endl;
    bool NO_SIM = true;
    std::unique_ptr<dv::io::CameraInputBase> reader;
    if (argc >= 2) {
        std::string arg1 = argv[1];

        // Handle different cases
        if (arg1 == "help") {
            std::cout << "Usage: " << argv[0] << " [help|camera|file]\n";
            std::cout << "  help    - Display this help message\n";
            std::cout << "  camera  - Read from camera\n";
            std::cout << "  file    - Read from a specified file\n";
            return 0;
        } else if (arg1 == "camera") {
            std::cout << "Reading from camera.\n";
            reader = std::make_unique<dv::io::CameraCapture>();
        } else {
            std::cout << "Reading from file: " << arg1 << std::endl;
            reader = std::make_unique<dv::io::MonoCameraRecording>(arg1);
            NO_SIM = false; // TODO: Change this, if the calibration file is passed then it's loaded
        }
    } else {
        NO_SIM = false;
        int target_freq = 700;
        std::cout << "No file provided. Using simulator with freq " << target_freq << " rad/s, " << rad2Hz(target_freq)
                  << " Hz" << std::endl;
        reader = std::make_unique<EventsFreqCalibPattern>(0, cv::Size(640, 480), target_freq, 5, 5, 0., false);
    }

    // if not simulation try to load the camera calibration
    std::shared_ptr<dv::camera::CameraGeometry> geometry;
    if (NO_SIM) {
        try {
            const auto calibrationSet = dv::camera::CalibrationSet::LoadFromFile("../camera/calib.xml");
            dv::camera::calibrations::CameraCalibration dvx_calib;

            if (!calibrationSet.getCameraList().empty()) {
                const auto &calibs = calibrationSet.getCameraCalibrations();
                dvx_calib = calibs.begin()->second;
                std::cout << "Found calibration for camera with name [" << dvx_calib.name << "]" <<
                          std::endl;
            }

            geometry = std::make_shared<dv::camera::CameraGeometry>(dvx_calib.getCameraGeometry());
        } catch (const std::exception &e) {
            std::cerr << "Failed to load camera calibration: " << e.what() << std::endl;
            return 1;
        }
    }

    cv::Size resolution = reader->getEventResolution().value();
    std::cout << "Resolution: " << resolution << std::endl;

    // visualization tool
    cv::namedWindow("Standard", cv::WINDOW_NORMAL);
    cv::namedWindow("Compensated", cv::WINDOW_NORMAL);
    // specify window size
    cv::resizeWindow("Standard", resolution.width, resolution.height);
    cv::resizeWindow("Compensated", resolution.width, resolution.height);

    cv::Mat compensated_frame = cv::Mat::zeros(resolution, CV_8UC3);

    // create accumulator
    dv::EdgeMapAccumulator accumulator(resolution);
    accumulator.setNeutralPotential(0.5f);
    accumulator.setEventContribution(0.25f);
    accumulator.setIgnorePolarity(false);

    dv::EdgeMapAccumulator comp_accumulator(resolution);
    comp_accumulator.setNeutralPotential(0.5f);
    comp_accumulator.setEventContribution(0.25f);
    comp_accumulator.setIgnorePolarity(false);

    // open3d vis
    auto vis = std::make_shared<Open3DVisualizer>();

    // create the bins for tracking regions in the image plane
    std::vector<std::shared_ptr<BinThreadFollower>> bins;

    // initialize the NUFFT to estimate the frequency
    FourierFreqEst fourier(1'000, 200, 800);

    double estimated_freq = 0;
    double phase_shift = 0;
    double amplitude = 0;

    // read the events
    bool estimate_freq = true;
    int skip = 0;
    dv::EventStore events;
    int64_t starting_timestamp_boot = 0;
    while (reader->isRunning() && estimate_freq) {
        if (skip < 10) { // skip the first 10 samples
            skip++;
            continue;
        }
        if (const auto events_dist = reader->getNextEventBatch(); events_dist.has_value()) {
            if (NO_SIM) {
                events = geometry->undistortEvents(events_dist.value());
            } else {
                events = events_dist.value();
            }
            for (const auto &event: events) {
                if (starting_timestamp_boot == 0) {
                    starting_timestamp_boot = event.timestamp();
                }

                if (fourier.feed(event.x(), event.y(),
                                 event.timestamp() / 1e6)) {
                    estimated_freq = fourier.getMainFreqRad();
                    phase_shift = fourier.getPhaseShift();
                    amplitude = fourier.getAmplitude();
                    estimate_freq = false;
                }
            }
        }
    }

    // int bin_h = 160, bin_w = 128;
    // int bin_h = resolution.height/, bin_w = resolution.width;
    int bin_h = 2 * 40, bin_w = 2 * 40;

    // if (resolution.width % bin_w != 0 || resolution.height % bin_h != 0) {
    //     throw std::runtime_error("Bin size must be a sub-multiple of the resolution.");
    // }
    int num_bins_h = resolution.height / bin_h;
    int num_bins_w = resolution.width / bin_w;

    // getMean() from fourier
    auto [c_x, c_y, c_t] = fourier.getMean();

    auto bin_follower = std::make_shared<BinThreadFollower>(3, compensated_frame, vis, bin_w, bin_h, 0.1, 0.1,
                                                            estimated_freq,
                                                            c_x, c_y, phase_shift, amplitude);

    // for (int i = 0; i < num_bins_h; i++) {
    //     for (int j = 0; j < num_bins_w; j++) {
    //         double c_x = j * bin_w + bin_w / 2;
    //         double c_y = i * bin_h + bin_h / 2;
    //         // process noise and measurement noise are set to 0.1
    //         // using the last 2 samples as window
    //         bins.emplace_back(
    //                 std::make_shared<BinThreadFollower>(3, compensated_frame, vis, bin_w, bin_h, 0.1, 0.1,
    //                                                     estimated_freq,
    //                                                     c_x, c_y, phase_shift, amplitude));
    //     }
    // }
    vis->update();

    // draw bins lines on compensated_frame


    // track the bins
    auto last_time = std::chrono::high_resolution_clock::now();

    int64_t starting_timestamp = 0;
    while (reader->isRunning()) {
        if (const auto events_dist = reader->getNextEventBatch(); events_dist.has_value()) {

            if (NO_SIM) {
                events = geometry->undistortEvents(events_dist.value());
            } else {
                events = events_dist.value();
            }

            // std::vector<EventStruct> events_struct;
            // for (const auto &e: events) {
            //     events_struct.emplace_back(EventStruct(e.timestamp(), e.x(), e.y(), e.polarity()));
            // }
            // // check the ev 2 image
            // cv::Mat out_img(resolution, CV_8U);
            // ev2img(events_struct, out_img, EventRepresentation::AVERAGE_TS);
            // cv::imshow("img", out_img);
            // cv::waitKey(1);

            // accumulator.accumulate(events);
            // make the compensated_frame image black

            // compensated_frame = cv::Mat::zeros(resolution, CV_8UC3);

            auto highest_time = events.getHighestTime();
            auto lowest_time = events.getLowestTime();
            auto time_diff = (highest_time - lowest_time) / 1e6;

            for (const auto &event: events) {
                if (starting_timestamp == 0) {
                    starting_timestamp = event.timestamp();
                }

                bin_follower->add_event(static_cast<double>(event.x()), static_cast<double>(event.y()),
                                        static_cast<double>(event.timestamp() - starting_timestamp) / 1e6);

            }

            // cv::imshow("Standard", accumulator.generateFrame().image);
            cv::imshow("Compensated", compensated_frame);
            cv::waitKey(1);

            if (cv::waitKey(1) == 27) {
                // stop and exit
                for (auto &bin: bins) {
                    bin->stop();
                }
                return 0;
            }

            // Measure the time elapsed for this loop iteration
            auto end_time = std::chrono::high_resolution_clock::now();
            const auto loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - last_time);
            last_time = end_time; // Update for next iteration

            // Calculate and print frequency in Hz
            const auto loop_time = loop_duration.count() / 1e6;
            const double frequency = 1.0 / loop_time; // Convert microseconds to seconds
            std::cout << "\rLoop frequency: " << frequency << " Hz, time diff: " << (time_diff - loop_time);
            std::cout.flush();

            vis->update();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        }
    }
    for (auto &bin: bins) {
        bin->stop();
    }
    vis->loop();
    return 0;
}