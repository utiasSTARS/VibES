#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/camera_capture.hpp>
#include <dv-processing/camera/calibration_set.hpp>


#include "../include/utils.hpp"
#include "../include/nufourier.hpp"
#include "../include/bin.hpp"
#include "../include/sim/ini_sim.hpp"
#include "../include/open3d_visualizer.hpp"

// Main example function
int main(int argc, char *argv[]) {
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
        }
    } else {
        NO_SIM = false;
        int target_freq = 700;
        std::cout << "No file provided. Using simulator with freq " << target_freq << " rad/s, " << rad2Hz(target_freq)
                  << " Hz" << std::endl;
        reader = std::make_unique<EventsFreqCalibPattern>(0, cv::Size(640, 480), target_freq, 5, 5, 0., false);
    }

    // if not simulation try to laod the camera calibration
    std::shared_ptr<dv::camera::CameraGeometry> geometry;
    if (NO_SIM) {
        const auto calibrationSet = dv::camera::CalibrationSet::LoadFromFile("../camera/calib.xml");
        dv::camera::calibrations::CameraCalibration dvx_calib;

        if (calibrationSet.getCameraList().size() > 0) {
            const auto calibs = calibrationSet.getCameraCalibrations();
            dvx_calib = calibs.begin()->second;
            std::cout << "Found calibration for camera with name [" << dvx_calib.name << "]" <<
                      std::endl;
        }

        geometry = std::make_shared<dv::camera::CameraGeometry>(dvx_calib.getCameraGeometry());
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
    Open3DVisualizer vis;

    // create the bins for tracking regions in the image plane
    std::vector<Bin> bins;
    int bin_h = 160, bin_w = 128;
    // int bin_h = resolution.height, bin_w = resolution.width;

    if (resolution.width % bin_w != 0 || resolution.height % bin_h != 0) {
        throw std::runtime_error("Bin size must be a sub-multiple of the resolution.");
    }
    int num_bins_h = resolution.height / bin_h;
    int num_bins_w = resolution.width / bin_w;


    // initiailize the NUFFT to estimate the frequency
    FourierFreqEst fourier(1000, 500, 900);

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


    for (int i = 0; i < num_bins_h; i++) {
        for (int j = 0; j < num_bins_w; j++) {
            double c_x = j * bin_w + bin_w / 2;
            double c_y = i * bin_h + bin_h / 2;
            // process noise and measurement noise are set to 0.1
            // using the last 2 samples as window
            bins.emplace_back(3, 0.1, 0.1, estimated_freq, c_x, c_y, phase_shift, amplitude);
        }
    }

    // draw bins lines on compensated_frame
    auto drawGridLines = [&]() {
        for (int i = 1; i < resolution.height / bin_h; i++) {
            int y = i * bin_h;
            cv::line(compensated_frame, cv::Point(0, y), cv::Point(resolution.width, y), cv::Scalar(100, 100, 100), 1);
        }
        for (int j = 1; j < resolution.width / bin_w; j++) {
            int x = j * bin_w;
            cv::line(compensated_frame, cv::Point(x, 0), cv::Point(x, resolution.height), cv::Scalar(100, 100, 100), 1);
        }
    };
    drawGridLines();

    // track the bins
    int64_t starting_timestamp = 0;
    while (reader->isRunning()) {
        if (const auto events_dist = reader->getNextEventBatch(); events_dist.has_value()) {
            if (NO_SIM) {
                events = geometry->undistortEvents(events_dist.value());
            } else {
                events = events_dist.value();
            }
            accumulator.accumulate(events);
            for (const auto &event: events) {
                if (starting_timestamp == 0) {
                    starting_timestamp = event.timestamp();
                }

                int bin_row = static_cast<int>(event.y() / bin_h);
                int bin_col = static_cast<int>(event.x() / bin_w);

                // Find the corresponding bin index
                int bin_index = bin_row * num_bins_w + bin_col;

                auto comp = bins[bin_index].update(static_cast<double>(event.x()), static_cast<double>(event.y()),
                                                   static_cast<double>(event.timestamp() - starting_timestamp) / 1e6);
                if (comp.has_value()) {
                    // std::cout << "\r" << bins[bin_index] << std::endl;
                    std::cout.flush();
                    // clean the compensated frame in the bin region
                    cv::Mat roi = compensated_frame(cv::Rect(bin_col * bin_w, bin_row * bin_h, bin_w, bin_h));
                    roi.setTo(cv::Scalar(0, 0, 0));

                    // add mean vals
                    auto [mean_x, mean_y, mean_t] = bins[bin_index].getMean();
                    // std::cout << mean_x << " " << mean_y << " " << mean_t << std::endl;
                    vis.addPoint(mean_x, mean_y, mean_t * 100, 0.1, 1.0);

                    // estimated curve
                    auto [est_x, est_y] = bins[bin_index].getEKF()->getPred();
                    vis.addPoint(est_x, est_y, mean_t * 100, 0.1, 0.1, 1.0);

                    // print mean vals on image
                    compensated_frame.at<cv::Vec3b>(mean_y, mean_x) = cv::Vec3b(255, 255, 255);

                    auto [comp_x, comp_y, comp_t] = comp.value();
                    // cv::putText(compensated_frame, "Freq: " + fp2str(bins[bin_index].getRad(), 1) + " Rad/s",
                    //             cv::Point(10, 30),
                    //             cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2);
                    cv::putText(
                            compensated_frame, "f:" + fp2str(bins[bin_index].getRad(), 1) + " rad/s",
                            cv::Point(5 + bin_col * (bin_w + 1), 30 + bin_row * (bin_h + 1)),
                            cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
                    drawGridLines();
                    auto [B, G, R] = color_map[bins[bin_index].getColor()];
                    auto cv_color = cv::Scalar(B, G, R);
                    int amp_x = bins[bin_index].getAmplitudeX() + 1;
                    int amp_y = bins[bin_index].getAmplitudeY() + 1;
                    cv::circle(compensated_frame, cv::Point(comp_x, comp_y),
                               std::min(std::max(std::abs(amp_x), std::abs(amp_y)), int(bin_w / 4)),
                               cv_color,
                               -1);
                }
            }
            cv::imshow("Standard", accumulator.generateFrame().image);
            cv::imshow("Compensated", compensated_frame);
            cv::waitKey(1);
            vis.update();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    vis.loop();
    return 0;
}