#include <torch/script.h>
#include <iostream>
#include <deque>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/camera_capture.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <thread>
#include <chrono>
#include "include/events_freq_calib_pattern.hpp"

int main() {
    // Load the scripted model
    torch::jit::Module model;
    try {
        model = torch::jit::load(
                "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/project/resnet1d_scripted.pt");
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model\n" << e.what() << std::endl;
        return -1;
    }
    std::cout << "Model loaded successfully\n";

    // start with dv camera
    dv::io::MonoCameraRecording reader(
            "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/file.aedat4");

    // dv::io::CameraCapture reader;
    std::cout << "Opened AEDAT4 file from [" << reader.getCameraName() << "] camera\n";

    const cv::Size resolution = *reader.getEventResolution();
    const cv::Size half_resolution = cv::Size(resolution.width / 2, resolution.height / 2);


    // read the events
    std::vector<float> input_data;
    int64_t x_mass = 0, y_mass = 0, initial_time = 0;
    int estimates = 0, counter = 0;

    float max_x = std::numeric_limits<float>::min(), max_y = std::numeric_limits<float>::min();
    float min_x = std::numeric_limits<float>::max(), min_y = std::numeric_limits<float>::max();

    int n_samples = 100;
    // read
    int square = 150;
    // while (reader.isRunning()) {
    //     if (const auto events = reader.getNextEventBatch(); events.has_value()) {
    EventsFreqCalibPattern pattern(0, resolution, 100, 10, 10, 0.01);

    for (int i=0; i<1000; i++) {
        // Generate events for 10 ms
        auto events = pattern.get_events();
        if (events) {
            for (const auto &e: *events) {
                // consider the events that are at the center only according to square
                if (e.x() < half_resolution.width - square || e.x() > half_resolution.width + square ||
                    e.y() < half_resolution.height - square || e.y() > half_resolution.height + square) {
                    continue;
                }
                x_mass += e.x();
                y_mass += e.y();
                if (initial_time == 0) {
                    initial_time = e.timestamp();
                }

                if (++counter == 1'000) {

                    float x = static_cast<float>(x_mass) / counter;
                    float y = static_cast<float>(y_mass) / counter;
                    x_mass = x;
                    y_mass = y;

                    float z = static_cast<float>(e.timestamp() - initial_time) / 1e6;
                    input_data.push_back(x);
                    input_data.push_back(y);
                    input_data.push_back(z);

                    if (x > max_x) max_x = x;
                    if (x < min_x) min_x = x;
                    if (y > max_y) max_y = y;
                    if (y < min_y) min_y = y;

                    if (input_data.size() >= 3 * n_samples) {

                        float mean_x = (max_x + min_x) / 2;
                        float mean_y = (max_y + min_y) / 2;

                        for (int i = 0; i < n_samples; i++) {
                            input_data[i * 3] -= mean_x;
                            input_data[i * 3 + 1] -= mean_y;
                        }

                        auto input_tensor = torch::from_blob(input_data.data(), {1, n_samples, 3}).to(
                                torch::kFloat32).to(
                                torch::kCUDA);

                        at::Tensor output = model.forward({input_tensor}).toTensor();
                        std::cout << "Model output: A = " << output[0][0].item<float>() << ", f = "
                                  << output[0][1].item<float>() << " rad/s, phi = "
                                  << output[0][2].item<float>()
                                  << std::endl;
                        std::cout.flush();

                        input_data.clear();

                        max_x = std::numeric_limits<float>::min(), max_y = std::numeric_limits<float>::min();
                        min_x = std::numeric_limits<float>::max(), min_y = std::numeric_limits<float>::max();
                    }
                    counter = 0;
                    estimates++;
                }
            }
        }
    }

    return 0;
}
