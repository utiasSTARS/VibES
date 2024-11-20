#include <torch/script.h>
#include <iostream>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/camera_capture.hpp>

#include <opencv2/imgproc.hpp>
#include <chrono>
#include "include/sim/ini_sim.hpp"
#include "include/utils.h"
// #include "include/open3d_visualizer.hpp"

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
    // dv::io::MonoCameraRecording reader(
    //         "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/file.aedat4");
    EventsFreqCalibPattern reader(0, cv::Size(640, 480), 500, 10, 10, 0.01, 10, true);

    // dv::io::CameraCapture reader;
    std::cout << "Opened AEDAT4 file from [" << reader.getCameraName() << "] camera\n";

    const cv::Size resolution = *reader.getEventResolution();
    const cv::Size half_resolution = cv::Size(resolution.width / 2, resolution.height / 2);

    // visualizer
    // Open3DVisualizer vis;

    // read the events
    std::vector<float> input_data;
    int64_t x_mass = 0, y_mass = 0, initial_time = 0;
    int estimates = 0, counter = 0;
    int samplingRate = 1e5;

    std::vector<std::tuple<double, double, int64_t>> events_buffer;
    float max_x = std::numeric_limits<float>::min(), max_y = std::numeric_limits<float>::min();
    float min_x = std::numeric_limits<float>::max(), min_y = std::numeric_limits<float>::max();

    int n_samples = 100;

    bool estimate = false;
    float A = 0, f = 0, phi = 0;

    int square = 150;
    int iters = 0;
    while (reader.isRunning()) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {
            for (const auto &event: *events) {
                // consider the events that are at the center only according to square
                if (event.x() < half_resolution.width - square || event.x() > half_resolution.width + square ||
                    event.y() < half_resolution.height - square || event.y() > half_resolution.height + square) {
                    continue;
                }
                x_mass += event.x();
                y_mass += event.y();
                if (initial_time == 0) {
                    initial_time = event.timestamp();
                }

                if (++counter == 1'000) {


                    events_buffer.emplace_back(static_cast<double>(x_mass) / counter,
                                               static_cast<double>(y_mass) / counter,
                                               (event.timestamp() - initial_time) / counter);


                    if (events_buffer.size() > 1) {
                        auto interpolated = interpolate_events(events_buffer, 1e6 / samplingRate);
                        if (interpolated.has_value()) {
                            for (const auto &e: *interpolated) {
                                auto x = static_cast<float>(std::get<0>(e));
                                auto y = static_cast<float>(std::get<1>(e));
                                auto t = static_cast<float>(std::get<2>(e));

                                if (x > max_x) max_x = x;
                                if (x < min_x) min_x = x;
                                if (y > max_y) max_y = y;
                                if (y < min_y) min_y = y;

                                input_data.push_back(x);
                                input_data.push_back(y);
                                input_data.push_back(t);


                                // if (iters < 100) {
                                //     vis.addPoint(std::get<0>(e), std::get<1>(e), std::get<2>(e), event.polarity());
                                // }
                            }
                        }
                    }


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
                        A = output[0][0].item<float>();
                        f = output[0][1].item<float>();
                        phi = output[0][2].item<float>();
                        std::cout << "Model output: A = " << A << ", f = " << f << ", phi = " << phi << std::endl;
                        std::cout.flush();

                        input_data.clear();

                        max_x = std::numeric_limits<float>::min(), max_y = std::numeric_limits<float>::min();
                        min_x = std::numeric_limits<float>::max(), min_y = std::numeric_limits<float>::max();
                    }
                    counter = 0;
                    estimates++;
                }
            }
            // vis.update();
            iters++;
        }
    }

    return 0;
}
