/**
 * @file vis_events.cpp
 * @brief Visualization tool for comparing different event-to-image conversion techniques.
 *
 * This utility captures events from a camera or file, accumulates them into
 * micro-batches (e.g., 2ms), and simultaneously renders them into multiple
 * representations:
 * 1. Binary Map
 * 2. Time Surface (Exponential Decay)
 * 3. Average Timestamp (Motion Flow)
 * 4. Event Count Heatmap
 *
 * It is useful for tuning the `ev2image` parameters and verifying data quality.
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/core/utils/rate_estimator.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/core/pipeline/stage.h>
#include <metavision/sdk/core/utils/misc.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <mutex>
#include <sstream>
#include <filesystem>
#include <csignal>

// Project Headers
#include "visualizer/ev2image.hpp"
#include "params_loader.hpp"
#include "event_frontend/undistort.hpp"
#include "profiler.hpp"

// Constants
namespace {
    constexpr double DEFAULT_FPS = 100.0;
    constexpr std::uint32_t DEFAULT_ACCUMULATION = 5000;
    constexpr int ESC_KEY = 27;
    constexpr int POLL_TIMEOUT_MS = 10;
    constexpr int ACCUMULATION_WINDOW_US = 2000; // 2ms accumulation for visualization
}

// Global flag for clean exit
static std::atomic<bool> keep_running{true};

void signal_handler(int signal) {
    std::cout << "\nSIGINT/SIGTERM received. Exiting..." << std::endl;
    keep_running = false;
}

int main(int argc, char *argv[]) {
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);

    // 1. Load Parameters
    VibES::ParamsLoader params(argc, argv);
    std::cout << params;

    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    // 2. Setup Output Directories
    // We create specific subfolders for each image type to allow batch saving
    std::vector<std::string> folder_names = {
            "img_bin", "img_cnt_gray", "img_ts", "img_avgts", "img_cnt_color", "img_ts_color", "img_avgts_color"
    };

    // Base path construction
    std::filesystem::path base_output_path = params.params->output_folder;
    if (!params.params->output_folder.empty()) {
        base_output_path /= "imgs";
        for (const auto &folder_name: folder_names) {
            std::filesystem::path folder_path = base_output_path / folder_name;
            if (!std::filesystem::exists(folder_path)) {
                try {
                    std::filesystem::create_directories(folder_path);
                } catch (const std::exception& e) {
                    std::cerr << "Error creating directory " << folder_path << ": " << e.what() << std::endl;
                }
            }
        }
    }

    // 3. Setup CD Frame Generator (Standard SDK Visualization)
    // This provides the "ground truth" standard view for comparison
    std::mutex cd_frame_mutex;
    cv::Mat cd_frame;
    Metavision::timestamp cd_frame_ts{0};

    Metavision::CDFrameGenerator cd_frame_generator(width, height);
    cd_frame_generator.set_display_accumulation_time_us(DEFAULT_ACCUMULATION);
    cd_frame_generator.start(DEFAULT_FPS,
                             [&cd_frame_mutex, &cd_frame, &cd_frame_ts](const Metavision::timestamp &ts,
                                                                        const cv::Mat &frame) {
                                 std::unique_lock<std::mutex> lock(cd_frame_mutex);
                                 cd_frame_ts = ts;
                                 frame.copyTo(cd_frame);
                             });

    // 4. Setup Custom Accumulation Buffers
    Metavision::Stage::EventBuffer amiev_events;
    Metavision::timestamp first_ts_in_batch = -1;
    int counter = 0;

    // 5. Main Callback
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (!keep_running) return;

        // Initialize timestamp reference
        if (first_ts_in_batch < 0 && begin != end) {
            first_ts_in_batch = begin->t;
        }

        // Copy events to local accumulation buffer
        amiev_events.insert(amiev_events.end(), begin, end);

        if (amiev_events.empty()) return;

        // Check if accumulation window is reached
        Metavision::timestamp current_duration = amiev_events.back().t - first_ts_in_batch;

        if (current_duration >= ACCUMULATION_WINDOW_US) {
            // --- Generate Images ---
            auto img = ev2img_metavision(amiev_events, height, width);

            // --- Display ---
            cv::imshow("Binary", img.img_bin);
            cv::imshow("Time Surface", img.img_ts_color);
            cv::imshow("Average TS", img.img_avgts_color);
            cv::imshow("Event Count", img.img_cnt_color);
            cv::waitKey(1);

            // --- Save Images (Optional) ---
            if (!params.params->output_folder.empty()) {
                std::stringstream ss;
                ss << std::setw(6) << std::setfill('0') << counter << ".png";
                std::string filename = ss.str();

                // Save specific debug views
                // cv::imwrite((base_output_path / "img_bin" / filename).string(), img.img_bin);
                // cv::imwrite((base_output_path / "img_ts_color" / filename).string(), img.img_ts_color);
                // cv::imwrite((base_output_path / "img_avgts_color" / filename).string(), img.img_avgts_color);
            }

            // --- Reset for next batch ---
            amiev_events.clear();
            first_ts_in_batch = -1;
            counter++;
        }
    });

    // 6. Start Camera
    params.camera.start();

    // 7. Main UI Loop
    std::cout << "Processing events... Press 'q' or 'ESC' to exit." << std::endl;
    while (params.camera.is_running() && keep_running) {

        // Handle global UI keys
        int key = processUI(POLL_TIMEOUT_MS);
        switch (key) {
            case ESC_KEY:
            case 'q':
                params.camera.stop();
                keep_running = false;
                break;
            case 'h':
                std::cout << "Controls:\n"
                          << "  ESC/q: Exit\n"
                          << "  h: This help\n";
                break;
            default:
                break;
        }

        // Poll SDK events
        Metavision::EventLoop::poll_and_dispatch(1);
    }

    // Cleanup
    cd_frame_generator.stop();
    if (params.camera.is_running()) {
        params.camera.stop();
    }

    std::cout << "Exiting." << std::endl;
    return 0;
}