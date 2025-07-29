//
// Enhanced Event-based Motion Tracking with Proper Visualization
// Based on Metavision SDK patterns
//

//#define FANCY_VISUALIZATION
//#define STORE

#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/core/utils/rate_estimator.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/core/pipeline/stage.h>
#include <metavision/sdk/core/utils/misc.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <mutex>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <csignal>
#include <thread>

#include "visualizer/ev2image.hpp"


#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
#include "estimator/iekf_sinusoid_fitter_multi_harmonic.hpp"
#include "event_frontend/undistort.hpp"
#include "haste_wrapper.hpp"
#include "profiler.hpp"


// Constants
namespace {
    constexpr double DEFAULT_FPS = 100.0;
    constexpr std::uint32_t DEFAULT_ACCUMULATION = 5000;
    constexpr int ESC_KEY = 27;
    constexpr int POLL_TIMEOUT_MS = 10;
}

// UI processing function similar to original
int processUI(int delay_ms) {
    auto then = std::chrono::high_resolution_clock::now();
    int key = cv::waitKey(delay_ms);
    auto now = std::chrono::high_resolution_clock::now();

    // Ensure consistent timing
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
    if (elapsed < delay_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms - elapsed));
    }

    return key;
}


int main(int argc, char *argv[]) {
    // Initialize parameters and camera
    HARMEDA::ParamsLoader params(argc, argv);
    std::cout << params;

    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    // create a folder for each of the images
    std::vector<std::string> folder_names = {
            "img_bin", "img_cnt_gray", "img_ts", "img_avgts", "img_cnt_color", "img_ts_color", "img_avgts_color"
    };
    for (const auto &folder_name: folder_names) {
        std::filesystem::path folder_path = params.params->output_folder + "/imgs/" + folder_name;
        if (!std::filesystem::exists(folder_path)) {
            std::filesystem::create_directories(folder_path);
        }
    }

    // Setup CD frame generator (similar to original)
    std::mutex cd_frame_mutex;
    cv::Mat cd_frame;
    Metavision::timestamp cd_frame_ts{0};

    Metavision::CDFrameGenerator cd_frame_generator(width, height);
    cd_frame_generator.set_display_accumulation_time_us(DEFAULT_ACCUMULATION);

    // Start frame generator with callback
    cd_frame_generator.start(DEFAULT_FPS,
                             [&cd_frame_mutex, &cd_frame, &cd_frame_ts](const Metavision::timestamp &ts,
                                                                        const cv::Mat &frame) {
                                 std::unique_lock<std::mutex> lock(cd_frame_mutex);
                                 cd_frame_ts = ts;
                                 frame.copyTo(cd_frame);
                             });

    // Setup event rate estimator
    double avg_rate = 0, peak_rate = 0;
    Metavision::RateEstimator cd_rate_estimator(
            [&avg_rate, &peak_rate](Metavision::timestamp ts, double arate, double prate) {
                avg_rate = arate;
                peak_rate = prate;
            },
            100000, 1000000, true);

    // Initialize fitters and estimator
    Metavision::Stage::EventBuffer compensated_events;
    unsigned short x_undistorted, y_undistorted;

    std::once_flag init_flag;

    Metavision::Stage::EventBuffer amiev_events;
    Metavision::timestamp duration_for_amiev = 0;

    int counter = 0;
    // Main event processing callback
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        unsigned short delta = end->t - begin->t;
        duration_for_amiev += delta;
        amiev_events.insert(amiev_events.end(), begin, end);
        if (duration_for_amiev > 2000) { //29997) {
            // do every 100
            auto img = ev2img_metavision(amiev_events, height, width);
            duration_for_amiev = 0; // reset the duration for next chunk
//            std::cout << "Processed " << compensated_images_amiev_vis.size() << " chunks of events." << std::endl;

            auto counter_str = std::to_string(counter);
            cv::imshow("Binary Image", img.img_bin);
//            cv::imshow("Count Gray Image", img.img_cnt_gray);
            cv::imshow("Timestamp Image", img.img_ts);
            cv::imshow("Average Timestamp Image", img.img_avgts);
            cv::imshow("Count Color Image", img.img_cnt_color);
//            cv::imshow("Timestamp Color Image", img.img_ts_color);
//            cv::imshow("Average Timestamp Color Image", img.img_avgts_color);
            cv::waitKey(1); // Wait for key press to show each image

//            cv::imwrite(params.params->output_folder + "/imgs/img_bin/img_" + counter_str + ".png", img.img_bin);
//            cv::imwrite(params.params->output_folder + "/imgs/img_cnt_gray/img_" + counter_str + ".png", img.img_cnt_gray);
//            cv::imwrite(params.params->output_folder + "/imgs/img_ts/img_" + counter_str + ".png", img.img_ts);
//            cv::imwrite(params.params->output_folder + "/imgs/img_avgts/img_" + counter_str + ".png", img.img_avgts);
//            cv::imwrite(params.params->output_folder + "/imgs/img_cnt_color/img_" + counter_str + ".png",
//                        img.img_cnt_color);
//            cv::imwrite(params.params->output_folder + "/imgs/img_ts_color/img_" + counter_str + ".png", img.img_ts_color);
//            cv::imwrite(params.params->output_folder + "/imgs/img_avgts_color/img_" + counter_str + ".png",
//                        img.img_avgts_color);
            counter++;
        }
    });

    // Start camera
    params.camera.start();

    // Main processing loop (similar to original)
    while (params.camera.is_running()) {
        // Process UI with consistent timing
        int key = processUI(POLL_TIMEOUT_MS);
        switch (key) {
            case ESC_KEY:
            case 'q':
                params.camera.stop();
                break;
            case 'h':
                std::cout << "Controls:\n"
                          << "  ESC/q: Exit\n"
                          << "  h: This help\n"
                          << "  Mouse click: Initialize tracker\n";
                break;
            default:
                break;
        }

        // Poll Metavision events
        Metavision::EventLoop::poll_and_dispatch(1);
    }

    // Cleanup
    cd_frame_generator.stop();
    if (params.camera.is_running()) {
        params.camera.stop();
    }

    return 0;
}