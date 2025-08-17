//
// Created by viciopoli on 17/08/25.
//
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

#include <metavision/sdk/driver/hdf5_event_file_writer.h>


#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
#include "estimator/iekf_sinusoid_fitter_multi_harmonic.hpp"
#include "event_frontend/undistort.hpp"
#include "haste_wrapper.hpp"


namespace {
    constexpr Metavision::timestamp kWindowUs = 10'000; // 10 ms in µs
}

// Small helper to write one window's buffer to a new HDF5 file
static void write_window_to_file(const std::string &base_dir,
                                 const std::vector<Metavision::EventCD> &buf,
                                 Metavision::timestamp t_start,
                                 Metavision::timestamp t_end,
                                 const Metavision::Camera &camera) {
    if (buf.empty()) return;

    std::ostringstream name;
    name << base_dir << "/events_" << t_start << "_" << t_end << ".hdf5";
    std::filesystem::path out_path{name.str()};
    if (!out_path.parent_path().empty() && !std::filesystem::exists(out_path.parent_path())) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    Metavision::HDF5EventFileWriter writer(out_path);
    writer.add_metadata_map_from_camera(camera);
    writer.add_events(buf.data(), buf.data() + buf.size());
    writer.close();
}

static Metavision::timestamp first_event_t = 0, last_event_t = 0;

int main(int argc, char *argv[]) {
    // Initialize parameters and camera
    HARMEDA::ParamsLoader params(argc, argv);
    std::cout << params;

    const auto width = params.camera.geometry().width();
    std::string output_images = params.params->output_folder + "/event_slices/";
    if (!std::filesystem::exists(output_images)) {
        std::filesystem::create_directories(output_images);
    }

    // Rolling window state
    Metavision::Stage::EventBuffer window_buf;
//    window_buf.reserve(200000); // reserve to reduce reallocs; tune if needed

    std::once_flag init_flag;
    long long slice_initial_time = 0;

    Metavision::EventCD e_tmp;
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::call_once(init_flag, [&]() {
            first_event_t = begin->t;
            slice_initial_time = first_event_t;
        });

        // Process the batch, splitting across as many windows as needed
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            last_event_t = ev->t;
            e_tmp.x = ev->x;
            e_tmp.y = ev->y;
            e_tmp.p = ev->p;
            e_tmp.t = ev->t;
            if (ev->t - slice_initial_time > 10000) {

                std::string out_path = output_images + "/events_" +
                                       std::to_string(slice_initial_time) + "_" + std::to_string(ev->t) + ".hdf5";

                Metavision::HDF5EventFileWriter writer(out_path);
                writer.add_metadata_map_from_camera(params.camera);
                writer.add_events(window_buf.data(), window_buf.data() + window_buf.size());
                writer.close();

                slice_initial_time = ev->t; // Reset slice start time
                window_buf.clear(); // Clear the buffer for the next slice
            }
            window_buf.push_back(e_tmp);

        }
    });

    // Start camera
    params.camera.start();

    std::cout << "Camera started, processing events..." << std::endl;
    while (params.camera.is_running()) {
        // Just keep the main thread alive
        Metavision::EventLoop::poll_and_dispatch(1);
    }

    return 0;
}
