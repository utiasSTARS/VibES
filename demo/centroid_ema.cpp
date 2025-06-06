#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <atomic>

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <boost/program_options.hpp>
#include <open3d/Open3D.h>

#include "ema.h"

namespace po = boost::program_options;


int main(int argc, char *argv[]) {
    std::string input_path;
    double tau;
    bool do_plot = false;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("input-event-file,i", po::value<std::string>(&input_path),
             "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
            ("tau", po::value<double>(&tau)->default_value(100000.0), "Time constant for EMA in microseconds.")
            ("plot", "Enable 3D visualization of events and centroids.");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const po::error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }


    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    do_plot = vm.count("plot");

    Metavision::Camera camera;
    try {
        if (!input_path.empty()) {
            camera = Metavision::Camera::from_file(input_path);
        } else {
            camera = Metavision::Camera::from_first_available();
        }
    } catch (const Metavision::CameraException &e) {
        std::cerr << "Camera initialization error: " << e.what() << std::endl;
        return 1;
    }

    int camera_width = camera.geometry().width();
    int camera_height = camera.geometry().height();

    CentroidEMA ema_calculator(tau, 1000);

    auto vis = std::make_shared<open3d::visualization::Visualizer>();
    std::shared_ptr<open3d::geometry::PointCloud> events_pcd_p0;
    std::shared_ptr<open3d::geometry::LineSet> centroids_trace;

    if (do_plot) {
        vis->CreateVisualizerWindow("Events and Centroids", 1600, 900);
        events_pcd_p0 = std::make_shared<open3d::geometry::PointCloud>();
        centroids_trace = std::make_shared<open3d::geometry::LineSet>();
        vis->AddGeometry(events_pcd_p0);
        vis->AddGeometry(centroids_trace);

        auto bounding_box = std::make_shared<open3d::geometry::AxisAlignedBoundingBox>(
                Eigen::Vector3d(0, 0, 0.0),
                Eigen::Vector3d(camera_width, camera_height, 1));

        // Set bounding box color for visibility
        bounding_box->color_ = Eigen::Vector3d(0.0, 1.0, 0.0);  // Green color

        vis->AddGeometry(bounding_box);
    }

    Metavision::timestamp last_print_time = 0;

    double time_scale = 1000.;
    std::mutex _mtx;
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::lock_guard<std::mutex> lock(_mtx);
        if (do_plot) {
            for (auto it = begin; it != end; ++it) {
                const auto &ev = *it;
                events_pcd_p0->points_.emplace_back(ev.x, ev.y, ev.t / time_scale);
                if (ev.p == 0) {
                    events_pcd_p0->colors_.emplace_back(0, 0, 1); // Blue
                } else {
                    events_pcd_p0->colors_.emplace_back(1, 0, 0); // Red
                }
            }
        }

        for (auto it = begin; it != end; ++it) {
            const auto &event = *it;
            auto updated_centroid = ema_calculator.update(event);
            if (do_plot && updated_centroid.has_value()) {
                auto centroid = updated_centroid.value();

                Eigen::Vector3d new_point(centroid[0], centroid[1], event.t / time_scale);
                centroids_trace->points_.push_back(new_point);

                if (centroids_trace->points_.size() > 1) {
                    Eigen::Vector2i line_indices(centroids_trace->points_.size() - 2,
                                                 centroids_trace->points_.size() - 1);
                    centroids_trace->lines_.push_back(line_indices);
                    centroids_trace->colors_.push_back(Eigen::Vector3d(0, 1, 0)); // Green
                }
            }
        }

        if (end > begin) {
            Metavision::timestamp current_t = (end - 1)->t;
            if (current_t - last_print_time > 100000) { // Print every ~100ms
                last_print_time = current_t;
                std::cout << "Timestamp: " << std::fixed << std::setprecision(2) << current_t / 1e6 << "s" << std::endl;
                const auto &centroids = ema_calculator.get_centroids();
                for (const auto &pair: centroids) {
                    std::cout << "  Polarity " << pair.first << ": (" << std::fixed << std::setprecision(2)
                              << pair.second[0] << ", " << pair.second[1] << ")" << std::endl;
                }
            }
        }
    });

    camera.start();

    if (do_plot) {
        while (camera.is_running()) {
            {
                std::lock_guard<std::mutex> lock(_mtx);
                vis->UpdateGeometry(events_pcd_p0);
                vis->UpdateGeometry(centroids_trace);
                vis->PollEvents();
                vis->UpdateRender();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        camera.stop();

    } else {
        if (input_path.empty()) {
            std::cout << "Processing events from live camera for 5 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
        } else {
            std::cout << "Processing events from file: " << input_path << std::endl;
            while (camera.is_running()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::cout << "Finished processing file." << std::endl;
        }
        camera.stop();
    }

    while (vis->PollEvents()) {
        vis->UpdateRender();
    }
    vis->DestroyVisualizerWindow();


    return 0;
}
