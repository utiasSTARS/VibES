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
#include <opencv2/opencv.hpp>

#include "ema.hpp"
#include "arma.hpp"
#include "sinusoid_fitter.hpp"

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

    auto t_window = 1000.;
    CentroidEMA ema_calculator(tau, t_window);

    // ARMA models for each polarity's x and y coordinates
    std::map<int, ARMA> arma_x_models;
    std::map<int, ARMA> arma_y_models;

    // Sinusoid fitter for each polarity's x and y coordinates
    std::map<int, SinusoidFitter> x_fitters;
    std::map<int, SinusoidFitter> y_fitters;

    // OpenCV visualization setup
    const int crop_size = 150;
    const int vis_width = 1920;
    cv::Mat crop_vis = cv::Mat::zeros(crop_size, vis_width, CV_8UC3);
    const int crop_x_start = camera_width / 2 - crop_size / 2;
    const int crop_y_start = camera_height / 2 - crop_size / 2;
    cv::namedWindow("X-values vs. Time", cv::WINDOW_NORMAL);

    auto vis = std::make_shared<open3d::visualization::Visualizer>();
    std::shared_ptr<open3d::geometry::PointCloud> events_pcd_p0;
    std::shared_ptr<open3d::geometry::LineSet> centroids_trace;
    std::shared_ptr<open3d::geometry::LineSet> arma_trace;
    std::shared_ptr<open3d::geometry::LineSet> sinusoid_trace;

    if (do_plot) {
        vis->CreateVisualizerWindow("Events, Centroids, and ARMA Prediction", 1600, 900);
        events_pcd_p0 = std::make_shared<open3d::geometry::PointCloud>();
        centroids_trace = std::make_shared<open3d::geometry::LineSet>();
        arma_trace = std::make_shared<open3d::geometry::LineSet>();
        sinusoid_trace = std::make_shared<open3d::geometry::LineSet>();
        vis->AddGeometry(events_pcd_p0);
        vis->AddGeometry(centroids_trace);
        vis->AddGeometry(arma_trace);
        vis->AddGeometry(sinusoid_trace);

        auto bounding_box = std::make_shared<open3d::geometry::AxisAlignedBoundingBox>(
                Eigen::Vector3d(0, 0, 0.0),
                Eigen::Vector3d(camera_width, camera_height, 1));
        bounding_box->color_ = Eigen::Vector3d(0.0, 1.0, 0.0);
        vis->AddGeometry(bounding_box);
    }

    Metavision::timestamp last_print_time = 0;
    Metavision::timestamp first_event_t = -1;
    const Metavision::timestamp fitting_duration = 2 * 1000 * 1000; // 2 seconds
    bool fitting_complete = false;

    double time_scale = 1000.;
    std::mutex _mtx;
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (first_event_t < 0 && begin != end) {
            first_event_t = begin->t;
        }

        std::lock_guard<std::mutex> lock(_mtx);
        for (auto it = begin; it != end; ++it) {
            const auto &event = *it;
            const Metavision::timestamp current_relative_t = (event.t - first_event_t);
            const double current_relative_t_sec = current_relative_t / 1.e6;

            if (do_plot) {
                events_pcd_p0->points_.emplace_back(event.x, event.y, current_relative_t_sec);
                if (event.p == 0) {
                    events_pcd_p0->colors_.emplace_back(0, 0, 1);
                } else {
                    events_pcd_p0->colors_.emplace_back(1, 0, 0);
                }
            }
            const int vis_x_time = static_cast<int>((event.t - first_event_t) / time_scale);

            if (vis_x_time >= vis_width) {
                if (camera.is_running()) {
                    camera.stop();
                }
                continue;
            }

            if (event.x >= crop_x_start && event.x < crop_x_start + crop_size &&
                event.y >= crop_y_start && event.y < crop_y_start + crop_size) {
                int vis_y_pos = event.x - crop_x_start;
                cv::Point point(vis_x_time, vis_y_pos);
                cv::circle(crop_vis, point, 1, (event.p == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255), -1);
            }

            auto updated_centroid = ema_calculator.update(event);
            if (updated_centroid.has_value()) {
                auto centroid = updated_centroid.value();
                int p = event.p;

                if (current_relative_t < fitting_duration) {
                     if (x_fitters.find(p) == x_fitters.end()){
                        x_fitters.emplace(p, SinusoidFitter());
                        y_fitters.emplace(p, SinusoidFitter());
                    }
                    x_fitters.at(p).add_point(current_relative_t_sec, centroid[0]);
                    y_fitters.at(p).add_point(current_relative_t_sec, centroid[1]);
                } else if (!fitting_complete) {
                    std::cout << "--- Fitting Sinusoid (Polarity " << p << ") ---" << std::endl;
                    std::cout << "X-Coordinate: ";
                    x_fitters.at(p).fit();
                    std::cout << "Y-Coordinate: ";
                    y_fitters.at(p).fit();
                }

                if (do_plot) {
                    Eigen::Vector3d new_point(centroid[0], centroid[1], (current_relative_t + t_window / 2) / 1.e6);
                    centroids_trace->points_.push_back(new_point);

                    if (centroids_trace->points_.size() > 1) {
                        Eigen::Vector2i line_indices(centroids_trace->points_.size() - 2,
                                                     centroids_trace->points_.size() - 1);
                        centroids_trace->lines_.push_back(line_indices);
                        centroids_trace->colors_.push_back(Eigen::Vector3d(0, 1, 0));
                    }
                }

                if (arma_x_models.find(p) == arma_x_models.end()) {
                    arma_x_models.emplace(p, ARMA({0.9}, {}, centroid[0]));
                    arma_y_models.emplace(p, ARMA({0.9}, {}, centroid[1]));
                }
                
                arma_x_models.at(p).update(centroid[0]);
                arma_y_models.at(p).update(centroid[1]);
                double pred_x = arma_x_models.at(p).predict();
                double pred_y = arma_y_models.at(p).predict();

                if (do_plot) {
                    Eigen::Vector3d pred_point(pred_x, pred_y, (current_relative_t + t_window * 1.5) / 1.e6);
                    arma_trace->points_.push_back(pred_point);

                    if(arma_trace->points_.size() > 1){
                         Eigen::Vector2i line_indices(arma_trace->points_.size() - 2, arma_trace->points_.size() - 1);
                         arma_trace->lines_.push_back(line_indices);
                         arma_trace->colors_.push_back(Eigen::Vector3d(1, 1, 0)); // Yellow
                    }
                }

                float cx = centroid[0];
                int vis_y_pos = cx - crop_x_start;
                if (vis_y_pos >= 0 && vis_y_pos < crop_size) {
                    cv::Point centroid_point(vis_x_time - (t_window / 2) / time_scale, vis_y_pos);
                    cv::circle(crop_vis, centroid_point, 2, cv::Scalar(0, 255, 0), -1);

                    int pred_vis_y_pos = pred_x - crop_x_start;
                    if(pred_vis_y_pos >=0 && pred_vis_y_pos < crop_size){
                        cv::Point pred_point(vis_x_time + (t_window / 2) / time_scale, pred_vis_y_pos);
                        cv::circle(crop_vis, pred_point, 2, cv::Scalar(255, 0, 255), -1); // Magenta
                    }

                    if (fitting_complete && x_fitters.count(p)) {
                        double sinusoid_pred_x = x_fitters.at(p).predict(current_relative_t_sec);
                        int sinusoid_pred_y_pos = sinusoid_pred_x - crop_x_start;
                         if(sinusoid_pred_y_pos >=0 && sinusoid_pred_y_pos < crop_size){
                            cv::Point pred_point(vis_x_time, sinusoid_pred_y_pos);
                            cv::circle(crop_vis, pred_point, 2, cv::Scalar(255, 165, 0), -1); // Orange for sinusoid
                        }
                    }
                }
                 if(fitting_complete && do_plot && x_fitters.count(p)){
                    double pred_x_sin = x_fitters.at(p).predict(current_relative_t_sec);
                    double pred_y_sin = y_fitters.at(p).predict(current_relative_t_sec);
                    Eigen::Vector3d pred_point(pred_x_sin, pred_y_sin, current_relative_t_sec);
                    sinusoid_trace->points_.push_back(pred_point);
                     if (sinusoid_trace->points_.size() > 1) {
                         Eigen::Vector2i line_indices(sinusoid_trace->points_.size() - 2, sinusoid_trace->points_.size() - 1);
                         sinusoid_trace->lines_.push_back(line_indices);
                         sinusoid_trace->colors_.push_back(Eigen::Vector3d(0.5, 0, 0.5)); // Purple
                     }
                }
            }
        }
        const Metavision::timestamp current_relative_t_end = (end-1 > begin) ? ((end-1)->t - first_event_t) : 0;
        if (!fitting_complete && current_relative_t_end >= fitting_duration){
             fitting_complete = true;
             std::cout << "\n--- Sinusoid Fitting Complete ---\n" << std::endl;
        }

        if (end > begin) {
            Metavision::timestamp current_t = (end - 1)->t;
            if (current_t - last_print_time > 100000) {
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
            cv::Mat frame_to_show;
            {
                std::lock_guard<std::mutex> lock(_mtx);
                vis->UpdateGeometry(events_pcd_p0);
                vis->UpdateGeometry(centroids_trace);
                vis->UpdateGeometry(arma_trace);
                if(fitting_complete) vis->UpdateGeometry(sinusoid_trace);
                vis->PollEvents();
                vis->UpdateRender();

                crop_vis.copyTo(frame_to_show);
            }

            if (!frame_to_show.empty()) {
                cv::imshow("X-values vs. Time", frame_to_show);
            }
            if (cv::waitKey(1) >= 0) {
                break;
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
    cv::destroyAllWindows();

    return 0;
} 