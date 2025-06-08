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
//#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>

#include "ema.hpp"
#include "iekf_sinusoid_fitter.hpp"

namespace po = boost::program_options;

// Helper function to initialize the IEKF
IEKFSinusoidFitter create_iekf() {
    IEKFSinusoidFitter::StateVector initial_state;
    // A1, B1, A2, B2, omega, C
    initial_state << 0, 0, 0, 0, 5 * 2 * M_PI, 0;

    IEKFSinusoidFitter::StateCovariance initial_covariance;
    initial_covariance.setIdentity();
    initial_covariance *= 1e2;

    IEKFSinusoidFitter::StateCovariance process_noise;
    process_noise.setIdentity();
    process_noise *= 1e-5;

    double measurement_noise = 1e-1;
    return IEKFSinusoidFitter(initial_state, initial_covariance, process_noise, measurement_noise);
}

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

    auto t_window = 500.;
    CentroidEMA ema_calculator(tau, t_window);

    // Sinusoid fitter for each polarity's x and y coordinates
    auto x_fitter = create_iekf();
    auto y_fitter = create_iekf();

    // OpenCV visualization setup
    const int crop_size = 200;
    const int vis_width = 1920;
    cv::Mat crop_vis = cv::Mat::ones(crop_size, vis_width, CV_8UC3);
    // Initialize the crop visualization with a white background
    crop_vis.setTo(cv::Scalar(255, 255, 255)); // Set to white background
    cv::Mat crop_vis_compensated = cv::Mat::zeros(crop_size, vis_width, CV_8UC3);
    const int crop_x_start = camera_width / 2 - crop_size / 2;
    const int crop_y_start = camera_height / 2 - crop_size / 2;
    if (do_plot) {
        cv::namedWindow("X-values vs. Time", cv::WINDOW_NORMAL);
        cv::namedWindow("Compensated Events", cv::WINDOW_NORMAL);
    }

    // auto vis = std::make_shared<open3d::visualization::Visualizer>();
    // std::shared_ptr<open3d::geometry::PointCloud> events_pcd_p0;
    // std::shared_ptr<open3d::geometry::LineSet> centroids_trace;
    // std::shared_ptr<open3d::geometry::LineSet> sinusoid_trace;

    // if (do_plot) {
    //     vis->CreateVisualizerWindow("Events and Centroids", 1600, 900);
    //     events_pcd_p0 = std::make_shared<open3d::geometry::PointCloud>();
    //     centroids_trace = std::make_shared<open3d::geometry::LineSet>();
    //     sinusoid_trace = std::make_shared<open3d::geometry::LineSet>();
    //     vis->AddGeometry(events_pcd_p0);
    //     vis->AddGeometry(centroids_trace);
    //     vis->AddGeometry(sinusoid_trace);

    //     auto bounding_box = std::make_shared<open3d::geometry::AxisAlignedBoundingBox>(
    //             Eigen::Vector3d(0, 0, 0.0),
    //             Eigen::Vector3d(camera_width, camera_height, 1));

    //     // Set bounding box color for visibility
    //     bounding_box->color_ = Eigen::Vector3d(0.0, 1.0, 0.0);  // Green color

    //     vis->AddGeometry(bounding_box);
    // }

    Metavision::timestamp last_print_time = 0;
    Metavision::timestamp first_event_t = -1; // To normalize time for visualization
    const Metavision::timestamp fitting_duration = 0.5 * 1000 * 1000; // 2 seconds
    bool fitting_complete = false;

    double time_scale = 1000.;
    bool filter_converged = false;
    std::mutex _mtx;
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (first_event_t < 0 && begin != end) {
            first_event_t = begin->t;
        }

        std::lock_guard<std::mutex> lock(_mtx);
        for (auto it = begin; it != end; ++it) {
            const auto &event = *it;
            const Metavision::timestamp current_relative_t = (event.t - first_event_t);
            const double current_t_sec = current_relative_t / 1.e6;

            // Normalize time for x-axis in the visualization
            const int vis_x_time = static_cast<int>(current_t_sec * (time_scale)); // Rescale time for visualization

            // Stop processing if the visualization reaches the edge of the window
            if (vis_x_time >= vis_width) {
                // save the opencv visualization to a file
                cv::imwrite("/home/viciopoli/STARS/courses/centroid_ema_visualization.png", crop_vis);
                cv::imwrite("/home/viciopoli/STARS/courses/centroid_ema_visualization_compensated.png",
                            crop_vis_compensated);
                std::cout << "Saved visualization to /home/viciopoli/STARS/courses/centroid_ema_visualization.png"
                          << std::endl;
                std::cout
                        << "Saved visualization to /home/viciopoli/STARS/courses/centroid_ema_visualization_compensated.png"
                        << std::endl;

                // print the fitting results
                std::cout << "\n--- Fitting Results ---\n" << std::endl;
                std::cout << "X Fitter Frequencies:" << std::endl;
                x_fitter.print_frequencies();
                std::cout << "Y Fitter Frequencies:" << std::endl;
                y_fitter.print_frequencies();

                if (camera.is_running()) {
                    camera.stop();
                }
                continue;
            }

            // Check if the event is within the crop region for OpenCV visualization
            if (event.x >= crop_x_start && event.x < crop_x_start + crop_size &&
                event.y >= crop_y_start && event.y < crop_y_start + crop_size) {
                int vis_y_pos = event.x - crop_x_start;
                cv::Point point(vis_x_time, vis_y_pos);
                cv::circle(crop_vis, point, 1, (event.p == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255), -1);
            }

            // Motion compensation for every event, once the filter is stable
            bool filter_is_ready = (event.t - first_event_t) > fitting_duration;
            if (filter_is_ready) {
                if (event.x >= crop_x_start && event.x < crop_x_start + crop_size &&
                    event.y >= crop_y_start && event.y < crop_y_start + crop_size) {

                    // Predict at the event's actual time for accurate compensation
                    double pred_x_for_comp = x_fitter.predict(current_t_sec);

                    int vis_y_compensated = (event.x - pred_x_for_comp) + (crop_size / 2);
                    if (vis_y_compensated >= 0 && vis_y_compensated < crop_size) {
                        cv::Point point_comp(vis_x_time, vis_y_compensated);
                        cv::circle(crop_vis_compensated, point_comp, 1,
                                   (event.p == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255), -1);
                    }
                }
            }

            auto result = ema_calculator.update(event);
            if (result.has_value()) {
                auto &[centroid, centroid_ts] = *result;

                double current_t_sec_update = (centroid_ts - first_event_t) / 1.e6;

                x_fitter.update(current_t_sec_update, centroid[0]);
                y_fitter.update(current_t_sec_update, centroid[1]);

                filter_converged = (centroid_ts - first_event_t) > fitting_duration;

                // Check if centroid x-coordinate is within the crop region
                float cx = centroid[0];
                int vis_y_pos = cx - crop_x_start;
                if (vis_y_pos >= 0 && vis_y_pos < crop_size) {
                    const int vis_x_time_centroid = static_cast<int>(current_t_sec_update * (time_scale));
                    cv::Point centroid_point(vis_x_time_centroid, vis_y_pos + 50);
                    cv::circle(crop_vis, centroid_point, 3, cv::Scalar(0, 255, 0), -1);
                }
                if (filter_converged) {
                    // Predict ahead to compensate for the processing delay and align the signals
                    double time_to_predict = current_t_sec_update - (t_window / 2.e6);
                    double pred_x = x_fitter.predict(time_to_predict);
                    int pred_vis_y_pos = pred_x - crop_x_start;

                    if (pred_vis_y_pos >= 0 && pred_vis_y_pos < crop_size) {
                        const int vis_x_time_pred = static_cast<int>(time_to_predict * time_scale) - 20;
                        cv::Point pred_point(vis_x_time_pred, pred_vis_y_pos + 60);
                        cv::circle(crop_vis, pred_point, 3, cv::Scalar(0, 165, 255), -1);
                    }
                }
            }
            if (filter_converged) {
                // Motion compensation
                if (event.x >= crop_x_start && event.x < crop_x_start + crop_size &&
                    event.y >= crop_y_start && event.y < crop_y_start + crop_size) {
                    // Predict at the event's actual time for accurate compensation
                    double pred_x_for_comp = x_fitter.predict(current_t_sec-((2000) / 1e6));
                    int vis_y_compensated = (event.x - pred_x_for_comp) + (crop_size / 2);
                    if (vis_y_compensated >= 0 && vis_y_compensated < crop_size) {
                        cv::Point point_comp(vis_x_time, vis_y_compensated);
                        cv::circle(crop_vis_compensated, point_comp, 1,
                                   (event.p == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255), -1);
                    }
                }
            }
        }
        const Metavision::timestamp current_relative_t_end = (end - 1 > begin) ? ((end - 1)->t - first_event_t) : 0;
        if (!fitting_complete && current_relative_t_end >= fitting_duration) {
            fitting_complete = true;
            std::cout << "\n--- Sinusoid Fitting Complete ---\n" << std::endl;
        }

        if (end > begin) {
            Metavision::timestamp current_t = (end - 1)->t;
            if (current_t - last_print_time > 100000) { // Print every ~100ms
                last_print_time = current_t;
                std::cout << "Timestamp: " << std::fixed << std::setprecision(2) << current_t / 1e6 << "s" << std::endl;
                const auto &centroid = ema_calculator.get_centroids();
                std::cout << "  Centroid: (" << std::fixed << std::setprecision(2)
                          << centroid[0] << ", " << centroid[1] << ")" << std::endl;
            }
        }
    });

    camera.start();

    if (do_plot) {
        while (camera.is_running()) {
            cv::Mat frame_to_show;
            cv::Mat frame_to_show_compensated;
            {
                std::lock_guard<std::mutex> lock(_mtx);
                // vis->UpdateGeometry(events_pcd_p0);
                // vis->UpdateGeometry(centroids_trace);
                // if (filter_converged) vis->UpdateGeometry(sinusoid_trace);
                // vis->PollEvents();
                // vis->UpdateRender();

                crop_vis.copyTo(frame_to_show);
                crop_vis_compensated.copyTo(frame_to_show_compensated);
//                crop_vis.setTo(cv::Scalar(0, 0, 0));
            }

            if (!frame_to_show.empty()) {
                cv::imshow("X-values vs. Time", frame_to_show);
            }
            if (!frame_to_show_compensated.empty()) {
                cv::imshow("Compensated Events", frame_to_show_compensated);
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
    // while (vis->PollEvents()) {
    //     vis->UpdateRender();
    // }
    // vis->DestroyVisualizerWindow();
    cv::destroyAllWindows();


    return 0;
}
