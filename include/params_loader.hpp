//
// Created by viciopoli on 29/06/25.
//

#ifndef PROJECT_PARAMS_LOADER_H
#define PROJECT_PARAMS_LOADER_H

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

#include "event_frontend/ema.hpp"
#include "estimator/iekf_sinusoid_fitter.hpp"

namespace po = boost::program_options;
namespace HARMEDA {
    struct Params {
        std::string input_path;
        std::string calib_file; // Path to camera calibration file
        std::string output_folder = "output"; // Default output folder for frames
        double tau = 100000.0; // Default time constant for EMA in microseconds
        bool do_plot = false;  // Flag to enable 3D visualization
        int tracker_x = 0; // Tracker x position
        int tracker_y = 0; // Tracker y position
        int tracker_size = 10; // Tracker size
    };

    class ParamsLoader {
    public:
        ParamsLoader(int argc, char *argv[]) {
            params = std::make_shared<Params>();

            po::options_description desc("Allowed options");
            desc.add_options()
                    ("help,h", "produce help message")
                    ("input-event-file,i", po::value<std::string>(&params->input_path),
                     "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
                    ("calibration-file,c", po::value<std::string>(&params->calib_file)->default_value(""),
                     "Path to camera calibration file (optional). If specified, events will be undistorted.")
                    ("tracker-x", po::value<int>(&params->tracker_x)->default_value(0),
                     "X position of the tracker in pixels (default: 0).")
                    ("tracker-y", po::value<int>(&params->tracker_y)->default_value(0),
                     "Y position of the tracker in pixels (default: 0).")
                    ("tracker-size", po::value<int>(&params->tracker_size)->default_value(10),
                     "Size of the tracker in pixels (default: 10).")
                    ("output-folder,o", po::value<std::string>(&params->output_folder)->default_value("output"),
                     "Folder to save output frames (default: 'output').");

            po::variables_map vm;

            try {
                po::store(po::parse_command_line(argc, argv, desc), vm);
                po::notify(vm);
            } catch (const po::error &e) {
                std::cerr << "Error: " << e.what() << std::endl;
                std::cerr << desc << std::endl;
                throw std::runtime_error("Failed to parse command line arguments.");
            }

            if (vm.count("help")) {
                std::cout << desc << std::endl;
                throw std::runtime_error("Help requested.");
            }

//            params->do_plot = vm.count("plot");

            try {
                if (!params->input_path.empty()) {
                    camera = Metavision::Camera::from_file(params->input_path);
                } else {
                    camera = Metavision::Camera::from_first_available();
                }
            } catch (const Metavision::CameraException &e) {
                std::cerr << "Camera initialization error: " << e.what() << std::endl;
                throw std::runtime_error("Failed to initialize camera.");
            }
        }

        std::shared_ptr<Params> params;

        Metavision::Camera camera;
    };
}
#endif //PROJECT_PARAMS_LOADER_H
