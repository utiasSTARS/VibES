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

namespace po = boost::program_options;
namespace HARMEDA {
    struct Params {
        std::string input_path;
        std::string calib_file; // Path to camera calibration file
        std::string output_folder = "output"; // Default output folder for frames
        int tracker_x = 0; // Tracker x position
        int tracker_y = 0; // Tracker y position
        int tracker_size = 10; // Tracker size
        int iekf_iterations = 1; // Number of iterations for IEKF fitting
        bool nocompensation = false; // Flag for no compensation
    };

    class ParamsLoader {
    public:

        friend std::ostream &operator<<(std::ostream &os, const ParamsLoader &loader) {
            os << "\033[1;34mParameters:\n";
            os << "Input Path: " << loader.params->input_path << "\n"
               << "Calibration File: " << loader.params->calib_file << "\n"
               << "Output Folder: " << loader.params->output_folder << "\n"
               << "Tracker Size: " << loader.params->tracker_size << "\n"
               << "IEKF Iterations: " << loader.params->iekf_iterations;
            os << "\033[0m" << std::endl;
            return os;
        }

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
                    ("iekf-iterations", po::value<int>(&params->iekf_iterations)->default_value(1),
                     "Number of iterations for IEKF fitting (default: 1).")
                    ("output-folder,o", po::value<std::string>(&params->output_folder)->default_value("output"),
                     "Folder to save output frames (default: 'output').")
                    ("nocompensation", po::bool_switch(&params->nocompensation)->default_value(false),
                     "Disable compensation of events (default: false).");

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
