/**
 * @file params_loader.h
 * @brief Command-line argument parser and application configuration loader.
 *
 * This file handles the parsing of runtime parameters using Boost Program Options
 * and initializes the Metavision camera object based on the input source (live stream or file).
 *
 * It centralizes configuration management, making it easy to add new parameters
 * without cluttering the main logic.
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

#ifndef PROJECT_PARAMS_LOADER_H
#define PROJECT_PARAMS_LOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <filesystem>

#include <boost/program_options.hpp>
#ifdef OPENEB
#include <metavision/sdk/stream/camera.h>
#else
#include <metavision/sdk/driver/camera.h>
#endif

namespace po = boost::program_options;

namespace VibES {

    /**
     * @struct Params
     * @brief Container for application configuration parameters.
     */
    struct Params {
        std::string input_path;          ///< Path to .raw or .hdf5 event file. Empty for live camera.
        std::string calib_file;          ///< Path to camera calibration JSON file.
        std::string output_folder = "output"; ///< Directory for saving results.

        // Single tracker initialization (legacy/simple mode)
        int tracker_x = 0;               ///< Initial X coordinate for single tracker.
        int tracker_y = 0;               ///< Initial Y coordinate for single tracker.

        // Multi-tracker initialization
        std::vector<int> trackers_x;     ///< List of X coordinates for multiple trackers.
        std::vector<int> trackers_y;     ///< List of Y coordinates for multiple trackers.

        int tracker_size = 10;           ///< Size of the tracking patch (radius or side length).
        int iekf_iterations = 1;         ///< Number of IEKF update iterations per event batch.
        bool nocompensation = false;     ///< If true, disables motion compensation (debugging).
        int front_trackers = 0;          ///< Number of specific "front" trackers (implementation specific).
        int time_window_us = 10000;      ///< Integration time window in microseconds.
    };

    /**
     * @class ParamsLoader
     * @brief Handles parsing of command line arguments and camera initialization.
     */
    class ParamsLoader {
    public:
        /**
         * @brief Constructor. Parses arguments and initializes the camera.
         * * @param argc Argument count from main().
         * @param argv Argument vector from main().
         * @throws std::runtime_error If parsing fails, help is requested, or camera init fails.
         */
        ParamsLoader(int argc, char *argv[]) {
            params = std::make_shared<Params>();

            po::options_description desc("Allowed options");
            desc.add_options()
                    ("help,h", "Produce help message")
                    ("input-event-file,i", po::value<std::string>(&params->input_path),
                     "Path to input event file (RAW or HDF5). If not specified, the first available live camera is used.")
                    ("calibration-file,c", po::value<std::string>(&params->calib_file)->default_value(""),
                     "Path to camera calibration file (JSON). If specified, events will be undistorted.")

                    // Tracker Configuration
                    ("tracker-x", po::value<int>(&params->tracker_x)->default_value(0),
                     "Initial X position for single tracker (default: 0).")
                    ("tracker-y", po::value<int>(&params->tracker_y)->default_value(0),
                     "Initial Y position for single tracker (default: 0).")
                    ("tracker-size", po::value<int>(&params->tracker_size)->default_value(10),
                     "Size of the tracker patch in pixels (default: 10).")
                    ("trackers-x", po::value<std::vector<int>>(&params->trackers_x)->multitoken(),
                     "List of X positions for multiple trackers (space separated).")
                    ("trackers-y", po::value<std::vector<int>>(&params->trackers_y)->multitoken(),
                     "List of Y positions for multiple trackers (space separated).")
                    ("front-trackers", po::value<int>(&params->front_trackers)->default_value(0),
                     "Number of front-layer trackers (default: 0).")

                    // Algorithm Parameters
                    ("iekf-iterations", po::value<int>(&params->iekf_iterations)->default_value(1),
                     "Number of iterations for the IEKF update step (default: 1).")
                    ("nocompensation", po::bool_switch(&params->nocompensation)->default_value(false),
                     "Disable motion compensation (for debugging/visualization only).")
                    ("time-window-us", po::value<int>(&params->time_window_us)->default_value(10000),
                     "Time window for event accumulation/visualization in microseconds (default: 10000).")

                    // Output
                    ("output-folder,o", po::value<std::string>(&params->output_folder)->default_value("output"),
                     "Folder to save output frames and logs (default: 'output').");

            po::variables_map vm;

            try {
                po::store(po::parse_command_line(argc, argv, desc), vm);
                po::notify(vm);
            } catch (const po::error &e) {
                std::cerr << "[ParamsLoader] CLI Error: " << e.what() << std::endl;
                std::cerr << desc << std::endl;
                throw std::runtime_error("Failed to parse command line arguments.");
            }

            if (vm.count("help")) {
                std::cout << desc << std::endl;
                // We throw to stop execution cleanly in main
                throw std::runtime_error("Help requested.");
            }

            // Ensure output directory exists
            if (!std::filesystem::exists(params->output_folder)) {
                try {
                    std::filesystem::create_directories(params->output_folder);
                } catch (const std::exception& e) {
                    std::cerr << "[ParamsLoader] Warning: Could not create output folder: " << e.what() << std::endl;
                }
            }

            // Camera Initialization
            try {
                if (!params->input_path.empty()) {
                    std::cout << "[ParamsLoader] Opening file: " << params->input_path << std::endl;
                    camera = Metavision::Camera::from_file(params->input_path);
                } else {
                    std::cout << "[ParamsLoader] Opening live camera..." << std::endl;
                    camera = Metavision::Camera::from_first_available();
                }
            } catch (const Metavision::CameraException &e) {
                std::cerr << "[ParamsLoader] Camera initialization error: " << e.what() << std::endl;
                throw std::runtime_error("Failed to initialize camera.");
            }

            // Validation
            if (params->trackers_x.size() != params->trackers_y.size()) {
                std::cerr << "[ParamsLoader] Error: The number of tracker X and Y positions must match." << std::endl;
                std::cerr << "  Count X: " << params->trackers_x.size() << std::endl;
                std::cerr << "  Count Y: " << params->trackers_y.size() << std::endl;
                throw std::runtime_error("Tracker positions mismatch.");
            }
        }

        /**
         * @brief Output stream operator for logging parameters.
         */
        friend std::ostream &operator<<(std::ostream &os, const ParamsLoader &loader) {
            os << "\033[1;34m=== Configuration ===\n";
            os << " Source:         " << (loader.params->input_path.empty() ? "Live Camera" : loader.params->input_path) << "\n";
            os << " Calibration:    " << (loader.params->calib_file.empty() ? "None" : loader.params->calib_file) << "\n";
            os << " Output Folder:  " << loader.params->output_folder << "\n";
            os << " Tracker Size:   " << loader.params->tracker_size << " px\n";
            os << " IEKF Iterations:" << loader.params->iekf_iterations << "\n";
            os << " Compensation:   " << (loader.params->nocompensation ? "DISABLED" : "ENABLED") << "\n";
            os << " Time Window:    " << loader.params->time_window_us << " us\n";

            if (!loader.params->trackers_x.empty()) {
                os << " Multi-trackers: " << loader.params->trackers_x.size() << " initialized.\n";
            }
            os << "\033[0m" << std::endl;
            return os;
        }

        std::shared_ptr<Params> params; ///< Access to parsed parameters.
        Metavision::Camera camera;      ///< Initialized Metavision camera object.
    };
}
#endif //PROJECT_PARAMS_LOADER_H