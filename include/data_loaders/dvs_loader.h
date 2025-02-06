//
// Created by viciopoli on 06/02/25.
//

#ifndef PROJECT_DVS_LOADER_H
#define PROJECT_DVS_LOADER_H

class DVSLoader {
public:

    DVSLoader() = default;

    DVSLoader(int argc, char **argv) {
        if (load(argc, argv) != 0) {
            throw std::runtime_error("Failed to load DVS data.");
        }
    }


    std::shared_ptr<dv::camera::CameraGeometry> getGeometry() {
        return geometry;
    }

    std::shared_ptr<dv::io::CameraInputBase> getReader() {
        return std::move(reader);
    }

private:
    // load() returns 0 on success and non-zero on failure.
    // Usage: <executable> [help|camera|file] [calibration_file]
    //   - "help"    : Displays this help message.
    //   - "camera"  : Reads from a camera.
    //   - "file"    : Reads from a specified file.
    //   - [calibration_file] is an optional path to a camera calibration file.
    int load(int argc, char **argv) {
        // Default calibration file path if not overridden.
        std::string calib_file;
        bool load_calibration = false;

        if (argc >= 2) {
            std::string arg1 = argv[1];

            // Handle help request.
            if (arg1 == "help") {
                std::cout << "Usage: " << argv[0] << " [help|camera|file] [calibration_file]\n";
                std::cout << "  help             - Display this help message\n";
                std::cout << "  camera           - Read from camera\n";
                std::cout << "  file             - Read from a specified file\n";
                std::cout << "  calibration_file - Optional path to camera calibration file\n";
                return 0;
            }
                // If the first argument is "camera", read from camera.
            else if (arg1 == "camera") {
                std::cout << "Reading from camera.\n";
                reader = std::make_shared<dv::io::CameraCapture>();
            } else {
                // Otherwise, assume the first argument is a file path.
                std::cout << "Reading from file: " << arg1 << std::endl;
                reader = std::make_shared<dv::io::MonoCameraRecording>(arg1);
            }

            // Check if a calibration file path is provided.
            if (argc >= 3) {
                calib_file = argv[2]; // dv processing is going to check for the file
                load_calibration = true;
            }
        } else {
            // No arguments provided: run in simulation mode.
            int target_freq = 700;
            std::cout << "No file provided. Using simulator with freq " << target_freq
                      << " rad/s (" << rad2Hz(target_freq) << " Hz)" << std::endl;
            reader = std::make_shared<EventsFreqCalibPattern>(0, cv::Size(640, 480),
                                                              target_freq, 5, 5, 0., true);
        }

        // Load calibration if a calibration file was provided.
        if (load_calibration) {
            try {
                const auto calibrationSet = dv::camera::CalibrationSet::LoadFromFile(calib_file);
                dv::camera::calibrations::CameraCalibration dvx_calib;

                if (!calibrationSet.getCameraList().empty()) {
                    const auto &calibs = calibrationSet.getCameraCalibrations();
                    // Use the first calibration available.
                    dvx_calib = calibs.begin()->second;
                    std::cout << "Found calibration for camera with name ["
                              << dvx_calib.name << "]" << std::endl;
                } else {
                    std::cerr << "No camera calibrations found in the file." << std::endl;
                    return 1;
                }

                geometry = std::make_shared<dv::camera::CameraGeometry>(dvx_calib.getCameraGeometry());
            } catch (const std::exception &e) {
                std::cerr << "Failed to load camera calibration from \""
                          << calib_file << "\": " << e.what() << std::endl;
                return 1;
            }
        }

        return 0;
    }

    // Public members for the reader and the camera calibration geometry.
    std::shared_ptr<dv::io::CameraInputBase> reader = nullptr;
    std::shared_ptr<dv::camera::CameraGeometry> geometry = nullptr;
};


#endif //PROJECT_DVS_LOADER_H
