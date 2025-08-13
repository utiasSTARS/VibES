//
// Created by viciopoli on 18/02/25.
//

#ifndef PROJECT_UNDISTORT_H
#define PROJECT_UNDISTORT_H

#include <vector>
#include <iostream>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include "json.hpp"
#include <opencv2/opencv.hpp>

class Undistort {
public:
    using coords_type = unsigned short;

    explicit Undistort(const std::string &filepath, int width = 0, int height = 0)
            : _width(width), _height(height), _is_calibrated(false) {
        if (filepath.empty()) {
            std::cout << "\033[33mWarning: No calibration file provided. Undistortion will not be applied.\033[0m"
                      << std::endl;
            if (width <= 0 || height <= 0) {
                throw std::runtime_error("Image dimensions must be set if no calibration file is provided.");
            }
            _is_calibrated = false;
            default_LUT_init();
            return;
        }

        std::vector<double> K, D;
        if (!parseFile(filepath, K, D)) {
            std::cout
                    << "\033[33mWarning: Failed to parse calibration file. Undistortion will not be applied.\033[0m"
                    << std::endl;
            _is_calibrated = false;
            default_LUT_init();
            return;
        }

        if (!validateCalibration(K, D)) {
            std::cout << "\033[33mWarning: Invalid calibration parameters. Undistortion will not be applied.\033[0m"
                      << std::endl;
            _is_calibrated = false;
            default_LUT_init();
            return;
        }

        _is_calibrated = true;
        initializeLUT(K, D);
    }

    // Move constructor and assignment for efficiency
    Undistort(Undistort &&other)
    noexcept
            : _undistort_x(std::move(other._undistort_x)), _undistort_y(std::move(other._undistort_y)),
              _mask_out_of_frame(std::move(other._mask_out_of_frame)), _width(other._width), _height(other._height),
              _is_calibrated(other._is_calibrated) {
        other._is_calibrated = false;
    }

    Undistort &operator=(Undistort &&other) noexcept {
        if (this != &other) {
            _undistort_x = std::move(other._undistort_x);
            _undistort_y = std::move(other._undistort_y);
            _mask_out_of_frame = std::move(other._mask_out_of_frame);
            _width = other._width;
            _height = other._height;
            _is_calibrated = other._is_calibrated;
            other._is_calibrated = false;
        }
        return *this;
    }

    // Deleted copy operations to prevent accidental expensive copies
    Undistort(
            const Undistort &) = delete;

    Undistort &operator=(const Undistort &) = delete;

    // Custom distortion function - replaces existing LUT
    void populate(std::function<std::pair<double, double>(int, int)> distortion) {
        if (_width <= 0 || _height <= 0) {
            throw std::runtime_error("Image dimensions not set. Cannot populate LUT.");
        }

        _undistort_x.resize(_width * _height);
        _undistort_y.resize(_width * _height);
        _mask_out_of_frame.resize(_width * _height);

        for (int y = 0; y < _height; ++y) {
            for (int x = 0; x < _width; ++x) {
                auto [x_undist, y_undist] = distortion(x, y);

                const auto idx = y * _width + x;
                _undistort_x[idx] = static_cast<coords_type>(std::clamp(x_undist, 0.0,
                                                                        static_cast<double>(_width - 1)));
                _undistort_y[idx] = static_cast<coords_type>(std::clamp(y_undist, 0.0,
                                                                        static_cast<double>(_height - 1)));
                _mask_out_of_frame[idx] = (x_undist < 0 || x_undist >= _width || y_undist < 0 ||
                                           y_undist >= _height);
            }
        }
        _is_calibrated = true;
    }

    // Fast runtime operators - minimal overhead
    inline std::pair<coords_type, coords_type> operator()(coords_type x, coords_type y) const noexcept {
        // we are sure the coordinates are within the image bounds

        const auto idx = y * _width + x;
        return {_undistort_x[idx], _undistort_y[idx]};
    }

    inline void operator()(coords_type x, coords_type y, coords_type &new_x, coords_type &new_y) const noexcept {
        // we are sure the coordinates are within the image bounds

        const auto idx = y * _width + x;
        new_x = _undistort_x[idx];
        new_y = _undistort_y[idx];
    }

    inline void operator()(coords_type x, coords_type y, coords_type &new_x, coords_type &new_y,
                           bool &out_of_frame) const noexcept {
        // we are sure the coordinates are within the image bounds

        const auto idx = y * _width + x;
        new_x = _undistort_x[idx];
        new_y = _undistort_y[idx];
        out_of_frame = _mask_out_of_frame[idx];
    }

    // Metavision event handling
    inline Metavision::EventCD operator()(const Metavision::EventCD &ev) const noexcept {
        // we are sure the coordinates are within the image bounds

        const auto idx = ev.y * _width + ev.x;
        return {_undistort_x[idx], _undistort_y[idx], ev.p, ev.t};
    }

    // Getters for debugging/validation
    bool is_calibrated() const noexcept { return _is_calibrated; }

    int width() const noexcept { return _width; }

    int height() const noexcept { return _height; }

private:
    std::vector<coords_type> _undistort_x, _undistort_y;
    std::vector<bool> _mask_out_of_frame;
    int _width = 0, _height = 0;
    bool _is_calibrated = false;

    bool parseFile(const std::string &filepath, std::vector<double> &K, std::vector<double> &D) {
        try {
            std::ifstream file(filepath);
            if (!file.is_open()) {
                std::cerr << "Failed to open: " << filepath << std::endl;
                return false;
            }

            nlohmann::json data;
            file >> data;

            // Extract image size
            if (!data.contains("image_size") || !data["image_size"].is_array() || data["image_size"].size() != 2) {
                std::cerr << "Invalid or missing image_size in calibration file" << std::endl;
                return false;
            }

            auto img_size = data["image_size"];
            _width = img_size[0].get<int>();
            _height = img_size[1].get<int>();

            if (_width <= 0 || _height <= 0) {
                std::cerr << "Invalid image dimensions: " << _width << "x" << _height << std::endl;
                return false;
            }

            // Extract camera matrix (K)
            if (!data.contains("camera_matrix") || !data["camera_matrix"].contains("data")) {
                std::cerr << "Missing camera_matrix in calibration file" << std::endl;
                return false;
            }
            K = data["camera_matrix"]["data"].get<std::vector<double>>();

            // Extract distortion coefficients (D)
            if (!data.contains("distortion_coefficients") || !data["distortion_coefficients"].contains("data")) {
                std::cerr << "Missing distortion_coefficients in calibration file" << std::endl;
                return false;
            }
            D = data["distortion_coefficients"]["data"].get<std::vector<double>>();

            return true;

        } catch (const std::exception &e) {
            std::cerr << "Error parsing calibration file: " << e.what() << std::endl;
            return false;
        }
    }

    bool validateCalibration(const std::vector<double> &K, const std::vector<double> &D) const {
        if (K.size() != 9) {
            std::cerr << "Camera matrix must have 9 elements, got: " << K.size() << std::endl;
            return false;
        }

        if (D.empty()) {
            std::cerr << "Distortion coefficients cannot be empty" << std::endl;
            return false;
        }

        // Check focal lengths are positive
        if (K[0] <= 0 || K[4] <= 0) {
            std::cerr << "Invalid focal lengths: fx=" << K[0] << ", fy=" << K[4] << std::endl;
            return false;
        }

        return true;
    }

    void initializeLUT(const std::vector<double> &K, const std::vector<double> &D) {
        _undistort_x.resize(_width * _height);
        _undistort_y.resize(_width * _height);
        _mask_out_of_frame.resize(_width * _height);

        cv::Mat K_mat = cv::Mat(3, 3, CV_64F, const_cast<double *>(K.data()));
        cv::Mat D_mat = cv::Mat(1, static_cast<int>(D.size()), CV_64F, const_cast<double *>(D.data()));

        std::vector<cv::Point2f> distorted_points(1);
        std::vector<cv::Point2f> undistorted_points;

        for (int y = 0; y < _height; ++y) {
            for (int x = 0; x < _width; ++x) {
                distorted_points[0] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));

                cv::undistortPoints(distorted_points, undistorted_points, K_mat, D_mat);

                // Convert normalized coordinates back to pixel coordinates
                double x_new = undistorted_points[0].x * K[0] + K[2];
                double y_new = undistorted_points[0].y * K[4] + K[5];

                const auto idx = y * _width + x;
                bool out_of_bounds = (x_new < 0 || x_new >= _width || y_new < 0 || y_new >= _height);

                _mask_out_of_frame[idx] = out_of_bounds;
                _undistort_x[idx] = static_cast<coords_type>(std::clamp(x_new, 0.0,
                                                                        static_cast<double>(_width - 1)));
                _undistort_y[idx] = static_cast<coords_type>(std::clamp(y_new, 0.0,
                                                                        static_cast<double>(_height - 1)));
            }
        }

        std::cout << "Initialized undistortion LUT for " << _width << "x" << _height << " image" << std::endl;
    }

    void default_LUT_init() {
        std::cout << "\033[33mUsing default undistortion LUT for " << _width << "x" << _height
                  << " image. No calibration file provided.\033[0m" << std::endl;
        const size_t total_size = _width * _height;
        if (_undistort_x.size() != total_size) {
            _undistort_x.resize(total_size);
            _undistort_y.resize(total_size);
            _mask_out_of_frame.resize(total_size);
        }

        for (int y = 0; y < _height; ++y) {
            for (int x = 0; x < _width; ++x) {
                const auto idx = y * _width + x;
                _undistort_x[idx] = static_cast<coords_type>(x);
                _undistort_y[idx] = static_cast<coords_type>(y);
                _mask_out_of_frame[idx] = false;
            }
        }
    }
};

#endif //PROJECT_UNDISTORT_H