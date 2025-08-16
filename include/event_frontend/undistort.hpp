//
// Created by viciopoli on 18/02/25.
//

#ifndef PROJECT_UNDISTORT_H
#define PROJECT_UNDISTORT_H

#include <vector>
#include <iostream>
#include <fstream>
#include <functional>
#include "json.hpp"
#include <opencv2/opencv.hpp>

class Undistort {
public:
    Undistort(std::string filepath) {
        std::vector<double> K, D;

        if (filepath.empty()) {
            // warning text yellow
            std::cout << "\033[33mWarning: No calibration file provided. Undistortion will not be applied.\033[0m"
                      << std::endl;
            _is_calibrated = false;
            return;
        }
        _is_calibrated = true;

        parseFile(filepath, K, D);

        // populate the look-up table
        _undistort_x.resize(_width * _height);
        _undistort_y.resize(_width * _height);

        if (K.empty() || D.empty()) {
            // warning text yellow
            std::cout << "\033[33mWarning: K or D is empty. Undistortion will not be applied.\033[0m" << std::endl;
            return;
        }
        // it's rad tan in the prophesee camera
        // populate the LUT with the distortion function rad tan

        cv::Mat K_mat = cv::Mat(3, 3, CV_64F, K.data());
        cv::Mat D_mat = cv::Mat(1, D.size(), CV_64F, D.data());

        std::vector<cv::Point2f> distorted_points = {cv::Point2f(0, 0)};
        std::vector<cv::Point2f> undistorted_points;
        for (int y = 0; y < _height; ++y) {
            for (int x = 0; x < _width; ++x) {
                distorted_points[0].x = x;
                distorted_points[0].y = y;
                // undistort points with termination criteria
                cv::undistortPoints(distorted_points, undistorted_points, K_mat, D_mat);

                // Convert normalized coordinates back to pixel coordinates
                _undistort_x[y * _width + x] = static_cast<coords_type>(
                        undistorted_points[0].x * K_mat.at<double>(0, 0) + K_mat.at<double>(0, 2));
                _undistort_y[y * _width + x] = static_cast<coords_type>(
                        undistorted_points[0].y * K_mat.at<double>(1, 1) + K_mat.at<double>(1, 2));
            }
        }
//        _populated = true;
    }

    // function that allows to populate the look-up table with a distortion function provided by the user
    void populate(std::function<std::pair<double, double>(int, int)> distortion) {
        for (int y = 0; y < _height; ++y) {
            for (int x = 0; x < _width; ++x) {
                auto [x_undist, y_undist] = distortion(x, y);
                _undistort_x[y * _width + x] = x_undist;
                _undistort_y[y * _width + x] = y_undist;
            }
        }
    }

    // override operator() to get the undistorted coordinates
    std::pair<unsigned short, unsigned short> operator()(int x, int y) const {
        if (!_is_calibrated) { return {static_cast<double>(x), static_cast<double>(y)}; }

        const auto idx = y * _width + x;
        return {_undistort_x[idx], _undistort_y[idx]};
    }

    void operator()(unsigned short x, unsigned short y, unsigned short &new_x, unsigned short &new_y) {
        if (!_is_calibrated) {
            new_x = x;
            new_y = y;
            return;
        }

        const auto idx = y * _width + x;
        new_x = _undistort_x[idx];
        new_y = _undistort_y[idx];
    }

    // override operator() to get the undistorted coordinates
    Metavision::EventCD operator()(const Metavision::EventCD &ev) const {
        if (!_is_calibrated) { return ev; }

        const auto idx = ev.y * _width + ev.x;
        return {
                _undistort_x[idx],
                _undistort_y[idx],
                ev.p, ev.t
        };
    }

private:
    using coords_type = unsigned short;
    // make a look-up table for the undistortion
    std::vector<coords_type> _undistort_x, _undistort_y;
    int _width, _height;
    bool _is_calibrated = false;

    void parseFile(std::string &filepath, std::vector<double> &K, std::vector<double> &D) {
        try {
            // Open the JSON file
            std::ifstream file(filepath.c_str());
            if (!file.is_open()) {
                std::cerr << "Failed to open intrinsics.json" << std::endl;
                return;
            }

            // Parse the JSON content
            auto data = nlohmann::json::parse(file);

            // Extract image size
            auto img_size = data["image_size"];
            _width = img_size[0];
            _height = img_size[1];

            // Extract camera matrix (K) data
            K = data["camera_matrix"]["data"].get<std::vector<double>>();

            // Extract distortion coefficients (D) data
            D = data["distortion_coefficients"]["data"].get<std::vector<double>>();

            // Print results (optional)
            std::cout << "Camera Matrix (K):" << std::endl;
            for (size_t i = 0; i < K.size(); i++) {
                std::cout << K[i] << ((i % 3 == 2) ? "\n" : "\t");
            }

            std::cout << "\nDistortion Coefficients (D):" << std::endl;
            for (const auto &d: D) {
                std::cout << d << " ";
            }
            std::cout << std::endl;

        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return;
        }

    }
};

#endif //PROJECT_UNDISTORT_H
