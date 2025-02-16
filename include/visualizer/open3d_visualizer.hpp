//
// Created by viciopoli on 19/11/24.
//

#ifndef PROJECT_OPEN3D_VISUALIZER_H
#define PROJECT_OPEN3D_VISUALIZER_H

#include <algorithm>
#include <vector>
#include <tuple>
#include <open3d/Open3D.h>

class Open3DVisualizer {
public:
    Open3DVisualizer(int width = 640, int height = 480) : _width(width), _height(height) {
        vis = std::make_shared<open3d::visualization::Visualizer>();
        vis->CreateVisualizerWindow("Events Visualizer", 1600, 900);
        point_cloud = std::make_shared<open3d::geometry::PointCloud>();
        vis->AddGeometry(point_cloud);

        auto bounding_box = std::make_shared<open3d::geometry::AxisAlignedBoundingBox>(
                Eigen::Vector3d(0, 0, 0.0),
                Eigen::Vector3d(_width, _height, 1));

        // Set bounding box color for visibility
        bounding_box->color_ = Eigen::Vector3d(0.0, 1.0, 0.0);  // Green color

        vis->AddGeometry(bounding_box);

        vis->AddGeometry(_line_set);
        vis->AddGeometry(_line_set2);
    }

    ~Open3DVisualizer() {
        vis->DestroyVisualizerWindow();
    }

    void addPoint(double x, double y, double z, double r = 0.1, double g = 0.1, double b = 0.1) {
        std::lock_guard<std::mutex> lock(_mtx);
        point_cloud->points_.emplace_back(x, y, z);
        point_cloud->colors_.emplace_back(r, g, b);
    }

    void addLine(double x, double y, double z, double r = 0.1, double g = 0.1, double b = 0.1) {
        std::lock_guard<std::mutex> lock(_mtx);


        // If this is the first point, store it and return
        if (_line_set->points_.empty()) {
            _line_set->points_.emplace_back(Eigen::Vector3d(x, y, z));
            return;
        }

        // Update LineSet
        _line_set->points_.emplace_back(Eigen::Vector3d(x, y, z));
        _line_set->lines_.emplace_back(Eigen::Vector2i(_line_set->points_.size() - 2, _line_set->points_.size() - 1));

        // Ensure colors array matches the number of lines
        _line_set->colors_.resize(_line_set->lines_.size(), Eigen::Vector3d(r, g, b));
    }

    void addLine2(double x, double y, double z, double r = 0.1, double g = 0.1, double b = 0.1) {
        std::lock_guard<std::mutex> lock(_mtx);

        // If this is the first point, store it and return
        if (_line_set2->points_.empty()) {
            _line_set2->points_.emplace_back(Eigen::Vector3d(x, y, z));
            return;
        }

        // Update LineSet
        _line_set2->points_.emplace_back(Eigen::Vector3d(x, y, z));
        _line_set2->lines_.emplace_back(
                Eigen::Vector2i(_line_set2->points_.size() - 2, _line_set2->points_.size() - 1));

        // Ensure colors array matches the number of lines
        _line_set2->colors_.resize(_line_set2->lines_.size(), Eigen::Vector3d(r, g, b));
    }

    void update() {
        std::lock_guard<std::mutex> lock(_mtx);
        vis->UpdateGeometry(point_cloud);
        vis->UpdateGeometry(_line_set);
        vis->UpdateGeometry(_line_set2);
        vis->PollEvents();
        vis->UpdateRender();
    }

    void loop() {
        while (vis->PollEvents()) {
            vis->UpdateRender();
        }
    }

    [[nodiscard]] int getWidth() const {
        return _width;
    }

    [[nodiscard]] int getHeight() const {
        return _height;
    }

private:
    std::shared_ptr<open3d::visualization::Visualizer> vis;
    std::shared_ptr<open3d::geometry::PointCloud> point_cloud;
    const int _width, _height;

    std::shared_ptr<open3d::geometry::LineSet> _line_set = std::make_shared<open3d::geometry::LineSet>();
    std::vector<Eigen::Vector3d> _points;
    std::vector<Eigen::Vector2i> _lines;


    std::shared_ptr<open3d::geometry::LineSet> _line_set2 = std::make_shared<open3d::geometry::LineSet>();
    std::vector<Eigen::Vector3d> _points2;
    std::vector<Eigen::Vector2i> _lines2;

    std::mutex _mtx;
};

#endif //PROJECT_OPEN3D_VISUALIZER_H
