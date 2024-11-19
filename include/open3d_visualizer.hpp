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
    Open3DVisualizer() {
        vis = std::make_shared<open3d::visualization::Visualizer>();
        vis->CreateVisualizerWindow("Events Visualizer", 1600, 900);
        point_cloud = std::make_shared<open3d::geometry::PointCloud>();
        vis->AddGeometry(point_cloud);

        auto bounding_box = std::make_shared<open3d::geometry::AxisAlignedBoundingBox>(
                Eigen::Vector3d(0, 0, 0.0),
                Eigen::Vector3d(640, 480, 1));

        // Set bounding box color for visibility
        bounding_box->color_ = Eigen::Vector3d(0.0, 1.0, 0.0);  // Green color

        vis->AddGeometry(bounding_box);
    }

    ~Open3DVisualizer() {
        vis->DestroyVisualizerWindow();
    }

    void addPoint(double x, double y, double z, bool polarity) {
        point_cloud->points_.emplace_back(x, y, z);
        point_cloud->colors_.emplace_back(0.1, 0.1, polarity);
    }

    void update() {
        vis->UpdateGeometry(point_cloud);
        vis->PollEvents();
        vis->UpdateRender();
    }

    void loop() {
        while (vis->PollEvents()) {
            vis->UpdateRender();
        }
    }


private:
    std::shared_ptr<open3d::visualization::Visualizer> vis;
    std::shared_ptr<open3d::geometry::PointCloud> point_cloud;

};

#endif //PROJECT_OPEN3D_VISUALIZER_H
