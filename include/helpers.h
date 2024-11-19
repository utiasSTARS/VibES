//
// Created by viciopoli on 18/11/24.
//
#ifndef PROJECT_HELPERS_H
#define PROJECT_HELPERS_H

#include <algorithm>
#include <vector>
#include <tuple>
#include <open3d/Open3D.h>

template<typename T>
inline T linear_interp(T alpha, T x0, T x1) {
    return x0 + alpha * (x1 - x0);
}

std::optional<std::vector<std::tuple<double, double, int64_t>>>
interpolate_events(std::vector<std::tuple<double, double, int64_t>> &events,
                   double dt) {
    if (events.empty()) return std::nullopt;
    std::vector<std::tuple<double, double, int64_t>> out;
    out.push_back(events[0]);
    size_t i = 1;
    for (; i < events.size(); ++i) {
        auto [x0, y0, t0] = events[i - 1];
        auto [x1, y1, t1] = events[i];

        // Ensure the time difference is valid
        if (t1 <= t0) continue;

        // Add interpolated points
        for (double t = t0 + dt; t < t1; t += dt) {
            double alpha = (t - t0) / (t1 - t0); // Normalized interpolation factor
            double x = linear_interp(alpha, x0, x1);
            double y = linear_interp(alpha, y0, y1);
            out.emplace_back(x, y, static_cast<int64_t>(t));
        }
        // Add the next event
        out.push_back(events[i]);
    }
    // remove the i events from the buffer
    events.erase(events.begin(), events.begin() + i - 1);

    return out;
}

class Open3DVisualizer {
public:
    Open3DVisualizer() {
        vis = std::make_shared<open3d::visualization::Visualizer>();
        vis->CreateVisualizerWindow("Open3D", 1600, 900);
        point_cloud = std::make_shared<open3d::geometry::PointCloud>();
        vis->AddGeometry(point_cloud);

        auto bounding_box = std::make_shared<open3d::geometry::AxisAlignedBoundingBox>(
                Eigen::Vector3d(0, 0, 0.0),
                Eigen::Vector3d(640, 480, 10));

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

#endif //PROJECT_HELPERS_H
