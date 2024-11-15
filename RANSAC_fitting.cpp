#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <open3d/Open3D.h>

struct Point3D {
    double x, y, z;
};

struct CylindricalPoint {
    double r, theta, z;
};

void cartesianToCylindrical(const Point3D &p, CylindricalPoint &cp) {
    cp.r = std::sqrt(p.x * p.x + p.y * p.y);
    cp.theta = std::atan2(p.y, p.x);
    cp.z = p.z;
}

double screwRadialModel(double a, double b, double theta) {
    return a + b * theta;
}

double screwVerticalModel(double c, double theta) {
    return c * theta;
}

std::tuple<double, double, double>
fitScrewRansac(const std::vector<Point3D> &points, int numIterations = 1000, double threshold = 0.1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points.size() - 1);

    int maxInliers = 0;
    double bestA = 0, bestB = 0, bestC = 0;

    for (int iter = 0; iter < numIterations; ++iter) {
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        if (idx1 == idx2) continue;

        CylindricalPoint cp1, cp2;
        cartesianToCylindrical(points[idx1], cp1);
        cartesianToCylindrical(points[idx2], cp2);

        double b = (cp2.r - cp1.r) / (cp2.theta - cp1.theta);
        double a = cp1.r - b * cp1.theta;
        double c = (cp2.z - cp1.z) / (cp2.theta - cp1.theta);

        int inliers = 0;
        for (const auto &point: points) {
            CylindricalPoint cp;
            cartesianToCylindrical(point, cp);

            double r_pred = screwRadialModel(a, b, cp.theta);
            double z_pred = screwVerticalModel(c, cp.theta);

            if (std::abs(cp.r - r_pred) < threshold && std::abs(cp.z - z_pred) < threshold) {
                ++inliers;
            }
        }

        if (inliers > maxInliers) {
            maxInliers = inliers;
            bestA = a;
            bestB = b;
            bestC = c;
        }
    }

    return {bestA, bestB, bestC};
}

int main() {
    std::vector<Point3D> data;
    double a_true = 1.0, b_true = 0.1, c_true = 0.05;
    double noise = 0.05;
    int numPoints = 100;

    std::default_random_engine generator;
    std::normal_distribution<double> noise_dist(0.0, noise);

// Create Open3D PointCloud from the generated points
    auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();

    // Visualize the PointCloud
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Event Camera Visualization", 800, 600);

    vis.AddGeometry(point_cloud);
    vis.GetViewControl().SetLookat(
            Eigen::Vector3d(1, 1, -1));
    vis.GetViewControl().SetZoom(0.8);
    vis.GetRenderOption().point_size_ = 2.0;


    for (int i = 0; i < numPoints; ++i) {
        double theta = i * 0.000001;
        double r = a_true + b_true * theta;
        double z = c_true * theta;
        Point3D p;
        p.x = r * std::cos(theta) + noise_dist(generator);
        p.y = r * std::sin(theta) + noise_dist(generator);
        p.z = z + noise_dist(generator);
        std::cout << "Point " << i << ": " << p.x << ", " << p.y << ", " << p.z << std::endl;
        point_cloud->points_.emplace_back(p.x, p.y, p.z);
        point_cloud->colors_.emplace_back(1, 1, 0);
        data.push_back(p);


        vis.UpdateGeometry(point_cloud);
        vis.PollEvents();
        vis.UpdateRender();
    }


    while (vis.PollEvents()) {
        vis.UpdateRender();
    }

    vis.DestroyVisualizerWindow();

    auto [bestA, bestB, bestC] = fitScrewRansac(data);

    std::cout << "Best fit parameters (a, b, c): " << bestA << ", " << bestB << ", " << bestC << std::endl;

    return 0;
}
