#include <dv-processing/io/mono_camera_recording.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <chrono>
#include <thread>

struct EventCluster {
    std::vector<dv::Event> events;  // Events in the cluster
    int64_t last_timestamp;         // Timestamp of the latest event in the cluster
    int avg_x, avg_y;               // Average x and y coordinates of the cluster

    EventCluster(const dv::Event &event)
            : last_timestamp(event.timestamp()), avg_x(event.x()), avg_y(event.y()) {
        events.push_back(event);
    }

    bool isWithinThreshold(const dv::Event &event, int64_t time_threshold, int spatial_threshold) const {
        int64_t time_diff = event.timestamp() - last_timestamp;
        int spatial_diff = std::sqrt(std::pow(event.x() - avg_x, 2) + std::pow(event.y() - avg_y, 2));
        return (time_diff <= time_threshold) && (spatial_diff <= spatial_threshold);
    }

    void addEvent(const dv::Event &event) {
        events.push_back(event);
        last_timestamp = event.timestamp();

        // Recalculate cluster center
        int sum_x = 0, sum_y = 0;
        for (const auto &ev : events) {
            sum_x += ev.x();
            sum_y += ev.y();
        }
        avg_x = sum_x / events.size();
        avg_y = sum_y / events.size();
    }
};

int main(int argc, char** argv) {

    // Initialize dv-processing event reader
    dv::io::MonoCameraRecording reader(
            "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/file.aedat4");

    if (!reader.isRunning()) {
        std::cerr << "Failed to open AEDAT4 file." << std::endl;
        return 1;
    }

    // Define thresholds
    const int64_t time_threshold_micro = 50;  // 5 ms in microseconds
    const int spatial_threshold = 5;            // 5 pixels

    // Image size (sensor resolution)
    const cv::Size resolution = *reader.getEventResolution();
    const int width = resolution.width, height = resolution.height;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

    std::deque<EventCluster> clusters;  // Store active clusters
    auto last_render_time = std::chrono::steady_clock::now();

    while (reader.isRunning()) {
        // Fetch a batch of events
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {
            for (const auto &event : events.value()) {
                bool added_to_cluster = false;

                // Check if event fits in any existing cluster
                for (auto &cluster : clusters) {
                    if (cluster.isWithinThreshold(event, time_threshold_micro, spatial_threshold)) {
                        cluster.addEvent(event);
                        added_to_cluster = true;
                        break;
                    }
                }

                // If not added to any cluster, create a new one
                if (!added_to_cluster) {
                    clusters.emplace_back(event);
                }
            }

            // Display clusters every 1000 ms
            auto current_time = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_render_time).count() >= 1000) {
                // Clear the image
                image.setTo(cv::Scalar(0, 0, 0));

                // Draw each cluster as a point on the image
                for (const auto &cluster : clusters) {
                    for (const auto &event : cluster.events) {
                        cv::Point point(event.x(), event.y());
                        cv::circle(image, point, 1, cv::Scalar(0, 255, 0), -1);  // Green dot for each event
                    }
                }

                // Show the image
                cv::imshow("Event Clusters", image);
                cv::waitKey(1);  // Allows OpenCV to update the window

                // Clear clusters for next iteration
                clusters.clear();
                last_render_time = current_time;
            }
        }

        // Optional: Sleep to reduce CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return 0;
}
