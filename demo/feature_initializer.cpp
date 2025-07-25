#include <iostream>
#include <thread>
#include <chrono>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>
#include <metavision/sdk/analytics/utils/tracking_drawing.h>


#include "event_frontend/initializers.hpp"

#include "opencv2/highgui/highgui.hpp"


int main(int argc, char * argv[])
{

  // auto cam = Metavision::Camera::from_file(argv[1]);
  auto cam = Metavision::Camera::from_file("/home/steph/datasets/harmeda/circle_files/circle_real.hdf5");

  Metavision::timestamp sample_ts = 200000;

  const auto w = cam.geometry().width();
  const auto h = cam.geometry().height();

  Initializer initializer(w, h, 10, 300, sample_ts);

  Metavision::PeriodicFrameGenerationAlgorithm frame_gen(w, h);  ///< Events Frame generator

  Metavision::Window window("Frames", w, h, Metavision::BaseWindow::RenderMode::BGR);
  Metavision::Window feature_window("Captured Frame", w, h,
    Metavision::BaseWindow::RenderMode::BGR);
  Metavision::Window feature1_window("Shi-Tomasi Features", w, h,
    Metavision::BaseWindow::RenderMode::BGR);
  Metavision::Window feature2_window("FAST Features", w, h,
    Metavision::BaseWindow::RenderMode::BGR);
  Metavision::Window track_window("Tracker", w, h, Metavision::BaseWindow::RenderMode::BGR);

  frame_gen.set_output_callback(
    [&](Metavision::timestamp, cv::Mat & frame) {
      window.show(frame);
    });


  cam.cd().add_callback(
    [&](const Metavision::EventCD * begin, const Metavision::EventCD * end) {
      frame_gen.process_events(begin, end);
      initializer.process_events(begin, end);
    });


  std::cout << "Camera started." << std::endl;
  const bool started = cam.start();
  if (!started) {
    std::cerr << "Failed to start camera." << std::endl;
    return 1;
  }

  cv::Mat init_frame;

  while (cam.is_running()) {
    Metavision::EventLoop::poll_and_dispatch(20);
    if (auto frame = initializer.get_frame()) {
      init_frame = *frame;
      feature_window.show(init_frame, false);

      // Using Shi-Tomasi features
      cv::Mat feat1_frame = init_frame.clone();
      auto shitomasi_features = initializer.get_shitomasi_features();
      if (!shitomasi_features.empty()) {
        for (const auto & feature : shitomasi_features) {
          cv::circle(feat1_frame, cv::Point2f(feature.x, feature.y), 3, cv::Scalar(0, 0, 255), -1);
        }
      } else {
        std::cerr << "No features found using Shi-Tomasi." << std::endl;
      }
      feature1_window.show(feat1_frame, false);

      // Using FAST features
      cv::Mat feat2_frame = init_frame.clone();
      auto fast_features = initializer.get_fast_features(50);
      if (!fast_features.empty()) {
        for (const auto & feature : fast_features) {
          cv::circle(feat2_frame, cv::Point2f(feature.x, feature.y), 3, cv::Scalar(0, 0, 255), -1);
        }
      } else {
        std::cerr << "No features found using FAST." << std::endl;
      }
      feature2_window.show(feat2_frame, false);

      cv::Mat tracked_frame = init_frame.clone();
      auto tracking_data = initializer.get_tracking_data();
      if (!tracking_data.empty()) {
        Metavision::draw_tracking_results(
          initializer.get_track_timestamp(), tracking_data.cbegin(), tracking_data.cend(),
          tracked_frame);
      } else {
        std::cerr << "No tracking data available." << std::endl;
      }
      track_window.show(tracked_frame, false);
    }
  }

  cam.stop();

  std::cout << "Camera stopped." << std::endl;

  return 0;
}
