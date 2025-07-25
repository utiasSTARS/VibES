//
// Created by viciopoli on 18/02/25.
//

#ifndef PROJECT_INITIALIZERS_H
#define PROJECT_INITIALIZERS_H

#include <vector>
#include <iostream>
#include <optional>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/rolling_event_buffer.h>

#include <metavision/sdk/analytics/algorithms/tracking_algorithm.h>
#include <metavision/sdk/analytics/configs/tracking_algorithm_config.h>


#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

class Initializer
{

  using EventBufferReslicer = Metavision::EventBufferReslicerAlgorithm;
  using RollingEventBuffer = Metavision::RollingEventBuffer<Metavision::EventCD>;
  using RollingEventBufferConfig = Metavision::RollingEventBufferConfig;
  using TrackingBuffer = std::vector<Metavision::EventTrackingData>;

public:
  Initializer(
    uint32_t width,
    uint32_t height,
    uint32_t min_track_size,
    uint32_t max_track_size,
    Metavision::timestamp capture_ts,
    Metavision::timestamp event_seq_ts = 10000)
  : _width(width),
    _height(height),
    _min_track_size(min_track_size),
    _max_track_size(max_track_size),
    _capture_ts(capture_ts),
    _event_seq_ts(event_seq_ts)
  {

    // initialize rolling event buffer
    using RollingEventBufferConfig = Metavision::RollingEventBufferConfig;
    _event_buffer = RollingEventBuffer(RollingEventBufferConfig::make_n_us(_event_seq_ts));

    // initialize the reslicer
    const auto cond = EventBufferReslicer::Condition::make_n_us(_capture_ts);
    _slicer = std::make_unique<EventBufferReslicer>(
      [&](EventBufferReslicer::ConditionStatus status, Metavision::timestamp ts,
      std::size_t n_events) {initializer_callback(status, ts, n_events);},
      cond);

    // initialize the tracker
    Metavision::TrackingConfig tracking_config;
    _tracker = std::make_unique<Metavision::TrackingAlgorithm>(
      _width, _height,
      tracking_config);
    _tracker->set_min_size(_min_track_size);
    _tracker->set_max_size(_max_track_size);


    // create frame
    _frame.create(height, width, CV_8UC3);
  }

  void initializer_callback(
    EventBufferReslicer::ConditionStatus status, Metavision::timestamp ts,
    std::size_t n_events)
  {
    if (!_captured) {
      Metavision::BaseFrameGenerationAlgorithm::generate_frame_from_events(
        _event_buffer.cbegin(),
        _event_buffer.cend(),
        _frame);
      _captured = true;
    }

    tracker_callback(status, ts, n_events);
  }

  void tracker_callback(
    EventBufferReslicer::ConditionStatus status, Metavision::timestamp ts,
    std::size_t n_events)
  {
    _tracked_objects.clear();

    _last_track_ts = ts;

    _tracker->process_events(
      _event_buffer.cbegin(), _event_buffer.cend(),
      std::back_inserter(_tracked_objects));

    for (const auto & obj : _tracked_objects) {
      _tracked_object_ids.insert(obj.object_id_);
    }
  }

  void process_events(const Metavision::EventCD * begin, const Metavision::EventCD * end)
  {
    _slicer->process_events(
      begin, end, [&](const auto sub_slice_begin_it, const auto sub_slice_end_it) {
        _event_buffer.insert_events(sub_slice_begin_it, sub_slice_end_it);
      });
  }

  std::vector<cv::Point2f> get_shitomasi_features(
    uint32_t max_corners = 16,
    double quality_level = 0.01,
    double min_distance = 10, uint32_t block_size = 3,
    uint32_t grad_size = 3, double k = 0.04) const
  {
    std::vector<cv::Point2f> features;
    if (_captured) {
      cv::Mat gray_frame;
      cv::cvtColor(_frame, gray_frame, cv::COLOR_BGR2GRAY);
      cv::goodFeaturesToTrack(
        gray_frame, features, max_corners, quality_level, min_distance,
        cv::Mat(), block_size, grad_size, k);
    }
    return features;
  }

  std::vector<cv::Point2f> get_fast_features(uint32_t threshold = 20) const
  {
    std::vector<cv::Point2f> features;
    if (_captured) {
      cv::Mat gray_frame;
      cv::cvtColor(_frame, gray_frame, cv::COLOR_BGR2GRAY);
      std::vector<cv::KeyPoint> keypoints;
      cv::FAST(gray_frame, keypoints, threshold, true);
      for (const auto & kp : keypoints) {
        features.emplace_back(kp.pt.x, kp.pt.y);
      }
    }
    return features;
  }

  TrackingBuffer get_tracking_data(bool visualize = true) const
  {
    // ASSUMPTION: the frame is captured and the tracker has been called
    if (!_captured) {
      return {};
    }

    return _tracked_objects;
  }

  std::optional<cv::Mat> get_frame()
  {
    if (_captured) {
      return _frame;
    } else {
      return std::nullopt;
    }
  }

  Metavision::timestamp get_track_timestamp() const
  {
    return _last_track_ts;
  }

  void reset()
  {
    _captured = false;
  }

private:
  uint32_t _width;
  uint32_t _height;

  uint32_t _min_track_size;
  uint32_t _max_track_size;

  Metavision::timestamp _capture_ts;
  Metavision::timestamp _event_seq_ts;
  Metavision::timestamp _last_track_ts = 0;

  cv::Mat _frame;
  bool _captured = false;

  std::unique_ptr<Metavision::EventBufferReslicerAlgorithm> _slicer;

  std::unique_ptr<Metavision::TrackingAlgorithm> _tracker;

  RollingEventBuffer _event_buffer;

  // tracked objects
  TrackingBuffer _tracked_objects;
  std::set<size_t> _tracked_object_ids;


};

#endif //PROJECT_INITIALIZERS_H
