//
// Created by viciopoli on 07/07/25.
//
//
// Created by viciopoli on 29/06/25.
//

#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/base_window.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
#include "slice_visualizer.hpp"
#include "estimator/iekf_sinusoid_fitter.hpp"
#include "event_frontend/centroid.hpp"
#include "haste/app/command_parser.hpp"
#include "haste/tracking.hpp"
#include "event_frontend/undistort.hpp"

IEKFSinusoidFitter create_iekf(double A, double B, double omega) {
    IEKFSinusoidFitter::StateVector initial_state;
    // A1, B1, omega, C
    initial_state << A, B, omega, 0;

    IEKFSinusoidFitter::StateCovariance initial_covariance;
    initial_covariance.setIdentity();
    initial_covariance *= 1e3;
    initial_covariance(2, 2) = 1e5; // Set a smaller initial uncertainty for omega

    IEKFSinusoidFitter::StateCovariance process_noise;
    process_noise.setIdentity();
    process_noise *= 1e-1;

    double measurement_noise = 1e-1;
    return IEKFSinusoidFitter(initial_state, initial_covariance, process_noise, measurement_noise);
}

int main(int argc, char *argv[]) {

    // load params and create camera
    HARMEDA::ParamsLoader params(argc, argv);

    const auto w = params.camera.geometry().width();
    const auto h = params.camera.geometry().height();

//    haste::HypothesisPatchTracker::kPatchSize = params.params->tracker_size;

    Undistort undistort(params.params->calib_file);

//    auto camera = haste::PinholeRadTanCamera<haste::HypothesisPatchTracker::Scalar>(w, h);
//    haste::RpgDataset::loadCalibration(params.params->calib_file, camera);
//    auto undistortion_map = camera.createUndistortionMap();// This undistort mapping could alternatively be used during an online process


    const std::uint32_t acc = 20000;
    double fps = 50;
    auto frame_gen = Metavision::PeriodicFrameGenerationAlgorithm(w, h, acc, fps);
    auto frame_gen_comp = Metavision::PeriodicFrameGenerationAlgorithm(w, h, acc, fps);

    Metavision::Window window("Frames", w, h, Metavision::BaseWindow::RenderMode::BGR);
    Metavision::Window window_compensated("Frames compensated", w, h, Metavision::BaseWindow::RenderMode::BGR);

    int shift_x = params.params->tracker_x; // -25; // -100;
    int shift_y = params.params->tracker_y; // -25; // 10;
    frame_gen.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
        // draw a square of size 113 in the center of the frame
        double x_square_center = w / 2.0 + shift_x;
        double y_square_center = h / 2.0 + shift_y;
        int size = haste::HypothesisPatchTracker::kPatchSize;
        int half_size = size / 2;
        cv::rectangle(frame, cv::Point(x_square_center - half_size, y_square_center - half_size + 1),
                      cv::Point(x_square_center + half_size, y_square_center + half_size + 1),
                      cv::Scalar(0, 0, 255), 2);

        window.show(frame);
    });

    frame_gen_comp.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
        window_compensated.show(frame);
    });

    std::unique_ptr<IEKFSinusoidFitter> x_fitter, y_fitter;

    double f_min = 10.0;   // Minimum frequency in Hz
    double f_max = 100.0;  // Maximum frequency in Hz
    int max_harmonics = 4; // Maximum number of harmonics to estimate
    NUFFTHelixEstimator nufft_estimator(f_min, f_max, max_harmonics);

    // create tracker
    std::shared_ptr<haste::HypothesisPatchTracker> tracker;

    // read data from camera and estimate the frequencies
    Metavision::timestamp first_event_t = -1;

    HARMEDA::SliceVisualizer slice_visualizer_x(h, w, 0, HARMEDA::X_AXIS, 3);
    HARMEDA::SliceVisualizer slice_visualizer_y(h, w, 0, HARMEDA::Y_AXIS, 3);

    cv::namedWindow("Slice X Visualizer", cv::WINDOW_NORMAL);
    cv::namedWindow("Slice Y Visualizer", cv::WINDOW_NORMAL);

    std::mutex mtx;
//    std::fstream filex, filey;
//    filex.open("centroid_data_x.txt", std::ios::out);
//    filey.open("centroid_data_y.txt", std::ios::out);

    double tracker_latency = 0.0; // seconds
    float x_undistorted, y_undistorted;
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::vector<Metavision::EventCD> events_undistored(std::distance(begin, end));
        std::vector<Metavision::EventCD> events_compensated;
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            const Metavision::EventCD &event = *ev;

            std::tie(x_undistorted, y_undistorted) = undistort(event.x, event.y);
//             make sure the undistorted coordinates are within the image bounds
            if (x_undistorted < 0 || x_undistorted >= w || y_undistorted < 0 || y_undistorted >= h) {
                continue; // Skip events that are out of bounds
            }
            Metavision::EventCD undist_event(
                    static_cast<unsigned short>(x_undistorted),
                    static_cast<unsigned short>(y_undistorted),
                    event.p,
                    event.t
            );

            events_undistored.push_back(undist_event);

            if (first_event_t < 0 && begin != end) {
                first_event_t = begin->t;
                tracker = std::make_shared<haste::HasteDifferenceStarTracker>(0.01,
                                                                              w / 2. + shift_x,
                                                                              h / 2. + shift_y,
                                                                              0.0);
                continue;
            }
            const auto current_relative_t = static_cast<double>(event.t - first_event_t);
            const double current_t_sec = current_relative_t / 1.e6;

            slice_visualizer_x.feed(undist_event);
            slice_visualizer_y.feed(undist_event);

            if (y_fitter && x_fitter) {
                auto y_pred = y_fitter->predict_rel(current_t_sec);
                auto x_pred = x_fitter->predict_rel(current_t_sec);
                // correct for the motion
                events_compensated.emplace_back(static_cast<unsigned short>(x_undistorted - x_pred),
                                                static_cast<unsigned short>(y_undistorted - y_pred),
                                                event.p,
                                                event.t);
            }

            slice_visualizer_y.editFrame([&](cv::Mat &frame) {
                if (y_fitter) {
                    auto y_pred = y_fitter->predict(current_t_sec);

                    cv::circle(frame,
                               cv::Point(slice_visualizer_x.time_value, y_pred),
                               1,
                               cv::Scalar(255, 255, 0),
                               -1);
                }
            });
            slice_visualizer_x.editFrame([&](cv::Mat &frame) {
                if (x_fitter) {
                    auto x_pred = x_fitter->predict(current_t_sec);
                    cv::circle(frame,
                               cv::Point(slice_visualizer_x.time_value, x_pred),
                               1,
                               cv::Scalar(255, 255, 0),
                               -1);
                }
            });

            const auto &update_type = tracker->pushEvent(current_t_sec,
                                                         x_undistorted,
                                                         y_undistorted);

            if (update_type == haste::HypothesisPatchTracker::EventUpdate::kStateEvent) {
                std::cout << "N samples: " << nufft_estimator.getCurrentIndex() << std::endl;
//                std::cout << "Tracker state updated to: {t=" << tracker->t() << ",\t x=" << tracker->x()
//                          << ",\t y=" << tracker->y() << ",\t theta=" << tracker->theta() << "}" << std::endl;
                if (!nufft_estimator.done() &&
                    nufft_estimator.feed(Centroid(tracker->t(), tracker->x(), tracker->y()))) {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (!nufft_estimator.compute()) {
                        continue;
                    }
                    nufft_estimator.printResults();

                    std::vector<double> Ax, Ay, Bx, By, omegas, offsets;
                    for (auto &h: nufft_estimator.getHarmonics()) {
                        auto phase_x = std::atan(h.amplitude_x / h.amplitude_y);
                        auto phase_y = std::atan(h.amplitude_y / h.amplitude_x);
                        auto ax = h.amplitude_x * std::cos(phase_x);
                        auto ay = h.amplitude_y * std::sin(phase_y);
                        auto bx = h.amplitude_x * std::sin(phase_x);
                        auto by = h.amplitude_y * std::cos(phase_y);
                        Ax.push_back(ax);
                        Ay.push_back(ay);
                        Bx.push_back(bx);
                        By.push_back(by);
                        omegas.push_back(h.frequency);
                        offsets.push_back(h.offset_x);
                        offsets.push_back(h.offset_y);
                    }

                    if (Ax.empty()) { throw std::runtime_error("No harmonics estimated yet."); }

                    auto fitter = create_iekf(Ax[0], Bx[0], omegas[0]);
                    x_fitter = std::make_unique<IEKFSinusoidFitter>(fitter);

                    y_fitter = std::make_unique<IEKFSinusoidFitter>(
                            create_iekf(Ay[0], By[0], omegas[0])
                    );
                }

                tracker_latency = tracker->t() - current_t_sec;
//                std::cout << "Tracker latency: " << tracker_latency << " seconds" << std::endl;

                // Print the centroid data
                slice_visualizer_y.editFrame([&](cv::Mat &frame) {
                    cv::circle(frame, cv::Point(
                                       tracker->t() * slice_visualizer_x.time_scale,
                                       int(tracker->y())), 1,
                               cv::Scalar(0, 255, 0), -1);
                });
                slice_visualizer_x.editFrame([&](cv::Mat &frame) {
                    cv::circle(frame, cv::Point(
                                       tracker->t() * slice_visualizer_y.time_scale,
                                       int(tracker->x())), 1,
                               cv::Scalar(0, 255, 255), -1);
                });

                if (x_fitter && y_fitter) {
                    x_fitter->update(tracker->t() + tracker_latency, tracker->x());
                    y_fitter->update(tracker->t() + tracker_latency, tracker->y());
                }
            }
        }
        frame_gen.process_events(events_undistored.begin(), events_undistored.end());
        frame_gen_comp.process_events(events_compensated.begin(), events_compensated.end());
    });

    params.camera.start();

    while (params.camera.is_running()) {
        // Normalize time for x-axis in the visualization
        if (tracker) {
            haste::ImshowEigenArrayNormalized(
                    "Feature Event Window Projection",
                    tracker->eventWindowToModel(tracker->event_window(), tracker->state()).transpose());
            haste::ImshowEigenArrayNormalized("Feature Template", tracker->tracker_template().transpose());
        }
        cv::imshow("Slice X Visualizer", slice_visualizer_x.getFrameSide());
        cv::imshow("Slice Y Visualizer", slice_visualizer_y.getFrameSide());
        cv::waitKey(1); // Allow OpenCV to process the window events
        if (cv::waitKey(1) == 27) { // Exit on ESC key
            break;
        }
        Metavision::EventLoop::poll_and_dispatch(20);
    }
    params.camera.stop();

    // print fitter
    if (x_fitter) {
        std::cout << "X Fitter: " << *x_fitter << std::endl;
    }
    if (y_fitter) {
        std::cout << "Y Fitter: " << *y_fitter << std::endl;
    }

//    filex.close();
//    filey.close();

    std::cout << "Processing complete." << std::endl;
    return 0;

}
