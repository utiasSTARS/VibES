//
// Created by viciopoli on 29/06/25.
//

#include "estimator/iekf_helix_fitter.hpp"
#include "estimator/nufft_multiharmonics.hpp"
#include "params_loader.hpp"
#include "slice_visualizer.hpp"
#include "estimator/iekf_sinusoid_fitter.hpp"
#include "event_frontend/centroid.hpp"

IEKFSinusoidFitter create_iekf(double A, double B, double omega) {
    IEKFSinusoidFitter::StateVector initial_state;
    // A1, B1, omega, C
    initial_state << A, B, omega, 0;

    IEKFSinusoidFitter::StateCovariance initial_covariance;
    initial_covariance.setIdentity();
    initial_covariance *= 1e2;

    IEKFSinusoidFitter::StateCovariance process_noise;
    process_noise.setIdentity();
    process_noise *= 1e-5;

    double measurement_noise = 1e-1;
    return IEKFSinusoidFitter(initial_state, initial_covariance, process_noise, measurement_noise);
}

int main(int argc, char *argv[]) {
    // load params and create camera
    HARMEDA::ParamsLoader params(argc, argv);

    std::unique_ptr<IEKFSinusoidFitter> x_fitter, y_fitter;

    double f_min = 10.0;   // Minimum frequency in Hz
    double f_max = 100.0;  // Maximum frequency in Hz
    int max_harmonics = 4; // Maximum number of harmonics to estimate
    NUFFTHelixEstimator nufft_estimator(f_min, f_max, max_harmonics);

    // create EMA centroid
    auto t_window = nufft_estimator.getWindowEMA(); // in us
    std::cout << "Using time window for EMA: " << t_window << " us" << std::endl;
    CentroidEMA ema_calculator(params.params->tau, 1000, params.camera.geometry().width(),
                               params.camera.geometry().height());

//    CMassCalculation centroid_calculator(
//            params.camera.geometry().width(), params.camera.geometry().height()
//    );

    // read data from camera and estimate the frequencies
    Metavision::timestamp first_event_t = 0;

    Metavision::timestamp prev_centroid_time = -1;

    HARMEDA::SliceVisualizer slice_visualizer_x(
            params.camera.geometry().height(), params.camera.geometry().width(), 0, HARMEDA::X_AXIS, 3);
    HARMEDA::SliceVisualizer slice_visualizer_y(
            params.camera.geometry().height(), params.camera.geometry().width(), 0, HARMEDA::Y_AXIS, 3);

    cv::namedWindow("Slice X Visualizer", cv::WINDOW_NORMAL);
    cv::namedWindow("Slice Y Visualizer", cv::WINDOW_NORMAL);

    std::mutex mtx;
//    std::fstream filex, filey;
//    filex.open("centroid_data_x.txt", std::ios::out);
//    filey.open("centroid_data_y.txt", std::ios::out);

    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            const Metavision::EventCD &event = *ev;
            if (first_event_t < 0 && begin != end) {
                first_event_t = begin->t;
            }

            if (first_event_t > ev->t) {
                std::cerr << "Warning: Non-increasing event timestamps detected. Previous: "
                          << first_event_t << ", Current: " << ev->t << std::endl;
                throw std::runtime_error("Event timestamps are not increasing.");
            }

            const Metavision::timestamp current_relative_t = event.t - first_event_t;
            const double current_t_sec = current_relative_t / 1.e6;


            // Print the centroid data
            double x_pred = 0, y_pred = 0;
            if (x_fitter) {
                x_pred = x_fitter->predict(current_t_sec);
            }
            if (y_fitter) {
                y_pred = y_fitter->predict(current_t_sec);
            }

            auto ev_comp = Metavision::EventCD(
                    static_cast<unsigned short>(event.x - x_pred),
                    static_cast<unsigned short>(event.y),
                    event.p,
                    event.t
            );
            slice_visualizer_x.feed(event);
            slice_visualizer_y.feed(event);

            // feed the EMA centroid calculator
            auto result = ema_calculator.feed(event);

            slice_visualizer_x.editFrame([&](cv::Mat &frame) {
                if (x_fitter) {
                    cv::circle(frame,
                               cv::Point(slice_visualizer_x.time_value, x_pred + 3),
                               1,
                               cv::Scalar(255, 255, 0),
                               -1);
                }
            });
            slice_visualizer_x.editFrame([&](cv::Mat &frame) {
                if (y_fitter) {
                    cv::circle(frame,
                               cv::Point(slice_visualizer_x.time_value, y_pred + 3),
                               1,
                               cv::Scalar(255, 255, 0),
                               -1);
                }
            });

            if (result.has_value()) {
//              Feed the centroid data to the NUFFT estimator
                if (!nufft_estimator.done() && nufft_estimator.feed(result.value())) {
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
                }

                // Print the centroid data
                slice_visualizer_x.editFrame([&](cv::Mat &frame) {
                    cv::circle(frame, cv::Point(slice_visualizer_x.time_value, result.value().x), 1,
                               cv::Scalar(0, 255, 0), -1);
                });
                slice_visualizer_y.editFrame([&](cv::Mat &frame) {
                    cv::circle(frame, cv::Point(slice_visualizer_y.time_value, result.value().y), 1,
                               cv::Scalar(0, 255, 255), -1);
                });

                if (x_fitter) {
                    x_fitter->update(
                            static_cast<double>(result.value().t - first_event_t) * 1.e-6,
                            result.value().x
                    );
                }
                if (y_fitter) {
                    y_fitter->update(
                            static_cast<double>(result.value().t - first_event_t) * 1.e-6,
                            result.value().y
                    );
                }
            }
        }
    });

    params.camera.start();

    while (params.camera.is_running()) {
        // Normalize time for x-axis in the visualization
        cv::imshow("Slice X Visualizer", slice_visualizer_x.getFrameSide());
        cv::imshow("Slice Y Visualizer", slice_visualizer_y.getFrameSide());
        cv::waitKey(0); // Allow OpenCV to process the window events
    }

//    filex.close();
//    filey.close();

    std::cout << "Processing complete." << std::endl;
    return 0;

}
