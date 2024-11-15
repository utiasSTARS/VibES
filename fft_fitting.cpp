#include <fftw3.h>
#include <unordered_map>
#include <deque>
#include <iostream>
#include <cmath>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/camera_capture.hpp>
#include <opencv2/highgui.hpp>


int main() {
    dv::io::MonoCameraRecording reader(
            "/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/file.aedat4");

    std::cout << "Opened an AEDAT4 file which contains data from [" << reader.getCameraName() << "] camera"
              << std::endl;

    cv::Size resolution = *reader.getEventResolution();

    int N = 1024;  // Number of samples to accumulate for FFT

    // FFT setup
    std::vector<double> x_data(N), y_data(N);
    fftw_complex *outX = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *outY = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan planX = fftw_plan_dft_r2c_1d(N, x_data.data(), outX, FFTW_ESTIMATE);
    fftw_plan planY = fftw_plan_dft_r2c_1d(N, y_data.data(), outY, FFTW_ESTIMATE);

    int64_t x_mean = 0, y_mean = 0, timestamp_mean = 0, timestamp = 0;
    int counter = 0;

    int index = 0;
    double samplingRate = 100'000;   // Fixed sampling rate (Hz)
    double deltaTime = 1.0 / samplingRate; // Time interval between samples (in seconds)
    double lastTimestamp = 0.0; // Track last event timestamp for interpolation
    std::deque<std::tuple<double, double, double>> events_buffer; // Buffer for events with (x, y, t)

    cv::Mat img = cv::Mat::zeros(resolution, CV_8UC1);
    double max_x = std::numeric_limits<double>::min(), max_y = std::numeric_limits<double>::min();
    double min_x = std::numeric_limits<double>::max(), min_y = std::numeric_limits<double>::max();

    while (reader.isRunning()) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {

            // Collect events in a buffer with timestamp normalization to seconds
            for (const auto &event: events.value()) {
                if (timestamp == 0) timestamp = event.timestamp();
                x_mean += event.x();
                y_mean += event.y();
                timestamp_mean += (event.timestamp() - timestamp);

                if (counter++ == 50) {

                    events_buffer.emplace_back(x_mean / counter, y_mean / counter, timestamp_mean / 1e6);
                    x_mean = 0;
                    y_mean = 0;
                    counter = 0;
                }

            }

            // Interpolation for fixed sampling
            while (events_buffer.size() > 1 && index < N) {
                auto [x1, y1, t1] = events_buffer.front();
                auto [x2, y2, t2] = *(++events_buffer.begin());

                if (lastTimestamp == 0.0) lastTimestamp = t1;

                // Interpolate to fill data at regular intervals
                while (lastTimestamp + deltaTime <= t2 && index < N) {
                    double ratio = (lastTimestamp + deltaTime - t1) / (t2 - t1);
                    x_data[index] = x1 + ratio * (x2 - x1);
                    y_data[index] = y1 + ratio * (y2 - y1);
                    // check max min
                    if (x_data[index] > max_x) max_x = x_data[index];
                    if (x_data[index] < min_x) min_x = x_data[index];
                    if (y_data[index] > max_y) max_y = y_data[index];
                    if (y_data[index] < min_y) min_y = y_data[index];

                    lastTimestamp += deltaTime;
                    index++;
                }

                // Remove processed events
                events_buffer.pop_front();
            }

            // visualize the interpolated x on cv::Mat
            // for (int i = 0; i < N; i++) {
            //     int x = x_data[i];
            //     int y = y_data[i];
            //     img.at<uchar>(y, x) = 255;
            // }
            // cv::imshow("Interpolated X", img);
            // cv::waitKey(0);

            // Execute FFT when we have N interpolated samples
            if (index >= N) {
                index = 0;  // Reset index after filling N samples

                // remove the mean max min from data
                double mean_x = (max_x + min_x) / 2;
                double mean_y = (max_y + min_y) / 2;

                for (int i = 0; i < N; i++) {
                    x_data[i] -= mean_x;
                    y_data[i] -= mean_y;
                }

                fftw_execute(planX);
                fftw_execute(planY);

                // Analyze FFT results to find peak frequency for both X and Y
                double maxMagnitudeX = 0.0, maxMagnitudeY = 0.0;
                int peakIndexX = -1, peakIndexY = -1;

                for (int i = 0; i < N / 2; ++i) {
                    double magnitudeX = std::sqrt(outX[i][0] * outX[i][0] + outX[i][1] * outX[i][1]);
                    double magnitudeY = std::sqrt(outY[i][0] * outY[i][0] + outY[i][1] * outY[i][1]);
                    if (magnitudeX > maxMagnitudeX) {
                        maxMagnitudeX = magnitudeX;
                        peakIndexX = i;
                    }
                    if (magnitudeY > maxMagnitudeY) {
                        maxMagnitudeY = magnitudeY;
                        peakIndexY = i;
                    }
                }

                double peakFrequencyX = peakIndexX * samplingRate / N;
                double peakFrequencyY = peakIndexY * samplingRate / N;

                std::cout << "Estimated " << peakIndexX << " Frequency for X: " << peakFrequencyX << " Hz" << std::endl;
                std::cout << "Estimated " << peakIndexY << " Frequency for Y: " << peakFrequencyY << " Hz" << std::endl;

                x_data.assign(N, 0.0);
                y_data.assign(N, 0.0);
            }
        }
    }

    // Clean up
    fftw_destroy_plan(planX);
    fftw_destroy_plan(planY);
    fftw_free(outX);
    fftw_free(outY);

    return 0;
}




// // Function to generate a noisy helical signal
// void generateNoisyHelix(int N, double frequency, double samplingRate, std::vector<double>& x, std::vector<double>& y) {
//     for (int i = 0; i < N; ++i) {
//         double time = i / samplingRate;
//         x[i] = cos(2 * M_PI * frequency * time) + 1.5 * ((double) rand() / RAND_MAX - 0.5);
//         y[i] = sin(2 * M_PI * frequency * time) + 1.5 * ((double) rand() / RAND_MAX - 0.5);
//     }
// }
//
// int main() {
//     int N = 1024; // Number of samples
//     double samplingRate = 1000.0; // in Hz
//     double signalFrequency = 50.0; // Hz
//
//     std::vector<double> x(N), y(N);
//     generateNoisyHelix(N, signalFrequency, samplingRate, x, y);
//
//     // Allocate memory for FFTW for X and Y components
//     fftw_complex* outX = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
//     fftw_complex* outY = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
//     fftw_plan planX = fftw_plan_dft_r2c_1d(N, x.data(), outX, FFTW_ESTIMATE);
//     fftw_plan planY = fftw_plan_dft_r2c_1d(N, y.data(), outY, FFTW_ESTIMATE);
//
//     // Perform FFT on both components
//     fftw_execute(planX);
//     fftw_execute(planY);
//
//     // Calculate magnitudes and find the peak frequencies
//     double maxMagnitudeX = 0.0, maxMagnitudeY = 0.0;
//     int peakIndexX = 0, peakIndexY = 0;
//     for (int i = 0; i < N / 2; ++i) {
//         double magnitudeX = sqrt(outX[i][0] * outX[i][0] + outX[i][1] * outX[i][1]);
//         double magnitudeY = sqrt(outY[i][0] * outY[i][0] + outY[i][1] * outY[i][1]);
//         if (magnitudeX > maxMagnitudeX) {
//             maxMagnitudeX = magnitudeX;
//             peakIndexX = i;
//         }
//         if (magnitudeY > maxMagnitudeY) {
//             maxMagnitudeY = magnitudeY;
//             peakIndexY = i;
//         }
//     }
//
//     // Calculate the frequencies corresponding to the peak indices
//     double peakFrequencyX = peakIndexX * samplingRate / N;
//     double peakFrequencyY = peakIndexY * samplingRate / N;
//
//     std::cout << "Estimated Frequency for X: " << peakFrequencyX << " Hz" << std::endl;
//     std::cout << "Estimated Frequency for Y: " << peakFrequencyY << " Hz" << std::endl;
//
//     // Clean up
//     fftw_destroy_plan(planX);
//     fftw_destroy_plan(planY);
//     fftw_free(outX);
//     fftw_free(outY);
//
//     return 0;
// }
//