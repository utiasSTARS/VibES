#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include "include/events_freq_calib_pattern.hpp"

// Function to convert events to a 2D histogram
cv::Mat eventsToHistogram(const std::vector<std::tuple<int, int, double, int>> &events, int width, int height) {
    cv::Mat histogram = cv::Mat::zeros(height, width, CV_64F);

    for (const auto &event: events) {
        int x = std::get<0>(event);
        int y = std::get<1>(event);
        int polarity = std::get<3>(event);
        if (x >= 0 && x < width && y >= 0 && y < height) {
            histogram.at<double>(y, x) += polarity; // Accumulate polarity
        }
    }

    return histogram;
}

// Function to compute 2D FFT and estimate motion frequency
void computeMotionFrequency(const cv::Mat &input, double dx, double dy) {
    int Nx = input.rows;
    int Ny = input.cols;

    // Allocate FFTW arrays
    fftw_complex *out = fftw_alloc_complex(Nx * Ny);
    double *in = fftw_alloc_real(Nx * Ny);

    // Copy input data into FFTW array
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            in[i * Ny + j] = input.at<double>(i, j);
        }
    }

    // Create and execute FFT plan
    fftw_plan plan = fftw_plan_dft_r2c_2d(Nx, Ny, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Compute magnitude and find the dominant frequency
    double max_magnitude = 0.0;
    int max_u = 0, max_v = 0;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny / 2 + 1; ++j) {
            int index = i * (Ny / 2 + 1) + j;
            double real = out[index][0];
            double imag = out[index][1];
            double mag = std::sqrt(real * real + imag * imag);

            if (mag > max_magnitude) {
                max_magnitude = mag;
                max_u = i;
                max_v = j;
            }
        }
    }

    // Map FFT indices to frequencies
    double freq_u = (max_u <= Nx / 2 ? max_u : max_u - Nx) / (Nx * dx);
    double freq_v = max_v / (Ny * dy);

    std::cout << "Dominant Motion Frequency:\n";
    std::cout << "  Frequency in X (u): " << freq_u << " cycles per unit length\n";
    std::cout << "  Frequency in Y (v): " << freq_v << " cycles per unit length\n";

    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

// Example usage
int main() {
    // Simulated event data: (x, y, t, p)
    EventsFreqCalibPattern reader(0, cv::Size(640, 480), 400, 10, 10, 0.01);

    std::vector<std::tuple<int, int, double, int>> events_data;

    while (reader.isRunning()) {
        if (const auto events = reader.getNextEventBatch(); events.has_value()) {
            for (const auto &e: *events) {
                events_data.push_back(std::make_tuple(e.x(), e.y(), e.timestamp() / 1e6, e.polarity()));
            }
            break;
        }
    }

    // Define image dimensions
    int width = 640, height = 480;

    // Convert events to a 2D histogram
    cv::Mat histogram = eventsToHistogram(events_data, width, height);

    // Display the 2D histogram
    cv::imshow("Event Histogram", histogram);
    cv::waitKey(0);

    // Compute motion frequency
    double dx = 1.0, dy = 1.0; // Spatial resolution
    computeMotionFrequency(histogram, dx, dy);

    return 0;
}
