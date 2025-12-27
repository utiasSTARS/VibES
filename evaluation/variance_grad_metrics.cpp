/**
 * @file event_focus_evaluator.cpp
 * @brief Utility to evaluate the "focus" quality of an event stream.
 *
 * This tool computes statistical metrics on event accumulation frames to quantify
 * sharpness and information content. It is commonly used for auto-focusing
 * algorithms or data quality assessment.
 *
 * Metrics Computed:
 * 1. **Variance of Counts:** Measures the contrast of the event accumulation.
 * Higher variance typically implies a sharper image (edges are distinct from background).
 * 2. **Gradient Energy:** Measures the sum of squared gradients.
 * High gradient energy indicates sharp edges.
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

#include <metavision/sdk/core/utils/rate_estimator.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/core/utils/misc.h>

#include <filesystem>
#include <mutex>
#include <sstream>
#include <numeric>
#include <cmath>
#include <vector>
#include <fstream>

#include "params_loader.hpp"

/**
 * @brief Computes the Variance of event counts across the image sensor.
 *
 * A simplified focus metric. In a focused image, events are concentrated on edges,
 * leading to high peaks (high count) and large empty areas (low count), thus high variance.
 * In a defocused image, events are spread out (blur), reducing variance.
 *
 * @param counts Linearized vector of event counts (size W*H).
 * @return The variance value.
 */
double computeVarCounts(const std::vector<long long> &counts) {
    long long N = counts.size();
    if (N < 2) return 0.0;

    double mean = std::accumulate(counts.begin(), counts.end(), 0.0) / static_cast<double>(N);

    double var = 0.0;
    for (auto c: counts) {
        double diff = c - mean;
        var += diff * diff;
    }
    return var / (N - 1);
}

/**
 * @brief Computes the Gradient Energy (sum of squared discrete derivatives).
 *
 * Uses simple forward differences to approximate the gradient magnitude.
 * Focus = Sum( sqrt( (dI/dx)^2 + (dI/dy)^2 ) )
 *
 * @param counts Linearized vector of event counts.
 * @param H Sensor height.
 * @param W Sensor width.
 * @return Average gradient energy per pixel.
 */
double computeGradEnergy(const std::vector<long long> &counts, int H, int W) {
    double energy = 0.0;

    // Iterate over the image (excluding the last row/col for finite diff check)
    for (int y = 0; y < H - 1; y++) {
        for (int x = 0; x < W - 1; x++) {
            long long c = counts[y * W + x];
            long long cx1 = counts[y * W + (x + 1)]; // Right neighbor
            long long cy1 = counts[(y + 1) * W + x]; // Bottom neighbor

            double dx = static_cast<double>(cx1 - c);
            double dy = static_cast<double>(cy1 - c);

            energy += std::sqrt(dx * dx + dy * dy);
        }
    }

    // Normalize by the number of processed pixels
    return energy / ((H - 1) * (W - 1));
}


// Global timestamps for tracking duration
static Metavision::timestamp first_event_t = 0, last_event_t = 0;

int main(int argc, char *argv[]) {
    // 1. Initialize Configuration
    VibES::ParamsLoader params(argc, argv);
    std::cout << params;

    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    // Setup output paths
    std::string output_file = params.params->output_folder;
    if (!std::filesystem::exists(output_file)) {
        std::filesystem::create_directories(output_file);
    }

    std::once_flag init_flag;
    long long slice_initial_time = 0;

    // Buffers for accumulation
    // We use a flat vector for the "image" to avoid OpenCV dependency in the core loop
    std::vector<long long> image_slice(width * height, 0);

    // Results storage
    std::vector<double> var_counts;
    std::vector<double> grad_energy;

    // 2. Main Processing Callback
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::call_once(init_flag, [&]() {
            first_event_t = begin->t;
            slice_initial_time = first_event_t;
        });

        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            last_event_t = ev->t;

            // Check if accumulation window is complete
            if (ev->t - slice_initial_time > params.params->time_window_us) {
                // Compute metrics for the completed slice
                var_counts.push_back(computeVarCounts(image_slice));
                grad_energy.push_back(computeGradEnergy(image_slice, height, width));

                // Reset for next slice
                slice_initial_time = ev->t;
                std::fill(image_slice.begin(), image_slice.end(), 0);
            }

            // Accumulate event
            // Note: Boundary check omitted for speed, assumed valid from SDK
            image_slice[ev->y * width + ev->x] += 1;
        }
    });

    // 3. Start Acquisition
    params.camera.start();
    std::cout << "Camera started. Accumulating statistics every "
              << params.params->time_window_us << " us..." << std::endl;

    while (params.camera.is_running()) {
        Metavision::EventLoop::poll_and_dispatch(1);
    }

    // 4. Final Analysis & Reporting
    if (!var_counts.empty()) {
        // Calculate Mean and StdDev for Variance Metric
        double mean_var = std::accumulate(var_counts.begin(), var_counts.end(), 0.0) / var_counts.size();
        double std_var = 0.0;
        for (const auto &v: var_counts) std_var += (v - mean_var) * (v - mean_var);
        std_var = std::sqrt(std_var / (var_counts.size() - 1));

        // Calculate Mean and StdDev for Gradient Energy Metric
        double mean_grad = std::accumulate(grad_energy.begin(), grad_energy.end(), 0.0) / grad_energy.size();
        double std_grad = 0.0;
        for (const auto &g: grad_energy) std_grad += (g - mean_grad) * (g - mean_grad);
        std_grad = std::sqrt(std_grad / (grad_energy.size() - 1));

        std::cout << "\n=== Focus Evaluation Results ===" << std::endl;
        std::cout << "Variance of counts: " << mean_var << " +/- " << std_var << std::endl;
        std::cout << "Gradient energy:    " << mean_grad << " +/- " << std_grad << std::endl;

        // 5. Save Logs
        std::ofstream var_file(output_file + "/variance.txt");
        std::ofstream grad_file(output_file + "/gradient_energy.txt");

        if (var_file.is_open() && grad_file.is_open()) {
            // Write Headers
            var_file << "Window(us): " << params.params->time_window_us << "\n";
            var_file << "Mean: " << mean_var << " Std: " << std_var << "\n";
            var_file << "Values:\n";

            grad_file << "Window(us): " << params.params->time_window_us << "\n";
            grad_file << "Mean: " << mean_grad << " Std: " << std_grad << "\n";
            grad_file << "Values:\n";

            // Write Data Series
            for (size_t i = 0; i < var_counts.size(); ++i) {
                var_file << var_counts[i] << "\n";
                grad_file << grad_energy[i] << "\n";
            }
            std::cout << "Detailed logs saved to " << output_file << std::endl;
        } else {
            std::cerr << "Error writing output files." << std::endl;
        }
    } else {
        std::cout << "No events processed or window size too large for sequence." << std::endl;
    }

    return 0;
}