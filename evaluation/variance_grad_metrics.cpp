//
// Created by viciopoli on 18/08/25.
//
#include <metavision/sdk/core/utils/rate_estimator.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/core/pipeline/stage.h>
#include <metavision/sdk/core/utils/misc.h>

#include <filesystem>
#include <mutex>
#include <sstream>

#include <metavision/sdk/driver/hdf5_event_file_writer.h>

#include "params_loader.hpp"

double computeVarCounts(const std::vector<long long> &counts) {
    long long N = counts.size();
    double mean = 0.0;
    mean = std::accumulate(counts.begin(), counts.end(), 0.0);
    mean /= N;

    double var = 0.0;
    for (auto c: counts) {
        double diff = c - mean;
        var += diff * diff;
    }
    return var / (N - 1);
}


double computeGradEnergy(const std::vector<long long> &counts, int H, int W) {
    double energy = 0.0;

    for (int y = 1; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {
            long long c = counts[y * W + x];
            long long cx1 = counts[y * W + (x + 1)];
            long long cy1 = counts[(y + 1) * W + x];

            double dx = double(cx1 - c);
            double dy = double(cy1 - c);

            energy += std::sqrt(dx * dx + dy * dy);
        }
    }

    return energy / ((H - 2) * (W - 2));  // normalize
}


static Metavision::timestamp first_event_t = 0, last_event_t = 0;

int main(int argc, char *argv[]) {
    // Initialize parameters and camera
    VibES::ParamsLoader params(argc, argv);
    std::cout << params;

    const auto width = params.camera.geometry().width();
    const auto height = params.camera.geometry().height();

    std::string output_file = params.params->output_folder;
    if (!std::filesystem::exists(output_file)) {
        std::filesystem::create_directories(output_file);
    }

    std::once_flag init_flag;
    long long slice_initial_time = 0;

    std::vector<long long> image_slice(width * height, 0);
    std::vector<double> var_counts;
    std::vector<double> grad_energy;

    Metavision::EventCD e_tmp;
    params.camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::call_once(init_flag, [&]() {
            first_event_t = begin->t;
            slice_initial_time = first_event_t;
        });

        // Process the batch, splitting across as many windows as needed
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            last_event_t = ev->t;
            e_tmp.x = ev->x;
            e_tmp.y = ev->y;
            e_tmp.p = ev->p;
            e_tmp.t = ev->t;
            if (ev->t - slice_initial_time > params.params->time_window_us) {
                var_counts.push_back(computeVarCounts(image_slice));
                grad_energy.push_back(computeGradEnergy(image_slice, height, width));

                slice_initial_time = ev->t; // Reset slice start time
                image_slice.clear();
                image_slice.resize(width * height, 0);
            }
            image_slice[ev->y * width + ev->x] += 1;

        }
    });

    // Start camera
    params.camera.start();

    std::cout << "Camera started, processing events..." << std::endl;
    while (params.camera.is_running()) {
        // Just keep the main thread alive
        Metavision::EventLoop::poll_and_dispatch(1);
    }


    // Final processing after camera stops
    if (!var_counts.empty()) {
        auto var = computeVarCounts(image_slice);
        if (var != 0) {
            auto grad = computeGradEnergy(image_slice, height, width);
            var_counts.push_back(computeVarCounts(image_slice));
            grad_energy.push_back(computeGradEnergy(image_slice, height, width));
        }

        // make some statistics
        double mean_var = std::accumulate(var_counts.begin(), var_counts.end(), 0.0);
        mean_var /= var_counts.size();
        double std_var = 0.0;
        for (const auto &v: var_counts) {
            double diff = v - mean_var;
            std_var += diff * diff;
        }
        std_var = std::sqrt(std_var / (var_counts.size() - 1));

        double mean_grad = std::accumulate(grad_energy.begin(), grad_energy.end(), 0.0);
        mean_grad /= grad_energy.size();
        double std_grad = 0.0;
        for (const auto &g: grad_energy) {
            double diff = g - mean_grad;
            std_grad += diff * diff;
        }
        std_grad = std::sqrt(std_grad / (grad_energy.size() - 1));
        std::cout << "Variance of counts: " << mean_var << " ± " << std_var << std::endl;
        std::cout << "Gradient energy: " << mean_grad << " ± " << std_grad << std::endl;

        // Save results to file
        std::ofstream var_file(output_file + "/variance.txt");
        std::ofstream grad_file(output_file + "/gradient_energy.txt");

        if (var_file.is_open() && grad_file.is_open()) {
            var_file << "Accumulation window: " << params.params->time_window_us << " us\n";
            var_file << "Variance of counts:\n";
            grad_file << "Accumulation window: " << params.params->time_window_us << " us\n";
            grad_file << "Gradient energy:\n";
            var_file << "Mean: " << mean_var << " ± " << std_var << "\n";
            grad_file << "Mean: " << mean_grad << " ± " << std_grad << "\n";
            var_file << "Counts:\n";
            grad_file << "Energy:\n";
            for (size_t i = 0; i < var_counts.size(); ++i) {
                var_file << var_counts[i] << "\n";
                grad_file << grad_energy[i] << "\n";
            }
            var_file.close();
            grad_file.close();
            std::cout << "Results saved to " << output_file << std::endl;
        } else {
            std::cerr << "Error opening output files." << std::endl;
        }
    } else {
        std::cout << "No events processed." << std::endl;
    }

    return 0;
}
