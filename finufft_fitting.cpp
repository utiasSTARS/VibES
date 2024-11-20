#include <finufft.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

int main() {
    // Parameters for the sinusoid
    double true_freq = 500.0;    // True frequency of the sinusoid (Hz)
    double sample_rate = 2000;   // Simulated sampling rate (Hz)
    double duration = 0.01;      // Duration of the signal (s)
    int64_t nj = 200;            // Number of non-uniform samples
    int64_t nk = 100;            // Number of frequency bins to scan
    double eps = 1e-6;           // Desired accuracy
    int iflag = 1;               // Forward transform (+1)

    // Generate non-uniform time samples and signal
    std::vector<double> t(nj);                    // Non-uniform time samples
    std::vector<std::complex<double>> signal(nj); // Sinusoidal signal
    for (int64_t j = 0; j < nj; ++j) {
        t[j] = M_PI * (2.0 * j / nj - 1.0); // Map [0, duration] to [-π, π)
        signal[j] = std::exp(std::complex<double>(0.0, 2 * M_PI * true_freq * j / sample_rate));
    }

    // Define a frequency range to scan
    double freq_min = 450.0; // Minimum frequency (Hz)
    double freq_max = 550.0; // Maximum frequency (Hz)
    std::vector<double> freqs(nk); // Target frequencies in radians/sec
    for (int64_t k = 0; k < nk; ++k) {
        freqs[k] = 2 * M_PI * (freq_min + k * (freq_max - freq_min) / (nk - 1)) / sample_rate;
    }

    // Allocate memory for the output Fourier coefficients
    std::vector<std::complex<double>> spectrum(nk);

    // Perform the NUFFT (type 3)
    int ier = finufft1d3(
            nj,                   // Number of input points
            t.data(),             // Input locations
            signal.data(),        // Input strengths
            iflag,                // Forward transform
            eps,                  // Desired accuracy
            nk,                   // Number of output frequencies
            freqs.data(),         // Output frequency locations
            spectrum.data(),      // Fourier coefficients
            nullptr               // Default options
    );

    // Check for errors
    if (ier != 0) {
        std::cerr << "FINUFFT type 3 failed with error code: " << ier << std::endl;
        return ier;
    }

    // Find the frequency with the maximum magnitude
    double max_magnitude = 0.0;
    double estimated_freq = 0.0;
    for (int64_t k = 0; k < nk; ++k) {
        double magnitude = std::abs(spectrum[k]);
        if (magnitude > max_magnitude) {
            max_magnitude = magnitude;
            estimated_freq = (freq_min + k * (freq_max - freq_min) / (nk - 1));
        }
    }

    // Output results
    std::cout << "True frequency: " << true_freq << " Hz" << std::endl;
    std::cout << "Estimated frequency: " << estimated_freq << " Hz" << std::endl;

    return 0;
}
