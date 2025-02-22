import numpy as np
import matplotlib.pyplot as plt


def load_signal(filename):
    """Load the signal from a text file."""
    # Load data from the provided text file
    data = np.loadtxt(filename,
                      delimiter=","
                      )
    return data[:, 0]


def compute_fft(signal, sampling_rate):
    """Compute the FFT of the signal and extract top 4 dominant frequencies."""
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1 / sampling_rate)  # Positive frequencies
    fft_values = np.fft.rfft(signal)  # Compute FFT
    magnitudes = np.abs(fft_values)  # Magnitude spectrum

    magnitudes[0:10] = 0  # Remove DC component

    # remove too high frequencies
    magnitudes[500:] = 0

    sorted_indices = np.argsort(magnitudes)[::-1]  # Sort by magnitude (descending)

    top_frequencies = []
    top_magnitudes = []

    min_separation = 5  # Minimum separation between dominant frequencies
    for idx in sorted_indices:
        if len(top_frequencies) >= 4:
            break
        freq = freqs[idx]

        # Check if this frequency is sufficiently far from already selected ones
        if all(abs(freq - f) >= min_separation for f in top_frequencies):
            top_frequencies.append(freq)
            top_magnitudes.append(magnitudes[idx])


    return top_frequencies, top_magnitudes, freqs, magnitudes


def plot_fft(freqs, magnitudes, top_frequencies):
    """Plot the FFT spectrum and highlight the dominant frequencies."""

    # reduce the number of magnitues to 500
    freqs = freqs[:500]
    magnitudes = magnitudes[:500]

    plt.figure(figsize=(8, 5))
    plt.plot(freqs, magnitudes, label='FFT Magnitude')
    plt.scatter(top_frequencies, [magnitudes[np.where(freqs == f)][0] for f in top_frequencies], color='red',
                label='Top 4 Frequencies')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Spectrum")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    sampling_rate = 100  # Define the correct sampling rate (Hz)

    signal = load_signal(
        "./scripts/centroid_data_real/centroids_x_dot.txt",
    )
    top_frequencies, top_magnitudes, freqs, magnitudes = compute_fft(signal, sampling_rate)

    print("Top 4 Frequencies:")
    for i, (freq, mag) in enumerate(zip(top_frequencies, top_magnitudes)):
        print(f"{i + 1}: {freq:.2f} Hz (Magnitude: {mag:.2f})")

    plot_fft(freqs, magnitudes, top_frequencies)
