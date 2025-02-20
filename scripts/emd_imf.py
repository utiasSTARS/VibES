import numpy as np
import scipy.signal
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def emd(x, max_imfs=10, tol=0.05, max_sift_iterations=100):
    """
    Perform Empirical Mode Decomposition (EMD) on the input signal x.

    Parameters:
        x : numpy array
            The input signal (1D).
        max_imfs : int
            Maximum number of IMFs to extract.
        tol : float
            Tolerance for the stopping criterion in the sifting process.
        max_sift_iterations : int
            Maximum iterations in the sifting process for each IMF.

    Returns:
        imfs : list of numpy arrays
            The extracted intrinsic mode functions.
        residue : numpy array
            The final residue after subtracting all extracted IMFs.
    """
    residue = x.copy()
    imfs = []
    n = len(x)
    t = np.arange(n)

    for i in range(max_imfs):
        h = residue.copy()
        sift_iter = 0

        while sift_iter < max_sift_iterations:
            # Identify local maxima and minima
            max_peaks, _ = scipy.signal.find_peaks(h)
            min_peaks, _ = scipy.signal.find_peaks(-h)

            # If there are not enough extrema, break the sifting process
            if len(max_peaks) + len(min_peaks) < 2:
                break

            # Interpolate to obtain the upper envelope
            if len(max_peaks) > 1:
                upper_env = CubicSpline(t[max_peaks], h[max_peaks], extrapolate=True)(t)
            else:
                upper_env = np.zeros(n)
            # Interpolate to obtain the lower envelope
            if len(min_peaks) > 1:
                lower_env = CubicSpline(t[min_peaks], h[min_peaks], extrapolate=True)(t)
            else:
                lower_env = np.zeros(n)

            # Compute the mean of the envelopes
            mean_env = (upper_env + lower_env) / 2.0

            # Sifting: remove the local mean from h
            h1 = h - mean_env

            # Stopping criterion: if the change is small, then h1 is an IMF.
            if np.linalg.norm(h - h1) < tol * np.linalg.norm(h):
                h = h1
                break

            h = h1
            sift_iter += 1

        imfs.append(h)
        residue = residue - h

        # Stop if the residue becomes a monotonic function (not enough extrema)
        max_peaks_res, _ = scipy.signal.find_peaks(residue)
        min_peaks_res, _ = scipy.signal.find_peaks(-residue)
        if len(max_peaks_res) + len(min_peaks_res) < 2:
            break

    return imfs, residue


# =============================================================================
# Example usage:
# =============================================================================

if __name__ == '__main__':
    # Create a sample signal: a combination of two sinusoids and a trend.
    # t = np.linspace(0, 1, 1000)
    # s1 = 0.5 * np.sin(2 * np.pi * 100 * t)  # 5 Hz component
    # s2 = 3 * np.sin(2 * np.pi * 20 * t)  # 20 Hz component
    # trend = np.random.rand(len(t))
    # signal = s1 + s2 + trend

    # read the signal from the file: centroids_x_april.txt that contains for each row "x,t" values
    data = np.loadtxt("/scripts/centroid_data_real/centroids_y_april.txt", delimiter=",")
    signal = data[:, 0]
    dc_component = np.mean(signal)
    signal = signal - dc_component
    t = data[:, 1]


    # Apply EMD to decompose the signal into IMFs and residue.
    imfs, residue = emd(signal, max_imfs=5, tol=0.05, max_sift_iterations=100)

    # Reconstruct the signal by summing the IMFs.
    reconstructed = np.sum(imfs, axis=0)

    # Plot the original signal, the extracted IMFs, and the final residue.
    plt.figure(figsize=(12, 8))


    plt.subplot(len(imfs) + 3, 1, 1)
    plt.plot(t, reconstructed, 'r')
    plt.plot(t, signal, label="Original Signal", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Original Signal vs. Reconstruction from IMFs")

    for i, imf in enumerate(imfs):
        plt.subplot(len(imfs) + 2, 1, i + 2)
        plt.plot(t, imf)
        plt.title(f"IMF {i + 1}")

    plt.subplot(len(imfs) + 2, 1, len(imfs) + 1)
    plt.plot(t, residue, 'r')
    plt.title("Residue")

    plt.tight_layout()
    plt.show()
