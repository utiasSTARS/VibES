import numpy as np
import matplotlib.pyplot as plt


def h(x):
    """
    Measurement function for the 7D state:
        x = [a1, omega1, phi1, a2, omega2, phi2, b]
    Measurement: z = b + a1 * sin(phi1) + a2 * sin(phi2)
    """
    a1, omega1, phi1, a2, omega2, phi2, b = x
    return b + a1 * np.sin(phi1) + a2 * np.sin(phi2)


def jacobian_h(x):
    """
    Jacobian of h(x) with respect to x.
    """
    a1, omega1, phi1, a2, omega2, phi2, b = x
    return np.array([[np.sin(phi1), 0, a1 * np.cos(phi1), np.sin(phi2), 0, a2 * np.cos(phi2), 1]])


def ekf_measurement_update(x, P, z, R):
    z_pred = h(x)
    y = z - z_pred
    H = jacobian_h(x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_new = x + (K * y).flatten()
    I = np.eye(len(x))
    P_new = (I - K @ H) @ P
    return x_new, P_new, y


def f(x, dt):
    """
    Process model assuming constant amplitudes and frequencies.
    """
    a1, omega1, phi1, a2, omega2, phi2, b = x
    return np.array([a1, omega1, phi1 + omega1 * dt, a2, omega2, phi2 + omega2 * dt, b])


def jacobian_f(x, dt):
    """
    Jacobian of f(x) with respect to x.
    """
    return np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, dt, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, dt, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])


def get_process_noise_covariance(dt, sigma_a, sigma_omega, sigma_b):
    """
    Process noise covariance for the 7D state.
    """
    Q = np.diag([dt * sigma_a ** 2, dt * sigma_omega ** 2, (dt ** 3 / 3) * sigma_omega ** 2,
                 dt * sigma_a ** 2, dt * sigma_omega ** 2, (dt ** 3 / 3) * sigma_omega ** 2,
                 dt * sigma_b ** 2])
    return Q


def run_ekf_on_real_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    signal, t = data[:, 0], data[:, 1]
    N = len(t)

    x_hat = np.array([20.0, 2 * np.pi * 130, 0.0, 7.0, 2 * np.pi * 0, np.pi / 8, 300.0])
    P = np.eye(7) * 100.0
    sigma_a, sigma_omega, sigma_b = 3.0, 3.0, 1.0
    R = 0.005

    ekf_predictions, x_estimates, t_est, residuals = [], [], [], []

    for i in range(N):
        res = 0
        if i > 0:
            dt = t[i] - t[i - 1]
            F = jacobian_f(x_hat, dt)
            x_pred = f(x_hat, dt)
            Q = get_process_noise_covariance(dt, sigma_a, sigma_omega, sigma_b)
            P_pred = F @ P @ F.T + Q
            x_hat, P, res = ekf_measurement_update(x_pred, P_pred, signal[i], R)
        residuals.append(res)
        ekf_predictions.append(h(x_hat))
        x_estimates.append(x_hat.copy())
        t_est.append(t[i])

    # print the state
    print("Final state estimate:")
    print("a1: {:.2f}".format(x_hat[0]))
    print("omega1: {:.2f}".format(x_hat[1]/(2*np.pi)))
    print("phi1: {:.2f}".format(x_hat[2]))
    print("a2: {:.2f}".format(x_hat[3]))
    print("omega2: {:.2f}".format(x_hat[4]/(2*np.pi)))
    print("phi2: {:.2f}".format(x_hat[5]))
    print("b: {:.2f}".format(x_hat[6]))
    return t, signal, np.array(t_est), np.array(ekf_predictions), residuals


def plot_results(t, signal, t_est, ekf_predictions, residuals):
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal, 'r.', label="Measurements")
    plt.plot(t_est, ekf_predictions, 'b-', label="EKF Estimate")
    plt.plot(t_est, residuals, 'g-', label="Residuals")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.show()


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


def fourier(signal):
    sampling_rate = 10000  # Define the correct sampling rate (Hz)

    top_frequencies, top_magnitudes, freqs, magnitudes = compute_fft(signal, sampling_rate)

    print("Top 4 Frequencies:")
    for i, (freq, mag) in enumerate(zip(top_frequencies, top_magnitudes)):
        print(f"{i + 1}: {freq:.2f} Hz (Magnitude: {mag:.2f})")

    plot_fft(freqs, magnitudes, top_frequencies)


# Example usage:
filename = "./scripts/centroid_data_real/centroids_y_dot.txt"
t, signal, t_est, ekf_predictions, residuals = run_ekf_on_real_data(filename)
fourier(residuals)
plot_results(t, signal, t_est, ekf_predictions, residuals)
