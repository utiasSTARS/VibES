import numpy as np
import scipy.signal
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Part 1: Empirical Mode Decomposition (EMD)
# ---------------------------------------------------------------------------
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
            The final residue (trend/DC) after subtracting all extracted IMFs.
    """
    residue = x.copy()
    imfs = []
    n = len(x)
    t = np.arange(n)

    for i in range(max_imfs):
        h = residue.copy()
        sift_iter = 0

        while sift_iter < max_sift_iterations:
            # Find local maxima and minima
            max_peaks, _ = scipy.signal.find_peaks(h)
            min_peaks, _ = scipy.signal.find_peaks(-h)

            # If there are not enough extrema, break out.
            if len(max_peaks) + len(min_peaks) < 2:
                break

            # Upper envelope interpolation
            if len(max_peaks) > 1:
                upper_env = CubicSpline(t[max_peaks], h[max_peaks], extrapolate=True)(t)
            else:
                upper_env = np.zeros(n)
            # Lower envelope interpolation
            if len(min_peaks) > 1:
                lower_env = CubicSpline(t[min_peaks], h[min_peaks], extrapolate=True)(t)
            else:
                lower_env = np.zeros(n)

            # Mean envelope
            mean_env = (upper_env + lower_env) / 2.0
            h1 = h - mean_env  # Sifting step

            # Check convergence: if change is small enough, accept h1 as an IMF.
            if np.linalg.norm(h - h1) < tol * np.linalg.norm(h):
                h = h1
                break

            h = h1
            sift_iter += 1

        imfs.append(h)
        residue = residue - h

        # Stop if residue has too few extrema (i.e. is monotonic)
        max_peaks_res, _ = scipy.signal.find_peaks(residue)
        min_peaks_res, _ = scipy.signal.find_peaks(-residue)
        if len(max_peaks_res) + len(min_peaks_res) < 2:
            break

    return imfs, residue

# ---------------------------------------------------------------------------
# Part 2: Automatic Initial Guess from IMFs using Hilbert Transform
# ---------------------------------------------------------------------------
def initial_guess_from_imfs(imfs, fs, freq_threshold=0.1):
    """
    Compute initial EKF state estimates from a list of IMFs.
    For each IMF, we use its analytic signal to estimate:
      - Amplitude: average magnitude of the analytic signal.
      - Phase: initial phase (first sample).
      - Frequency: average instantaneous frequency (in rad/s).
    Only IMFs with an average frequency above (freq_threshold in Hz) are selected.

    The state for each mode is defined as:
      [x_c, x_s, omega, sigma]
    where:
      x_c = amplitude*cos(phase)
      x_s = amplitude*sin(phase)
      omega = estimated frequency (rad/s)
      sigma = initial damping (set here to 0)

    Parameters:
      imfs : list of 1D numpy arrays
      fs : Sampling rate (Hz)
      freq_threshold : minimum frequency (Hz) for an IMF to be considered oscillatory.

    Returns:
      x0 : numpy array of shape (4*L,) containing the stacked initial state estimates,
           where L is the number of selected modes.
      selected_imfs : list of selected IMFs (those with avg frequency > threshold)
    """
    dt = 1/fs
    x0_list = []
    selected_imfs = []
    for imf in imfs:
        # Compute analytic signal using SciPy's hilbert transform
        analytic_signal = scipy.signal.hilbert(imf)
        phase = np.unwrap(np.angle(analytic_signal))
        if len(phase) < 2:
            continue
        # Estimate instantaneous frequency (rad/s)
        inst_freq = np.diff(phase) / dt
        avg_freq = np.mean(inst_freq)
        # Only select IMFs with average frequency above the threshold (converted to rad/s)
        if avg_freq < freq_threshold * 2 * np.pi:
            continue
        amplitude = np.mean(np.abs(analytic_signal))
        phase0 = phase[0]
        x0_list.append([amplitude * np.cos(phase0),
                        amplitude * np.sin(phase0),
                        avg_freq,
                        0.0])  # initial damping assumed 0
        selected_imfs.append(imf)
    if not x0_list:
        return np.array([]), []
    return np.array(x0_list).flatten(), selected_imfs

# ---------------------------------------------------------------------------
# Part 3: Multi-frequency Extended Kalman Filter (EKF)
# ---------------------------------------------------------------------------
def multi_freq_ekf(y, x0, P0, Q, R, fs):
    """
    Multi-frequency EKF for oscillation estimation.

    The state vector for L modes is:
      x = [x_c1, x_s1, ω1, σ1, x_c2, x_s2, ω2, σ2, ..., x_cL, x_sL, ωL, σL]^T.

    For each mode l:
      A_l = exp(-σ_l/fs), theta_l = ω_l/fs,
      x_c[l][k+1] = A_l * (x_c[l][k]*cos(theta_l) - x_s[l][k]*sin(theta_l))
      x_s[l][k+1] = A_l * (x_c[l][k]*sin(theta_l) + x_s[l][k]*cos(theta_l))
      ω[l][k+1]   = ω[l][k]
      σ[l][k+1]   = σ[l][k]

    Measurement model:
      y[k] = sum_{l}(x_c[l][k] + x_s[l][k]) + noise.

    Parameters:
      y   : 1D numpy array of measurements (length N).
      x0  : Initial state estimate (numpy array of size (4*L,)).
      P0  : Initial covariance matrix (4*L x 4*L).
      Q   : Process noise covariance matrix (4*L x 4*L).
      R   : Measurement noise variance (scalar).
      fs  : Sampling rate.

    Returns:
      x_est      : (N+1) x (4*L) state estimates.
      y_pred_all : 1D array of EKF predicted measurements.
    """
    N = len(y)
    state_dim = len(x0)
    L = state_dim // 4  # number of modes
    x_est = np.zeros((N+1, state_dim))
    x_est[0] = x0
    P = P0.copy()

    # Measurement matrix: each mode contributes (x_c + x_s)
    H = np.zeros((1, state_dim))
    for l in range(L):
        H[0, 4*l] = 1.0
        H[0, 4*l+1] = 1.0

    y_pred_all = np.zeros(N)

    for k in range(N):
        x = x_est[k].copy()
        x_pred = np.zeros(state_dim)
        F = np.zeros((state_dim, state_dim))  # Jacobian

        for l in range(L):
            idx = 4 * l
            x_c   = x[idx]
            x_s   = x[idx+1]
            omega = x[idx+2]
            sigma = x[idx+3]

            A = np.exp(-sigma/fs)
            theta = omega/fs

            # Nonlinear state prediction for mode l:
            x_pred[idx]   = A * (x_c * np.cos(theta) - x_s * np.sin(theta))
            x_pred[idx+1] = A * (x_c * np.sin(theta) + x_s * np.cos(theta))
            x_pred[idx+2] = omega     # frequency remains constant
            x_pred[idx+3] = sigma     # damping remains constant

            # Compute Jacobian entries for mode l:
            F[idx, idx]     = A * np.cos(theta)
            F[idx, idx+1]   = -A * np.sin(theta)
            F[idx, idx+2]   = -A/ fs * (x_c * np.sin(theta) + x_s * np.cos(theta))
            F[idx, idx+3]   = -A/ fs * (x_c * np.cos(theta) - x_s * np.sin(theta))

            F[idx+1, idx]   = A * np.sin(theta)
            F[idx+1, idx+1] = A * np.cos(theta)
            F[idx+1, idx+2] = A/ fs * (x_c * np.cos(theta) - x_s * np.sin(theta))
            F[idx+1, idx+3] = -A/ fs * (x_c * np.sin(theta) + x_s * np.cos(theta))

            F[idx+2, idx+2] = 1.0
            F[idx+3, idx+3] = 1.0

        y_pred = H @ x_pred  # predicted measurement (scalar)
        y_pred_all[k] = y_pred[0]

        P_pred = F @ P @ F.T + Q
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T / S
        innov = y[k] - y_pred[0]
        x_upd = x_pred + (K.flatten() * innov)
        P = P_pred - K @ H @ P_pred

        x_est[k+1] = x_upd

    return x_est, y_pred_all

# ---------------------------------------------------------------------------
# Part 4: Putting It All Together (EMD + EKF)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    fs = 30       # Sampling rate in Hz
    duration = 5  # seconds
    N = int(duration * fs)
    t = np.linspace(0, duration, N)

    # -----------------------------------------------------------------------
    # Simulate a "true" oscillatory signal (two modes) and add trend & noise.
    # -----------------------------------------------------------------------
    # Mode 1: 2π rad/s frequency, damping -0.1, amplitude 1.0, phase 0.
    # Mode 2: 3π rad/s frequency, damping 0.01, amplitude 0.5, phase π/4.
    data = np.loadtxt("/scripts/centroid_data_real/centroids_y_april.txt", delimiter=",")
    signal = data[:, 0]
    dc_component = np.mean(signal)
    # signal = signal - dc_component
    t = data[:, 1]


    # Ground truth oscillatory signal (sum of cosine & sine parts)
    y_clean = signal
    y_raw = signal

    # -----------------------------------------------------------------------
    # EMD Stage: Decompose y_raw into IMFs and residue.
    # -----------------------------------------------------------------------
    imfs, residue = emd(y_raw, max_imfs=8, tol=0.05, max_sift_iterations=100)

    # Use the Hilbert-based initial guess to select oscillatory IMFs.
    # (Only those with dominant frequency > 0.1 Hz are selected.)
    x0, selected_imfs = initial_guess_from_imfs(imfs, fs, freq_threshold=0.1)

    if len(selected_imfs) == 0:
        print("No oscillatory modes found by EMD!")
        exit()

    # Form the EMD-filtered signal by summing the selected IMFs.
    y_ekf = np.sum(np.array(selected_imfs), axis=0)

    # Determine the number of modes from the selected IMFs.
    L = len(x0) // 4
    print(f"Number of oscillatory modes detected: {L}")

    # Set initial covariance and process noise matrices according to dimension.
    P0 = np.diag([1.] * (4 * L))
    Q = np.diag([1e-3] * (4 * L))
    R = 8.

    # -----------------------------------------------------------------------
    # EKF Stage: Run the multi-frequency EKF on the EMD-filtered signal.
    # -----------------------------------------------------------------------
    x_est, y_pred_all = multi_freq_ekf(y_ekf, x0, P0, Q, R, fs)

    # -----------------------------------------------------------------------
    # Plotting Results
    # -----------------------------------------------------------------------
    plt.figure(figsize=(14, 12))

    # 1. Plot the raw measured signal, the EMD-filtered signal, and the ground truth.
    plt.subplot(3, 1, 1)
    plt.plot(t, y_raw, 'k.', markersize=3, label="Raw Measured Signal")
    plt.plot(t, y_ekf, 'r-', linewidth=2, label="EMD-filtered (sum of selected IMFs)")
    plt.plot(t, y_clean, 'b--', linewidth=2, label="Ground Truth Oscillatory")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Signal: Raw vs. EMD-filtered vs. Ground Truth")

    # 2. Plot EKF predicted signal vs. the EMD-filtered signal.
    plt.subplot(3, 1, 2)
    plt.plot(t, y_ekf, 'r-', linewidth=2, label="EMD-filtered Signal")
    plt.plot(t, y_pred_all, 'g--', linewidth=2, label="EKF Predicted Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("EKF Prediction vs. EMD-filtered Signal")

    # 3. Plot the estimated frequencies for all detected modes.
    plt.subplot(3, 1, 3)
    # For each mode, plot its estimated frequency over time.
    for l in range(L):
        idx = 4 * l + 2
        plt.plot(t, x_est[1:, idx], label=f"Estimated ω (mode {l+1})")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [rad/s]")
    plt.legend()
    plt.title("Estimated Frequencies via EKF")

    plt.tight_layout()
    plt.show()
