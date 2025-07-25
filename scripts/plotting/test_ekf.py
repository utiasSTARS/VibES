import numpy as np
import matplotlib.pyplot as plt

def multi_freq_ekf(y, x0, P0, Q, R, fs):
    """
    Extended Kalman Filter for multi-frequency oscillation estimation.

    The state vector for L modes is defined as:
      x = [x_c1, x_s1, ω1, σ1, x_c2, x_s2, ω2, σ2, ..., x_cL, x_sL, ωL, σL]^T.

    For each mode l (l = 1,...,L), the nonlinear state transition is:
      Let A_l = exp(-σ_l/fs) and theta_l = ω_l/fs, then
        x_c_l[k+1] = A_l * (x_c_l[k]*cos(theta_l) - x_s_l[k]*sin(theta_l))
        x_s_l[k+1] = A_l * (x_c_l[k]*sin(theta_l) + x_s_l[k]*cos(theta_l))
        ω_l[k+1]   = ω_l[k]
        σ_l[k+1]   = σ_l[k]

    The measurement model is:
      y[k] = sum_{l=1}^L (x_c_l[k] + x_s_l[k]) + noise.

    Parameters:
      y   : 1D numpy array of measurements (length N).
      x0  : Initial state estimate, numpy array of size (4*L,).
      P0  : Initial covariance matrix, shape (4*L, 4*L).
      Q   : Process noise covariance matrix, shape (4*L, 4*L).
      R   : Measurement noise variance (scalar).
      fs  : Sampling rate.

    Returns:
      x_est    : (N+1) x (4*L) numpy array containing state estimates over time.
      y_pred_all : 1D numpy array of length N containing the EKF predicted measurements.
    """
    N = len(y)
    state_dim = len(x0)
    L = state_dim // 4  # number of oscillation modes
    x_est = np.zeros((N+1, state_dim))
    x_est[0] = x0
    P = P0.copy()

    # Build the measurement matrix H (1 x 4L): for each mode, we measure x_c + x_s.
    H = np.zeros((1, state_dim))
    for l in range(L):
        H[0, 4*l] = 1.0
        H[0, 4*l + 1] = 1.0

    # Array to store the predicted measurement at each step.
    y_pred_all = np.zeros(N)

    for k in range(N):
        x = x_est[k].copy()
        x_pred = np.zeros(state_dim)
        # Build the overall Jacobian as a block diagonal matrix.
        F = np.zeros((state_dim, state_dim))

        # Process each mode independently
        for l in range(L):
            idx = 4 * l
            x_c   = x[idx]
            x_s   = x[idx+1]
            omega = x[idx+2]
            sigma = x[idx+3]

            A = np.exp(-sigma / fs)
            theta = omega / fs

            # Nonlinear prediction for this mode:
            x_pred[idx]   = A * (x_c * np.cos(theta) - x_s * np.sin(theta))
            x_pred[idx+1] = A * (x_c * np.sin(theta) + x_s * np.cos(theta))
            x_pred[idx+2] = omega     # frequency remains constant
            x_pred[idx+3] = sigma     # damping remains constant

            # Compute the Jacobian for mode l (partial derivatives)
            # For x_c:
            F[idx, idx]     = A * np.cos(theta)
            F[idx, idx+1]   = -A * np.sin(theta)
            F[idx, idx+2]   = -A/ fs * (x_c * np.sin(theta) + x_s * np.cos(theta))
            F[idx, idx+3]   = -A/ fs * (x_c * np.cos(theta) - x_s * np.sin(theta))
            # For x_s:
            F[idx+1, idx]   = A * np.sin(theta)
            F[idx+1, idx+1] = A * np.cos(theta)
            F[idx+1, idx+2] = A/ fs * (x_c * np.cos(theta) - x_s * np.sin(theta))
            F[idx+1, idx+3] = -A/ fs * (x_c * np.sin(theta) + x_s * np.cos(theta))
            # ω and σ remain constant:
            F[idx+2, idx+2] = 1.0
            F[idx+3, idx+3] = 1.0

        # Save the predicted measurement for plotting:
        y_pred = H @ x_pred   # shape (1,)
        y_pred_all[k] = y_pred[0]

        # Predict covariance for the full state
        P_pred = F @ P @ F.T + Q

        # ----- Measurement update -----
        # Predicted measurement already computed above (y_pred)
        S = H @ P_pred @ H.T + R  # Innovation covariance (scalar)
        K = P_pred @ H.T / S      # Kalman gain (state_dim x 1)
        innov = y[k] - y_pred[0]  # Innovation (measurement residual)
        x_upd = x_pred + (K.flatten() * innov)
        P = P_pred - K @ H @ P_pred

        x_est[k+1] = x_upd

    return x_est, y_pred_all

# =============================================================================
# Example usage: Simulating two oscillation modes and applying the multi-frequency EKF.
# =============================================================================

if __name__ == '__main__':
    fs = 100          # Sampling rate in Hz
    N = 150          # Number of samples
    t = np.arange(N) / fs

    # Define two oscillation modes:
    # Mode 1: frequency = 2π rad/s, damping = -0.1, amplitude = 1.0, phase = 0.
    # Mode 2: frequency = 3π rad/s, damping = 0.01, amplitude = 0.5, phase = π/4.
    A1, phi1 = .50, 0.0
    omega1, sigma1 = 500 * np.pi, -0.1
    A2, phi2 = 5, np.pi/4
    omega2, sigma2 = 100 * np.pi, 0.01

    # Simulate the true state evolution for both modes.
    x_true = np.zeros((N, 8))  # 8 = 4 states * 2 modes

    # Initialize true states:
    # Mode 1:
    x_true[0, 0] = A1 * np.cos(phi1)
    x_true[0, 1] = A1 * np.sin(phi1)
    x_true[0, 2] = omega1
    x_true[0, 3] = sigma1
    # Mode 2:
    x_true[0, 4] = A2 * np.cos(phi2)
    x_true[0, 5] = A2 * np.sin(phi2)
    x_true[0, 6] = omega2
    x_true[0, 7] = sigma2

    # Propagate the state over time using the same nonlinear model.
    for k in range(N-1):
        for l in range(2):
            idx = 4 * l
            x_c = x_true[k, idx]
            x_s = x_true[k, idx+1]
            omega = x_true[k, idx+2]
            sigma = x_true[k, idx+3]
            A_exp = np.exp(-sigma/fs)
            theta = omega/fs
            x_true[k+1, idx]   = A_exp * (x_c * np.cos(theta) - x_s * np.sin(theta))
            x_true[k+1, idx+1] = A_exp * (x_c * np.sin(theta) + x_s * np.cos(theta))
            x_true[k+1, idx+2] = omega
            x_true[k+1, idx+3] = sigma

    # The ground truth measurement is the sum of the cosine and sine components for both modes.
    y_clean = np.sum(x_true[:, [0, 1, 4, 5]], axis=1)
    noise_std = 1.05
    # Noisy measurement (simulating sensor noise)
    y_meas = y_clean + np.random.normal(0, noise_std, size=y_clean.shape)

    # Initial state guess: use approximate values (possibly obtained from a trigger like FFT or HHT)
    x0 = np.zeros(8)
    # Mode 1 initial guess:
    x0[0] = A1 * np.cos(phi1) * 1.1   # slight perturbation
    x0[1] = A1 * np.sin(phi1) * 1.1
    x0[2] = omega1 * 0.9
    x0[3] = sigma1 * 1.1
    # Mode 2 initial guess:
    x0[4] = A2 * np.cos(phi2) * 1.1
    x0[5] = A2 * np.sin(phi2) * 1.1
    x0[6] = omega2 * 0.9
    x0[7] = sigma2 * 1.1

    # Initial covariance, process noise, and measurement noise
    P0 = np.diag([0.1] * 8)
    Q = np.diag([1e-6] * 8)
    R = noise_std**2

    # Run the multi-frequency EKF. The function now returns both the state estimates and the predicted measurements.
    x_est, y_pred_all = multi_freq_ekf(y_meas, x0, P0, Q, R, fs)

    # Plotting results:
    plt.figure(figsize=(12, 10))

    # Plot the measurement predictions along with the GT and the noisy measurements.
    plt.subplot(3, 1, 1)
    plt.plot(t, y_clean, 'k-', label='Ground Truth')
    plt.plot(t, y_meas, 'o', label='Noisy Measurement', markersize=3, alpha=0.6)
    plt.plot(t, y_pred_all, 'r--', label='EKF Prediction')
    plt.xlabel('Time [s]')
    plt.ylabel('Signal')
    plt.legend()
    plt.title('Signal Comparison')

    # Plot frequencies for both modes.
    plt.subplot(3, 1, 2)
    plt.plot(t, x_true[:, 2], 'k--', label='True ω1')
    plt.plot(t, x_est[1:, 2], 'b', label='Estimated ω1')
    plt.plot(t, x_true[:, 6], 'k--', label='True ω2')
    plt.plot(t, x_est[1:, 6], 'r', label='Estimated ω2')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [rad/s]')
    plt.legend()
    plt.title('Frequency Estimation')

    # Plot damping factors for both modes.
    plt.subplot(3, 1, 3)
    plt.plot(t, x_true[:, 3], 'k--', label='True σ1')
    plt.plot(t, x_est[1:, 3], 'b', label='Estimated σ1')
    plt.plot(t, x_true[:, 7], 'k--', label='True σ2')
    plt.plot(t, x_est[1:, 7], 'r', label='Estimated σ2')
    plt.xlabel('Time [s]')
    plt.ylabel('Damping Factor')
    plt.legend()
    plt.title('Damping Factor Estimation')

    plt.tight_layout()
    plt.show()
