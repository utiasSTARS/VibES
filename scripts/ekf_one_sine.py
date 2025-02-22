import numpy as np
import matplotlib.pyplot as plt


# ----- 1) Measurement Model and EKF Measurement Update -----

def h(x):
    """
    Measurement function for the 4D state:
        x = [a, omega, phi, b]
    Measurement: z = b + a * sin(phi)
    """
    a, omega, phi, b = x
    return b + a * np.sin(phi)


def jacobian_h(x):
    """
    Jacobian of h(x) = b + a * sin(phi) with respect to x = [a, omega, phi, b].

    Partial derivatives:
      ∂h/∂a = sin(phi)
      ∂h/∂omega = 0
      ∂h/∂phi = a * cos(phi)
      ∂h/∂b = 1
    """
    a, omega, phi, b = x
    return np.array([[np.sin(phi), 0.0, a * np.cos(phi), 1.0]])


def ekf_measurement_update(x, P, z, R):
    """
    EKF measurement update for a single scalar measurement z.

    Args:
        x (np.ndarray): 4D predicted state vector [a, omega, phi, b].
        P (np.ndarray): 4x4 predicted state covariance.
        z (float): Current measurement.
        R (float): Measurement noise variance (scalar).

    Returns:
        x_new (np.ndarray): Updated 4D state.
        P_new (np.ndarray): Updated 4x4 covariance.
    """
    # Predicted measurement
    z_pred = h(x)
    # Residual (innovation)
    y = z - z_pred
    # Jacobian of the measurement function
    H = jacobian_h(x)  # shape = (1, 4)
    # Innovation covariance
    S = H @ P @ H.T + R  # shape = (1, 1)
    # Kalman gain
    K = P @ H.T @ np.linalg.inv(S)  # shape = (4, 1)
    # State update
    x_new = x + (K * y).flatten()
    # Covariance update
    I = np.eye(len(x))
    P_new = (I - K @ H) @ P
    return x_new, P_new


# ----- 2) Process Model and Its Jacobian -----

def f(x, dt):
    """
    Process model for the 4D state x = [a, omega, phi, b].
    We assume:
        a_{k+1}   = a_k
        omega_{k+1} = omega_k
        phi_{k+1}   = phi_k + omega_k * dt
        b_{k+1}     = b_k
    """
    a, omega, phi, b = x
    return np.array([a, omega, phi + omega * dt, b])


def jacobian_f(x, dt):
    """
    Jacobian of f(x) w.r.t x for the 4D state.

    f(x) = [ a,
             omega,
             phi + omega*dt,
             b ]

    So the Jacobian is:
       [1,   0,     0,   0]
       [0,   1,     0,   0]
       [0,  dt,     1,   0]
       [0,   0,     0,   1]
    """
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, dt, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


# ----- 3) Process Noise Covariance -----

def get_process_noise_covariance(dt, sigma_a, sigma_omega, sigma_b):
    """
    The 4D process noise covariance Q.
    Here we combine:
      - a small random walk for amplitude (a)
      - integrated frequency noise for (omega, phi)
      - a small random walk for offset (b)

    The block for [omega, phi] can follow the integrated noise approach:
      [ dt * sigma_omega^2,      (dt^2/2) * sigma_omega^2 ]
      [ (dt^2/2)* sigma_omega^2, (dt^3/3)* sigma_omega^2 ]

    So Q can be:
      Q = [
        [ dt*sigma_a^2,               0,                          0,                    0 ],
        [ 0,               dt*sigma_omega^2,    (dt^2/2)*sigma_omega^2,   0 ],
        [ 0,     (dt^2/2)*sigma_omega^2, (dt^3/3)*sigma_omega^2,          0 ],
        [ 0,                          0,                          0,       dt*sigma_b^2 ]
      ]
    """
    Q = np.array([
        [dt * sigma_a ** 2, 0.0, 0.0, 0.0],
        [0.0, dt * sigma_omega ** 2, (dt ** 2 / 2) * sigma_omega ** 2, 0.0],
        [0.0, (dt ** 2 / 2) * sigma_omega ** 2, (dt ** 3 / 3) * sigma_omega ** 2, 0.0],
        [0.0, 0.0, 0.0, dt * sigma_b ** 2]
    ])
    return Q


# ----- 4) Putting It All Together: EKF with Real Data -----

def run_ekf_on_real_data():
    # 1) Load your real data.
    #    Columns assumed: [ measurement (signal), time_stamp ].
    data = np.loadtxt(
        "./scripts/centroid_data_real/centroids_y_dot.txt",
        delimiter=","
    )
    signal = data[:, 0]  # measured signal (e.g., centroid y position)
    t = data[:, 1]  # time stamps
    N = len(t)

    # 2) Set an initial guess for the state [a, omega, phi, b].
    x_hat = np.array([20.0, 2.0 * np.pi * 100.0, 0.0, 300.0])

    # 3) Initialize the covariance matrix
    P = np.eye(4) * 100.0  # diagonal matrix with large values

    # 4) Define noise parameters (tuning). Adjust as needed:
    sigma_a = 3.  # amplitude random walk
    sigma_omega = 3.  # frequency noise
    sigma_b = 1.  # offset random walk
    R = 0.05  # measurement noise variance (try bigger if data is very noisy)

    # 5) Storage for results
    ekf_predictions = []
    x_estimates = []
    t_est = []
    residuals = []

    # 6) Run the filter over the data
    for i in range(N):
        if i == 0:
            # First measurement, just record initial state
            ekf_predictions.append(h(x_hat))
            x_estimates.append(x_hat.copy())
            t_est.append(t[i])
            residuals.append(signal[i] - ekf_predictions[-1])
        else:
            # (a) Prediction step
            dt = t[i] - t[i - 1]
            F = jacobian_f(x_hat, dt)
            x_pred = f(x_hat, dt)
            Q = get_process_noise_covariance(dt, sigma_a, sigma_omega, sigma_b)
            P_pred = F @ P @ F.T + Q

            # (b) Measurement residual before update
            z = signal[i]
            z_pred = h(x_pred)
            y = z - z_pred
            residuals.append(y)

            # (c) Measurement update
            x_hat, P = ekf_measurement_update(x_pred, P_pred, z, R)

            # Store results
            ekf_predictions.append(h(x_hat))
            x_estimates.append(x_hat.copy())
            t_est.append(t[i])

    print(x_hat/(2*np.pi))
    # Convert to arrays
    t_est = np.array(t_est)
    ekf_predictions = np.array(ekf_predictions)
    residuals = np.array(residuals)

    # ----- 7) Plot the results -----
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, 'o', label="Real Measurement", markersize=3, alpha=0.6)
    plt.plot(t_est, ekf_predictions, '-', label="EKF Prediction", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Signal")
    plt.title("Real Measurements and EKF Estimated Signal (4D State with Offset)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot residuals
    plt.figure(figsize=(10, 4))
    plt.plot(t_est, residuals, 'r-', label="Residual (z - h(x_pred))")
    plt.xlabel("Time [s]")
    plt.ylabel("Residual")
    plt.title("Measurement Residuals Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_ekf_on_real_data()
