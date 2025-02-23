import numpy as np
import cv2
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Function to generate motion maps based on sinusoidal parameters
def generate_motion_map(frequency, amplitude_x, amplitude_y, phase_shift, time_window, num_samples):
    t = np.linspace(0, time_window, num_samples)
    motion_map_x = amplitude_x * np.sin(2 * np.pi * frequency * t + phase_shift[0])
    motion_map_y = amplitude_y * np.sin(2 * np.pi * frequency * t + phase_shift[1])
    return motion_map_x, motion_map_y


def accumulate_events(events, width, height, time_window):
    image = np.zeros((height, width), dtype=np.float32)
    time_us = time_window * 1e6
    for timestamp, x, y, polarity in events:
        if timestamp < time_us:
            if 0 <= x < width and 0 <= y < height:
                image[y, x] += 1 if polarity else -1
    return image  # Return raw float32 image instead of normalizing


compensated_x_values = []
timestamps = []


# Define function to update plot
def update_plot():
    scatter.set_offsets(np.column_stack((timestamps, compensated_x_values)))
    plt.pause(0.05)


def compensate_motion_with_sinusoidal(events, frequency, amplitude_x, amplitude_y, phase_shift, time_window, width,
                                      height):
    image = np.zeros((height, width), dtype=np.float64)
    freq_rads = 2 * np.pi * frequency
    frequency_rads_us = freq_rads / 1e-6
    init_t = events[0][0]
    idx = 0
    for timestamp, x, y, polarity in events:
        compensated_x = int(x - amplitude_x * np.sin(frequency_rads_us * (timestamp - init_t) + phase_shift[0]))
        compensated_x_values[idx] = compensated_x
        idx += 1

        compensated_y = int(y - amplitude_y * np.sin(frequency_rads_us * (timestamp - init_t) + phase_shift[1]))
        if 0 <= compensated_x < width and 0 <= compensated_y < height:
            image[compensated_y, compensated_x] += 1 # if polarity else -1
    update_plot()
    return image  # Return raw float32 image instead of normalizing


# def compute_sharpness(image):
#     if image.dtype != np.float64:
#         image = image.astype(np.float64)
#
#     laplacian = cv2.Laplacian(image, cv2.CV_64F)
#     variance = laplacian.var()
#     return variance

def compute_sharpness(image):
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.sum(np.abs(dx) + np.abs(dy))


# Objective function for optimization
sharpness_values = []


def objective_function(params, events, time_window, width, height):
    A_x, A_y, frequency, phase_x, phase_y = params
    compensated_image = compensate_motion_with_sinusoidal(
        events, frequency, A_x, A_y, (phase_x, phase_y), time_window, width, height
    )
    # compensated_image = cv2.normalize(compensated_image, None, 0, 1, cv2.NORM_MINMAX)
    sharpness = compute_sharpness(compensated_image)
    sharpness_values.append(sharpness)
    # write sharpness level on image
    # normalize image
    compensated_image = cv2.normalize(compensated_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.putText(compensated_image, f"Sharpness: {sharpness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.imshow("Compensated Image", compensated_image)
    cv2.waitKey(1)
    return -sharpness  # Negate to maximize sharpness


def read_events_from_hdf5(file_path, time_window_us):
    import h5py
    initial_time = -1
    events = []
    with h5py.File(file_path, 'r') as f:
        dataset = f['CD/events']
        initial_time = dataset[0][3]  # First timestamp in dataset
        for e in dataset:
            x, y, p, ts = e
            if x < 0 or y < 0 or x >= 1280 or y >= 720:
                continue
            delta = ts - initial_time
            events.append((ts, x, y, p))

            if delta > time_window_us:
                break  # Stop when reaching the time window or max events

    # print info
    print(f"Read {len(events)} events from {file_path}")
    return events


# Main script
if __name__ == "__main__":
    # Parameters
    width, height = 1280, 720  # Image dimensions
    time_window = .1  # Time window in seconds
    time_window_us = time_window * 1e6  # Time window in microseconds

    file_path = '/home/viciopoli/datasets/event_harmeda/dot_static_undist.hdf5'
    events = read_events_from_hdf5(file_path, time_window_us)

    # scatter plot events x and time with small dots
    compensated_x_values = [x for _, x, _, _ in events]
    timestamps = [t for t, _, _, _ in events]

    # Initialize plot
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], c='b', s=5)  # Blue scatter points
    ax.set_xlim(0, time_window_us)  # Will be updated dynamically
    ax.set_ylim(0, 1280)  # Adjust based on expected range
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Compensated X")
    plt.ion()  # Interactive mode on

    # Initial guess for the parameters: A_x, A_y, frequency, phase_x, phase_y
    initial_guess = [1.5, 1.5, 45.0, 0, np.pi / 4]  # Adjust these based on your expectations

    # Optimization process
    result = minimize(
        objective_function,
        initial_guess,
        args=(events, time_window, width, height),
        method='Nelder-Mead',
        options={'maxiter': 5000, 'disp': True}
    )

    # print opt message
    print(result.message)

    # Display optimal parameters
    optimal_params = result.x
    print("Optimal parameters:", optimal_params)

    plt.ioff()  # Turn off interactive mode
    plt.show()

    # Optionally: visualize the final compensated image using optimal parameters
    # final_compensated_image = compensate_motion_with_sinusoidal(
    #     events, optimal_params[2], optimal_params[0], optimal_params[1],
    #     (optimal_params[3], optimal_params[4]), time_window, width, height
    # )
    # final_compensated_image = cv2.normalize(final_compensated_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Display the final compensated image
    # cv2.imshow("Final Compensated Image", final_compensated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.plot(np.array(sharpness_values))
    plt.xlabel("Iteration")
    plt.ylabel("Sharpness Score")
    plt.title("Sharpness Evolution")
    plt.show()
