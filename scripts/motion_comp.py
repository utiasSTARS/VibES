import numpy as np
import cv2
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


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


compensated_x_values_pos = []
compensated_x_values_neg = []
timestamps_pos = []
timestamps_neg = []


# Define function to update plot
def update_plot():
    scatter.set_offsets(np.column_stack((timestamps_pos, compensated_x_values_pos)))
    scatter_neg.set_offsets(np.column_stack((timestamps_neg, compensated_x_values_neg)))
    plt.pause(0.05)


def compensation_vis(events, A_x, A_y, B_x, B_y, omega, width,
                     height):
    image = np.zeros((height, width), dtype=np.float64)
    freq_rads = 2 * np.pi * omega
    frequency_rads_us = freq_rads * 1e-6
    init_t = events[0][0]
    for timestamp, x, y, polarity in events:
        compensated_x = x - (A_x * np.sin(frequency_rads_us * (timestamp - init_t)) + B_x * np.sin(
            frequency_rads_us * (timestamp - init_t)))

        compensated_y = y - (A_y * np.sin(frequency_rads_us * (timestamp - init_t)) + B_y * np.sin(
            frequency_rads_us * (timestamp - init_t)))

        compensated_x = int(compensated_x)
        compensated_y = int(compensated_y)
        if 0 <= compensated_x < width and 0 <= compensated_y < height:
            image[compensated_y, compensated_x] += 1 if polarity else -1
    # cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    return image


def compensate_motion_with_sinusoidal(events, A_x, A_y, B_x, B_y, omega, width,
                                      height):
    image = np.zeros((height, width), dtype=np.float64)
    freq_rads = 2 * np.pi * omega
    frequency_rads_us = freq_rads * 1e-6
    init_t = events[0][0]
    idx_neg = 0
    idx_pos = 0
    for timestamp, x, y, polarity in events:
        compensated_x = x - (A_x * np.sin(frequency_rads_us * (timestamp - init_t)) + B_x * np.sin(
            frequency_rads_us * (timestamp - init_t)))
        if polarity:
            compensated_x_values_pos[idx_pos] = compensated_x
            idx_pos += 1
        else:
            compensated_x_values_neg[idx_neg] = compensated_x
            idx_neg += 1

        compensated_y = y - (A_y * np.sin(frequency_rads_us * (timestamp - init_t)) + B_y * np.sin(
            frequency_rads_us * (timestamp - init_t)))

        compensated_x = int(compensated_x)
        compensated_y = int(compensated_y)
        if 0 <= compensated_x < width and 0 <= compensated_y < height:
            image[compensated_y, compensated_x] += 1  # if polarity else -1
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


def objective_function(params, events, width, height):
    A_x, A_y, B_x, B_y, omega = params
    compensated_image = compensate_motion_with_sinusoidal(
        events, A_x, A_y, B_x, B_y, omega, width, height
    )
    # compensated_image = cv2.normalize(compensated_image, None, 0, 1, cv2.NORM_MINMAX)
    # smooth the values in compensated_image with a Gaussian filter
    compensated_image_smooth = gaussian_filter(compensated_image, sigma=2.)
    cv2.imshow("smooth", compensated_image_smooth)
    # get values that are greater than 0
    # compensated_image_smooth = compensated_image_smooth[compensated_image_smooth > 0]
    sharpness = np.var(compensated_image_smooth - np.mean(compensated_image_smooth))
    # sharpness = compute_sharpness(compensated_image_smooth)
    sharpness_values.append(sharpness)
    # write sharpness level on image
    # normalize image
    compensated_image = cv2.normalize(compensated_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.putText(compensated_image, f"Sharpness: {sharpness:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
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

    file_path = '/home/viciopoli/datasets/event_harmeda/circle_real.hdf5'
    events = read_events_from_hdf5(file_path, time_window_us)

    # scatter plot events x and time with small dots
    compensated_x_values_pos = [x for _, x, _, p in events if p]
    compensated_x_values_neg = [x for _, x, _, p in events if not p]
    timestamps_pos = [t for t, _, _, p in events if p]
    timestamps_neg = [t for t, _, _, p in events if not p]

    # Initialize plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(compensated_x_values_pos, timestamps_pos, c='b', s=2)  # Blue scatter points
    scatter_neg = ax.scatter(compensated_x_values_neg, timestamps_neg, c='r', s=2)  # Red scatter points
    ax.set_xlim(0, time_window_us)  # Will be updated dynamically
    ax.set_ylim(0, 1280)  # Adjust based on expected range
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Compensated X")
    plt.ion()  # Interactive mode on

    # origin
    original_img_unn = compensation_vis(
        events, 0, 0, 0, 0, 0, width, height
    )
    original_img = cv2.normalize(original_img_unn.copy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Initial guess for the parameters: A_x, A_y, frequency, phase_x, phase_y
    omega_init = 15.
    phi_init = np.pi / 2.
    A = 20.
    A1 = 10.
    A_x = A * np.sin(0.)
    B_x = A * np.cos(0.)
    A_y = A1 * np.sin(phi_init)
    B_y = A1 * np.cos(phi_init)

    initial_guess = [A_x, A_y, B_x, B_y, omega_init]  # Adjust these based on your expectations

    # Optimization process
    result = minimize(
        objective_function,
        initial_guess,
        args=(events, width, height),
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

    final_compensated_image_unn = compensation_vis(
        events, optimal_params[0], optimal_params[1], optimal_params[2], optimal_params[3],
        optimal_params[4], width, height
    )
    final_compensated_image = cv2.normalize(final_compensated_image_unn.copy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    print(f"Sharpness original {np.array(original_img_unn).var()}")
    print(f"Sharpness compensated {np.array(final_compensated_image_unn).var()}")
    print("-----------------------")
    print(f"Omega: {optimal_params[4]:2f}")
    print(f"A x: {(np.sqrt(optimal_params[0] ** 2 + optimal_params[2] ** 2)):2f}")
    print(f"A y: {(np.sqrt(optimal_params[1] ** 2 + optimal_params[3] ** 2)):2f}")
    print(f"phi x: {(np.arctan2(abs(optimal_params[2]), abs(optimal_params[0]))):2f}")
    print(f"phi y: {(np.arctan2(abs(optimal_params[3]), abs(optimal_params[1]))):2f}")

    # Display the final compensated image
    cv2.imshow("Final Compensated Image", final_compensated_image)
    cv2.imshow("Original image", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.plot(np.array(sharpness_values))
    plt.xlabel("Iteration")
    plt.ylabel("Sharpness Score")
    plt.title("Sharpness Evolution")
    plt.show()

    # do an interactive thersholding and canny edge detection
    max_lowThreshold = 200
    window_name = 'Edge Map'
    title_trackbar = 'Min Threshold:'
    ratio = 3
    kernel_size = 3


    def CannyThreshold(val):
        low_threshold = val
        img_blur = cv2.blur(final_compensated_image, (3, 3))
        detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
        mask = detected_edges != 0
        dst = final_compensated_image[..., None] * (mask[:, :, None].astype(final_compensated_image.dtype))
        cv2.imshow(window_name, dst)


    cv2.namedWindow(window_name)
    cv2.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)

    CannyThreshold(0)
    cv2.waitKey()

    # find edges using canny edge detection
    edges_orig = cv2.Canny(original_img, 100, 200)
    cv2.imshow("Edges original", edges_orig)

    edges_comp = cv2.Canny(final_compensated_image, 100, 200)
    cv2.imshow("Edges compensated", edges_comp)

    cv2.waitKey(0)
