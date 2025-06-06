import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize

compensated_x_values = []
timestamps = []


def update_plot(val):
    A_x = sliders['A_x'].val
    A_y = sliders['A_y'].val
    freq = sliders['freq'].val
    phase_x = sliders['phase_x'].val
    phase_y = sliders['phase_y'].val

    image = np.zeros((height, width), dtype=np.float64)
    freq_rads = 2 * np.pi * freq
    frequency_rads_us = freq_rads / 1e-6
    init_t = events[0][0]
    idx = 0
    for timestamp, x, y, polarity in events:
        compensated_x_values[idx] = x - A_x * np.sin(frequency_rads_us * (timestamp - init_t) + phase_x)
        compensated_x = int(compensated_x_values[idx])
        idx += 1

        compensated_y = int(y - A_y * np.sin(frequency_rads_us * (timestamp - init_t) + phase_y))
        if 0 <= compensated_x < width and 0 <= compensated_y < height:
            image[compensated_y, compensated_x] += 1 if polarity else -1

    scatter.set_offsets(np.column_stack((timestamps, compensated_x_values)))

    compensated_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("Compensated Image", compensated_image)
    cv2.waitKey(1)

    fig.canvas.draw_idle()


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


# Parameters
time_window_us = .03  # 1 second window
width, height = 1280, 720  # Image dimensions
timestamps = np.linspace(0, time_window_us, 1000)

if __name__ == "__main__":
    # Parameters
    width, height = 1280, 720  # Image dimensions
    time_window = .1  # Time window in seconds
    time_window_us = time_window * 1e6  # Time window in microseconds

    file_path = '/home/viciopoli/datasets/event_harmeda/rotate.hdf5'
    events = read_events_from_hdf5(file_path, time_window_us)

    # scatter plot events x and time with small dots
    compensated_x_values = [x for _, x, _, _ in events]
    timestamps = [t for t, _, _, _ in events]

    # Initial guess for parameters
    initial_guess = [2, 2, 50.0, 0, np.pi / 4]

    # Setup figure and axes
    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    ax = axes
    scatter = ax.scatter(timestamps, compensated_x_values, c='b', s=5)
    ax.set_xlim(0, time_window_us)
    ax.set_ylim(-50, width + 50)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Compensated X")

    # Add sliders
    slider_axs = {
        'A_x': plt.axes([0.2, 0.02, 0.65, 0.02]),
        'A_y': plt.axes([0.2, 0.05, 0.65, 0.02]),
        'freq': plt.axes([0.2, 0.08, 0.65, 0.02]),
        'phase_x': plt.axes([0.2, 0.11, 0.65, 0.02]),
        'phase_y': plt.axes([0.2, 0.14, 0.65, 0.02]),
    }

    sliders = {
        'A_x': Slider(slider_axs['A_x'], 'A_x', 0.1, 50.0, valinit=initial_guess[0]),
        'A_y': Slider(slider_axs['A_y'], 'A_y', 0.1, 50.0, valinit=initial_guess[1]),
        'freq': Slider(slider_axs['freq'], 'freq', 1.0, 200.0, valinit=initial_guess[2]),
        'phase_x': Slider(slider_axs['phase_x'], 'phase_x', -np.pi, np.pi, valinit=initial_guess[3]),
        'phase_y': Slider(slider_axs['phase_y'], 'phase_y', -np.pi, np.pi, valinit=initial_guess[4]),
    }

    for slider in sliders.values():
        slider.on_changed(update_plot)

    # Initial plot update
    update_plot(None)
    plt.show()
