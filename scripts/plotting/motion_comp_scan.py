import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import h5py
from tqdm import tqdm


def accumulate(events, params):
    acc = np.zeros((height, width), dtype=np.float64)
    A_x, A_y, frequency, phase_x, phase_y = params
    for timestamp, x, y, polarity in events:
        compensated_x = int(x - A_x * np.sin(frequency * timestamp + phase_x))
        compensated_y = int(y - A_y * np.sin(frequency * timestamp + phase_y))
        if 0 <= compensated_x < width and 0 <= compensated_y < height:
            acc[compensated_y, compensated_x] += 1  # or -=1 if polarity is 0
    return acc


def update_plot(val):
    if val == None:
        return
    A_x = sliders['A_x'].val
    A_y = sliders['A_y'].val
    freq = sliders['freq'].val
    phase_x = sliders['phase_x'].val
    phase_y = sliders['phase_y'].val

    freq_rads = 2 * np.pi * freq
    frequency_rads_us = freq_rads / 1e6

    acc = accumulate(events, [A_x, A_y, frequency_rads_us, phase_x, phase_y])

    # Normalize for display
    img_display.set_data(acc)
    img_display.set_clim(vmin=0, vmax=np.max(acc) if np.max(acc) > 0 else 1)
    fig.canvas.draw_idle()


def read_events_from_hdf5(file_path, time_window_us):
    events = []
    with h5py.File(file_path, 'r') as f:
        dataset = f['CD/events']
        initial_time = dataset[0][3]
        for e in tqdm(dataset):
            x, y, p, ts = e
            # if x < 0 or y < 0 or x >= width/2 or y >= height/2:
            #     continue
            events.append((ts, x, y, p))
            delta = ts - initial_time
            if delta > time_window_us:
                break
    print(f"Read {len(events)} events from {file_path}")
    return events


def sharpness(image):
    # compute the variance of the image
    var = np.var(image - np.mean(image))
    return var


# Parameters
width, height = 640, 480
time_window = 0.1
time_window_us = time_window * 1e6
file_path = '/home/viciopoli/datasets/event_harmeda/harmeda_newpattern_hz10_3mm.hdf5'

# Load events
events = read_events_from_hdf5(file_path, time_window_us)

# Initial guess for sliders
initial_guess = [2, 2, 10.0, np.pi / 2, 0]

# Initial accumulation
accumulation = accumulate(events, initial_guess)

# Test the accumulation function with different parameters
var_variation = []

# iterate between -[pi, pi] for phase_x and phase_y
for p in np.linspace(-np.pi, np.pi, 100):
    initial_guess[3] += p
    initial_guess[4] += p
    for i in range(-50, 50):
        initial_guess[0] = i / 10
        # for j in range(10):
        # initial_guess[1] = j
        accumulation = accumulate(events, initial_guess)
        var_variation.append(sharpness(accumulation))

# plot the variation
plt.figure()
plt.plot(var_variation)
plt.title("Sharpness Variation")
plt.xlabel("Parameter Variation")
plt.ylabel("Sharpness")
plt.show()

# Plot setup
fig, ax = plt.subplots(figsize=(10, 6))
img_display = ax.imshow(accumulation, cmap='viridis', origin='lower')
ax.set_title("Accumulated Event Image")

# Sliders
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

update_plot(None)
plt.show()
