import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(
    '/home/viciopoli/datasets/event_harmeda/harmeda_dataset/results/logo/edge_analysis_continuity_20250814_181906.csv')

# Convert from string to numpy arrays and extract max
max_novib_length = [np.array(ast.literal_eval(a)).max() for a in df['novib_lengths'][100:1400]]
max_vib_length = [np.array(ast.literal_eval(a)).max() for a in df['vib_lengths'][100:1400]]
image_ids = df['image_id'][100:1400].to_numpy()

window_size = 10

def rolling_stats(values, window):
    """Compute rolling min, median, and max."""
    values = np.array(values)
    min_vals, med_vals, max_vals = [], [], []
    for i in range(len(values) - window + 1):
        window_data = values[i:i+window]
        min_vals.append(window_data.min())
        med_vals.append(np.median(window_data))
        max_vals.append(window_data.max())
    return np.array(min_vals), np.array(med_vals), np.array(max_vals)

# Compute rolling statistics
novib_min, novib_med, novib_max = rolling_stats(max_novib_length, window_size)
vib_min, vib_med, vib_max = rolling_stats(max_vib_length, window_size)

# Fix: match x-axis length
x_vals = image_ids[window_size - 1:]

plt.figure(figsize=(12, 6))

# No VIB mode
plt.plot(x_vals, novib_med, 'r-', label='No VIB (median)', alpha=0.8)
plt.fill_between(x_vals, novib_min, novib_max, color='r', alpha=0.2)

# VIB mode
plt.plot(x_vals, vib_med, 'b-', label='VIB (median)', alpha=0.8)
plt.fill_between(x_vals, vib_min, vib_max, color='b', alpha=0.2)

plt.xlabel('Image ID')
plt.ylabel('Total Edge Length (pixels)')
plt.title('Edge Length Comparison with Rolling Median and Min/Max Shading')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 2. Plot number of connected components
plt.figure(figsize=(12, 6))
plt.plot(df['image_id'], df['vib_num_components'], 'b-', label='VIB Mode', alpha=0.7)
plt.plot(df['image_id'], df['novib_num_components'], 'r-', label='No VIB Mode', alpha=0.7)
plt.xlabel('Image ID')
plt.ylabel('Number of Connected Components')
plt.title('Connected Components Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
