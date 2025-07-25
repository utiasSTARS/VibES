import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(
    '/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/project/cmake-build-release/output/centroids.txt')

# Extract columns
time = data[:, 0]
x = data[:, 1]
y = data[:, 2]

# Generate a 10 Hz sine wave using the same time vector
frequency = 10.  # in Hz
sine_wave = 5. * np.sin(2. * np.pi * frequency * time)

# Plot all
plt.figure(figsize=(10, 5))
plt.plot(time, x, label='x')
plt.plot(time, y, label='y')
plt.plot(time, sine_wave, label='10 Hz sine wave', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.title('x, y, and 10 Hz Sine Wave over Time')
plt.legend()
plt.grid(True)
plt.show()
