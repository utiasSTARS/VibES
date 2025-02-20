import matplotlib.pyplot as plt
import numpy as np
import sys

# Ensure correct usage
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <filename>")
    sys.exit(1)

filename = sys.argv[1]  # Get filename from argument

try:
    # Read the data
    with open(filename, "r") as file:
        residuals = [float(line.strip()) for line in file]

    # Ensure the length is a multiple of 3
    if len(residuals) % 3 != 0:
        print("Warning: The residuals vector length is not a multiple of 3!")

    # Split residuals into x, y, and theta
    residuals_x = residuals[0::2]
    residuals_y = residuals[1::2]
    # residuals_theta = residuals[2::3]

    indices = np.arange(len(residuals_x))  # Common index

    # Plot
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(indices, residuals_x, marker="o", linestyle="-", color="r", label="Residual X")
    plt.axhline(0, color="gray", linestyle="--")
    plt.ylabel("X Residual")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(indices, residuals_y, marker="o", linestyle="-", color="g", label="Residual Y")
    plt.axhline(0, color="gray", linestyle="--")
    plt.ylabel("Y Residual")
    plt.legend()
    plt.grid()

    # plt.subplot(3, 1, 3)
    # plt.plot(indices, residuals_theta, marker="o", linestyle="-", color="b", label="Residual Theta")
    # plt.axhline(0, color="gray", linestyle="--")
    # plt.xlabel("Index")
    # plt.ylabel("Theta Residual")
    # plt.legend()
    # plt.grid()

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    sys.exit(1)
except ValueError:
    print(f"Error: The file '{filename}' contains non-numeric values.")
    sys.exit(1)
