import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_all_densities(folder):
    """Load all .pkl density files from a folder."""
    densities = []
    for file in os.listdir(folder):
        if file.endswith(".pkl"):
            path = os.path.join(folder, file)
            with open(path, "rb") as f:
                d = pickle.load(f)
                densities.append(d)
    if not densities:
        raise FileNotFoundError(f"No .pkl files found in {folder}")
    return np.concatenate(densities)


def plot_histogram(densities, bins=50, save_path=None):
    """Plot normalized histogram of densities."""
    plt.figure(figsize=(8, 6))
    plt.hist(densities, bins=bins, color="skyblue", edgecolor="black",
             alpha=0.7, density=True)  # normalized
    plt.xlabel("Density Value")
    plt.ylabel("Probability Density")
    plt.title("Normalized Histogram of KDE Densities")
    plt.grid(True, linestyle="--", alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot histogram of densities from KDE .pkl files")
    parser.add_argument("folder", help="Folder containing .pkl density files")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for histogram")
    parser.add_argument("--output", help="Path to save histogram image (png/jpg/pdf)")
    args = parser.parse_args()

    print(f"Loading densities from {args.folder}...")
    all_densities = load_all_densities(args.folder)

    min_density = np.min(all_densities)
    max_density = np.max(all_densities)
    if max_density > min_density:
        normalized_densities = (all_densities - min_density) / (
                max_density - min_density
        )

    print(f"Loaded {normalized_densities.shape[0]} density values from {args.folder}")

    # Compute statistics
    mean_val = np.mean(normalized_densities)
    var_val = np.var(normalized_densities)
    print(f"Mean of distribution: {mean_val:.6f}")
    print(f"Variance of distribution: {var_val:.6f}")

    print("Plotting histogram...")
    plot_histogram(normalized_densities, bins=args.bins, save_path=args.output)


if __name__ == "__main__":
    main()
