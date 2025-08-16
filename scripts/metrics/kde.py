import numpy as np
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
import logging
import pickle
import os

from tqdm import tqdm
from pathlib import Path

from readers import *

logging.basicConfig(level=logging.INFO)

# Optional imports with fallbacks
try:
    from sklearn.neighbors import KernelDensity

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("sklearn not available, some KDE features will be limited")

try:
    from scipy.stats import gaussian_kde

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("Warning: scipy not available, using alternative methods")

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logging.warning("Warning: seaborn not available, using matplotlib for plotting")


class KDEMetrics:
    """
    Compute Kernel Density Estimation metrics for event camera data.
    """

    def __init__(self, event_loader: BaseEventReader, name="Dataset"):
        self.event_loader = event_loader
        self.cam_w = event_loader.get_geom_width()
        self.cam_h = event_loader.get_geom_height()
        self.name = name

    def _compute_kde_densities(self, pts):
        """Compute KDE densities for given points using available libraries."""
        if HAS_SKLEARN:
            kde2d = KernelDensity(kernel="gaussian", bandwidth=0.05)
            kde2d.fit(pts)
            log_densities = kde2d.score_samples(pts)
            return np.exp(log_densities)
        elif HAS_SCIPY:
            kde = gaussian_kde(pts.T, bw_method=0.05)
            return kde(pts.T)
        else:
            # Fallback to histogram-based density estimation
            logging.warning("Using histogram-based density estimation")
            hist, x_edges, y_edges = np.histogram2d(pts[:, 0], pts[:, 1], bins=50)

            x_indices = np.clip(
                np.digitize(pts[:, 0], x_edges) - 1, 0, hist.shape[0] - 1
            )
            y_indices = np.clip(
                np.digitize(pts[:, 1], y_edges) - 1, 0, hist.shape[1] - 1
            )

            return hist[x_indices, y_indices]

    def point_distribution_single_window(self, events_data):
        """Compute point distribution metrics for a single time window."""
        if len(events_data) < 10:  # Skip windows with too few events
            return None

        # Extract and normalize coordinates
        pts = events_data[["x", "y"]].values.astype(float)
        pts[:, 0] /= self.cam_w  # Normalize x coordinates
        pts[:, 1] /= self.cam_h  # Normalize y coordinates

        densities = self._compute_kde_densities(pts)

        return {
            "variance": np.var(densities),
            "mean_density": np.mean(densities),
            "std_density": np.std(densities),
            "min_density": np.min(densities),
            "max_density": np.max(densities),
            "num_events": len(events_data),
            "densities": densities,
            "points": pts,
        }

    def plot_kde_density_histogram(
        self, densities, title="KDE Density Distribution", bins=50
    ):
        """Plot histogram of KDE densities."""
        plt.figure(figsize=(10, 6))

        plt.hist(densities, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")
        plt.xlabel("KDE Density Value")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f"""Statistics:
            Mean: {np.mean(densities):.4f}
            Std: {np.std(densities):.4f}
            Min: {np.min(densities):.4f}
            Max: {np.max(densities):.4f}
            Count: {len(densities)}"""

        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

    def point_distribution_time_windows(self, window_size_us, overlap_us=0):
        """Compute point distribution metrics for multiple time windows."""
        results = []
        all_densities = []

        logging.info(
            f"Processing time windows of {window_size_us} us with {overlap_us} us overlap..."
        )

        for start_time, end_time, window_data in tqdm(
            self.event_loader.get_time_windows(window_size_us, overlap_us),
            desc="Processing time windows",
        ):
            result = self.point_distribution_single_window(window_data)
            if result is not None:
                result["start_time"] = start_time
                result["end_time"] = end_time
                result["duration"] = end_time - start_time
                results.append(result)
                all_densities.extend(result["densities"])

        # Plot histogram of all densities
        if all_densities:
            self.plot_kde_density_histogram(
                all_densities,
                f"KDE Density Distribution (All Windows, {len(results)} windows)",
            )

        return results, all_densities

    def compute_kde_densities_for_comparison(
        self, window_size_us=50000, max_events=50000, downsample_factor=1
    ):
        """
        Compute KDE densities using time-windowed approach instead of random sampling.
        Returns normalized densities for comparison between datasets.

        Args:
            window_size_us: Size of time windows in microseconds
            max_events: Maximum number of events to collect from time windows
        """
        logging.info(
            f"Computing KDE densities for {self.name} using time-windowed approach..."
        )

        all_densities = []
        events_collected = 0

        counter = 0
        # Collect events from time windows until we reach max_events
        for start_time, end_time, window_data in self.event_loader.get_time_windows(
            window_size_us, overlap_us=0
        ):
            # if events_collected >= max_events:
            #     break

            # if counter>100:
            #     break

            if len(window_data) < 10:  # Skip windows with too few events
                continue

            # Extract and normalize coordinates
            pts = window_data[["x", "y"]].values.astype(float)
            pts[:, 0] /= self.cam_w
            pts[:, 1] /= self.cam_h

            if downsample_factor > 1:
                # Downsample points if needed
                pre_downsample_shape = pts.shape
                pts = pts[::downsample_factor]
                logging.info(
                    f"Downsampled points from shape: {pre_downsample_shape} to {pts.shape}"
                )

            # Limit events from this window if needed
            remaining_capacity = max_events - events_collected
            if len(pts) > remaining_capacity:
                pts = pts[:remaining_capacity]

            # Compute densities for this window
            window_densities = self._compute_kde_densities(pts)
            all_densities.extend(window_densities)
            events_collected += len(pts)

            # counter+=1
            # print(f"{counter} Processed window {start_time}-{end_time}, collected {events_collected}/{max_events} events")

        if not all_densities:
            logging.warning(f"No densities computed for {self.name}")
            return np.array([])

        all_densities = np.array(all_densities)

        # Normalize densities to [0, 1] range
        min_density = np.min(all_densities)
        max_density = np.max(all_densities)
        if max_density > min_density:
            normalized_densities = (all_densities - min_density) / (
                max_density - min_density
            )
        else:
            normalized_densities = np.zeros_like(all_densities)

        logging.info(
            f"Computed {len(normalized_densities)} normalized KDE densities for {self.name}"
        )
        logging.info(
            f"Time-windowed approach: collected events from {events_collected} total events"
        )
        return normalized_densities

    def point_distribution_full_data(self):
        """Compute and visualize point distribution for the entire dataset."""
        logging.info("Computing density visualization for entire dataset...")

        df = self.event_loader.df
        pts = df[["x", "y"]].values.astype(float)
        pts[:, 0] /= self.cam_w
        pts[:, 1] /= self.cam_h

        logging.info(f"Processing {len(pts)} events...")

        # Compute densities for histogram
        densities = self._compute_kde_densities(pts)

        # Create subplots: density map and histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Density map
        if HAS_SEABORN:
            sns.kdeplot(
                x=pts[:, 0],
                y=pts[:, 1],
                cmap="viridis",
                fill=True,
                thresh=0,
                levels=100,
                ax=ax1,
            )
        else:
            im = ax1.hist2d(
                pts[:, 0], pts[:, 1], bins=100, cmap="viridis", density=True
            )
            plt.colorbar(im[3], ax=ax1, label="Density")

        ax1.set_xlabel("Normalized X coordinate")
        ax1.set_ylabel("Normalized Y coordinate")
        ax1.set_title(f"Event Density Map ({len(pts)} events)")

        # Density histogram
        ax2.hist(densities, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        ax2.set_xlabel("KDE Density Value")
        ax2.set_ylabel("Frequency")
        ax2.set_title("KDE Density Distribution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return densities

    def analyze_temporal_kde_variance(self, window_size_us, overlap_us=0):
        """Analyze how KDE variance and density distribution change over time."""
        results, all_densities = self.point_distribution_time_windows(
            window_size_us, overlap_us
        )

        if not results:
            logging.error("No valid time windows found")
            return

        # Extract metrics
        times = [r["start_time"] for r in results]
        variances = [r["variance"] for r in results]
        mean_densities = [r["mean_density"] for r in results]
        num_events = [r["num_events"] for r in results]

        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # KDE Variance over time
        axes[0, 0].plot(times, variances, "b-", alpha=0.7)
        axes[0, 0].set_xlabel("Time (us)")
        axes[0, 0].set_ylabel("KDE Variance")
        axes[0, 0].set_title(f"KDE Variance Over Time (Window: {window_size_us} us)")
        axes[0, 0].grid(True, alpha=0.3)

        # Event count over time
        axes[0, 1].plot(times, num_events, "r-", alpha=0.7)
        axes[0, 1].set_xlabel("Time (us)")
        axes[0, 1].set_ylabel("Number of Events")
        axes[0, 1].set_title("Event Count per Time Window")
        axes[0, 1].grid(True, alpha=0.3)

        # Mean density over time
        axes[1, 0].plot(times, mean_densities, "g-", alpha=0.7)
        axes[1, 0].set_xlabel("Time (us)")
        axes[1, 0].set_ylabel("Mean KDE Density")
        axes[1, 0].set_title("Mean KDE Density Over Time")
        axes[1, 0].grid(True, alpha=0.3)

        # Variance vs Event count scatter
        axes[1, 1].scatter(num_events, variances, alpha=0.6, color="purple")
        axes[1, 1].set_xlabel("Number of Events")
        axes[1, 1].set_ylabel("KDE Variance")
        axes[1, 1].set_title("KDE Variance vs Event Count")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        logging.info(f"\nSummary Statistics ({len(results)} windows):")
        logging.info(
            f"Mean KDE variance: {np.mean(variances):.6f} ± {np.std(variances):.6f}"
        )
        logging.info(
            f"Mean density: {np.mean(mean_densities):.6f} ± {np.std(mean_densities):.6f}"
        )
        logging.info(
            f"Mean events per window: {np.mean(num_events):.1f} ± {np.std(num_events):.1f}"
        )


def compare_kde_densities(
    kde_metrics_list,
    bins=50,
    window_size_us=50000,
    max_events=50000,
    downsample_factor=1,
    output_name="kde_comparison",
):
    """
    Compare KDE density distributions between multiple datasets using time-windowed approach.

    Args:
        kde_metrics_list: List of KDEMetrics objects to compare
        bins: Number of histogram bins
        window_size_us: Size of time windows in microseconds
        max_events: Maximum number of events to collect per dataset
    """
    plt.figure(figsize=(12, 8))

    colors = ["skyblue", "lightcoral", "lightgreen", "gold", "plum"]
    all_densities = []

    for i, kde_metrics in enumerate(kde_metrics_list):
        # Compute normalized densities using time-windowed approach
        normalized_densities = kde_metrics.compute_kde_densities_for_comparison(
            window_size_us=window_size_us,
            max_events=max_events,
            downsample_factor=downsample_factor,
        )

        if len(normalized_densities) == 0:
            logging.warning(
                f"No densities computed for {kde_metrics.name}, skipping..."
            )
            continue

        all_densities.append(normalized_densities)

        # Plot histogram
        color = colors[i % len(colors)]
        plt.hist(
            normalized_densities,
            bins=bins,
            alpha=0.6,
            label=kde_metrics.name,
            color=color,
            edgecolor="black",
            density=True,
        )

    plt.xlabel("Normalized KDE Density (0-1)")
    plt.ylabel("Density")
    plt.title(
        f"Comparison of Normalized KDE Density Distributions\n(Time-windowed approach: {window_size_us}μs windows)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = "Statistics:\n"
    for i, (kde_metrics, densities) in enumerate(zip(kde_metrics_list, all_densities)):
        stats_text += f"{kde_metrics.name}:\n"
        stats_text += f"  Mean: {np.mean(densities):.3f}\n"
        stats_text += f"  Std: {np.std(densities):.3f}\n"
        stats_text += f"  Events: {len(densities)}\n"

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(f"{output_name}.png")
    plt.close()
    # plt.show()

    return all_densities


def load_folder_datasets(folder_path):
    """
    Load datasets from harmeda and ev subfolders.

    Args:
        folder_path: Path to the main folder containing harmeda and ev subfolders

    Returns:
        List of KDEMetrics objects
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise ValueError(f"Folder {folder_path} does not exist")

    subfolders = ["harmeda", "ev"]
    kde_metrics_list = []

    for subfolder in subfolders:
        subfolder_path = folder_path / subfolder
        events_file = subfolder_path / "events.hdf5"

        if not events_file.exists():
            logging.warning(f"{events_file} not found, skipping {subfolder}")
            continue

        logging.info(f"\nLoading {subfolder} dataset from {events_file}")
        try:
            event_loader = choose_event_reader(str(events_file))
            kde_metrics = KDEMetrics(event_loader, name=subfolder.upper())
            kde_metrics_list.append(kde_metrics)
        except Exception as e:
            logging.error(f"Error loading {subfolder}: {e}")
            continue

    if not kde_metrics_list:
        raise ValueError("No valid datasets found in the specified folder")

    return kde_metrics_list


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze HDF5 event camera data with KDE metrics."
    )

    # Make the input argument more flexible
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to HDF5 file OR folder containing harmeda/ev subfolders",
    )
    parser.add_argument(
        "--time_window_us",
        "-t",
        type=float,
        default=50000.0,
        help="Time window in microseconds for analysis (default: 50ms)",
    )
    parser.add_argument(
        "--overlap_us",
        type=float,
        default=0.0,
        help="Overlap between time windows in microseconds",
    )
    parser.add_argument(
        "--full_data",
        action="store_true",
        help="Analyze full dataset instead of time windows",
    )
    parser.add_argument(
        "--histogram_only",
        action="store_true",
        help="Only show KDE density histogram for time windows",
    )
    parser.add_argument(
        "--compare_folders",
        action="store_true",
        help="Compare KDE densities between harmeda and ev folders",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=50000,
        help="Maximum number of events to collect from time windows for comparison (default: 50000)",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=1,
        help="Downsample factor for event data (default: 1, no downsampling)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="kde_comparison",
        help="Base name for output files (default: kde_comparison)",
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save KDE raw results to file"
    )

    args = parser.parse_args()

    try:
        input_path = Path(args.input_path)

        # Check if input is a folder with harmeda/ev structure or a single file
        if input_path.is_dir():
            harmeda_file = input_path / "harmeda" / "events.hdf5"
            ev_file = input_path / "ev" / "events.hdf5"

            if harmeda_file.exists() or ev_file.exists():
                logging.info(f"Found folder structure, processing both datasets...")
                kde_metrics_list = load_folder_datasets(args.input_path)

                if args.compare_folders or len(kde_metrics_list) > 1:
                    # Compare datasets using time-windowed approach
                    logging.info(
                        f"\nComparing {len(kde_metrics_list)} datasets using time-windowed approach..."
                    )
                    densities = compare_kde_densities(
                        kde_metrics_list,
                        window_size_us=args.time_window_us,
                        max_events=args.max_events,
                        downsample_factor=args.downsample_factor,
                        output_name=args.output_name,
                    )

                    if args.save_results:
                        with open(f"{args.output_name}_densities.pkl", "wb") as f:
                            pickle.dump(densities, f)
                else:
                    # Process single dataset found
                    kde_metrics = kde_metrics_list[0]
                    logging.info(f"Processing single dataset: {kde_metrics.name}")
                    process_single_dataset(kde_metrics, args)

                return 0
            else:
                logging.warning(
                    f"Folder {input_path} does not contain harmeda/ev structure, treating as single file path"
                )

        # Process as single file
        if not input_path.exists():
            logging.error(f"{input_path} does not exist")
            return 1

        logging.info(f"Processing single file: {args.input_path}")
        event_loader = choose_event_reader(str(args.input_path))
        kde_metrics = KDEMetrics(event_loader, name="Dataset")

        process_single_dataset(kde_metrics, args)

    except Exception as e:
        logging.error(f"Error: {e}")
        return 1

    return 0


def process_single_dataset(kde_metrics, args):
    """Process a single dataset based on command line arguments."""
    logging.info(
        f"Camera Geometry: {kde_metrics.event_loader.get_geom_height()}x{kde_metrics.event_loader.get_geom_width()}"
    )

    if args.full_data:
        # Analyze entire dataset
        kde_metrics.point_distribution_full_data()
    elif args.histogram_only:
        # Only compute and show histogram
        _, all_densities = kde_metrics.point_distribution_time_windows(
            args.time_window_us, args.overlap_us
        )
    else:
        # Full temporal analysis
        kde_metrics.analyze_temporal_kde_variance(args.time_window_us, args.overlap_us)


if __name__ == "__main__":
    exit(main())
