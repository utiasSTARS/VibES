import numpy as np
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path

# Optional imports with fallbacks
try:
    from sklearn.neighbors import KernelDensity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, some KDE features will be limited")

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using alternative methods")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib for plotting")


class FastEventLoader:
    """
    Efficient loader for HDF5 event camera data.
    Loads entire dataset into memory for fast time-based queries.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.cam_w = None
        self.cam_h = None
        self._load_entire_file()

    def _load_entire_file(self):
        """Load the entire HDF5 file into a pandas DataFrame with standardized columns."""
        print(f"Loading HDF5 file: {self.file_path}")

        with h5py.File(self.file_path, 'r') as f:
            # Access the CD/events dataset
            if 'CD/events' not in f:
                raise ValueError("Dataset 'CD/events' not found in HDF5 file")

            events_dataset = f['CD/events']
            print(f"Found {len(events_dataset)} total events")
            print(f"Event dataset shape: {events_dataset.shape}")

            # Load all events
            all_events = events_dataset[:]
            df_dict = self._parse_events_structure(all_events)

            # Get camera geometry
            self._extract_camera_geometry(f, df_dict)

        # Create DataFrame with standardized column names
        self.df = self._create_standardized_dataframe(df_dict)

        # Sort by timestamp for efficient time-based queries
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

        self._print_summary()

    def _parse_events_structure(self, all_events):
        """Parse the structure of events data (structured vs regular array)."""
        if len(all_events.dtype.names) > 0:
            # Structured array with named fields
            print(f"Event fields: {all_events.dtype.names}")
            return {field: all_events[field] for field in all_events.dtype.names}
        else:
            # Regular array - assume columns are [t, x, y, p] or similar
            print("Events are in regular array format")
            if all_events.shape[1] >= 4:
                return {
                    't': all_events[:, 0],
                    'x': all_events[:, 1],
                    'y': all_events[:, 2],
                    'p': all_events[:, 3]
                }
            elif all_events.shape[1] == 3:
                return {
                    't': all_events[:, 0],
                    'x': all_events[:, 1],
                    'y': all_events[:, 2]
                }
            else:
                raise ValueError(f"Unexpected event array shape: {all_events.shape}")

    def _extract_camera_geometry(self, f, df_dict):
        """Extract camera width and height from HDF5 attributes or infer from data."""
        try:
            # Try different attribute locations
            if 'width' in f.attrs:
                self.cam_w = f.attrs['width']
            elif 'CD' in f and 'width' in f['CD'].attrs:
                self.cam_w = f['CD'].attrs['width']
            elif 'sensor_size' in f.attrs:
                self.cam_w = f.attrs['sensor_size'][0]
            else:
                self.cam_w = int(np.max(df_dict['x'])) + 1
                print(f"Inferred camera width from data: {self.cam_w}")

            if 'height' in f.attrs:
                self.cam_h = f.attrs['height']
            elif 'CD' in f and 'height' in f['CD'].attrs:
                self.cam_h = f['CD'].attrs['height']
            elif 'sensor_size' in f.attrs:
                self.cam_h = f.attrs['sensor_size'][1]
            else:
                self.cam_h = int(np.max(df_dict['y'])) + 1
                print(f"Inferred camera height from data: {self.cam_h}")

        except Exception as e:
            print(f"Warning: Could not determine camera geometry ({e})")
            self.cam_w, self.cam_h = 640, 480
            print(f"Using default camera geometry: {self.cam_h}x{self.cam_w}")

    def _create_standardized_dataframe(self, df_dict):
        """Create DataFrame with standardized column names."""
        field_mapping = {
            't': 'timestamp', 'time': 'timestamp', 'ts': 'timestamp',
            'x': 'x', 'y': 'y',
            'p': 'polarity', 'pol': 'polarity', 'polarity': 'polarity'
        }

        standardized_dict = {}
        for field, data in df_dict.items():
            standard_name = field_mapping.get(field, field)
            standardized_dict[standard_name] = data

        df = pd.DataFrame(standardized_dict)

        # Validate required columns
        if 'timestamp' not in df.columns:
            raise ValueError("No timestamp column found in event data")
        if 'x' not in df.columns or 'y' not in df.columns:
            raise ValueError("No x,y coordinate columns found in event data")

        return df

    def _print_summary(self):
        """Print summary information about the loaded data."""
        print(f"Loaded {len(self.df)} events")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Time range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"Camera geometry: {self.cam_h}x{self.cam_w}")
        print(f"Sample events:\n{self.df.head()}")

    def get_geom_width(self):
        return self.cam_w

    def get_geom_height(self):
        return self.cam_h

    def get_time_slice(self, start_time, end_time):
        """Get events within a specific time window."""
        mask = (self.df['timestamp'] >= start_time) & (self.df['timestamp'] < end_time)
        return self.df[mask]

    def get_time_windows(self, window_size_us, overlap_us=0, max_windows=100):
        """Generator that yields time windows of specified size."""
        start_time = self.df['timestamp'].min()
        end_time = self.df['timestamp'].max()

        # Start from 10 seconds into the data
        current_start = start_time + 10_000_000  # 10 seconds
        step_size = window_size_us - overlap_us

        print(f"Generating time windows from {current_start} to {end_time}")
        print(f"Window size: {window_size_us} us, overlap: {overlap_us} us")

        counter = 0
        while current_start < end_time and counter < max_windows:
            current_end = min(current_start + window_size_us, end_time)
            window_data = self.get_time_slice(current_start, current_end)

            if len(window_data) > 0:  # Only yield non-empty windows
                yield current_start, current_end, window_data
                counter += 1

            current_start += step_size

    def load_all(self):
        """Return all events as numpy array for backward compatibility."""
        cols = ['timestamp', 'x', 'y']
        if 'polarity' in self.df.columns:
            cols.append('polarity')
        return self.df[cols].values


class KDEMetrics:
    """
    Compute Kernel Density Estimation metrics for event camera data.
    """

    def __init__(self, event_loader: FastEventLoader, name="Dataset"):
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
            print("Warning: Using histogram-based density estimation")
            hist, x_edges, y_edges = np.histogram2d(pts[:, 0], pts[:, 1], bins=50)

            x_indices = np.clip(np.digitize(pts[:, 0], x_edges) - 1, 0, hist.shape[0] - 1)
            y_indices = np.clip(np.digitize(pts[:, 1], y_edges) - 1, 0, hist.shape[1] - 1)

            return hist[x_indices, y_indices]

    def point_distribution_single_window(self, events_data):
        """Compute point distribution metrics for a single time window."""
        if len(events_data) < 10:  # Skip windows with too few events
            return None

        # Extract and normalize coordinates
        pts = events_data[['x', 'y']].values.astype(float)
        pts[:, 0] /= self.cam_w  # Normalize x coordinates
        pts[:, 1] /= self.cam_h  # Normalize y coordinates

        densities = self._compute_kde_densities(pts)

        return {
            'variance': np.var(densities),
            'mean_density': np.mean(densities),
            'std_density': np.std(densities),
            'min_density': np.min(densities),
            'max_density': np.max(densities),
            'num_events': len(events_data),
            'densities': densities,
            'points': pts
        }

    def plot_kde_density_histogram(self, densities, title="KDE Density Distribution", bins=50):
        """Plot histogram of KDE densities."""
        plt.figure(figsize=(10, 6))

        plt.hist(densities, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('KDE Density Value')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f"""Statistics:
Mean: {np.mean(densities):.4f}
Std: {np.std(densities):.4f}
Min: {np.min(densities):.4f}
Max: {np.max(densities):.4f}
Count: {len(densities)}"""

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def point_distribution_time_windows(self, window_size_us, overlap_us=0):
        """Compute point distribution metrics for multiple time windows."""
        results = []
        all_densities = []

        print(f"Processing time windows of {window_size_us} us with {overlap_us} us overlap...")

        for start_time, end_time, window_data in tqdm(
                self.event_loader.get_time_windows(window_size_us, overlap_us),
                desc="Processing time windows"
        ):
            result = self.point_distribution_single_window(window_data)
            if result is not None:
                result['start_time'] = start_time
                result['end_time'] = end_time
                result['duration'] = end_time - start_time
                results.append(result)
                all_densities.extend(result['densities'])

        # Plot histogram of all densities
        if all_densities:
            self.plot_kde_density_histogram(
                all_densities,
                f"KDE Density Distribution (All Windows, {len(results)} windows)"
            )

        return results, all_densities

    def compute_kde_densities_for_comparison(self, window_size_us=50000, max_events=50000):
        """
        Compute KDE densities using time-windowed approach instead of random sampling.
        Returns normalized densities for comparison between datasets.

        Args:
            window_size_us: Size of time windows in microseconds
            max_events: Maximum number of events to collect from time windows
        """
        print(f"Computing KDE densities for {self.name} using time-windowed approach...")

        all_densities = []
        events_collected = 0

        counter = 0
        # Collect events from time windows until we reach max_events
        for start_time, end_time, window_data in self.event_loader.get_time_windows(window_size_us, overlap_us=0):
            # if events_collected >= max_events:
            #     break

            if counter>100:
                break

            if len(window_data) < 10:  # Skip windows with too few events
                continue

            # Extract and normalize coordinates
            pts = window_data[['x', 'y']].values.astype(float)
            pts[:, 0] /= self.cam_w
            pts[:, 1] /= self.cam_h

            # Limit events from this window if needed
            remaining_capacity = max_events - events_collected
            if len(pts) > remaining_capacity:
                pts = pts[:remaining_capacity]

            # Compute densities for this window
            window_densities = self._compute_kde_densities(pts)
            all_densities.extend(window_densities)
            events_collected += len(pts)

            counter+=1
            print(f"{counter} Processed window {start_time}-{end_time}, collected {events_collected}/{max_events} events")

        if not all_densities:
            print(f"Warning: No densities computed for {self.name}")
            return np.array([])

        all_densities = np.array(all_densities)

        # Normalize densities to [0, 1] range
        min_density = np.min(all_densities)
        max_density = np.max(all_densities)
        if max_density > min_density:
            normalized_densities = (all_densities - min_density) / (max_density - min_density)
        else:
            normalized_densities = np.zeros_like(all_densities)

        print(f"Computed {len(normalized_densities)} normalized KDE densities for {self.name}")
        print(f"Time-windowed approach: collected events from {events_collected} total events")
        return normalized_densities

    def point_distribution_full_data(self):
        """Compute and visualize point distribution for the entire dataset."""
        print("Computing density visualization for entire dataset...")

        df = self.event_loader.df
        pts = df[['x', 'y']].values.astype(float)
        pts[:, 0] /= self.cam_w
        pts[:, 1] /= self.cam_h

        print(f"Processing {len(pts)} events...")

        # Compute densities for histogram
        densities = self._compute_kde_densities(pts)

        # Create subplots: density map and histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Density map
        if HAS_SEABORN:
            sns.kdeplot(x=pts[:, 0], y=pts[:, 1], cmap="viridis",
                        fill=True, thresh=0, levels=100, ax=ax1)
        else:
            im = ax1.hist2d(pts[:, 0], pts[:, 1], bins=100, cmap='viridis', density=True)
            plt.colorbar(im[3], ax=ax1, label='Density')

        ax1.set_xlabel("Normalized X coordinate")
        ax1.set_ylabel("Normalized Y coordinate")
        ax1.set_title(f"Event Density Map ({len(pts)} events)")

        # Density histogram
        ax2.hist(densities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('KDE Density Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('KDE Density Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return densities

    def analyze_temporal_kde_variance(self, window_size_us, overlap_us=0):
        """Analyze how KDE variance and density distribution change over time."""
        results, all_densities = self.point_distribution_time_windows(window_size_us, overlap_us)

        if not results:
            print("No valid time windows found")
            return

        # Extract metrics
        times = [r['start_time'] for r in results]
        variances = [r['variance'] for r in results]
        mean_densities = [r['mean_density'] for r in results]
        num_events = [r['num_events'] for r in results]

        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # KDE Variance over time
        axes[0, 0].plot(times, variances, 'b-', alpha=0.7)
        axes[0, 0].set_xlabel('Time (us)')
        axes[0, 0].set_ylabel('KDE Variance')
        axes[0, 0].set_title(f'KDE Variance Over Time (Window: {window_size_us} us)')
        axes[0, 0].grid(True, alpha=0.3)

        # Event count over time
        axes[0, 1].plot(times, num_events, 'r-', alpha=0.7)
        axes[0, 1].set_xlabel('Time (us)')
        axes[0, 1].set_ylabel('Number of Events')
        axes[0, 1].set_title('Event Count per Time Window')
        axes[0, 1].grid(True, alpha=0.3)

        # Mean density over time
        axes[1, 0].plot(times, mean_densities, 'g-', alpha=0.7)
        axes[1, 0].set_xlabel('Time (us)')
        axes[1, 0].set_ylabel('Mean KDE Density')
        axes[1, 0].set_title('Mean KDE Density Over Time')
        axes[1, 0].grid(True, alpha=0.3)

        # Variance vs Event count scatter
        axes[1, 1].scatter(num_events, variances, alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Number of Events')
        axes[1, 1].set_ylabel('KDE Variance')
        axes[1, 1].set_title('KDE Variance vs Event Count')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print(f"\nSummary Statistics ({len(results)} windows):")
        print(f"Mean KDE variance: {np.mean(variances):.6f} ± {np.std(variances):.6f}")
        print(f"Mean density: {np.mean(mean_densities):.6f} ± {np.std(mean_densities):.6f}")
        print(f"Mean events per window: {np.mean(num_events):.1f} ± {np.std(num_events):.1f}")


def compare_kde_densities(kde_metrics_list, bins=50, window_size_us=50000, max_events=50000):
    """
    Compare KDE density distributions between multiple datasets using time-windowed approach.

    Args:
        kde_metrics_list: List of KDEMetrics objects to compare
        bins: Number of histogram bins
        window_size_us: Size of time windows in microseconds
        max_events: Maximum number of events to collect per dataset
    """
    plt.figure(figsize=(12, 8))

    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
    all_densities = []

    for i, kde_metrics in enumerate(kde_metrics_list):
        # Compute normalized densities using time-windowed approach
        normalized_densities = kde_metrics.compute_kde_densities_for_comparison(
            window_size_us=window_size_us,
            max_events=max_events
        )

        if len(normalized_densities) == 0:
            print(f"Warning: No densities computed for {kde_metrics.name}, skipping...")
            continue

        all_densities.append(normalized_densities)

        # Plot histogram
        color = colors[i % len(colors)]
        plt.hist(normalized_densities, bins=bins, alpha=0.6, label=kde_metrics.name,
                 color=color, edgecolor='black', density=True)

    plt.xlabel('Normalized KDE Density (0-1)')
    plt.ylabel('Density')
    plt.title(f'Comparison of Normalized KDE Density Distributions\n(Time-windowed approach: {window_size_us}μs windows)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = "Statistics:\n"
    for i, (kde_metrics, densities) in enumerate(zip(kde_metrics_list, all_densities)):
        stats_text += f"{kde_metrics.name}:\n"
        stats_text += f"  Mean: {np.mean(densities):.3f}\n"
        stats_text += f"  Std: {np.std(densities):.3f}\n"
        stats_text += f"  Events: {len(densities)}\n"

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

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

    subfolders = ['harmeda', 'ev']
    kde_metrics_list = []

    for subfolder in subfolders:
        subfolder_path = folder_path / subfolder
        events_file = subfolder_path / 'events.hdf5'

        if not events_file.exists():
            print(f"Warning: {events_file} not found, skipping {subfolder}")
            continue

        print(f"\nLoading {subfolder} dataset from {events_file}")
        try:
            event_loader = FastEventLoader(str(events_file))
            kde_metrics = KDEMetrics(event_loader, name=subfolder.upper())
            kde_metrics_list.append(kde_metrics)
        except Exception as e:
            print(f"Error loading {subfolder}: {e}")
            continue

    if not kde_metrics_list:
        raise ValueError("No valid datasets found in the specified folder")

    return kde_metrics_list


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Analyze HDF5 event camera data with KDE metrics.")

    # Make the input argument more flexible
    parser.add_argument("input_path", type=str,
                        help="Path to HDF5 file OR folder containing harmeda/ev subfolders")
    parser.add_argument(
        "--time_window_us", "-t", type=float, default=50000.0,
        help="Time window in microseconds for analysis (default: 50ms)"
    )
    parser.add_argument(
        "--overlap_us", type=float, default=0.0,
        help="Overlap between time windows in microseconds"
    )
    parser.add_argument(
        "--full_data", action="store_true",
        help="Analyze full dataset instead of time windows"
    )
    parser.add_argument(
        "--histogram_only", action="store_true",
        help="Only show KDE density histogram for time windows"
    )
    parser.add_argument(
        "--compare_folders", action="store_true",
        help="Compare KDE densities between harmeda and ev folders"
    )
    parser.add_argument(
        "--max_events", type=int, default=50000,
        help="Maximum number of events to collect from time windows for comparison (default: 50000)"
    )

    args = parser.parse_args()

    try:
        input_path = Path(args.input_path)

        # Check if input is a folder with harmeda/ev structure or a single file
        if input_path.is_dir():
            harmeda_file = input_path / 'harmeda' / 'events.hdf5'
            ev_file = input_path / 'ev' / 'events.hdf5'

            if harmeda_file.exists() or ev_file.exists():
                print(f"Found folder structure, processing both datasets...")
                kde_metrics_list = load_folder_datasets(args.input_path)

                if args.compare_folders or len(kde_metrics_list) > 1:
                    # Compare datasets using time-windowed approach
                    print(f"\nComparing {len(kde_metrics_list)} datasets using time-windowed approach...")
                    compare_kde_densities(
                        kde_metrics_list,
                        window_size_us=args.time_window_us,
                        max_events=args.max_events
                    )
                else:
                    # Process single dataset found
                    kde_metrics = kde_metrics_list[0]
                    print(f"Processing single dataset: {kde_metrics.name}")
                    process_single_dataset(kde_metrics, args)

                return 0
            else:
                print(f"Folder {input_path} does not contain harmeda/ev structure, treating as single file path")

        # Process as single file
        if not input_path.exists():
            print(f"Error: {input_path} does not exist")
            return 1

        print(f"Processing single file: {args.input_path}")
        event_loader = FastEventLoader(str(args.input_path))
        kde_metrics = KDEMetrics(event_loader, name="Dataset")

        process_single_dataset(kde_metrics, args)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def process_single_dataset(kde_metrics, args):
    """Process a single dataset based on command line arguments."""
    print(f"Camera Geometry: {kde_metrics.event_loader.get_geom_height()}x{kde_metrics.event_loader.get_geom_width()}")

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