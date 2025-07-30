import numpy as np
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

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

    def __init__(self, event_loader: FastEventLoader):
        self.event_loader = event_loader
        self.cam_w = event_loader.get_geom_width()
        self.cam_h = event_loader.get_geom_height()

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


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Analyze HDF5 event camera data with KDE metrics.")
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file")
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

    args = parser.parse_args()

    try:
        # Load data
        print(f"Processing file: {args.file_path}")
        event_loader = FastEventLoader(args.file_path)

        print(f"Camera Geometry: {event_loader.get_geom_height()}x{event_loader.get_geom_width()}")

        # Initialize KDE metrics
        kde_metrics = KDEMetrics(event_loader=event_loader)

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

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())