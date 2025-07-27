import numpy as np
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm

from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import seaborn as sns

from loader import EventLoader


class KDEMetrics:
    def __init__(self, event_loader: EventLoader):
        self.event_loader = event_loader
        self.cam_w = event_loader.get_geom_width()
        self.cam_h = event_loader.get_geom_height()
        self.events = event_loader.load()

    def point_distribution(self):
        # Compute point distribution metric

        pts = self.events[:, 1:3].astype(float)  # Extract x, y coordinates
        pts[:, 0] /= self.cam_w  # Normalize x coordinates by camera width
        pts[:, 1] /= self.cam_h  # Normalize y coordinates by camera height
        print(pts)

        kde = KDEUnivariate(pts)
        kde.fit(bw=0.05, kernel="gaussian")  # Tune bandwidth

        # Use sklearn's KernelDensity for 2D density estimation
        # kde2d = KernelDensity(
        #     kernel="gaussian", bandwidth=0.05
        # )  # tune bandwidth as needed
        # kde2d.fit(pts)

        # gaussian_kde_2d = gaussian_kde(pts.T, bw_method=0.05)  # Use Gaussian KDE

        # # Evaluate KDE at each event location (returns log densities)
        # log_densities = kde2d.score_samples(pts)
        # densities = np.exp(log_densities)  # Convert to actual densities

        # print("Variance of KDE densities:", np.var(densities))

        # plt.figure()
        # plt.hist(densities, bins=80, density=True)
        # plt.xlabel("KDE density at event locations")
        # plt.ylabel("Probability density")
        # plt.title("Distribution of event densities (1-D)")
        # plt.tight_layout()
        # plt.show()

        # Use seaborn's kdeplot for 2D density estimation
        sns.kdeplot(
            data=pts,
            cmap="viridis",
            fill=True,
            thresh=0,
            levels=100,
        )
        plt.show()

        pass


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 event data.")
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file")
    parser.add_argument(
        "--time_window_us",
        "-t",
        type=float,
        default=-1.0,
        help="Time window in microseconds",
    )
    parser.add_argument(
        "--delta_t",
        type=int,
        default=10000,
        help="Delta time for reading events (default: 1000 us)",
    )
    args = parser.parse_args()

    el = EventLoader(args.file_path, delta_t=args.delta_t)
    width = el.get_geom_width()
    height = el.get_geom_height()

    print(f"Processing file: {args.file_path}")
    print(f"Camera Geometry: {height}x{width}")

    m = KDEMetrics(event_loader=el)

    m.point_distribution()
