import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def create_color_map(n_colors, colormap='gist_rainbow'):
    cmap = plt.get_cmap(colormap)
    return [cmap(i / n_colors)[:3] for i in range(n_colors)]


def visualize_3d_tracks(tracks_3d, point_size=3.0, line_width=2.0):
    """
    Visualize 3D tracks using Open3D.

    Args:
        tracks_3d (np.ndarray): Array of shape (T, N, 3) representing tracks.
        point_size (float): Size of the points.
        line_width (float): Width of the lines (Open3D doesn't support this directly in all renderers).
    """
    T, N, _ = tracks_3d.shape
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Tracks", width=960, height=720)

    color_map = create_color_map(N)

    geometries = []

    for i in range(N):  # for each track
        # Collect all 3D points over time for this track
        points = tracks_3d[:, i, :]
        if np.any(np.isnan(points)):
            continue

        # Create line set
        lines = [[j, j + 1] for j in range(len(points) - 1)]
        colors = [color_map[i] for _ in lines]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(line_set)

        # Optionally show points
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.paint_uniform_color(color_map[i])
        geometries.append(point_cloud)

    for g in geometries:
        vis.add_geometry(g)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # Example: generate synthetic spiral tracks
    # Load the npy file
    data = np.load("./thirdparty/ETAP/output/checkerboard/predictions.npy")  # shape (1, 97, 300, 2)
    tracks_2d = data[0]  # (T, N, 2)
    T, N, _ = tracks_2d.shape
    z = np.arange(T).reshape(-1, 1, 1)  # shape (T, 1, 1)

    # Stack to get shape (T, N, 3): [x, y, t]
    tracks_3d = np.concatenate([tracks_2d, z.repeat(N, axis=1)], axis=-1)

    visualize_3d_tracks(tracks_3d)