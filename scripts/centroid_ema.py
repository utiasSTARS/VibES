import numpy as np
from metavision_sdk_stream import Camera, CameraStreamSlicer
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CentroidEMA:
    """
    Calculates the exponential moving average (EMA) of centroids for event data.
    This class maintains separate centroids for different polarities.
    """

    def __init__(self, tau, initial_event=None):
        """
        Initializes the CentroidEMA calculator.

        Args:
            tau (float): The time constant for the EMA. A smaller tau gives more weight to recent events.
            initial_event (tuple, optional): The first event (x, y, p, t) to initialize a centroid. Defaults to None.
        """
        self.tau = tau
        self.centroids = {}  # Dictionary to store centroids for each polarity
        self.last_timestamps = {}  # Dictionary to store the last timestamp for each polarity

        if initial_event is not None:
            self.initialize(initial_event)

    def initialize(self, event):
        """
        Initializes a new centroid for a given polarity using an event.

        Args:
            event (tuple): The event (x, y, p, t) to initialize with.
        """
        x, y, p, t = event
        self.centroids[p] = np.array([x, y], dtype=np.float32)
        self.last_timestamps[p] = t

    def update(self, event):
        """
        Updates a centroid with a new event.

        Args:
            event (tuple): The new event (x, y, p, t).

        Returns:
            np.ndarray: The updated centroid for the event's polarity.
        """
        x, y, p, t = event

        if p not in self.centroids:
            self.initialize(event)
            return self.centroids[p]

        # Time-dependent alpha calculation
        delta_t = t - self.last_timestamps[p]
        if self.tau > 0 and delta_t > 0:
            alpha = 1 - np.exp(-delta_t / self.tau)
        else:
            alpha = 1.0  # If time does not advance, new event takes over

        current_centroid = self.centroids[p]
        new_position = np.array([x, y], dtype=np.float32)

        updated_centroid = (1 - alpha) * current_centroid + alpha * new_position

        self.centroids[p] = updated_centroid
        self.last_timestamps[p] = t

        return updated_centroid

    def get_centroid(self, polarity):
        """
        Gets the current centroid for a given polarity.

        Args:
            polarity: The polarity of the desired centroid.

        Returns:
            np.ndarray or None: The centroid coordinates, or None if it doesn't exist.
        """
        return self.centroids.get(polarity, None)


def compute_ema_centroids(events, tau):
    """
    Computes the Exponential Moving Average of centroids for a sequence of events.

    Args:
        events (list or np.ndarray): A sequence of events.
                                     Each event is a tuple or array (x, y, p, t).
                                     Events are assumed to be sorted by timestamp t.
        tau (float): The time constant for the EMA.

    Returns:
        tuple: A tuple containing:
            - A list of updated centroid dictionaries for each event.
            - The final CentroidEMA object.
    """
    if not events:
        return [], None

    if not isinstance(events, np.ndarray):
        events = np.array(events)

    # Sort events by timestamp just in case they are not
    if events.shape[0] > 1:
        events = events[events[:, 3].argsort()]

    initial_event = events[0]
    ema_calculator = CentroidEMA(tau, initial_event=initial_event)

    centroid_history = []

    # The first event was for initialization.
    p_initial = initial_event[2]
    centroid_history.append({p_initial: ema_calculator.get_centroid(p_initial).copy()})

    for event in events[1:]:
        ema_calculator.update(event)
        # Store a copy of the current state of centroids
        history_item = {p: c.copy() for p, c in ema_calculator.centroids.items()}
        centroid_history.append(history_item)

    return centroid_history, ema_calculator


def plot_events_and_centroids(events, centroid_history):
    """
    Plots events and EMA centroids in a 3D (x, y, t) space.

    Args:
        events (np.ndarray): The structured array of events.
        centroid_history (list): A list of (t, p, x, y) tuples for centroids.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Separate events by polarity for different colors
    events_p1 = events[events['p'] == 1]
    events_p0 = events[events['p'] == 0]

    # Plot events
    ax.scatter(events_p1['x'], events_p1['y'], events_p1['t'], label='Events (Polarity 1)', c='red', marker='.')
    ax.scatter(events_p0['x'], events_p0['y'], events_p0['t'], label='Events (Polarity 0)', c='blue', marker='.')

    # Plot centroid history
    if centroid_history:
        history_arr = np.array(centroid_history)
        for p_val in np.unique(history_arr[:, 1]):
            p_history = history_arr[history_arr[:, 1] == p_val]
            # Sort by time to ensure lines are drawn correctly
            p_history = p_history[p_history[:, 0].argsort()]
            t, _, x, y = p_history.T
            ax.plot(x, y, t, label=f'Centroid (Polarity {int(p_val)})', linewidth=3)

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Time (microseconds)')
    ax.set_title('3D Visualization of Events and EMA Centroids')
    ax.legend()
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision SDK Exponential Moving Average Centroid.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file',
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
    parser.add_argument(
        '--tau', type=float, default=100000,
        help="Time constant for EMA in microseconds.")
    parser.add_argument(
        '--plot', action='store_true',
        help="Generate a 3D plot of events and centroids at the end (for file processing).")
    args = parser.parse_args()
    return args


def main():
    """Main function to process events and compute EMA centroids."""
    args = parse_args()

    # Plotting is only done when processing a file and requested by the user
    do_plot = args.plot and args.input_event_file

    # Open camera or file
    if args.input_event_file:
        camera = Camera.from_file(args.input_event_file)
    else:
        camera = Camera.from_first_available()

    # We use a slicer to get events in batches
    slicer = CameraStreamSlicer(camera.move())

    # Initialize CentroidEMA
    ema_calculator = CentroidEMA(tau=args.tau)

    # Lists to store data for plotting
    all_events = [] if do_plot else None
    centroid_history = [] if do_plot else None

    print("Processing events...")
    print("Processing only 5 seconds of events...")
    start_time = slicer.get_last_timestamp_s()
    end_time = start_time + 5

    # Process events
    for event_slice in slicer:
        if event_slice.events.size == 0:
            continue

        if do_plot:
            all_events.append(event_slice.events)

        # Update centroids with new events
        for event in event_slice.events:
            updated_centroid = ema_calculator.update((event['x'], event['y'], event['p'], event['t']))
            if do_plot:
                centroid_history.append((event['t'], event['p'], updated_centroid[0], updated_centroid[1]))

        # Print the latest centroids periodically
        print(f"Timestamp: {slicer.get_last_timestamp_s():.2f}s")
        for p, centroid in ema_calculator.centroids.items():
            print(f"  Polarity {p}: ({centroid[0]:.2f}, {centroid[1]:.2f})")

        if event_slice.get_last_timestamp_s() > end_time:
            break

    # After processing, generate the plot if enabled
    if do_plot:
        if not all_events:
            print("No events were processed, nothing to plot.")
            return

        print("Generating 3D plot...")
        all_events_arr = np.concatenate(all_events)
        plot_events_and_centroids(all_events_arr, centroid_history)


if __name__ == '__main__':
    main()
