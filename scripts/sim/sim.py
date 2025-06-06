import cv2
import numpy as np
from tqdm import tqdm


class Sim:
    def circle(self):
        h, w = self.im_size
        y, x = np.ogrid[:h, :w]
        x_center = self.shift_x
        y_center = self.shift_y
        radius = 20.
        inner_circle = np.array((x - x_center) ** 2 + (y - y_center) ** 2).astype(np.int64)

        mask_in = inner_circle >= int(radius ** 2)
        mask_out = inner_circle <= int(radius ** 2 + 15.)
        mask = mask_in & mask_out

        y, x = np.where(mask)
        events = np.column_stack((x, y))
        return events

    def __init__(self):
        self.im_size = (320, 480)
        self.phase = np.pi / 2.
        self.omega_hz_us = 50. * 1e-6  # Hz
        self.omega_rad = 2. * np.pi * self.omega_hz_us  # Hz
        self.amp_x = 10.
        self.amp_y = 15.
        self.shift_x = 240
        self.shift_y = 160
        self.figure = self.circle

    def gen_events_stream(self, time_us) -> np.ndarray:
        events = []
        for i in tqdm(range(0, time_us)):
            x = self.amp_x * np.sin(self.omega_rad * i)
            y = self.amp_y * np.sin(self.omega_rad * i + self.phase)
            events_fig = self.figure()
            events_fig[:, 0] += int(x)
            events_fig[:, 1] += int(y)
            # add time and polarity to events
            events_fig = np.hstack((events_fig, np.ones((events_fig.shape[0], 1), np.int64)))
            events_fig = np.hstack((events_fig, i * np.ones((events_fig.shape[0], 1), np.int64)))

            events.append(events_fig)

        return np.vstack(events).astype(np.int64)


if __name__ == "__main__":
    cv2.namedWindow("events", cv2.WINDOW_NORMAL)

    sim = Sim()

    t_max = 1000
    events = sim.gen_events_stream(t_max)

    frame = np.zeros((320, 480, 1), np.uint8)
    cv2.imshow("events", frame)
    cv2.waitKey(1)

    t_0 = 0
    for ev in events:
        frame[ev[1], ev[0]] = 255 * (ev[3] / t_max)
        cv2.imshow("events", frame)
        cv2.waitKey(1)
        # if t_0 != ev[3]:
        #     cv2.imshow("events", frame)
        #     cv2.waitKey(1)
        #     t_0 = ev[3]
        #     # reset the image
        #     frame = np.zeros((320, 480, 1), np.uint8)
