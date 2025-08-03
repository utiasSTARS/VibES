#!/usr/bin/env python3
"""
Vibrating Triangle - Interactive Frequency Control
Python port of the C++ OpenCV application
Created by viciopoli on 01/08/25.
"""

import cv2
import numpy as np
import math
import time
import sys
import argparse


class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val):
        self.rect = (x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.dragging = False
        self.handle_radius = height // 2

        self.track_rect = (
            x + self.handle_radius,
            y + height // 4,
            width - 2 * self.handle_radius,
            height // 2
        )
        self.update_handle_pos()

    def update_handle_pos(self):
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        self.handle_x = self.track_rect[0] + int(ratio * self.track_rect[2])
        self.handle_y = self.rect[1] + self.rect[3] // 2

    def handle_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is within handle
            handle_rect = (
                self.handle_x - self.handle_radius,
                self.handle_y - self.handle_radius,
                self.handle_radius * 2,
                self.handle_radius * 2
            )
            if (handle_rect[0] <= x <= handle_rect[0] + handle_rect[2] and
                    handle_rect[1] <= y <= handle_rect[1] + handle_rect[3]):
                self.dragging = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # Clamp to track bounds
            clamped_x = max(self.track_rect[0],
                            min(self.track_rect[0] + self.track_rect[2], x))

            # Calculate new value
            ratio = (clamped_x - self.track_rect[0]) / self.track_rect[2]
            self.val = self.min_val + ratio * (self.max_val - self.min_val)
            self.update_handle_pos()

    def draw(self, img):
        # Colors (BGR format in OpenCV)
        gray = (128, 128, 128)
        blue = (255, 150, 100)
        white = (255, 255, 255)

        # Draw track
        cv2.rectangle(img,
                      (self.track_rect[0], self.track_rect[1]),
                      (self.track_rect[0] + self.track_rect[2],
                       self.track_rect[1] + self.track_rect[3]),
                      gray, -1)

        # Draw handle
        cv2.circle(img, (self.handle_x, self.handle_y), self.handle_radius, blue, -1)
        cv2.circle(img, (self.handle_x, self.handle_y), self.handle_radius, white, 2)

    def get_value(self):
        return self.val

    def update_position(self, new_y):
        self.rect = (self.rect[0], new_y, self.rect[2], self.rect[3])
        self.track_rect = (
            self.track_rect[0],
            new_y + self.rect[3] // 4,
            self.track_rect[2],
            self.track_rect[3]
        )
        self.update_handle_pos()


# Global slider for mouse callback
g_frequency_slider = None


def on_mouse(event, x, y, flags, param):
    global g_frequency_slider
    if g_frequency_slider:
        g_frequency_slider.handle_mouse_event(event, x, y, flags, param)


def main():
    global g_frequency_slider

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vibrating Triangle - Interactive Frequency Control')
    parser.add_argument('amplitude', nargs='?', type=float, default=30.0,
                        help='vibration amplitude in pixels (positive number, default: 30)')

    args = parser.parse_args()
    amplitude = args.amplitude

    if amplitude <= 0:
        print("Error: Amplitude must be positive")
        return 1

    # Window settings
    WIDTH = 800
    HEIGHT = 600
    WINDOW_TITLE = "Vibrating Triangle - Interactive Frequency Control"

    # Colors (BGR format in OpenCV)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    DARK_GRAY = (50, 50, 50)

    # Triangle settings
    TRIANGLE_SIZE = 80

    # Create window
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, WIDTH, HEIGHT)

    # Create frequency slider
    frequency_slider = Slider(50, HEIGHT - 120, 300, 30, 0.1, 100.0, 2.5)
    g_frequency_slider = frequency_slider

    # Set mouse callback
    cv2.setMouseCallback(WINDOW_TITLE, on_mouse)

    # Animation variables
    start_time = time.time()

    # FPS calculation variables
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    fps_update_interval = 1.0  # Update FPS display every second

    print(f"Vibrating triangle started with amplitude: {amplitude}px")
    print("Use the frequency slider to adjust vibration speed")
    print("Press ESC or close window to exit")
    print("Window is resizable - drag the corners or edges")

    while True:
        # FPS calculation
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_update_interval:
            current_fps = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time

        # Create image
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img[:] = BLACK

        # Get current frequency from slider
        frequency = frequency_slider.get_value()

        # Calculate center position
        CENTER_X = WIDTH // 2
        CENTER_Y = HEIGHT // 2

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Calculate vibration offset using sine wave
        vibration_offset = amplitude * math.sin(2 * math.pi * frequency * elapsed_time)
        vibration_offset_Y = amplitude * math.sin(2 * math.pi * frequency * elapsed_time + math.pi / 2)

        # Calculate triangle position (vibrating both horizontally and vertically)
        triangle_x = CENTER_X + int(vibration_offset) - TRIANGLE_SIZE // 2
        triangle_y = CENTER_Y + int(vibration_offset_Y) - TRIANGLE_SIZE // 2

        # Draw the vibrating triangle
        triangle_points = np.array([
            [triangle_x + TRIANGLE_SIZE // 2, triangle_y],  # Top point
            [triangle_x, triangle_y + TRIANGLE_SIZE],       # Bottom left
            [triangle_x + TRIANGLE_SIZE, triangle_y + TRIANGLE_SIZE]  # Bottom right
        ], np.int32)

        cv2.fillPoly(img, [triangle_points], RED)

        # Draw control panel background
        cv2.rectangle(img, (0, HEIGHT - 150), (WIDTH, HEIGHT), DARK_GRAY, -1)
        cv2.line(img, (0, HEIGHT - 150), (WIDTH, HEIGHT - 150), WHITE, 2)

        # Draw frequency slider
        frequency_slider.draw(img)

        # Draw info text
        freq_text = f"Frequency: {frequency:.2f} Hz"
        amp_text = f"Amplitude: {int(amplitude)}px"
        fps_text = f"FPS: {current_fps:.1f}"

        cv2.putText(img, freq_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        cv2.putText(img, amp_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        cv2.putText(img, fps_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Green FPS text

        # Draw large frequency display in center-top
        large_freq_text = f"{frequency:.1f} Hz"

        # Add background rectangle for better visibility
        padding = 20

        # Draw slider label
        cv2.putText(img, "Frequency (Hz):", (50, HEIGHT - 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        # Draw frequency value near slider
        freq_value = f"{frequency:.2f}"
        cv2.putText(img, freq_value, (360, HEIGHT - 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        # Instructions
        cv2.putText(img, "Drag slider to adjust frequency • Press ESC to exit",
                    (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        # Display the image
        cv2.imshow(WINDOW_TITLE, img)

        # Handle key events (~75 FPS)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
            break

        # Check if window was closed
        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            break

    # Cleanup
    cv2.destroyAllWindows()
    print("Animation closed")

    return 0


if __name__ == "__main__":
    sys.exit(main())