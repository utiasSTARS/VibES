#!/usr/bin/env python3
"""
Camera Intrinsics Calibration Script

This script performs camera calibration using a chessboard pattern to determine
intrinsic camera parameters including focal length, principal point, and distortion coefficients.

Requirements:
    - OpenCV (cv2)
    - NumPy
    - Images of a chessboard pattern taken from different angles and positions

Usage:
    python camera_calibration.py --input_path /path/to/images --pattern_size 9x6
"""

import cv2
import numpy as np
import os
import glob
import argparse
import json
from pathlib import Path


class CameraCalibrator:
    def __init__(self, pattern_size=(9, 6), square_size=1.0):
        """
        Initialize the camera calibrator.

        Args:
            pattern_size (tuple): Number of inner corners per chessboard row and column (width, height)
            square_size (float): Size of a chessboard square in your chosen units (e.g., mm, cm)
        """
        self.pattern_size = pattern_size
        self.square_size = square_size

        # Termination criteria for corner sub-pixel accuracy
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        # Arrays to store object points and image points from all images
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane

        self.image_size = None
        self.valid_images = []

    def find_chessboard_corners(self, image_path, visualize=False):
        """
        Find chessboard corners in a single image.

        Args:
            image_path (str): Path to the image file
            visualize (bool): Whether to display the detected corners

        Returns:
            bool: True if corners were found successfully
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Store image size (should be consistent across all images)
        if self.image_size is None:
            self.image_size = gray.shape[::-1]  # (width, height)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

        if ret:
            # Refine corner positions to sub-pixel accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            self.objpoints.append(self.objp)
            self.imgpoints.append(corners2)
            self.valid_images.append(image_path)

            if visualize:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, self.pattern_size, corners2, ret)
                cv2.imshow('Chessboard Corners', img)
                cv2.waitKey(500)

            print(f"✓ Corners found in: {os.path.basename(image_path)}")
            return True
        else:
            print(f"✗ No corners found in: {os.path.basename(image_path)}")
            return False

    def calibrate_camera(self):
        """
        Perform camera calibration using the collected corner points.

        Returns:
            tuple: (camera_matrix, distortion_coefficients, reprojection_error)
        """
        if len(self.objpoints) < 10:
            print(f"Warning: Only {len(self.objpoints)} valid images found. Recommend at least 10 for good calibration.")

        print(f"Calibrating camera with {len(self.objpoints)} images...")

        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None
        )

        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(self.objpoints)

        return mtx, dist, mean_error, rvecs, tvecs

    def save_calibration_results(self, camera_matrix, dist_coeffs, reprojection_error,
                                 output_path="camera_calibration.json"):
        """
        Save calibration results to a JSON file.

        Args:
            camera_matrix (np.ndarray): 3x3 camera matrix
            dist_coeffs (np.ndarray): Distortion coefficients
            reprojection_error (float): Mean reprojection error
            output_path (str): Path to save the calibration file
        """
        calibration_data = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist(),
            "reprojection_error": float(reprojection_error),
            "image_size": list(self.image_size),
            "pattern_size": list(self.pattern_size),
            "square_size": self.square_size,
            "num_images_used": len(self.valid_images),
            "valid_images": [os.path.basename(path) for path in self.valid_images]
        }

        with open(output_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"Calibration results saved to: {output_path}")

    def print_calibration_summary(self, camera_matrix, dist_coeffs, reprojection_error):
        """Print a summary of the calibration results."""
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        print("\n" + "="*50)
        print("CAMERA CALIBRATION RESULTS")
        print("="*50)
        print(f"Images used for calibration: {len(self.valid_images)}")
        print(f"Image size: {self.image_size[0]} x {self.image_size[1]}")
        print(f"Reprojection error: {reprojection_error:.4f} pixels")
        print(f"\nCamera Matrix:")
        print(f"  fx = {fx:.2f}    fy = {fy:.2f}")
        print(f"  cx = {cx:.2f}    cy = {cy:.2f}")
        print(f"\nDistortion Coefficients:")
        print(f"  k1 = {dist_coeffs[0][0]:.6f}")
        print(f"  k2 = {dist_coeffs[0][1]:.6f}")
        print(f"  p1 = {dist_coeffs[0][2]:.6f}")
        print(f"  p2 = {dist_coeffs[0][3]:.6f}")
        print(f"  k3 = {dist_coeffs[0][4]:.6f}")
        print("="*50)

        # Quality assessment
        if reprojection_error < 0.5:
            print("✓ Excellent calibration quality!")
        elif reprojection_error < 1.0:
            print("✓ Good calibration quality.")
        else:
            print("⚠ Fair calibration quality. Consider taking more images or improving lighting.")


def main():
    parser = argparse.ArgumentParser(description="Perform camera intrinsics calibration using chessboard pattern")
    parser.add_argument("--input_path", "-i", required=True,
                        help="Path to directory containing calibration images")
    parser.add_argument("--pattern_size", "-p", default="9x6",
                        help="Chessboard pattern size as WIDTHxHEIGHT (default: 9x6)")
    parser.add_argument("--square_size", "-s", type=float, default=1.0,
                        help="Size of chessboard squares in your units (default: 1.0)")
    parser.add_argument("--output", "-o", default="camera_calibration.json",
                        help="Output file for calibration results (default: camera_calibration.json)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Visualize detected corners (will pause for each image)")
    parser.add_argument("--extensions", default="jpg,jpeg,png,bmp,tiff",
                        help="Comma-separated list of image extensions (default: jpg,jpeg,png,bmp,tiff)")

    args = parser.parse_args()

    # Parse pattern size
    try:
        width, height = map(int, args.pattern_size.split('x'))
        pattern_size = (width, height)
    except ValueError:
        print("Error: Pattern size must be in format WIDTHxHEIGHT (e.g., 9x6)")
        return

    # Get image files
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist")
        return

    extensions = [ext.strip().lower() for ext in args.extensions.split(',')]
    image_files = []
    for ext in extensions:
        pattern = f"*.{ext}"
        image_files.extend(glob.glob(str(input_path / pattern), recursive=False))

    if not image_files:
        print(f"Error: No image files found in '{input_path}' with extensions: {extensions}")
        return

    print(f"Found {len(image_files)} image files")
    print(f"Chessboard pattern: {pattern_size[0]}x{pattern_size[1]} inner corners")
    print(f"Square size: {args.square_size} units")

    # Initialize calibrator
    calibrator = CameraCalibrator(pattern_size, args.square_size)

    # Process images
    print("\nProcessing images...")
    successful_detections = 0
    for image_file in sorted(image_files):
        if calibrator.find_chessboard_corners(image_file, args.visualize):
            successful_detections += 1

    if args.visualize:
        cv2.destroyAllWindows()

    if successful_detections == 0:
        print("Error: No chessboard corners were detected in any image!")
        print("Make sure:")
        print("- Images contain a chessboard pattern")
        print("- Pattern size matches your chessboard")
        print("- Images are clear and well-lit")
        return

    # Perform calibration
    camera_matrix, dist_coeffs, reprojection_error, rvecs, tvecs = calibrator.calibrate_camera()

    # Save and display results
    calibrator.save_calibration_results(camera_matrix, dist_coeffs, reprojection_error, args.output)
    calibrator.print_calibration_summary(camera_matrix, dist_coeffs, reprojection_error)


if __name__ == "__main__":
    main()