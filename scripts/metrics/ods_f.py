#!/usr/bin/env python3
"""
Image Processing Pipeline - Python Version
Converted from C++ OpenCV code
Created by viciopoli on 08/08/25
Enhanced with NoVib plotting
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.IMG_NUM = 1500
        self.SEARCH_RADIUS = 2

        # Initialize result arrays for vibration images
        self.grey_num = [0] * self.IMG_NUM
        self.vib_match_num_in_gray = [0] * self.IMG_NUM
        self.vib_no_match_num = [0] * self.IMG_NUM
        self.vib_match_num = [0] * self.IMG_NUM
        self.vib_all_num = [0] * self.IMG_NUM

        # Initialize result arrays for no-vibration images
        self.novib_match_num_in_gray = [0] * self.IMG_NUM
        self.novib_no_match_num = [0] * self.IMG_NUM
        self.novib_match_num = [0] * self.IMG_NUM
        self.novib_all_num = [0] * self.IMG_NUM

        # Update paths to match MATLAB structure
        self.frame_gray_path = "/home/viciopoli/datasets/event_harmeda/harmeda_dataset/frames_amiev/"
        self.frame_novib_path = "/home/viciopoli/datasets/event_harmeda/harmeda_dataset/results/amiev/ev/img_bin/"
        self.frame_vib_path = "/home/viciopoli/datasets/event_harmeda/harmeda_dataset/results/amiev/harmeda/img_bin/"

        # Store the transformation matrix for registration
        self.transformation_matrix = np.eye(2, 3, dtype=np.float32)
        self.has_initial_transform = False

    def apply_threshold(self, image: np.ndarray, thresh: float) -> np.ndarray:
        """Apply binary thresholding"""
        _, binary = cv2.threshold(image, thresh * 255, 255, cv2.THRESH_BINARY)
        return binary

    def apply_otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply Otsu's thresholding"""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations as in the original code"""
        result = image.copy()

        # Close operation with square kernel (7x7)
        kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_square)

        # Dilate with vertical line (1x3) - 90 degrees
        kernel_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        result = cv2.dilate(result, kernel_line_v)

        # Dilate with horizontal line (3x1) - 0 degrees
        kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        result = cv2.dilate(result, kernel_line_h)

        return result

    def bwareaopen(self, binary: np.ndarray, min_area: int) -> np.ndarray:
        """Remove small connected components (equivalent to MATLAB's bwareaopen)"""
        result = np.zeros_like(binary)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(result, [contour], -1, 255, -1)

        return result

    def bwareaopen_large(self, binary: np.ndarray, max_area: int) -> np.ndarray:
        """Remove large connected components (keep only smaller ones)"""
        result = np.zeros_like(binary)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) <= max_area:
                cv2.drawContours(result, [contour], -1, 255, -1)

        return result

    def register_images(self, fixed: np.ndarray, moving: np.ndarray,
                        update_transform: bool = True) -> np.ndarray:
        """Register images using ECC algorithm"""
        try:
            # Convert images to appropriate format
            moving_f = moving.astype(np.float32)
            fixed_f = fixed.astype(np.float32)

            # Create proper 2x3 affine matrix
            warp_matrix = np.eye(2, 3, dtype=np.float32)

            # Set termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-2)

            try:
                # Perform registration
                _, warp_matrix = cv2.findTransformECC(fixed_f, moving_f, warp_matrix,
                                                      cv2.MOTION_AFFINE, criteria)
                logger.info("ECC registration successful")
            except cv2.error as e:
                logger.warning("ECC registration failed, using identity transform")
                # warp_matrix remains as identity

            # Apply transformation
            registered = cv2.warpAffine(moving, warp_matrix, (fixed.shape[1], fixed.shape[0]))

            return registered

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return moving.copy()

    def process_gray_image(self, index: int) -> np.ndarray:
        """Process the gray reference image"""
        # Build filename with proper formatting
        filename = f"image_{index}.jpg"
        gray_path = os.path.join(self.frame_gray_path, filename)

        logger.info(f"Loading gray image: {gray_path}")
        gray_img = cv2.imread(gray_path, cv2.IMREAD_COLOR)

        if gray_img is None:
            raise FileNotFoundError(f"Could not load gray image: {gray_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

        # Apply Otsu thresholding (equivalent to graythresh + im2bw)
        gray_thresh = self.apply_otsu_threshold(gray)

        # Invert (~gray_edge in MATLAB)
        gray_edge = cv2.bitwise_not(gray_thresh)

        # Apply bwareaopen_large (remove large areas > 100000)
        gray_edge = self.bwareaopen_large(gray_edge, 100000)

        # Apply bwareaopen (remove small areas < 100)
        gray_edge = self.bwareaopen(gray_edge, 100)

        # Apply edge detection
        canny_edges = cv2.Canny(gray_edge, 50, 150)

        # Convert back to binary format for consistency
        _, result = cv2.threshold(canny_edges, 127, 255, cv2.THRESH_BINARY)

        # Show the processed gray image
        cv2.imshow("Processed Gray Image", result)
        cv2.waitKey(1)

        return result

    def analyze_matches(self, image: np.ndarray, gray_img: np.ndarray, i: int, is_vib: bool):
        """Analyze matches between an image and gray reference"""
        rows, cols = gray_img.shape
        image_tmp = image.copy()

        if is_vib:
            match_count = 0
        else:
            match_count = 0

        for ii in range(rows):
            for jj in range(cols):
                if gray_img[ii, jj] == 255:
                    # Define search region
                    up = max(0, ii - self.SEARCH_RADIUS)
                    down = min(rows - 1, ii + self.SEARCH_RADIUS)
                    left = max(0, jj - self.SEARCH_RADIUS)
                    right = min(cols - 1, jj + self.SEARCH_RADIUS)

                    # Extract region
                    region = image[up:down+1, left:right+1]

                    # Check for matches
                    if cv2.countNonZero(region) > 0:
                        match_count += 1

                    # Clear the region in temporary image
                    image_tmp[up:down+1, left:right+1] = 0

        no_match_count = cv2.countNonZero(image_tmp)
        total_pixels = cv2.countNonZero(image)
        match_pixels = total_pixels - no_match_count

        if is_vib:
            self.vib_match_num_in_gray[i - 1] = match_count
            self.vib_no_match_num[i - 1] = no_match_count
            self.vib_match_num[i - 1] = match_pixels
            self.vib_all_num[i - 1] = total_pixels
        else:
            self.novib_match_num_in_gray[i - 1] = match_count
            self.novib_no_match_num[i - 1] = no_match_count
            self.novib_match_num[i - 1] = match_pixels
            self.novib_all_num[i - 1] = total_pixels

    def process_image_pair(self, i: int, gray_img: np.ndarray):
        """Process a pair of vibration and no-vibration images"""
        # Build filenames
        novib_path = os.path.join(self.frame_novib_path, f"{i}.png")
        vib_path = os.path.join(self.frame_vib_path, f"{i}.png")

        novib = cv2.imread(novib_path, cv2.IMREAD_COLOR)
        vib = cv2.imread(vib_path, cv2.IMREAD_COLOR)

        if novib is None or vib is None:
            logger.warning(f"Could not load images for index {i}")
            return

        # Convert to grayscale
        novib_gray = cv2.cvtColor(novib, cv2.COLOR_BGR2GRAY)
        vib_gray = cv2.cvtColor(vib, cv2.COLOR_BGR2GRAY)

        # Apply thresholding with fixed values from MATLAB
        vib_edge = self.apply_threshold(vib_gray, 0.22)
        novib_edge = self.apply_threshold(novib_gray, 0.26)

        # Morphological operations on novib_edge
        novib_edge = self.morphological_operations(novib_edge)

        # Median filtering
        vib_edge = cv2.medianBlur(vib_edge, 3)
        novib_edge = cv2.medianBlur(novib_edge, 3)

        # Remove small areas
        novib_edge = self.bwareaopen(novib_edge, 100)

        # Register both images to gray_img
        mv_vib_edge = self.register_images(vib_edge, gray_img)
        mv_novib_edge = self.register_images(novib_edge, gray_img)

        # Create visualizations
        imgcolor = np.zeros((vib_edge.shape[0], vib_edge.shape[1], 3), dtype=np.uint8)
        imgcolor[:, :, 0] = 0  # Blue channel
        imgcolor[:, :, 1] = novib_edge  # Green channel
        imgcolor[:, :, 2] = vib_edge    # Red channel

        imgcolor2 = np.zeros((vib_edge.shape[0], vib_edge.shape[1], 3), dtype=np.uint8)
        imgcolor2[:, :, 0] = 0  # Blue channel
        imgcolor2[:, :, 1] = gray_img    # Green channel
        imgcolor2[:, :, 2] = mv_vib_edge # Red channel

        # Convert single channel images to color for display
        novib_color = cv2.cvtColor(novib_edge, cv2.COLOR_GRAY2BGR)
        vib_color = cv2.cvtColor(vib_edge, cv2.COLOR_GRAY2BGR)

        # Display results in subplots
        display1 = np.hstack([novib_color, vib_color])
        display2 = np.hstack([imgcolor, imgcolor2])
        full_display = np.vstack([display1, display2])

        # Resize for display
        full_display = cv2.resize(full_display,
                                  (full_display.shape[1] // 2, full_display.shape[0] // 2))
        cv2.imshow("Processing Results", full_display)
        cv2.waitKey(1)

        # Count grey pixels (only once per iteration)
        self.grey_num[i - 1] = cv2.countNonZero(gray_img)

        # Analyze matches for both vibration and no-vibration images
        self.analyze_matches(mv_vib_edge, gray_img, i, is_vib=True)
        self.analyze_matches(mv_novib_edge, gray_img, i, is_vib=False)

        # Progress indicator
        if i % 100 == 0 or i <= 10:
            logger.info(f"Processed {i}/{self.IMG_NUM} images")
            logger.info(f"  Grey pixels: {self.grey_num[i - 1]}")
            logger.info(f"  Vib matches in gray: {self.vib_match_num_in_gray[i - 1]}, Total vib pixels: {self.vib_all_num[i - 1]}")
            logger.info(f"  NoVib matches in gray: {self.novib_match_num_in_gray[i - 1]}, Total novib pixels: {self.novib_all_num[i - 1]}")

    def plot_results(self):
        """Plot the match ratios for both vibration and no-vibration images"""
        # Calculate match ratios for vibration images
        vib_match_ratios = []
        novib_match_ratios = []

        for i in range(self.IMG_NUM):
            if self.grey_num[i] > 0:
                vib_match_ratios.append(self.vib_match_num_in_gray[i] / self.grey_num[i])
                novib_match_ratios.append(self.novib_match_num_in_gray[i] / self.grey_num[i])
            else:
                vib_match_ratios.append(0.0)
                novib_match_ratios.append(0.0)

        if not vib_match_ratios and not novib_match_ratios:
            logger.warning("No match ratios to plot")
            return

        # Calculate statistics
        vib_max_ratio = max(vib_match_ratios) if vib_match_ratios else 0
        vib_min_ratio = min(vib_match_ratios) if vib_match_ratios else 0
        novib_max_ratio = max(novib_match_ratios) if novib_match_ratios else 0
        novib_min_ratio = min(novib_match_ratios) if novib_match_ratios else 0

        logger.info(f"Vib match ratio range: {vib_min_ratio} to {vib_max_ratio}")
        logger.info(f"NoVib match ratio range: {novib_min_ratio} to {novib_max_ratio}")

        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Both ratios on the same graph
        axes[0, 0].plot(range(1, len(vib_match_ratios) + 1), vib_match_ratios, 'r-', linewidth=2, label='Vibration', alpha=0.8)
        axes[0, 0].plot(range(1, len(novib_match_ratios) + 1), novib_match_ratios, 'b-', linewidth=2, label='No Vibration', alpha=0.8)
        axes[0, 0].set_title('Match Ratios Comparison (match_num_in_gray / grey_num)')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Match Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Plot 2: Vibration ratios only
        axes[0, 1].plot(range(1, len(vib_match_ratios) + 1), vib_match_ratios, 'r-', linewidth=2)
        axes[0, 1].set_title('Vibration Match Ratios')
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('Match Ratio')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: No-vibration ratios only
        axes[1, 0].plot(range(1, len(novib_match_ratios) + 1), novib_match_ratios, 'b-', linewidth=2)
        axes[1, 0].set_title('No-Vibration Match Ratios')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Match Ratio')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Difference between vibration and no-vibration ratios
        difference_ratios = [vib - novib for vib, novib in zip(vib_match_ratios, novib_match_ratios)]
        axes[1, 1].plot(range(1, len(difference_ratios) + 1), difference_ratios, 'g-', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Difference (Vibration - No-Vibration)')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Ratio Difference')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Additional statistics plot
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Create histograms
        ax.hist(vib_match_ratios, bins=50, alpha=0.7, label='Vibration', color='red', density=True)
        ax.hist(novib_match_ratios, bins=50, alpha=0.7, label='No Vibration', color='blue', density=True)
        ax.set_title('Distribution of Match Ratios')
        ax.set_xlabel('Match Ratio')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run(self):
        """Run the complete image processing pipeline"""
        logger.info("Starting image processing...")

        try:
            # Process gray image (first loop in MATLAB)
            gray_img = self.process_gray_image(1)
            logger.info("Gray image processed successfully")

            # Process all image pairs (second loop in MATLAB)
            for i in range(400, self.IMG_NUM + 1):
                self.process_image_pair(i, gray_img)

            # Plot results
            self.plot_results()

            logger.info("Processing complete!")

            # Save results to file
            results_data = {
                'Index': range(400, self.IMG_NUM + 1),
                'GreyNum': self.grey_num,
                'VibMatchInGray': self.vib_match_num_in_gray,
                'VibNoMatch': self.vib_no_match_num,
                'VibMatch': self.vib_match_num,
                'VibAll': self.vib_all_num,
                'VibRatio': [self.vib_match_num_in_gray[i] / self.grey_num[i] if self.grey_num[i] > 0 else 0.0
                             for i in range(self.IMG_NUM)],
                'NoVibMatchInGray': self.novib_match_num_in_gray,
                'NoVibNoMatch': self.novib_no_match_num,
                'NoVibMatch': self.novib_match_num,
                'NoVibAll': self.novib_all_num,
                'NoVibRatio': [self.novib_match_num_in_gray[i] / self.grey_num[i] if self.grey_num[i] > 0 else 0.0
                               for i in range(self.IMG_NUM)],
                'RatioDifference': [(self.vib_match_num_in_gray[i] - self.novib_match_num_in_gray[i]) / self.grey_num[i] if self.grey_num[i] > 0 else 0.0
                                    for i in range(self.IMG_NUM)]
            }

            df = pd.DataFrame(results_data)
            df.to_csv('evaluation_results_enhanced.csv', index=False)

            logger.info("Results saved to evaluation_results_enhanced.csv")

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise


def main():
    """Main function"""
    try:
        processor = ImageProcessor()
        processor.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        return -1

    return 0


if __name__ == "__main__":
    exit(main())