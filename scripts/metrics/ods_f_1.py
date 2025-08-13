#!/usr/bin/env python3
"""
Image Processing Pipeline - Python Version with Feature Matching
Converted from C++ OpenCV code
Created by viciopoli on 08/08/25
Enhanced with feature matching registration and NoVib plotting
Fixed indexing issues for configurable start/end range
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Tuple, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, start_index: int = 700, end_index: int = 1500):
        self.START_INDEX = start_index
        self.END_INDEX = end_index
        self.IMG_COUNT = end_index - start_index + 1  # Total number of images to process
        self.SEARCH_RADIUS = 2

        # Initialize result arrays with correct size
        self.grey_num = [0] * self.IMG_COUNT
        self.vib_match_num_in_gray = [0] * self.IMG_COUNT
        self.vib_no_match_num = [0] * self.IMG_COUNT
        self.vib_match_num = [0] * self.IMG_COUNT
        self.vib_all_num = [0] * self.IMG_COUNT

        # Initialize result arrays for no-vibration images
        self.novib_match_num_in_gray = [0] * self.IMG_COUNT
        self.novib_no_match_num = [0] * self.IMG_COUNT
        self.novib_match_num = [0] * self.IMG_COUNT
        self.novib_all_num = [0] * self.IMG_COUNT

        # Update paths to match MATLAB structure
        self.frame_gray_path = "/home/viciopoli/datasets/event_harmeda/harmeda_dataset/"
        self.frame_novib_path = "/home/viciopoli/datasets/event_harmeda/harmeda_dataset/results/checkerpattern/ev/img_gray_10000/"
        self.frame_vib_path = "/home/viciopoli/datasets/event_harmeda/harmeda_dataset/results/checkerpattern/harmeda/img_gray_10000/"

        # Store separate transformation matrices for both registration types
        self.vib_transformation_matrix = np.eye(2, 3, dtype=np.float32)
        self.novib_transformation_matrix = np.eye(2, 3, dtype=np.float32)
        self.has_vib_transform = False
        self.has_novib_transform = False

        # Feature matching parameters
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_match_count = 10
        self.use_previous_transform = False

        # Store registration quality metrics
        self.vib_registration_quality = []
        self.novib_registration_quality = []

    def get_array_index(self, image_index: int) -> int:
        """Convert image index to array index"""
        return image_index - self.START_INDEX

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

    def estimate_transform_from_matches(self, kp1: List, kp2: List, matches: List) -> Tuple[Optional[np.ndarray], int]:
        """
        Estimate affine transformation from feature matches using RANSAC

        Args:
            kp1: Keypoints from reference image
            kp2: Keypoints from moving image
            matches: List of matches between keypoints

        Returns:
            Tuple of (transformation_matrix, num_inliers)
        """
        if len(matches) < self.min_match_count:
            logger.warning(f"Not enough matches found: {len(matches)} < {self.min_match_count}")
            return None, 0

        # Extract matched points
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Use RANSAC to find robust transformation
        try:
            # Estimate affine transformation
            transform_matrix, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.99,
                refineIters=10
            )

            if transform_matrix is not None and inliers is not None:
                num_inliers = np.sum(inliers)
                logger.info(f"Feature matching successful: {num_inliers}/{len(matches)} inliers")
                return transform_matrix, num_inliers
            else:
                logger.warning("Feature matching failed to find transformation")
                return None, 0

        except Exception as e:
            logger.error(f"Error in transform estimation: {e}")
            return None, 0

    def register_images_ecc(self, fixed: np.ndarray, moving: np.ndarray, warp_matrix: np.ndarray) -> [np.ndarray,
                                                                                                      np.ndarray]:
        """
        Register images using ECC algorithm with previous transformation as initial guess

        Args:
            fixed: Reference image (grayscale)
            moving: Image to be registered (grayscale)
            registration_type: Either 'vib' or 'novib' to track different transform matrices

        Returns:
            Registered image
        """
        try:
            # Convert images to appropriate format
            # Convert to float32
            fixed_f = fixed.astype(np.float32)
            moving_f = moving.astype(np.float32)

            # --- Improve ECC stability ---
            # 1. Normalize to [0,1]
            fixed_f = cv2.normalize(fixed_f, None, 0, 1, cv2.NORM_MINMAX)
            moving_f = cv2.normalize(moving_f, None, 0, 1, cv2.NORM_MINMAX)

            # Set termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        500, 1e-5)
            if warp_matrix == []:
                warp_matrix = np.eye(2, 3, dtype=np.float32)  # Initialize warp matrix
            else:
                warp_matrix = warp_matrix.astype(np.float32)  # Ensure warp_matrix is float32
            # Perform registration with previous transform as initial guess
            correlation_coeff, warp_matrix = cv2.findTransformECC(
                fixed_f, moving_f, warp_matrix, cv2.MOTION_AFFINE, criteria)

            # Apply transformation
            registered = cv2.warpAffine(moving, warp_matrix,
                                        (fixed.shape[1], fixed.shape[0]))

            return registered, warp_matrix
        except Exception as e:
            return moving.copy(), []

    def register_images_ecc_translation(self, fixed: np.ndarray, moving: np.ndarray, warp_matrix: np.ndarray) -> [
        np.ndarray, np.ndarray]:
        """
        Register images using ECC algorithm with previous transformation as initial guess

        Args:
            fixed: Reference image (grayscale)
            moving: Image to be registered (grayscale)
            registration_type: Either 'vib' or 'novib' to track different transform matrices

        Returns:
            Registered image
        """
        try:
            # Convert images to appropriate format
            # Convert to float32
            fixed_f = fixed.astype(np.float32)
            moving_f = moving.astype(np.float32)

            # --- Improve ECC stability ---
            # 1. Normalize to [0,1]
            fixed_f = cv2.normalize(fixed_f, None, 0, 1, cv2.NORM_MINMAX)
            moving_f = cv2.normalize(moving_f, None, 0, 1, cv2.NORM_MINMAX)

            # Set termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        500, 1e-5)
            if warp_matrix == []:
                warp_matrix = np.eye(2, 3, dtype=np.float32)  # Initialize warp matrix
            else:
                warp_matrix = warp_matrix.astype(np.float32)  # Ensure warp_matrix is float32
            # Perform registration with previous transform as initial guess
            correlation_coeff, warp_matrix = cv2.findTransformECC(
                fixed_f, moving_f, warp_matrix, cv2.MOTION_TRANSLATION, criteria)

            # Apply transformation
            registered = cv2.warpAffine(moving, warp_matrix,
                                        (fixed.shape[1], fixed.shape[0]))

            return registered, warp_matrix
        except Exception as e:
            print(e)
            return moving.copy(), []

    def register_images(self, fixed: np.ndarray, moving: np.ndarray,
                        registration_type: str = 'vib') -> [np.ndarray, np.ndarray]:
        """
        Register images using feature matching with ORB features

        Args:
            fixed: Reference image (grayscale)
            moving: Image to be registered (grayscale)
            registration_type: Either 'vib' or 'novib' to track different transform matrices

        Returns:
            Registered image
        """
        try:
            # Convert images to uint8 if needed
            if fixed.dtype != np.uint8:
                fixed = (fixed * 255).astype(np.uint8) if fixed.max() <= 1.0 else fixed.astype(np.uint8)
            if moving.dtype != np.uint8:
                moving = (moving * 255).astype(np.uint8) if moving.max() <= 1.0 else moving.astype(np.uint8)

            # Detect keypoints and descriptors
            kp1, des1 = self.feature_detector.detectAndCompute(fixed, None)
            kp2, des2 = self.feature_detector.detectAndCompute(moving, None)

            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                logger.warning(f"Insufficient features detected for {registration_type}")
                quality_score = 0.0
            else:
                # Match features
                matches = self.matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                # Filter good matches (distance threshold)
                good_matches = [m for m in matches if m.distance < 50]

                logger.info(
                    f"Found {len(good_matches)} good matches out of {len(matches)} total for {registration_type}")

                # Estimate transformation
                transform_matrix, num_inliers = self.estimate_transform_from_matches(kp1, kp2, good_matches)

                # Calculate quality score based on inliers ratio
                quality_score = num_inliers / len(good_matches) if len(good_matches) > 0 else 0.0

                if transform_matrix is not None and num_inliers >= self.min_match_count:
                    # Store the successful transformation matrix
                    if registration_type == 'vib':
                        self.vib_transformation_matrix = transform_matrix.copy()
                        self.has_vib_transform = True
                        self.vib_registration_quality.append(quality_score)
                    elif registration_type == 'novib':
                        self.novib_transformation_matrix = transform_matrix.copy()
                        self.has_novib_transform = True
                        self.novib_registration_quality.append(quality_score)

                    # Apply transformation
                    registered = cv2.warpAffine(moving, transform_matrix,
                                                (fixed.shape[1], fixed.shape[0]))

                    logger.info(f"Registration successful for {registration_type}, quality: {quality_score:.3f}")
                    return registered, transform_matrix
                else:
                    # Use previous transform if available
                    if registration_type == 'vib' and self.has_vib_transform and self.use_previous_transform:
                        warp_matrix = self.vib_transformation_matrix
                        logger.info(f"Using previous vibration transform (quality: {quality_score:.3f})")
                    elif registration_type == 'novib' and self.has_novib_transform and self.use_previous_transform:
                        warp_matrix = self.novib_transformation_matrix
                        logger.info(f"Using previous no-vibration transform (quality: {quality_score:.3f})")
                    else:
                        logger.warning(f"No reliable transform found for {registration_type}")
                        return moving.copy(), []

                    # Apply previous transformation
                    registered = cv2.warpAffine(moving, warp_matrix,
                                                (fixed.shape[1], fixed.shape[0]))
                    return registered, warp_matrix

            # Store quality even if registration failed
            if registration_type == 'vib':
                self.vib_registration_quality.append(quality_score)
            elif registration_type == 'novib':
                self.novib_registration_quality.append(quality_score)

            return moving.copy(), []

        except Exception as e:
            logger.error(f"Registration error for {registration_type}: {e}")
            return moving.copy(), []

    def visualize_feature_matches(self, fixed: np.ndarray, moving: np.ndarray,
                                  registration_type: str = 'vib') -> Optional[np.ndarray]:
        """
        Visualize feature matches between two images

        Args:
            fixed: Reference image
            moving: Image to be registered
            registration_type: Type of registration for logging

        Returns:
            Visualization image or None if no matches
        """
        try:
            # Ensure images are uint8
            if fixed.dtype != np.uint8:
                fixed = (fixed * 255).astype(np.uint8) if fixed.max() <= 1.0 else fixed.astype(np.uint8)
            if moving.dtype != np.uint8:
                moving = (moving * 255).astype(np.uint8) if moving.max() <= 1.0 else moving.astype(np.uint8)

            # Detect keypoints and descriptors
            kp1, des1 = self.feature_detector.detectAndCompute(fixed, None)
            kp2, des2 = self.feature_detector.detectAndCompute(moving, None)

            if des1 is None or des2 is None:
                return None

            # Match features
            matches = self.matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Filter good matches
            good_matches = [m for m in matches if m.distance < 50]

            if len(good_matches) > 0:
                # Draw matches
                match_img = cv2.drawMatches(fixed, kp1, moving, kp2,
                                            good_matches[:50], None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                return match_img

        except Exception as e:
            logger.error(f"Error visualizing matches for {registration_type}: {e}")

        return None

    def get_transform_info(self):
        """Get current transformation matrices information"""
        vib_info = {
            'has_transform': self.has_vib_transform,
            'matrix': self.vib_transformation_matrix.copy()
        }
        novib_info = {
            'has_transform': self.has_novib_transform,
            'matrix': self.novib_transformation_matrix.copy()
        }
        return {'vib': vib_info, 'novib': novib_info}

    def process_gray_image(self, index: int) -> np.ndarray:
        """Process the gray reference image"""
        # Build filename with proper formatting
        filename = "checkerpattern.png"
        gray_path = os.path.join(self.frame_gray_path, filename)

        logger.info(f"Loading gray image: {gray_path}")
        gray_img = cv2.imread(gray_path, cv2.IMREAD_COLOR)

        if gray_img is None:
            raise FileNotFoundError(f"Could not load gray image: {gray_path}")

        gray_img = cv2.resize(gray_img, (gray_img.shape[1] // 2, gray_img.shape[0] // 2))

        # Convert to grayscale
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

        # Apply Otsu thresholding (equivalent to graythresh + im2bw)
        gray_thresh = self.apply_otsu_threshold(gray)

        # Apply edge detection
        canny_edges = cv2.Canny(gray_thresh, 50, 150)

        # Convert back to binary format for consistency
        _, result = cv2.threshold(canny_edges, 127, 255, cv2.THRESH_BINARY)

        # Show the processed gray image
        cv2.imshow("Processed Gray Image", result)

        return result

    def analyze_matches(self, image: np.ndarray, gray_img: np.ndarray, image_index: int, is_vib: bool):
        """Analyze matches between an image and gray reference"""
        rows, cols = gray_img.shape
        image_tmp = image.copy()

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
                    region = image[up:down + 1, left:right + 1]

                    # Check for matches
                    if cv2.countNonZero(region) > 0:
                        match_count += 1

                    # Clear the region in temporary image
                    image_tmp[up:down + 1, left:right + 1] = 0

        no_match_count = cv2.countNonZero(image_tmp)
        total_pixels = cv2.countNonZero(image)
        match_pixels = total_pixels - no_match_count

        # Use array index instead of image index
        array_idx = self.get_array_index(image_index)

        if is_vib:
            self.vib_match_num_in_gray[array_idx] = match_count
            self.vib_no_match_num[array_idx] = no_match_count
            self.vib_match_num[array_idx] = match_pixels
            self.vib_all_num[array_idx] = total_pixels
        else:
            self.novib_match_num_in_gray[array_idx] = match_count
            self.novib_no_match_num[array_idx] = no_match_count
            self.novib_match_num[array_idx] = match_pixels
            self.novib_all_num[array_idx] = total_pixels

    def proc_img(self, img, C):
        """Process input image to extract features with Zhang-Suen thinning"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Apply adaptive threshold to handle varying lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, C)

        # Invert so lines are white on black background
        thresh = cv2.bitwise_not(thresh)

        # Apply Zhang-Suen thinning algorithm

        return thresh

    def zhang_suen_thinning(self, image):
        """
        Apply Zhang-Suen thinning algorithm to binary image

        Args:
            image: Binary image (0 and 255 values)

        Returns:
            Thinned binary image
        """
        # Convert to binary (0 and 1)
        binary = (image > 0).astype(np.uint8)

        # Make a copy for processing
        skeleton = binary.copy()

        # Continue until no more changes
        changed = True
        while changed:
            changed = False

            # Step 1
            to_remove = []
            for i in range(1, skeleton.shape[0] - 1):
                for j in range(1, skeleton.shape[1] - 1):
                    if skeleton[i, j] == 1:
                        # Get 8-neighborhood (clockwise from top)
                        p = [skeleton[i-1, j], skeleton[i-1, j+1], skeleton[i, j+1],
                             skeleton[i+1, j+1], skeleton[i+1, j], skeleton[i+1, j-1],
                             skeleton[i, j-1], skeleton[i-1, j-1]]

                        # Condition 1: 2 <= B(P1) <= 6
                        B_P1 = sum(p)
                        if not (2 <= B_P1 <= 6):
                            continue

                        # Condition 2: A(P1) = 1
                        A_P1 = 0
                        for k in range(8):
                            if p[k] == 0 and p[(k + 1) % 8] == 1:
                                A_P1 += 1
                        if A_P1 != 1:
                            continue

                        # Condition 3: P2 * P4 * P6 = 0
                        if p[0] * p[2] * p[4] != 0:
                            continue

                        # Condition 4: P4 * P6 * P8 = 0
                        if p[2] * p[4] * p[6] != 0:
                            continue

                        # Mark for removal
                        to_remove.append((i, j))

            # Remove marked pixels
            for (i, j) in to_remove:
                skeleton[i, j] = 0
                changed = True

            # Step 2
            to_remove = []
            for i in range(1, skeleton.shape[0] - 1):
                for j in range(1, skeleton.shape[1] - 1):
                    if skeleton[i, j] == 1:
                        # Get 8-neighborhood (clockwise from top)
                        p = [skeleton[i-1, j], skeleton[i-1, j+1], skeleton[i, j+1],
                             skeleton[i+1, j+1], skeleton[i+1, j], skeleton[i+1, j-1],
                             skeleton[i, j-1], skeleton[i-1, j-1]]

                        # Condition 1: 2 <= B(P1) <= 6
                        B_P1 = sum(p)
                        if not (2 <= B_P1 <= 6):
                            continue

                        # Condition 2: A(P1) = 1
                        A_P1 = 0
                        for k in range(8):
                            if p[k] == 0 and p[(k + 1) % 8] == 1:
                                A_P1 += 1
                        if A_P1 != 1:
                            continue

                        # Condition 3: P2 * P4 * P8 = 0
                        if p[0] * p[2] * p[6] != 0:
                            continue

                        # Condition 4: P2 * P6 * P8 = 0
                        if p[0] * p[4] * p[6] != 0:
                            continue

                        # Mark for removal
                        to_remove.append((i, j))

            # Remove marked pixels
            for (i, j) in to_remove:
                skeleton[i, j] = 0
                changed = True

        # Convert back to 0-255 format
        return (skeleton * 255).astype(np.uint8)

    def process_image_pair(self, image_index: int, gray_img: np.ndarray):
        """Process a pair of vibration and no-vibration images"""
        # Build filenames
        novib_path = os.path.join(self.frame_novib_path, f"{image_index}.png")
        vib_path = os.path.join(self.frame_vib_path, f"{image_index}.png")

        novib = cv2.imread(novib_path, cv2.IMREAD_COLOR)
        vib = cv2.imread(vib_path, cv2.IMREAD_COLOR)

        if novib is None or vib is None:
            logger.warning(f"Could not load images for index {image_index}")
            return

        # Half the size for faster processing
        novib = cv2.resize(novib, (novib.shape[1] // 2, novib.shape[0] // 2))
        vib = cv2.resize(vib, (vib.shape[1] // 2, vib.shape[0] // 2))

        # gaussian blur
        # novib = cv2.GaussianBlur(novib, (5, 5), 0)
        # novib = cv2.cvtColor(novib, cv2.COLOR_BGR2GRAY)
        # novib_edge = cv2.threshold(novib,250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # novib_edge = cv2.cvtColor(novib, cv2.COLOR_GRAY2BGR)

        novib_edge = self.proc_img(novib, C=2)
        vib_edge = self.proc_img(vib, C=2)

        # Convert to grayscale
        # vib_edge = cv2.cvtColor(vib, cv2.COLOR_BGR2GRAY)
        # novib_edge = cv2.cvtColor(novib, cv2.COLOR_BGR2GRAY)
        #
        # vib_edge = cv2.GaussianBlur(vib_edge, (11, 11), 0)
        # novib_edge = cv2.GaussianBlur(novib_edge, (7, 7), 0)
        #
        # # Apply thresholding with fixed values from MATLAB
        # vib_edge = self.apply_threshold(vib_edge, 0.5)
        # novib_edge = self.apply_threshold(novib_edge, 0.2)
        #
        # # Median filtering
        # vib_edge = cv2.medianBlur(vib_edge, 3)
        # novib_edge = cv2.medianBlur(novib_edge, 3)
        #
        # # Remove small areas
        # vib_edge = self.bwareaopen(vib_edge, 100)
        # novib_edge = self.bwareaopen(novib_edge, 100)

        # cv2.imshow("novib_edge", novib_edge)
        # cv2.imshow("vib_edge", vib_edge)
        # cv2.waitKey(0)

        # novib_edge = cv2.cvtColor(novib_edge, cv2.COLOR_GRAY2BGR)
        # vib_edge = cv2.cvtColor(vib_edge, cv2.COLOR_GRAY2BGR)

        novib_edge, _ = self.register_images_ecc_translation(vib_edge, novib_edge, [])

        novib_edge = self.zhang_suen_thinning(novib_edge)
        vib_edge = self.zhang_suen_thinning(vib_edge)

        # Register both images to gray_img using feature matching
        mv_vib_edge, T = self.register_images(gray_img, vib_edge, registration_type='vib')
        if T != []:
            mv_novib_edge = cv2.warpAffine(novib_edge, T,
                                           (novib_edge.shape[1], novib_edge.shape[0]))
        else:
            mv_novib_edge = novib_edge

        mv_novib_edge, _ = self.register_images_ecc_translation(gray_img, mv_novib_edge, [])
        # if T==[]:
        #     mv_vib_edge = self.register_images_ecc(gray_img, vib_edge)
        # if T!=[]:
        #     mv_novib_edge, T_1 = self.register_images_ecc(gray_img, novib_edge, T)
        #     if T_1==[]:
        #         mv_novib_edge, T = self.register_images(gray_img, novib_edge, registration_type='novib')
        # else:
        #     mv_novib_edge, T = self.register_images(gray_img, novib_edge, registration_type='novib')
        #     if T==[]:
        #         mv_novib_edge = self.register_images_ecc(gray_img, novib_edge, T)

        # Visualize feature matches every 100 images for debugging
        if image_index % 100 == 0:
            vib_match_vis = self.visualize_feature_matches(gray_img, vib_edge, 'vib')
            novib_match_vis = self.visualize_feature_matches(gray_img, novib_edge, 'novib')

            if vib_match_vis is not None:
                cv2.imshow(f"Vib Feature Matches - Frame {image_index}",
                           cv2.resize(vib_match_vis, (800, 300)))
            if novib_match_vis is not None:
                cv2.imshow(f"NoVib Feature Matches - Frame {image_index}",
                           cv2.resize(novib_match_vis, (800, 300)))

        # Create comprehensive visualizations
        imgcolor = np.zeros((vib_edge.shape[0], vib_edge.shape[1], 3), dtype=np.uint8)
        imgcolor[:, :, 0] = 0  # Blue channel
        imgcolor[:, :, 1] = novib_edge  # Green channel
        imgcolor[:, :, 2] = vib_edge  # Red channel

        imgcolor2 = np.zeros((vib_edge.shape[0], vib_edge.shape[1], 3), dtype=np.uint8)
        imgcolor2[:, :, 0] = 0  # Blue channel
        imgcolor2[:, :, 1] = gray_img  # Green channel
        imgcolor2[:, :, 2] = mv_vib_edge  # Red channel

        # Create visualization for NoVib vs Gray alignment
        imgcolor3 = np.zeros((vib_edge.shape[0], vib_edge.shape[1], 3), dtype=np.uint8)
        imgcolor3[:, :, 0] = 0  # Blue channel
        imgcolor3[:, :, 1] = gray_img  # Green channel (Gray reference)
        imgcolor3[:, :, 2] = mv_novib_edge  # Red channel (Registered NoVib)

        # Convert single channel images to color for display
        novib_color = cv2.cvtColor(novib_edge, cv2.COLOR_GRAY2BGR)
        vib_color = cv2.cvtColor(vib_edge, cv2.COLOR_GRAY2BGR)
        gray_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        # Create a comprehensive 2x3 display layout
        # Top row: Original images
        display1 = np.hstack([novib_color, vib_color, gray_color])

        # Bottom row: Overlays
        display2 = np.hstack([imgcolor,  # NoVib(G) + Vib(R)
                              imgcolor2,  # Gray(G) + RegVib(R)
                              imgcolor3])  # Gray(G) + RegNoVib(R)

        full_display = np.vstack([display1, display2])

        # Add text labels for clarity
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Top row labels
        cv2.putText(full_display, "NoVib Original", (10, 30), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(full_display, "Vib Original", (novib_color.shape[1] + 10, 30), font, font_scale, (255, 255, 255),
                    thickness)
        cv2.putText(full_display, "Gray Reference", (novib_color.shape[1] + vib_color.shape[1] + 10, 30), font,
                    font_scale, (255, 255, 255), thickness)

        # Bottom row labels
        y_offset = display1.shape[0] + 30
        cv2.putText(full_display, "NoVib(G)+Vib(R)", (10, y_offset), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(full_display, "Gray(G)+RegVib(R)", (novib_color.shape[1] + 10, y_offset), font, font_scale,
                    (255, 255, 255), thickness)
        cv2.putText(full_display, "Gray(G)+RegNoVib(R)", (novib_color.shape[1] + vib_color.shape[1] + 10, y_offset),
                    font, font_scale, (255, 255, 255), thickness)

        # Resize for display
        full_display = cv2.resize(full_display,
                                  (full_display.shape[1] // 2, full_display.shape[0] // 2))
        cv2.imshow("Processing Results", full_display)
        cv2.waitKey(1)

        # Count grey pixels (only once per iteration) - use array index
        array_idx = self.get_array_index(image_index)
        self.grey_num[array_idx] = cv2.countNonZero(gray_img)

        # Analyze matches for both vibration and no-vibration images
        self.analyze_matches(mv_vib_edge, gray_img, image_index, is_vib=True)
        self.analyze_matches(mv_novib_edge, gray_img, image_index, is_vib=False)

        # Progress indicator with transform info
        if image_index % 100 == 0 or image_index <= self.START_INDEX + 10:
            logger.info(f"Processed {image_index}/{self.END_INDEX} images")
            logger.info(f"  Grey pixels: {self.grey_num[array_idx]}")
            logger.info(
                f"  Vib matches in gray: {self.vib_match_num_in_gray[array_idx]}, Total vib pixels: {self.vib_all_num[array_idx]}")
            logger.info(
                f"  NoVib matches in gray: {self.novib_match_num_in_gray[array_idx]}, Total novib pixels: {self.novib_all_num[array_idx]}")

            # Log current transform status and quality
            transform_info = self.get_transform_info()
            logger.info(f"  Transform status - Vib: {transform_info['vib']['has_transform']}, "
                        f"NoVib: {transform_info['novib']['has_transform']}")

            # Log recent registration quality
            if len(self.vib_registration_quality) > 0:
                logger.info(f"  Recent Vib quality: {self.vib_registration_quality[-1]:.3f}")
            if len(self.novib_registration_quality) > 0:
                logger.info(f"  Recent NoVib quality: {self.novib_registration_quality[-1]:.3f}")

    def plot_results(self):
        """Plot the match ratios for both vibration and no-vibration images"""
        # Calculate match ratios for vibration images
        vib_match_ratios = []
        novib_match_ratios = []

        for i in range(self.IMG_COUNT):
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

        # Create x-axis with actual image indices
        x_indices = list(range(self.START_INDEX, self.END_INDEX + 1))

        # Plot 1: Both ratios on the same graph
        axes[0, 0].plot(x_indices, vib_match_ratios, 'r-', linewidth=2, label='Vibration', alpha=0.8)
        axes[0, 0].plot(x_indices, novib_match_ratios, 'b-', linewidth=2, label='No Vibration', alpha=0.8)
        axes[0, 0].set_title('Match Ratios Comparison (match_num_in_gray / grey_num)')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Match Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Plot 2: Vibration ratios only
        axes[0, 1].plot(x_indices, vib_match_ratios, 'r-', linewidth=2)
        axes[0, 1].set_title('Vibration Match Ratios')
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('Match Ratio')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: No-vibration ratios only
        axes[1, 0].plot(x_indices, novib_match_ratios, 'b-', linewidth=2)
        axes[1, 0].set_title('No-Vibration Match Ratios')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Match Ratio')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Difference between vibration and no-vibration ratios
        difference_ratios = [vib - novib for vib, novib in zip(vib_match_ratios, novib_match_ratios)]
        axes[1, 1].plot(x_indices, difference_ratios, 'g-', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Difference (Vibration - No-Vibration)')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Ratio Difference')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # NEW: Create alignment quality visualization
        self.plot_alignment_quality()

    def plot_alignment_quality(self):
        """Plot alignment quality metrics between NoVib/Vib and Gray images"""
        # Calculate correlation coefficients for alignment quality assessment
        novib_correlations = []
        vib_correlations = []

        # We'll need to recalculate these during processing or store them
        # For now, create a placeholder that shows the concept
        logger.info("Creating alignment quality visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Alignment quality over time (using actual registration quality if available)
        x_range = list(range(self.START_INDEX, self.END_INDEX + 1))

        # Use actual registration quality if available, otherwise simulate
        if len(self.vib_registration_quality) > 0 and len(self.novib_registration_quality) > 0:
            # Pad quality arrays to match image count if needed
            vib_quality = self.vib_registration_quality[:self.IMG_COUNT]
            novib_quality = self.novib_registration_quality[:self.IMG_COUNT]

            # Extend with last value if arrays are shorter
            while len(vib_quality) < self.IMG_COUNT:
                vib_quality.append(vib_quality[-1] if vib_quality else 0.0)
            while len(novib_quality) < self.IMG_COUNT:
                novib_quality.append(novib_quality[-1] if novib_quality else 0.0)

            axes[0, 0].plot(x_range, vib_quality, 'r-', linewidth=2, label='Vib-Gray Alignment', alpha=0.8)
            axes[0, 0].plot(x_range, novib_quality, 'b-', linewidth=2, label='NoVib-Gray Alignment', alpha=0.8)
            axes[0, 0].set_title('Registration Quality Over Time')
        else:
            # Simulate alignment quality data if no actual data available
            simulated_novib_quality = [0.8 + 0.1 * np.sin(i / 100) + np.random.normal(0, 0.05) for i in
                                       range(self.IMG_COUNT)]
            simulated_vib_quality = [0.75 + 0.1 * np.cos(i / 100) + np.random.normal(0, 0.05) for i in
                                     range(self.IMG_COUNT)]

            axes[0, 0].plot(x_range, simulated_vib_quality, 'r-', linewidth=2, label='Vib-Gray Alignment', alpha=0.8)
            axes[0, 0].plot(x_range, simulated_novib_quality, 'b-', linewidth=2, label='NoVib-Gray Alignment',
                            alpha=0.8)
            axes[0, 0].set_title('Alignment Quality Over Time (Simulated)')

        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Quality Score')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)

        # Plot 2: Transform magnitude over time
        # Show how much transformation is being applied
        transform_magnitudes_vib = []
        transform_magnitudes_novib = []

        # Calculate transform magnitudes (translation + rotation approximation)
        vib_tx = self.vib_transformation_matrix[0, 2] if self.has_vib_transform else 0
        vib_ty = self.vib_transformation_matrix[1, 2] if self.has_vib_transform else 0
        vib_magnitude = np.sqrt(vib_tx ** 2 + vib_ty ** 2)

        novib_tx = self.novib_transformation_matrix[0, 2] if self.has_novib_transform else 0
        novib_ty = self.novib_transformation_matrix[1, 2] if self.has_novib_transform else 0
        novib_magnitude = np.sqrt(novib_tx ** 2 + novib_ty ** 2)

        axes[0, 1].bar(['NoVib Transform', 'Vib Transform'], [novib_magnitude, vib_magnitude],
                       color=['blue', 'red'], alpha=0.7)
        axes[0, 1].set_title('Current Transform Magnitudes')
        axes[0, 1].set_ylabel('Translation Magnitude (pixels)')

        # Plot 3: Transform parameters
        if self.has_vib_transform and self.has_novib_transform:
            vib_params = self.vib_transformation_matrix.flatten()
            novib_params = self.novib_transformation_matrix.flatten()

            param_names = ['M00', 'M01', 'Tx', 'M10', 'M11', 'Ty']
            x_pos = np.arange(len(param_names))

            width = 0.35
            axes[1, 0].bar(x_pos - width / 2, vib_params, width, label='Vib Transform', color='red', alpha=0.7)
            axes[1, 0].bar(x_pos + width / 2, novib_params, width, label='NoVib Transform', color='blue', alpha=0.7)
            axes[1, 0].set_title('Transform Parameters Comparison')
            axes[1, 0].set_ylabel('Parameter Value')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(param_names)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No transforms available yet',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Transform Parameters (Not Available)')

        # Plot 4: Registration convergence info
        axes[1, 1].text(0.1, 0.9, f"Registration Status:", fontsize=12, weight='bold', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.8, f"Processing Range: {self.START_INDEX} to {self.END_INDEX}", fontsize=10,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"Total Images: {self.IMG_COUNT}", fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Vib Transform Available: {self.has_vib_transform}", fontsize=10,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"NoVib Transform Available: {self.has_novib_transform}", fontsize=10,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f"Max Iterations: {self.max_iterations}", fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.3, f"Termination EPS: {self.termination_eps}", fontsize=10,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f"Using Previous Transform: {self.use_previous_transform}", fontsize=10,
                        transform=axes[1, 1].transAxes)

        if self.has_vib_transform:
            axes[1, 1].text(0.1, 0.1, f"Vib Translation: ({vib_tx:.2f}, {vib_ty:.2f})", fontsize=10,
                            transform=axes[1, 1].transAxes)
        if self.has_novib_transform:
            axes[1, 1].text(0.1, 0.0, f"NoVib Translation: ({novib_tx:.2f}, {novib_ty:.2f})", fontsize=10,
                            transform=axes[1, 1].transAxes)

        axes[1, 1].set_title('Processing Information')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

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
        logger.info(f"Starting image processing for range {self.START_INDEX} to {self.END_INDEX}...")

        try:
            # Process gray image (first loop in MATLAB)
            gray_img = self.process_gray_image(self.START_INDEX)
            logger.info("Gray image processed successfully")

            # Process all image pairs (second loop in MATLAB)
            for i in range(self.START_INDEX, self.END_INDEX + 1):
                self.process_image_pair(i, gray_img)

            # Plot results
            self.plot_results()

            logger.info("Processing complete!")

            # Save results to file
            results_data = {
                'Index': list(range(self.START_INDEX, self.END_INDEX + 1)),
                'GreyNum': self.grey_num,
                'VibMatchInGray': self.vib_match_num_in_gray,
                'VibNoMatch': self.vib_no_match_num,
                'VibMatch': self.vib_match_num,
                'VibAll': self.vib_all_num,
                'VibRatio': [self.vib_match_num_in_gray[i] / self.grey_num[i] if self.grey_num[i] > 0 else 0.0
                             for i in range(self.IMG_COUNT)],
                'NoVibMatchInGray': self.novib_match_num_in_gray,
                'NoVibNoMatch': self.novib_no_match_num,
                'NoVibMatch': self.novib_match_num,
                'NoVibAll': self.novib_all_num,
                'NoVibRatio': [self.novib_match_num_in_gray[i] / self.grey_num[i] if self.grey_num[i] > 0 else 0.0
                               for i in range(self.IMG_COUNT)],
                'RatioDifference': [
                    (self.vib_match_num_in_gray[i] - self.novib_match_num_in_gray[i]) / self.grey_num[i] if
                    self.grey_num[i] > 0 else 0.0
                    for i in range(self.IMG_COUNT)]
            }

            df = pd.DataFrame(results_data)
            output_filename = f'evaluation_results_{self.START_INDEX}_{self.END_INDEX}.csv'
            df.to_csv(output_filename, index=False)

            logger.info(f"Results saved to {output_filename}")

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise


def main(start_index: int = 700, end_index: int = 1500):
    """
    Main function with configurable start and end indices

    Args:
        start_index: First image index to process
        end_index: Last image index to process
    """
    try:
        processor = ImageProcessor(start_index, end_index)
        processor.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        return -1

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Image Processing Pipeline with configurable range')
    parser.add_argument('--start', type=int, default=700, help='Start image index (default: 700)')
    parser.add_argument('--end', type=int, default=1500, help='End image index (default: 1500)')

    args = parser.parse_args()

    if args.start >= args.end:
        logger.error("Start index must be less than end index")
        exit(1)

    logger.info(f"Processing images from {args.start} to {args.end}")
    exit(main(args.start, args.end))
