#!/usr/bin/env python3
"""
Simple Image Edge Dataloader
Extracts edges from images using the same processing pipeline
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, Iterator
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import logging
from skimage import measure, morphology
from skimage import color

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_edge_connectivity(edge_map, stats, path):
    """
    Visualize connectivity analysis:
    - Colored connected components
    - Junction points marked
    - Contour length histogram

    Args:
        edge_map (np.ndarray): Binary edge map.
        stats (dict): Output from edge_connectivity_stats.
    """

    labeled, _ = measure.label(edge_map, connectivity=2, return_num=True)
    colored_components = color.label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0), kind='overlay')

    # Mark junctions in red
    junction_overlay = colored_components.copy()
    coords = np.argwhere(edge_map)
    for y, x in coords:
        ymin, ymax = max(0, y - 1), min(edge_map.shape[0], y + 2)
        xmin, xmax = max(0, x - 1), min(edge_map.shape[1], x + 2)
        neigh_count = edge_map[ymin:ymax, xmin:xmax].sum() - 1
        if neigh_count > 2:
            junction_overlay[y, x] = [1, 0, 0]  # red in RGB

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: connected components + junctions
    axes[0].imshow(junction_overlay)
    axes[0].set_title(f"Connected Components & Junctions\n"
                      f"{stats['num_components']} components, "
                      f"{stats['junction_count']} junctions")
    axes[0].axis("off")

    # Right: contour length histogram
    axes[1].hist(stats["lengths"], bins=30, color='blue', alpha=0.7)
    axes[1].set_xlabel("Contour Length (px)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Contour Length Distribution\n"
                      f"Avg={stats['avg_contour_length']:.2f}, "
                      f"Median={stats['median_contour_length']:.2f}")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(path)
    # plt.show()


def plot_pr_curve(precisions, recalls, precision_novib, recalls_novib):
    """
    Plot Precision-Recall curve.

    Args:
        precisions (list or np.ndarray): Precision values at each threshold.
        recalls    (list or np.ndarray): Recall values at each threshold.
    """
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, marker='o', linewidth=2)
    plt.plot(recalls_novib, precision_novib, marker='x', linewidth=2, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

    # ==== Aggregate statistics ====


def summarize_continuity_stats(stats_list):
    # Convert list of dicts to dict of arrays
    keys = stats_list[0].keys()
    summary = {}
    for k in keys:
        values = [s[k] for s in stats_list if not isinstance(s[k], list)]
        if not values:
            continue
        summary[k] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
    return summary


def edge_connectivity_stats(edge_map):
    """
    Compute connectivity / fragmentation stats for a 1-pixel wide binary edge map.

    Args:
        edge_map (np.ndarray): Binary 2D array (edges=1, background=0).

    Returns:
        dict: {
            "num_components": number of connected components,
            "avg_contour_length": average contour length (px),
            "median_contour_length": median contour length (px),
            "junction_count": number of junction pixels,
            "lengths": list of all contour lengths
        }
    """
    edge_map = (edge_map > 0).astype(np.uint8)

    # Label connected components (8-connectivity)
    labeled, num_components = measure.label(edge_map, connectivity=2, return_num=True)

    # Get lengths (pixel counts) of each contour
    props = measure.regionprops(labeled)
    lengths = [p.area for p in props]  # since edges are 1px wide, area ≈ length

    avg_len = np.mean(lengths) if lengths else 0
    median_len = np.median(lengths) if lengths else 0

    # Count junction points: >2 neighbors in 8-neighborhood
    junction_count = 0
    coords = np.argwhere(edge_map)
    for y, x in coords:
        # Extract 3x3 neighborhood excluding center
        ymin, ymax = max(0, y - 1), min(edge_map.shape[0], y + 2)
        xmin, xmax = max(0, x - 1), min(edge_map.shape[1], x + 2)
        neigh_count = edge_map[ymin:ymax, xmin:xmax].sum() - 1
        if neigh_count > 2:
            junction_count += 1

    return {
        "num_components": num_components,
        "avg_contour_length": avg_len,
        "median_contour_length": median_len,
        "junction_count": junction_count,
        "lengths": lengths
    }


def evaluate_edges(pred_edges, gt_edges, tolerance=2):
    """
    Evaluate predicted edges against ground truth edges.

    Args:
        pred_edges (np.ndarray): Binary 2D array (1px edges), predicted.
        gt_edges   (np.ndarray): Binary 2D array (1px edges), ground truth.
        tolerance  (float): Pixel distance tolerance for matching.

    Returns:
        metrics (dict): Precision, Recall, F1, Chamfer Distance, Localization Error.
        overlay (np.ndarray): RGB overlay of TP (green), FP (red), FN (blue).
    """
    assert pred_edges.shape == gt_edges.shape, "Image sizes must match."
    pred_edges = (pred_edges > 0).astype(np.uint8)
    gt_edges = (gt_edges > 0).astype(np.uint8)

    # Distance transforms
    dist_gt = distance_transform_edt(1 - gt_edges)
    dist_pred = distance_transform_edt(1 - pred_edges)

    # True positives, false positives, false negatives
    TP = np.logical_and(pred_edges, dist_gt <= tolerance)
    FP = np.logical_and(pred_edges, dist_gt > tolerance)
    FN = np.logical_and(gt_edges, dist_pred > tolerance)

    tp_count = TP.sum()
    fp_count = FP.sum()
    fn_count = FN.sum()

    precision = tp_count / (tp_count + fp_count + 1e-9)
    recall = tp_count / (tp_count + fn_count + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # Chamfer Distance (symmetric)
    chamfer = (dist_gt[pred_edges].mean() + dist_pred[gt_edges].mean()) / 2.0

    # Localization error (distance for matched edges)
    loc_error = dist_gt[TP].mean() if tp_count > 0 else np.nan

    # Create overlay: TP=green, FP=red, FN=blue
    overlay = np.zeros((*pred_edges.shape, 3), dtype=np.uint8)
    overlay[TP] = [0, 255, 0]  # green
    overlay[FP] = [255, 0, 0]  # red
    overlay[FN] = [0, 0, 255]  # blue

    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ChamferDistance": chamfer,
        "LocalizationError": loc_error
    }

    return metrics, overlay


class GTImageDataloader:
    def __init__(self,
                 image_path: str,
                 hz: int = 1000,
                 start_index: int = 1,
                 end_index: int = 100,
                 resize_factor: float = 0.5,
                 canny_min: int = 50,
                 canny_max: int = 150):
        """
        Dataloader for Ground Truth images with process_gray_image pipeline

        Args:
            image_path: Path to directory containing GT images
            start_index: First image index to load (1-based)
            end_index: Last image index to load
            resize_factor: Factor to resize images (0.5 = half size)
        """
        self.multiplier = int(1000 / hz)  # Convert Hz to img id
        if self.multiplier < 1:
            raise ValueError("Hz must be less than 1000")
        self.image_path = image_path
        self.start_index = start_index
        self.end_index = end_index
        self.resize_factor = resize_factor
        self.current_index = start_index
        self.canny_min = canny_min
        self.canny_max = canny_max

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Iterator that yields (index, original_image, processed_image)"""
        self.current_index = self.start_index
        return self

    def __next__(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """Get next GT image and its processed version"""
        if self.current_index > self.end_index:
            raise StopIteration

        # Load GT image
        original_img = self.load_gt_image(self.current_index)
        if original_img is None:
            logger.warning(f"Could not load GT image at index {self.current_index}")
            self.current_index += self.multiplier
            return self.__next__()  # Try next image

        # Process using gray image pipeline
        processed_img = self.process_gray_image(original_img)

        result = (self.current_index, original_img, processed_img)
        self.current_index += self.multiplier

        return result

    def __len__(self) -> int:
        """Return total number of images"""
        return self.end_index - self.start_index + 1

    def load_gt_image(self, index: int) -> Optional[np.ndarray]:
        """
        Load GT image with proper naming format (0001.png, 0002.png, etc.)
        """
        # Format index with leading zeros (4 digits)
        filename = f"{index:04d}.png"
        image_path = os.path.join(self.image_path, filename)

        logger.debug(f"Loading GT image: {image_path}")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # Resize if needed (same as original code)
        if self.resize_factor != 1.0:
            new_width = int(img.shape[1] * self.resize_factor)
            new_height = int(img.shape[0] * self.resize_factor)
            img = cv2.resize(img, (new_width, new_height))

        return img

    def process_gray_image(self, img: np.ndarray) -> np.ndarray:
        """
        Process GT image using the same pipeline as process_gray_image from original code
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Otsu thresholding (equivalent to graythresh + im2bw)
        # gray_thresh = self.apply_otsu_threshold(gray)

        # Apply edge detection
        canny_edges = cv2.Canny(gray, self.canny_min, self.canny_max)

        # Convert back to binary format for consistency
        _, result = cv2.threshold(canny_edges, 127, 255, cv2.THRESH_BINARY)

        return result

    def apply_otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply Otsu's thresholding"""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def get_item(self, index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get a specific GT image by index (random access)"""
        if index < self.start_index or index > self.end_index:
            raise IndexError(f"Index {index} out of range [{self.start_index}, {self.end_index}]")

        original_img = self.load_gt_image(index)
        if original_img is None:
            return None

        processed_img = self.process_gray_image(original_img)
        return original_img, processed_img


class ImageEdgeDataloader:
    def __init__(self,
                 image_path: str,
                 start_index: int = 700,
                 end_index: int = 1500,
                 resize_factor: float = 0.5,
                 adaptive_c: int = 2):
        """
        Simple dataloader for loading images and extracting edges

        Args:
            image_path: Path to directory containing images
            start_index: First image index to load
            end_index: Last image index to load
            resize_factor: Factor to resize images (0.5 = half size)
            adaptive_c: Constant for adaptive thresholding
        """
        self.image_path = image_path
        self.start_index = start_index
        self.end_index = end_index
        self.resize_factor = resize_factor
        self.adaptive_c = adaptive_c
        self.current_index = start_index

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Iterator that yields (index, original_image, edge_image)"""
        self.current_index = self.start_index
        return self

    def __next__(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """Get next image and its edge extraction"""
        if self.current_index > self.end_index:
            raise StopIteration

        # Load image
        original_img = self.load_image(self.current_index)
        if original_img is None:
            logger.warning(f"Could not load image at index {self.current_index}")
            self.current_index += 1
            return self.__next__()  # Try next image

        # Extract edges
        edge_img = self.extract_edges(original_img)

        result = (self.current_index, original_img, edge_img)
        self.current_index += 1

        return result

    def __len__(self) -> int:
        """Return total number of images"""
        return self.end_index - self.start_index + 1

    def load_image(self, index: int) -> Optional[np.ndarray]:
        """Load image at given index"""
        filename = f"{index}.png"
        image_path = os.path.join(self.image_path, filename)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # Resize if needed
        if self.resize_factor != 1.0:
            new_width = int(img.shape[1] * self.resize_factor)
            new_height = int(img.shape[0] * self.resize_factor)
            img = cv2.resize(img, (new_width, new_height))

        return img

    def extract_edges(self, img: np.ndarray, gaussian_kernel: int) -> np.ndarray:
        """Extract edges from image using the same pipeline as original code"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)

        # invert colors for edge detection
        blurred = cv2.bitwise_not(blurred)

        # Apply adaptive threshold to handle varying lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 3, self.adaptive_c)

        # Invert so lines are white on black background
        thresh = cv2.bitwise_not(thresh)

        # Apply Zhang-Suen thinning for skeleton extraction
        thinned = self.zhang_suen_thinning(thresh)

        return thinned

    def zhang_suen_thinning(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Zhang-Suen thinning algorithm to binary image
        """
        # Convert to binary (0 and 1)
        binary = (image > 0).astype(np.uint8)
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
                        p = [skeleton[i - 1, j], skeleton[i - 1, j + 1], skeleton[i, j + 1],
                             skeleton[i + 1, j + 1], skeleton[i + 1, j], skeleton[i + 1, j - 1],
                             skeleton[i, j - 1], skeleton[i - 1, j - 1]]

                        # Check Zhang-Suen conditions
                        if self._check_zhang_suen_conditions(p, step=1):
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
                        # Get 8-neighborhood
                        p = [skeleton[i - 1, j], skeleton[i - 1, j + 1], skeleton[i, j + 1],
                             skeleton[i + 1, j + 1], skeleton[i + 1, j], skeleton[i + 1, j - 1],
                             skeleton[i, j - 1], skeleton[i - 1, j - 1]]

                        # Check Zhang-Suen conditions
                        if self._check_zhang_suen_conditions(p, step=2):
                            to_remove.append((i, j))

            # Remove marked pixels
            for (i, j) in to_remove:
                skeleton[i, j] = 0
                changed = True

        # Convert back to 0-255 format
        return (skeleton * 255).astype(np.uint8)

    def _check_zhang_suen_conditions(self, p: list, step: int) -> bool:
        """Check Zhang-Suen algorithm conditions"""
        # Condition 1: 2 <= B(P1) <= 6
        B_P1 = sum(p)
        if not (2 <= B_P1 <= 6):
            return False

        # Condition 2: A(P1) = 1
        A_P1 = 0
        for k in range(8):
            if p[k] == 0 and p[(k + 1) % 8] == 1:
                A_P1 += 1
        if A_P1 != 1:
            return False

        # Step-specific conditions
        if step == 1:
            # Condition 3: P2 * P4 * P6 = 0
            if p[0] * p[2] * p[4] != 0:
                return False
            # Condition 4: P4 * P6 * P8 = 0
            if p[2] * p[4] * p[6] != 0:
                return False
        else:  # step == 2
            # Condition 3: P2 * P4 * P8 = 0
            if p[0] * p[2] * p[6] != 0:
                return False
            # Condition 4: P2 * P6 * P8 = 0
            if p[0] * p[4] * p[6] != 0:
                return False

        return True

    def get_item(self, index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get a specific image by index (random access)"""
        if index < self.start_index or index > self.end_index:
            raise IndexError(f"Index {index} out of range [{self.start_index}, {self.end_index}]")

        original_img = self.load_image(index)
        if original_img is None:
            return None

        edge_img = self.extract_edges(original_img)
        return original_img, edge_img
