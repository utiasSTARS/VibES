from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple, Iterator

import cv2
import os
import numpy as np

class ImageEdgeDataloader:
    def __init__(self,
                 image_path: str,
                 start_index: int = 700,
                 end_index: int = 1500,
                 resize_factor: float = 0.5,
                 adaptive_c: int = 2,
                 gaussian_kernel: int = 5,
                 block_size: int = 11):
        self.image_path = image_path
        self.start_index = start_index
        self.end_index = end_index
        self.resize_factor = resize_factor
        self.adaptive_c = adaptive_c
        self.gaussian_kernel = gaussian_kernel
        self.block_size = block_size

        # Ensure gaussian_kernel is odd
        if self.gaussian_kernel % 2 == 0:
            self.gaussian_kernel += 1

        # Ensure block_size is odd
        if self.block_size % 2 == 0:
            self.block_size += 1

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Iterator to yield (index, original_image, edge_image) tuples."""
        for idx in range(self.start_index, self.end_index + 1):
            img = self.load_image(idx)
            if img is not None:
                edges = self.extract_edges(img)
                yield idx, img, edges

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and process all images into memory in one go."""
        indices = []
        originals = []
        edges = []

        for idx, orig, edge in self:
            indices.append(idx)
            originals.append(orig)
            edges.append(edge)

        return np.array(originals, dtype=object), np.array(edges, dtype=object), np.array(indices)

    def load_image(self, index: int) -> Optional[np.ndarray]:
        """Load a single image by index."""
        filename = f"{index}.png"
        path = os.path.join(self.image_path, filename)

        if not os.path.exists(path):
            return None

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        if self.resize_factor != 1.0:
            new_width = int(img.shape[1] * self.resize_factor)
            new_height = int(img.shape[0] * self.resize_factor)
            if new_width > 0 and new_height > 0:
                img = cv2.resize(img, (new_width, new_height))
            else:
                return None

        return img

    def extract_edges(self, img: np.ndarray) -> np.ndarray:
        """Extract edges from an image using adaptive threshold and Zhang-Suen thinning."""
        if img is None or img.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.gaussian_kernel, self.gaussian_kernel), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, self.block_size, self.adaptive_c)
        thresh = cv2.bitwise_not(thresh)
        return self.zhang_suen_thinning(thresh)

    def zhang_suen_thinning(self, image: np.ndarray) -> np.ndarray:
        """Apply Zhang-Suen thinning algorithm to create skeleton."""
        if image.size == 0:
            return image

        binary = (image > 0).astype(np.uint8)
        skeleton = binary.copy()

        if skeleton.shape[0] <= 2 or skeleton.shape[1] <= 2:
            return (skeleton * 255).astype(np.uint8)

        changed = True
        max_iterations = 1000  # Prevent infinite loops
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for step in (1, 2):
                to_remove = []
                for i in range(1, skeleton.shape[0] - 1):
                    for j in range(1, skeleton.shape[1] - 1):
                        if skeleton[i, j] == 1:
                            p = [skeleton[i-1, j], skeleton[i-1, j+1], skeleton[i, j+1],
                                 skeleton[i+1, j+1], skeleton[i+1, j], skeleton[i+1, j-1],
                                 skeleton[i, j-1], skeleton[i-1, j-1]]
                            if self._check_zhang_suen_conditions(p, step):
                                to_remove.append((i, j))

                if to_remove:
                    for (i, j) in to_remove:
                        skeleton[i, j] = 0
                    changed = True

        return (skeleton * 255).astype(np.uint8)

    def _check_zhang_suen_conditions(self, p: list, step: int) -> bool:
        """Check Zhang-Suen thinning conditions."""
        # Number of non-zero neighbors
        B_P1 = sum(p)
        if not (2 <= B_P1 <= 6):
            return False

        # Number of 0-1 transitions in clockwise direction
        A_P1 = sum(p[k] == 0 and p[(k + 1) % 8] == 1 for k in range(8))
        if A_P1 != 1:
            return False

        # Step-specific conditions
        if step == 1:
            if p[0] * p[2] * p[4] != 0:
                return False
            if p[2] * p[4] * p[6] != 0:
                return False
        else:
            if p[0] * p[2] * p[6] != 0:
                return False
            if p[0] * p[4] * p[6] != 0:
                return False

        return True


class GTImageDataloader:
    def __init__(self,
                 image_path: str,
                 hz: int = 1000,
                 start_index: int = 1,
                 end_index: int = 100,
                 resize_factor: float = 0.5,
                 canny_min: int = 50,
                 canny_max: int = 150):
        self.multiplier = max(1, int(1000 / hz))  # Ensure multiplier is at least 1
        self.image_path = image_path
        self.start_index = start_index
        self.end_index = end_index
        self.resize_factor = resize_factor
        self.canny_min = canny_min
        self.canny_max = canny_max

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Iterator to yield (index, original_image, edge_image) tuples."""
        indices = range(self.start_index, self.end_index + 1, self.multiplier)
        for idx in indices:
            img = self.load_gt_image(idx)
            if img is not None:
                edges = self.process_gray_image(img)
                yield idx, img, edges

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and process all images into memory in one go."""
        indices = []
        originals = []
        edges = []

        for idx, orig, edge in self:
            indices.append(idx)
            originals.append(orig)
            edges.append(edge)

        return np.array(originals, dtype=object), np.array(edges, dtype=object), np.array(indices)

    def load_gt_image(self, index: int) -> Optional[np.ndarray]:
        """Load a single ground truth image by index."""
        filename = f"{index:04d}.png"
        path = os.path.join(self.image_path, filename)

        if not os.path.exists(path):
            return None

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        if self.resize_factor != 1.0:
            new_width = int(img.shape[1] * self.resize_factor)
            new_height = int(img.shape[0] * self.resize_factor)
            if new_width > 0 and new_height > 0:
                img = cv2.resize(img, (new_width, new_height))
            else:
                return None

        return img

    def process_gray_image(self, img: np.ndarray) -> np.ndarray:
        """Process image to extract edges using Otsu thresholding and Canny edge detection."""
        if img is None or img.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply Canny edge detection
        edges = cv2.Canny(binary, self.canny_min, self.canny_max)

        # Final thresholding to ensure binary output
        _, result = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

        return result