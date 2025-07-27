import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import logging
from dataclasses import dataclass
from sklearn.metrics import f1_score
from scipy import ndimage
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """Data class to store registration results"""

    transformation_matrix: np.ndarray
    registered_image: np.ndarray
    registration_score: float
    method_used: str
    success: bool


class BinaryEdgeRegistration:
    """
    A comprehensive class for registering binary edge images from multiple sources.

    This class implements multiple registration methods suitable for binary edge maps:
    1. Phase Correlation - Fast and robust for translation-only transformations
    2. Feature-based Registration - Handles complex transformations using corner detection
    3. Enhanced ECC - Modified Enhanced Correlation Coefficient for binary images
    4. Template Matching - Fallback method for challenging cases
    """

    def __init__(self, registration_method: str = "auto"):
        """
        Initialize the registration system.

        Args:
            registration_method: 'auto', 'phase_correlation', 'feature_based',
                               'ecc', or 'template_matching'
        """
        self.registration_method = registration_method
        self.registration_results = {}

        # Parameters for different methods
        self.phase_corr_params = {"hann_window": True, "upsample_factor": 10}

        self.feature_params = {
            "max_corners": 1000,
            "quality_level": 0.01,
            "min_distance": 10,
            "block_size": 5,
        }

        self.ecc_params = {
            "max_iterations": 100,
            "termination_eps": 1e-6,
            "motion_type": cv2.MOTION_EUCLIDEAN,
        }

    def preprocess_binary_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess binary edge image for registration.

        Args:
            image: Binary edge image (0 or 255 values)

        Returns:
            Preprocessed image
        """
        # Ensure binary format (0 or 255)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Apply morphological operations to clean up edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        return image

    def phase_correlation_registration(
        self, moving_img: np.ndarray, fixed_img: np.ndarray
    ) -> RegistrationResult:
        """
        Register images using phase correlation method.
        Best for translation-only transformations with good accuracy.

        Args:
            moving_img: Image to be aligned
            fixed_img: Reference image

        Returns:
            RegistrationResult object
        """
        try:
            # Convert to float32 for FFT operations
            moving_float = moving_img.astype(np.float32) / 255.0
            fixed_float = fixed_img.astype(np.float32) / 255.0

            # Apply Hann window to reduce edge artifacts
            if self.phase_corr_params["hann_window"]:
                hann_2d = np.outer(
                    np.hanning(moving_float.shape[0]), np.hanning(moving_float.shape[1])
                )
                moving_float *= hann_2d
                fixed_float *= hann_2d

            # Compute phase correlation
            shift, error, diffphase = cv2.phaseCorrelate(moving_float, fixed_float)

            # Create transformation matrix
            transform_matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])

            # Apply transformation
            registered_img = cv2.warpAffine(
                moving_img, transform_matrix, (fixed_img.shape[1], fixed_img.shape[0])
            )

            # Calculate registration quality (higher is better)
            registration_score = 1.0 - error if error < 1.0 else 0.0

            return RegistrationResult(
                transformation_matrix=transform_matrix,
                registered_image=registered_img,
                registration_score=registration_score,
                method_used="phase_correlation",
                success=True,
            )

        except Exception as e:
            logger.warning(f"Phase correlation failed: {e}")
            return RegistrationResult(
                transformation_matrix=np.eye(2, 3, dtype=np.float32),
                registered_image=moving_img.copy(),
                registration_score=0.0,
                method_used="phase_correlation",
                success=False,
            )

    def feature_based_registration(
        self, moving_img: np.ndarray, fixed_img: np.ndarray
    ) -> RegistrationResult:
        """
        Register images using feature-based method with corner detection.
        Handles more complex transformations (rotation, scaling, affine).

        Args:
            moving_img: Image to be aligned
            fixed_img: Reference image

        Returns:
            RegistrationResult object
        """
        try:
            # Detect corners in both images
            corners_moving = cv2.goodFeaturesToTrack(moving_img, **self.feature_params)
            corners_fixed = cv2.goodFeaturesToTrack(fixed_img, **self.feature_params)

            if corners_moving is None or corners_fixed is None:
                raise ValueError("Insufficient features detected")

            # Convert to proper format
            corners_moving = np.float32(corners_moving).reshape(-1, 1, 2)
            corners_fixed = np.float32(corners_fixed).reshape(-1, 1, 2)

            # Create feature descriptors using local patches
            def extract_patch_descriptors(img, corners, patch_size=16):
                descriptors = []
                valid_corners = []

                for corner in corners:
                    x, y = int(corner[0][0]), int(corner[0][1])

                    # Check bounds
                    if (
                        patch_size // 2 <= x < img.shape[1] - patch_size // 2
                        and patch_size // 2 <= y < img.shape[0] - patch_size // 2
                    ):

                        patch = img[
                            y - patch_size // 2 : y + patch_size // 2,
                            x - patch_size // 2 : x + patch_size // 2,
                        ]
                        descriptors.append(patch.flatten())
                        valid_corners.append(corner)

                return np.array(descriptors), np.array(valid_corners)

            desc_moving, corners_moving_valid = extract_patch_descriptors(
                moving_img, corners_moving
            )
            desc_fixed, corners_fixed_valid = extract_patch_descriptors(
                fixed_img, corners_fixed
            )

            if len(desc_moving) < 4 or len(desc_fixed) < 4:
                raise ValueError("Insufficient valid features for matching")

            # Match features using normalized cross-correlation
            matches = []
            for i, desc_m in enumerate(desc_moving):
                best_match_idx = -1
                best_score = -1

                for j, desc_f in enumerate(desc_fixed):
                    # Normalized cross-correlation
                    correlation = np.corrcoef(desc_m, desc_f)[0, 1]
                    if not np.isnan(correlation) and correlation > best_score:
                        best_score = correlation
                        best_match_idx = j

                if (
                    best_match_idx >= 0 and best_score > 0.7
                ):  # Threshold for good matches
                    matches.append([i, best_match_idx, best_score])

            if len(matches) < 4:
                raise ValueError("Insufficient good matches found")

            # Extract matched points
            src_pts = np.float32([corners_moving_valid[m[0]][0] for m in matches])
            dst_pts = np.float32([corners_fixed_valid[m[1]][0] for m in matches])

            # Find transformation using RANSAC
            transform_matrix, mask = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=2000,
            )

            if transform_matrix is None:
                raise ValueError("Could not estimate transformation")

            # Apply transformation
            registered_img = cv2.warpAffine(
                moving_img, transform_matrix, (fixed_img.shape[1], fixed_img.shape[0])
            )

            # Calculate registration score based on inlier ratio
            inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0.0
            registration_score = inlier_ratio

            return RegistrationResult(
                transformation_matrix=transform_matrix,
                registered_image=registered_img,
                registration_score=registration_score,
                method_used="feature_based",
                success=True,
            )

        except Exception as e:
            logger.warning(f"Feature-based registration failed: {e}")
            return RegistrationResult(
                transformation_matrix=np.eye(2, 3, dtype=np.float32),
                registered_image=moving_img.copy(),
                registration_score=0.0,
                method_used="feature_based",
                success=False,
            )

    def ecc_registration(
        self, moving_img: np.ndarray, fixed_img: np.ndarray
    ) -> RegistrationResult:
        """
        Register images using Enhanced Correlation Coefficient method.
        Good for small transformations with subpixel accuracy.

        Args:
            moving_img: Image to be aligned
            fixed_img: Reference image

        Returns:
            RegistrationResult object
        """
        try:
            # Convert to float32
            moving_float = moving_img.astype(np.float32)
            fixed_float = fixed_img.astype(np.float32)

            # Initialize transformation matrix
            if self.ecc_params["motion_type"] == cv2.MOTION_TRANSLATION:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

            # Define termination criteria
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.ecc_params["max_iterations"],
                self.ecc_params["termination_eps"],
            )

            # Run ECC algorithm
            try:
                cc, warp_matrix = cv2.findTransformECC(
                    fixed_float,
                    moving_float,
                    warp_matrix,
                    self.ecc_params["motion_type"],
                    criteria,
                )
            except cv2.error:
                # If ECC fails, try with Gaussian pyramid
                fixed_pyr = cv2.pyrDown(fixed_float)
                moving_pyr = cv2.pyrDown(moving_float)

                cc, warp_matrix_pyr = cv2.findTransformECC(
                    fixed_pyr,
                    moving_pyr,
                    warp_matrix,
                    self.ecc_params["motion_type"],
                    criteria,
                )

                # Scale transformation back up
                warp_matrix = warp_matrix_pyr.copy()
                warp_matrix[0, 2] *= 2  # Scale translation
                warp_matrix[1, 2] *= 2

                cc = 0.5  # Reduced confidence for pyramid approach

            # Apply transformation
            registered_img = cv2.warpAffine(
                moving_img, warp_matrix, (fixed_img.shape[1], fixed_img.shape[0])
            )

            return RegistrationResult(
                transformation_matrix=warp_matrix,
                registered_image=registered_img,
                registration_score=cc,
                method_used="ecc",
                success=True,
            )

        except Exception as e:
            logger.warning(f"ECC registration failed: {e}")
            return RegistrationResult(
                transformation_matrix=np.eye(2, 3, dtype=np.float32),
                registered_image=moving_img.copy(),
                registration_score=0.0,
                method_used="ecc",
                success=False,
            )

    def template_matching_registration(
        self, moving_img: np.ndarray, fixed_img: np.ndarray
    ) -> RegistrationResult:
        """
        Register images using template matching as fallback method.
        Uses the center region of the moving image as template.

        Args:
            moving_img: Image to be aligned
            fixed_img: Reference image

        Returns:
            RegistrationResult object
        """
        try:
            # Extract template from center of moving image
            h, w = moving_img.shape
            template_size = min(h, w) // 3

            center_y, center_x = h // 2, w // 2
            template = moving_img[
                center_y - template_size // 2 : center_y + template_size // 2,
                center_x - template_size // 2 : center_x + template_size // 2,
            ]

            # Perform template matching
            result = cv2.matchTemplate(fixed_img, template, cv2.TM_CCOEFF_NORMED)

            # Find best match location
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # Calculate shift
            shift_x = max_loc[0] - (center_x - template_size // 2)
            shift_y = max_loc[1] - (center_y - template_size // 2)

            # Create transformation matrix
            transform_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

            # Apply transformation
            registered_img = cv2.warpAffine(
                moving_img, transform_matrix, (fixed_img.shape[1], fixed_img.shape[0])
            )

            return RegistrationResult(
                transformation_matrix=transform_matrix,
                registered_image=registered_img,
                registration_score=max_val,
                method_used="template_matching",
                success=True,
            )

        except Exception as e:
            logger.warning(f"Template matching registration failed: {e}")
            return RegistrationResult(
                transformation_matrix=np.eye(2, 3, dtype=np.float32),
                registered_image=moving_img.copy(),
                registration_score=0.0,
                method_used="template_matching",
                success=False,
            )

    def register_image_pair(
        self, moving_img: np.ndarray, fixed_img: np.ndarray
    ) -> RegistrationResult:
        """
        Register a pair of images using the specified or automatic method selection.

        Args:
            moving_img: Image to be aligned
            fixed_img: Reference image

        Returns:
            RegistrationResult object
        """
        # Preprocess images
        moving_processed = self.preprocess_binary_image(moving_img)
        fixed_processed = self.preprocess_binary_image(fixed_img)

        if self.registration_method == "auto":
            # Try methods in order of preference for binary edge images
            methods_to_try = [
                self.phase_correlation_registration,
                self.feature_based_registration,
                self.ecc_registration,
                self.template_matching_registration,
            ]

            best_result = None

            for method in methods_to_try:
                result = method(moving_processed, fixed_processed)

                if result.success and (
                    best_result is None
                    or result.registration_score > best_result.registration_score
                ):
                    best_result = result

                # If we get a good result, stop trying other methods
                if result.success and result.registration_score > 0.8:
                    break

            return (
                best_result
                if best_result
                else RegistrationResult(
                    transformation_matrix=np.eye(2, 3, dtype=np.float32),
                    registered_image=moving_img.copy(),
                    registration_score=0.0,
                    method_used="auto_failed",
                    success=False,
                )
            )

        # Use specific method
        method_map = {
            "phase_correlation": self.phase_correlation_registration,
            "feature_based": self.feature_based_registration,
            "ecc": self.ecc_registration,
            "template_matching": self.template_matching_registration,
        }

        if self.registration_method in method_map:
            return method_map[self.registration_method](
                moving_processed, fixed_processed
            )
        else:
            raise ValueError(f"Unknown registration method: {self.registration_method}")


def load_images_from_folder(folder_path: str) -> Dict[str, np.ndarray]:
    """
    Load all images from a folder and return as dictionary.

    Args:
        folder_path: Path to folder containing images

    Returns:
        Dictionary mapping filename to image array
    """
    images = {}
    folder = Path(folder_path)

    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        return images

    # Supported image extensions
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    for img_path in folder.iterdir():
        if img_path.suffix.lower() in extensions:
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images[img_path.stem] = img
                else:
                    logger.warning(f"Could not load image: {img_path}")
            except Exception as e:
                logger.error(f"Error loading {img_path}: {e}")

    logger.info(f"Loaded {len(images)} images from {folder_path}")
    return images


def register_image_sets(
    gt_folder: str,
    sev_folder: str,
    amiev_folder: str,
    output_folder: str = "registered_outputs",
    registration_method: str = "auto",
) -> Dict:
    """
    Main function to register image sets from three folders.

    Args:
        gt_folder: Path to ground truth edges folder
        sev_folder: Path to S-EV edges folder
        amiev_folder: Path to AMI-EV edges folder
        output_folder: Path to save registered images
        registration_method: Registration method to use

    Returns:
        Dictionary containing registration results and statistics
    """
    # Create output directories
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    (output_path / "registered_sev").mkdir(exist_ok=True)
    (output_path / "registered_amiev").mkdir(exist_ok=True)
    (output_path / "visualizations").mkdir(exist_ok=True)

    # Load images from all folders
    gt_images = load_images_from_folder(gt_folder)
    sev_images = load_images_from_folder(sev_folder)
    amiev_images = load_images_from_folder(amiev_folder)

    # Initialize registration system
    registrator = BinaryEdgeRegistration(registration_method)

    # Find common image names across all folders
    common_names = (
        set(gt_images.keys()) & set(sev_images.keys()) & set(amiev_images.keys())
    )

    if not common_names:
        logger.error("No common image names found across all folders")
        return {"success": False, "message": "No matching images found"}

    logger.info(f"Processing {len(common_names)} common images")

    results = {
        "sev_registrations": {},
        "amiev_registrations": {},
        "statistics": {
            "total_processed": 0,
            "sev_successful": 0,
            "amiev_successful": 0,
            "average_sev_score": 0.0,
            "average_amiev_score": 0.0,
        },
    }

    sev_scores = []
    amiev_scores = []

    for img_name in sorted(common_names):
        logger.info(f"Processing image: {img_name}")

        gt_img = gt_images[img_name]
        sev_img = sev_images[img_name]
        amiev_img = amiev_images[img_name]

        # Register S-EV image to ground truth
        sev_result = registrator.register_image_pair(sev_img, gt_img)
        results["sev_registrations"][img_name] = sev_result

        # Register AMI-EV image to ground truth
        amiev_result = registrator.register_image_pair(amiev_img, gt_img)
        results["amiev_registrations"][img_name] = amiev_result

        # Save registered images
        if sev_result.success:
            cv2.imwrite(
                str(output_path / "registered_sev" / f"{img_name}.png"),
                sev_result.registered_image,
            )
            sev_scores.append(sev_result.registration_score)
            results["statistics"]["sev_successful"] += 1

        if amiev_result.success:
            cv2.imwrite(
                str(output_path / "registered_amiev" / f"{img_name}.png"),
                amiev_result.registered_image,
            )
            amiev_scores.append(amiev_result.registration_score)
            results["statistics"]["amiev_successful"] += 1

        # Create visualization
        create_registration_visualization(
            gt_img,
            sev_img,
            amiev_img,
            sev_result.registered_image if sev_result.success else sev_img,
            amiev_result.registered_image if amiev_result.success else amiev_img,
            str(output_path / "visualizations" / f"{img_name}_comparison.png"),
            sev_result,
            amiev_result,
        )

        results["statistics"]["total_processed"] += 1

    # Calculate final statistics
    results["statistics"]["average_sev_score"] = (
        np.mean(sev_scores) if sev_scores else 0.0
    )
    results["statistics"]["average_amiev_score"] = (
        np.mean(amiev_scores) if amiev_scores else 0.0
    )

    # Save summary report
    save_registration_report(results, output_path / "registration_report.txt")

    logger.info("Registration completed successfully!")
    logger.info(f"Results saved to: {output_folder}")

    return results


def create_registration_visualization(
    gt_img,
    sev_orig,
    amiev_orig,
    sev_reg,
    amiev_reg,
    output_path,
    sev_result,
    amiev_result,
):
    """Create a visualization comparing original and registered images."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Ground truth (reference)
    axes[0, 0].imshow(gt_img, cmap="gray")
    axes[0, 0].set_title("Ground Truth\n(Reference)")
    axes[0, 0].axis("off")

    # S-EV original and registered
    axes[0, 1].imshow(sev_orig, cmap="gray")
    axes[0, 1].set_title("S-EV Original")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(sev_reg, cmap="gray")
    title = f"S-EV Registered\n{sev_result.method_used}\nScore: {sev_result.registration_score:.3f}"
    axes[0, 2].set_title(title)
    axes[0, 2].axis("off")

    # AMI-EV original and registered
    axes[1, 1].imshow(amiev_orig, cmap="gray")
    axes[1, 1].set_title("AMI-EV Original")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(amiev_reg, cmap="gray")
    title = f"AMI-EV Registered\n{amiev_result.method_used}\nScore: {amiev_result.registration_score:.3f}"
    axes[1, 2].set_title(title)
    axes[1, 2].axis("off")

    # Overlay comparison
    overlay = np.zeros((gt_img.shape[0], gt_img.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 0] = gt_img  # Red channel for GT
    overlay[:, :, 1] = sev_reg  # Green channel for S-EV
    overlay[:, :, 2] = amiev_reg  # Blue channel for AMI-EV

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Overlay Comparison\n(R:GT, G:S-EV, B:AMI-EV)")
    axes[1, 0].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_registration_report(results, output_path):
    """Save a detailed registration report."""
    with open(output_path, "w") as f:
        f.write("BINARY EDGE IMAGE REGISTRATION REPORT\n")
        f.write("=" * 50 + "\n\n")

        stats = results["statistics"]
        f.write(f"Total images processed: {stats['total_processed']}\n")
        f.write(f"S-EV successful registrations: {stats['sev_successful']}\n")
        f.write(f"AMI-EV successful registrations: {stats['amiev_successful']}\n")
        f.write(f"Average S-EV registration score: {stats['average_sev_score']:.4f}\n")
        f.write(
            f"Average AMI-EV registration score: {stats['average_amiev_score']:.4f}\n\n"
        )

        f.write("DETAILED RESULTS BY IMAGE:\n")
        f.write("-" * 30 + "\n")

        for img_name in sorted(results["sev_registrations"].keys()):
            sev_res = results["sev_registrations"][img_name]
            amiev_res = results["amiev_registrations"][img_name]

            f.write(f"\nImage: {img_name}\n")
            f.write(
                f"  S-EV: {sev_res.method_used} | Score: {sev_res.registration_score:.4f} | Success: {sev_res.success}\n"
            )
            f.write(
                f"  AMI-EV: {amiev_res.method_used} | Score: {amiev_res.registration_score:.4f} | Success: {amiev_res.success}\n"
            )


def calculate_ods_f1_score(
    gt_edges: np.ndarray, pred_edges: np.ndarray, thresholds: np.ndarray = None
) -> float:
    """
    Calculate Optimal Dataset Scale F1 score for edge detection evaluation.

    Args:
        gt_edges: Ground truth binary edges
        pred_edges: Predicted binary edges
        thresholds: Array of thresholds to evaluate (for non-binary predictions)

    Returns:
        ODS F1 score
    """
    if thresholds is None:
        # For binary images, calculate F1 directly
        gt_flat = (gt_edges > 0).flatten()
        pred_flat = (pred_edges > 0).flatten()

        if np.sum(gt_flat) == 0 and np.sum(pred_flat) == 0:
            return 1.0  # Perfect match for empty edges
        elif np.sum(gt_flat) == 0 or np.sum(pred_flat) == 0:
            return 0.0  # No overlap possible

        return f1_score(gt_flat, pred_flat, average="binary")

    # For non-binary predictions, find optimal threshold
    best_f1 = 0.0
    gt_flat = (gt_edges > 0).flatten()

    for threshold in thresholds:
        pred_flat = (pred_edges > threshold).flatten()

        if np.sum(pred_flat) == 0:
            continue

        f1 = f1_score(gt_flat, pred_flat, average="binary", zero_division=0)
        best_f1 = max(best_f1, f1)

    return best_f1


# Example usage and main execution
if __name__ == "__main__":
    # Example paths - adjust these to your actual folder structure
    gt_folder = "gt_edges"
    sev_folder = "s_ev_edges"
    amiev_folder = "ami_ev_edges"
    output_folder = "registered_outputs"

    # Run registration with automatic method selection
    try:
        results = register_image_sets(
            gt_folder=gt_folder,
            sev_folder=sev_folder,
            amiev_folder=amiev_folder,
            output_folder=output_folder,
            registration_method="auto",  # or 'phase_correlation', 'feature_based', 'ecc', 'template_matching'
        )

        # Print summary
        if "statistics" in results:
            stats = results["statistics"]
            print(f"\nREGISTRATION SUMMARY:")
            print(f"Total processed: {stats['total_processed']}")
            print(
                f"S-EV success rate: {stats['sev_successful']}/{stats['total_processed']}"
            )
            print(
                f"AMI-EV success rate: {stats['amiev_successful']}/{stats['total_processed']}"
            )
            print(f"Average S-EV score: {stats['average_sev_score']:.4f}")
            print(f"Average AMI-EV score: {stats['average_amiev_score']:.4f}")

    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise
