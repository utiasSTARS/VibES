import numpy as np
from scipy import ndimage
from skimage import morphology
from typing import Tuple, Union
import matplotlib.pyplot as plt


def calculate_ods_f_score(
        ground_truth: np.ndarray,
        predicted_edges: np.ndarray,
        thresholds: Union[np.ndarray, None] = None,
        tolerance: int = 2,
        plot_results: bool = False
) -> Tuple[float, float, dict]:
    """
    Calculate the Optimal Dataset Scale F1 (ODS-F) score for edge detection.

    Parameters:
    -----------
    ground_truth : np.ndarray
        Binary ground truth edge image (0s and 1s or 0s and 255s)
    predicted_edges : np.ndarray
        Grayscale or binary predicted edge image
    thresholds : np.ndarray, optional
        Array of thresholds to test. If None, uses 100 evenly spaced values
        between the min and max of predicted_edges
    tolerance : int, default=2
        Distance tolerance for matching predicted edges to ground truth edges
    plot_results : bool, default=False
        Whether to plot precision-recall curve and F1 scores

    Returns:
    --------
    tuple: (ods_f_score, optimal_threshold, results_dict)
        - ods_f_score: The optimal F1 score
        - optimal_threshold: Threshold that achieves the optimal F1 score
        - results_dict: Dictionary containing arrays of thresholds, precisions,
          recalls, and f1_scores
    """

    # Normalize inputs to ensure proper format
    gt_binary = _normalize_binary_image(ground_truth)
    pred_edges = predicted_edges.astype(np.float64)

    # Generate thresholds if not provided
    if thresholds is None:
        min_val = np.min(pred_edges)
        max_val = np.max(pred_edges)
        thresholds = np.linspace(min_val, max_val, 100)

    precisions = []
    recalls = []
    f1_scores = []

    # Calculate precision, recall, and F1 for each threshold
    for threshold in thresholds:
        # Binarize predicted edges at current threshold
        binary_pred = (pred_edges >= threshold).astype(np.uint8)

        # Calculate precision, recall, and F1
        precision, recall, f1 = _calculate_edge_metrics(
            gt_binary, binary_pred, tolerance
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Convert to numpy arrays
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

    # Find optimal F1 score and corresponding threshold
    optimal_idx = np.argmax(f1_scores)
    ods_f_score = f1_scores[optimal_idx]
    optimal_threshold = thresholds[optimal_idx]

    # Store results
    results = {
        'thresholds': thresholds,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'optimal_precision': precisions[optimal_idx],
        'optimal_recall': recalls[optimal_idx]
    }

    # Plot results if requested
    if plot_results:
        _plot_ods_results(results, ods_f_score, optimal_threshold)

    return ods_f_score, optimal_threshold, results


def _normalize_binary_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to binary (0 and 1) format."""
    if image.dtype == np.uint8 and np.max(image) > 1:
        # Assume 255 represents edges
        return (image > 127).astype(np.uint8)
    else:
        # Assume already binary or needs simple thresholding
        return (image > 0.5).astype(np.uint8)


def _calculate_edge_metrics(
        gt_binary: np.ndarray,
        pred_binary: np.ndarray,
        tolerance: int
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score for edge detection with tolerance.

    Parameters:
    -----------
    gt_binary : np.ndarray
        Binary ground truth edges
    pred_binary : np.ndarray
        Binary predicted edges
    tolerance : int
        Distance tolerance for matching edges

    Returns:
    --------
    tuple: (precision, recall, f1_score)
    """

    # Handle edge case where no edges are predicted
    if np.sum(pred_binary) == 0:
        return 0.0, 0.0, 0.0

    # Handle edge case where no ground truth edges exist
    if np.sum(gt_binary) == 0:
        if np.sum(pred_binary) == 0:
            return 1.0, 1.0, 1.0  # Perfect match (both empty)
        else:
            return 0.0, 0.0, 0.0  # False positives with no ground truth

    # Create distance transform of ground truth edges
    # This gives the distance from each pixel to the nearest ground truth edge
    gt_distance = ndimage.distance_transform_edt(1 - gt_binary)

    # Find predicted edge pixels
    pred_edge_pixels = np.where(pred_binary > 0)

    # Count true positives: predicted edges within tolerance of ground truth
    distances_to_gt = gt_distance[pred_edge_pixels]
    true_positives = np.sum(distances_to_gt <= tolerance)

    # Calculate precision
    total_predicted = np.sum(pred_binary)
    precision = true_positives / total_predicted if total_predicted > 0 else 0.0

    # For recall, we need to find how many ground truth edges are matched
    # Create distance transform of predicted edges
    pred_distance = ndimage.distance_transform_edt(1 - pred_binary)

    # Find ground truth edge pixels
    gt_edge_pixels = np.where(gt_binary > 0)

    # Count ground truth edges that have a predicted edge within tolerance
    distances_to_pred = pred_distance[gt_edge_pixels]
    matched_gt_edges = np.sum(distances_to_pred <= tolerance)

    # Calculate recall
    total_gt = np.sum(gt_binary)
    recall = matched_gt_edges / total_gt if total_gt > 0 else 0.0

    # Calculate F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score


def _plot_ods_results(results: dict, ods_f_score: float, optimal_threshold: float):
    """Plot precision-recall curve and F1 scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot precision-recall curve
    ax1.plot(results['recalls'], results['precisions'], 'b-', linewidth=2)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Mark optimal point
    optimal_idx = np.argmax(results['f1_scores'])
    ax1.plot(results['recalls'][optimal_idx], results['precisions'][optimal_idx],
             'ro', markersize=8, label=f'ODS-F: {ods_f_score:.3f}')
    ax1.legend()

    # Plot F1 scores vs thresholds
    ax2.plot(results['thresholds'], results['f1_scores'], 'g-', linewidth=2)
    ax2.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.7,
                label=f'Optimal threshold: {optimal_threshold:.3f}')
    ax2.axhline(y=ods_f_score, color='r', linestyle='--', alpha=0.7,
                label=f'ODS-F score: {ods_f_score:.3f}')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Example usage and demonstration
def demo_ods_f_calculation():
    """
    Demonstrate ODS-F calculation with synthetic data.
    """
    print("ODS-F Score Calculator Demo")
    print("=" * 40)

    # Create synthetic ground truth edges (a circle)
    size = 100
    center = size // 2
    y, x = np.ogrid[:size, :size]
    mask = ((x - center) ** 2 + (y - center) ** 2) < (size // 4) ** 2

    # Create ground truth as circle edge
    gt_edges = mask.astype(np.uint8)
    gt_edges = gt_edges - morphology.binary_erosion(gt_edges)

    # Create noisy predicted edges
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, (size, size))
    pred_edges = gt_edges.astype(float) + noise
    pred_edges = np.clip(pred_edges, 0, 1)

    # Add some random false positives
    false_positives = np.random.random((size, size)) > 0.95
    pred_edges[false_positives] = np.random.uniform(0.5, 1.0, np.sum(false_positives))

    # Calculate ODS-F score
    ods_score, opt_threshold, results = calculate_ods_f_score(
        gt_edges, pred_edges, plot_results=True, tolerance=2
    )

    print(f"Optimal Dataset Scale F1 Score: {ods_score:.4f}")
    print(f"Optimal Threshold: {opt_threshold:.4f}")
    print(f"Precision at optimal threshold: {results['optimal_precision']:.4f}")
    print(f"Recall at optimal threshold: {results['optimal_recall']:.4f}")

    return ods_score, opt_threshold, results


if __name__ == "__main__":
    # Run demonstration
    demo_ods_f_calculation()