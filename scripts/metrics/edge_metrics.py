"""
Edge Analysis and Evaluation Tool.

This script performs batch evaluation of edge detection quality on image sequences.
It compares "Vibrating" (VIB) and "Non-Vibrating" (NoVIB) image sets against
Ground Truth (GT) using two primary metrics:
1. **Continuity:** Connectivity analysis of edge segments.
2. **Precision/Recall (PR):** Pixel-wise comparison against Ground Truth.

It utilizes multiprocessing to handle large datasets efficiently.

Usage:
    python edge_analysis.py --path /data/dataset --analysis both

Dependencies:
    - utils_batched.py (Custom Dataloaders)
    - utils.py (Metric calculations)
"""

from concurrent.futures import ProcessPoolExecutor
import cv2
import os
import argparse
import numpy as np
import logging
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Any, Dict, List

# Custom modules for data loading and metric computation
from utils_batched import (
    GTImageDataloader, ImageEdgeDataloader
)

from utils import (
    edge_connectivity_stats, summarize_continuity_stats,
    evaluate_edges, plot_pr_curve
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Edge extraction and evaluation tool")

    # Paths
    parser.add_argument("--path", required=True, help="Base path to binary images folder")
    parser.add_argument("--gt_path", required=False, help="Path to ground truth images folder")

    # Index range control
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of image sequence")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index (exclusive), -1 for all")

    # Image processing params
    parser.add_argument("--resize", type=float, default=0.5, help="Resize factor for faster processing")
    parser.add_argument("--adaptive_c", type=int, default=2, help="Constant C for adaptive thresholding")

    # Canny parameters (for GT generation)
    parser.add_argument("--canny_min", type=int, default=50, help="Min threshold for Canny")
    parser.add_argument("--canny_max", type=int, default=150, help="Max threshold for Canny")

    # Preprocessing params
    parser.add_argument("--gaussian_kernel", type=int, default=7, help="Gaussian blur kernel size")
    parser.add_argument("--block_size", type=int, default=11, help="Block size for adaptive threshold")

    # Evaluation tolerance
    parser.add_argument("--tolerance", type=float, default=10.0,
                        help="Pixel distance tolerance for edge matching (Buffer method)")

    # Analysis mode
    parser.add_argument(
        "--analysis",
        choices=["continuity", "pr", "both"],
        default="both",
        help="Type of edge analysis to perform"
    )

    # Output configuration
    parser.add_argument("--output_dir", default="./results", help="Directory to save CSV results")
    parser.add_argument("--save_prefix", default="edge_analysis", help="Prefix for output CSV files")

    return parser.parse_args()


def safe_load_image(loader, target_idx: int) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Safely load a single image from a loader instance.

    Args:
        loader: An iterator/generator yielding (index, original_image, edge_image).
        target_idx: The specific index we are looking for.

    Returns:
        Tuple of (index, original_image, edge_image) or (None, None, None) if not found.
    """
    try:
        for idx, orig, edges in loader:
            if idx == target_idx:
                return idx, orig, edges
        return None, None, None
    except Exception as e:
        logger.warning(f"Error loading image {target_idx}: {e}")
        return None, None, None


def process_image(idx: int, vib_path: str, novib_path: str, gt_path: Optional[str], args) -> Dict[str, Any]:
    """
    Worker function to process a single image index.

    This function is designed to be picklable for use with ProcessPoolExecutor.
    It instantiates its own dataloaders for the specific index to ensure thread safety.

    Args:
        idx: Image index to process.
        vib_path: Path to Vibrating image directory.
        novib_path: Path to Non-Vibrating image directory.
        gt_path: Path to Ground Truth directory (optional).
        args: Parsed command line arguments.

    Returns:
        Dictionary containing calculated statistics and success flag.
    """
    result = {
        'image_id': idx,
        'vib_stats': None,
        'novib_stats': None,
        'vib_pr_metrics': None,
        'novib_pr_metrics': None,
        'success': False
    }

    try:
        # Initialize loaders specific to this index to avoid race conditions
        # Note: start_index=idx, end_index=idx ensures we only read the relevant file
        vib_loader = ImageEdgeDataloader(
            vib_path, start_index=idx, end_index=idx,
            resize_factor=args.resize, adaptive_c=args.adaptive_c,
            gaussian_kernel=args.gaussian_kernel, block_size=args.block_size
        )
        novib_loader = ImageEdgeDataloader(
            novib_path, start_index=idx, end_index=idx,
            resize_factor=args.resize, adaptive_c=args.adaptive_c,
            gaussian_kernel=args.gaussian_kernel, block_size=args.block_size
        )

        # Load images
        _, _, vib_edges = safe_load_image(vib_loader, idx)
        _, _, novib_edges = safe_load_image(novib_loader, idx)

        # Fail fast if loading failed
        if vib_edges is None or novib_edges is None:
            logger.warning(f"Skipping index {idx}: Failed to load images")
            return result

        # --- 1. Continuity Analysis ---
        # Measures edge fragmentation
        if args.analysis in ("continuity", "both"):
            try:
                result['vib_stats'] = edge_connectivity_stats(vib_edges)
                result['novib_stats'] = edge_connectivity_stats(novib_edges)
            except Exception as e:
                logger.warning(f"Error in continuity analysis for index {idx}: {e}")

        # --- 2. Precision/Recall Analysis ---
        # Compares against Ground Truth
        if args.analysis in ("pr", "both") and gt_path:
            try:
                # Synchronization assumption: GT index is ahead by 1 frame.
                # TODO: Verify this alignment for your specific dataset!
                gt_idx = idx + 1

                gt_loader = GTImageDataloader(
                    gt_path, hz=100, start_index=gt_idx, end_index=gt_idx,
                    resize_factor=args.resize, canny_min=args.canny_min, canny_max=args.canny_max
                )

                _, _, gt_edges = safe_load_image(gt_loader, gt_idx)

                if gt_edges is not None:
                    # evaluate_edges returns (metrics_dict, visualization_image)
                    # We discard the visualization image to save memory/bandwidth
                    result['vib_pr_metrics'], _ = evaluate_edges(vib_edges, gt_edges, tolerance=args.tolerance)
                    result['novib_pr_metrics'], _ = evaluate_edges(novib_edges, gt_edges, tolerance=args.tolerance)
                else:
                    logger.warning(f"No ground truth image found for index {idx} (mapped to {gt_idx})")
            except Exception as e:
                logger.warning(f"Error in PR analysis for index {idx}: {e}")

        result['success'] = True
        return result

    except Exception as e:
        logger.error(f"Fatal error processing image {idx}: {e}")
        return result


def save_results_to_csv(results: List[Dict[str, Any]], args) -> Optional[str]:
    """
    Aggregates results and saves them to a CSV file.

    Args:
        results: List of result dictionaries from workers.
        args: Argument object containing output directory paths.

    Returns:
        Path to the saved CSV file, or None if failed.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_rows = []

    for result in results:
        if not result['success']:
            continue

        # Flatten the nested dictionary structure for CSV format
        row = {'image_id': result['image_id']}

        # Flatten VIB statistics
        if result['vib_stats']:
            for key, value in result['vib_stats'].items():
                row[f'vib_{key.lower().replace(" ", "_")}'] = value

        # Flatten NoVIB statistics
        if result['novib_stats']:
            for key, value in result['novib_stats'].items():
                row[f'novib_{key.lower().replace(" ", "_")}'] = value

        # Flatten VIB PR metrics
        if result['vib_pr_metrics']:
            for key, value in result['vib_pr_metrics'].items():
                row[f'vib_{key.lower()}'] = value

        # Flatten NoVIB PR metrics
        if result['novib_pr_metrics']:
            for key, value in result['novib_pr_metrics'].items():
                row[f'novib_{key.lower()}'] = value

        csv_rows.append(row)

    if csv_rows:
        df = pd.DataFrame(csv_rows)
        csv_filename = f"{args.save_prefix}_{args.analysis}_{timestamp}.csv"
        csv_path = os.path.join(args.output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")

        print(f"\nCSV saved with {len(csv_rows)} rows. Columns:")
        for col in sorted(df.columns):
            print(f"  - {col}")

        return csv_path
    else:
        logger.warning("No valid results collected to save.")
        return None


def main():
    args = parse_args()

    # --- Directory Setup ---
    # TODO: Update these subfolder names if your dataset structure differs
    vib_path = os.path.join(args.path, "vib_img_gray_33333/")
    novib_path = os.path.join(args.path, "novib_img_gray_33333/")
    gt_path = args.gt_path if args.analysis in ("pr", "both") else None

    # --- Validation ---
    if not os.path.exists(vib_path):
        logger.error(f"VIB path does not exist: {vib_path}")
        return
    if not os.path.exists(novib_path):
        logger.error(f"NoVIB path does not exist: {novib_path}")
        return
    if gt_path and not os.path.exists(gt_path):
        logger.error(f"Ground truth path does not exist: {gt_path}")
        return

    # --- Index Discovery ---
    if args.end_idx < 0:
        logger.info("Scanning directories to determine index range...")
        vib_files = sorted(os.listdir(vib_path))
        novib_files = sorted(os.listdir(novib_path))
        if not vib_files or not novib_files:
            logger.error("No images found in the specified directories.")
            return
        args.end_idx = min(len(vib_files), len(novib_files))

    indices = range(args.start_idx, args.end_idx)
    logger.info(f"Processing {len(indices)} images (Index {args.start_idx} to {args.end_idx-1})")

    # --- Main Processing Loop ---
    results = []

    # Containers for aggregate stats (used for printing summary at the end)
    vib_stats_all, novib_stats_all = [], []
    precisions, recalls, precisions_nv, recalls_nv = [], [], [], []

    try:
        # Determine CPU count safely
        max_workers = min(os.cpu_count() or 4, 8)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Dispatch jobs
            futures = [
                executor.submit(process_image, idx, vib_path, novib_path, gt_path, args)
                for idx in indices
            ]

            # Collect results as they finish (or in order)
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)

                    if result['success']:
                        # Collect data for summary printing
                        if result['vib_stats']:
                            vib_stats_all.append(result['vib_stats'])
                        if result['novib_stats']:
                            novib_stats_all.append(result['novib_stats'])
                        if result.get('vib_pr_metrics'):
                            precisions.append(result['vib_pr_metrics'].get("Precision", 0))
                            recalls.append(result['vib_pr_metrics'].get("Recall", 0))
                        if result.get('novib_pr_metrics'):
                            precisions_nv.append(result['novib_pr_metrics'].get("Precision", 0))
                            recalls_nv.append(result['novib_pr_metrics'].get("Recall", 0))

                    if (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i + 1}/{len(indices)}")

                except Exception as e:
                    logger.error(f"Error retrieving future result {i}: {e}")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
        return
    except Exception as e:
        logger.error(f"Multiprocessing error: {e}")
        return

    # --- Save Results ---
    logger.info("Saving individual results to CSV...")
    save_results_to_csv(results, args)

    # --- Generate Summary Report ---
    logger.info("Generating statistical summary...")

    # 1. Continuity Summary
    if args.analysis in ("continuity", "both"):
        if vib_stats_all and novib_stats_all:
            vib_summary = summarize_continuity_stats(vib_stats_all)
            novib_summary = summarize_continuity_stats(novib_stats_all)
            print("\n" + "="*50)
            print("CONTINUITY ANALYSIS SUMMARY")
            print("="*50)
            print(f"VIB MODE (n={len(vib_stats_all)}):")
            for key, value in vib_summary.items():
                print(f"  {key}: {value}")
            print(f"\nNO VIB MODE (n={len(novib_stats_all)}):")
            for key, value in novib_summary.items():
                print(f"  {key}: {value}")
        else:
            print("No valid continuity statistics collected.")

    # 2. Precision/Recall Summary
    if args.analysis in ("pr", "both"):
        if precisions and recalls and precisions_nv and recalls_nv:
            print("\n" + "="*50)
            print("PRECISION/RECALL ANALYSIS SUMMARY")
            print("="*50)
            print(f"VIB MODE    - Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}, "
                  f"Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
            print(f"NO VIB MODE - Precision: {np.mean(precisions_nv):.3f} ± {np.std(precisions_nv):.3f}, "
                  f"Recall: {np.mean(recalls_nv):.3f} ± {np.std(recalls_nv):.3f}")

            # Optional: Plot PR Curve if configured in utils
            try:
                plot_pr_curve(precisions, recalls, precisions_nv, recalls_nv)
            except Exception as e:
                logger.error(f"Error plotting PR curve: {e}")
        else:
            print("No valid precision/recall metrics collected.")

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()