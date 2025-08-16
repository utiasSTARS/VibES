from concurrent.futures import ProcessPoolExecutor
import cv2
import os
import argparse
import numpy as np
import logging
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Any, Dict, List

from utils_batched import (
    GTImageDataloader, ImageEdgeDataloader
)

from utils import (
    edge_connectivity_stats, summarize_continuity_stats,
    evaluate_edges, plot_pr_curve
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Edge extraction and evaluation tool")

    # Paths
    parser.add_argument("--path", required=True, help="Path to bin images folder before ev/harmeda")
    parser.add_argument("--gt_path", required=False, help="Path to ground truth images folder")

    # Index range
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of images")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index of images")

    # Resize factor
    parser.add_argument("--resize", type=float, default=0.5, help="Resize factor for images")

    # Edge detection params
    parser.add_argument("--adaptive_c", type=int, default=2, help="C value for adaptive threshold")
    parser.add_argument("--canny_min", type=int, default=50, help="Min threshold for Canny")
    parser.add_argument("--canny_max", type=int, default=150, help="Max threshold for Canny")

    parser.add_argument("--gaussian_kernel", type=int, default=7, help="Gaussian kernel size")
    parser.add_argument("--block_size", type=int, default=11, help="Block size for adaptive threshold")

    # Evaluation params
    parser.add_argument("--tolerance", type=float, default=10.0, help="Pixel distance tolerance")

    # Analysis mode
    parser.add_argument(
        "--analysis",
        choices=["continuity", "pr", "both"],
        default="both",
        help="Type of edge analysis to perform"
    )

    # Output options
    parser.add_argument("--output_dir", default="./results", help="Directory to save CSV results")
    parser.add_argument("--save_prefix", default="edge_analysis", help="Prefix for output CSV files")

    return parser.parse_args()


def safe_load_image(loader, target_idx: int) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """Safely load a single image from a loader, returning None values if not found."""
    try:
        for idx, orig, edges in loader:
            if idx == target_idx:
                return idx, orig, edges
        return None, None, None
    except Exception as e:
        logger.warning(f"Error loading image {target_idx}: {e}")
        return None, None, None


def process_image(idx: int, vib_path: str, novib_path: str, gt_path: Optional[str], args) -> Dict[str, Any]:
    """Process a single image index and return statistics and metrics."""
    result = {
        'image_id': idx,
        'vib_stats': None,
        'novib_stats': None,
        'vib_pr_metrics': None,
        'novib_pr_metrics': None,
        'success': False
    }

    try:
        # Create loaders for single image
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
        vib_idx, vib_orig, vib_edges = safe_load_image(vib_loader, idx)
        novib_idx, novib_orig, novib_edges = safe_load_image(novib_loader, idx)

        # Skip if either image failed to load
        if vib_edges is None or novib_edges is None:
            logger.warning(f"Skipping index {idx}: Failed to load images")
            return result

        # Continuity analysis
        if args.analysis in ("continuity", "both"):
            try:
                result['vib_stats'] = edge_connectivity_stats(vib_edges)
                result['novib_stats'] = edge_connectivity_stats(novib_edges)
            except Exception as e:
                logger.warning(f"Error in continuity analysis for index {idx}: {e}")

        # Precision/Recall analysis
        if args.analysis in ("pr", "both") and gt_path:
            try:
                # For ground truth, we need to map the index appropriately
                gt_idx = idx + 1  # Adjust based on your indexing scheme
                gt_loader = GTImageDataloader(
                    gt_path, hz=100, start_index=gt_idx, end_index=gt_idx,
                    resize_factor=args.resize, canny_min=args.canny_min, canny_max=args.canny_max
                )

                _, _, gt_edges = safe_load_image(gt_loader, gt_idx)

                if gt_edges is not None:
                    result['vib_pr_metrics'], _ = evaluate_edges(vib_edges, gt_edges, tolerance=args.tolerance)
                    result['novib_pr_metrics'], _ = evaluate_edges(novib_edges, gt_edges, tolerance=args.tolerance)
                else:
                    logger.warning(f"No ground truth image found for index {idx}")
            except Exception as e:
                logger.warning(f"Error in PR analysis for index {idx}: {e}")

        result['success'] = True
        return result

    except Exception as e:
        logger.error(f"Error processing image {idx}: {e}")
        return result


def save_results_to_csv(results: List[Dict[str, Any]], args) -> None:
    """Save analysis results to CSV files."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare data for CSV
    csv_rows = []

    for result in results:
        if not result['success']:
            continue

        row = {'image_id': result['image_id']}

        # Add VIB statistics
        if result['vib_stats']:
            for key, value in result['vib_stats'].items():
                row[f'vib_{key.lower().replace(" ", "_")}'] = value

        # Add NoVIB statistics
        if result['novib_stats']:
            for key, value in result['novib_stats'].items():
                row[f'novib_{key.lower().replace(" ", "_")}'] = value

        # Add VIB PR metrics
        if result['vib_pr_metrics']:
            for key, value in result['vib_pr_metrics'].items():
                row[f'vib_{key.lower()}'] = value

        # Add NoVIB PR metrics
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

        # Print column info for user reference
        print(f"\nCSV saved with {len(csv_rows)} rows and the following columns:")
        for col in sorted(df.columns):
            print(f"  - {col}")

        return csv_path
    else:
        logger.warning("No valid results to save to CSV")
        return None


def main():
    args = parse_args()

    # Construct paths
    vib_path = os.path.join(args.path, "harmeda/img_gray_33333/")
    novib_path = os.path.join(args.path, "ev/img_gray_33333/")
    gt_path = args.gt_path if args.analysis in ("pr", "both") else None

    # Validate paths
    if not os.path.exists(vib_path):
        logger.error(f"VIB path does not exist: {vib_path}")
        return
    if not os.path.exists(novib_path):
        logger.error(f"NoVIB path does not exist: {novib_path}")
        return
    if gt_path and not os.path.exists(gt_path):
        logger.error(f"Ground truth path does not exist: {gt_path}")
        return

    if args.end_idx < 0:
        logger.error("Reading all images in folders")
        # find the minimum and maximum indices from the directories
        vib_files = sorted(os.listdir(vib_path))
        novib_files = sorted(os.listdir(novib_path))
        if not vib_files or not novib_files:
            logger.error("No images found in the specified directories.")
            return
        args.end_idx = min(len(vib_files), len(novib_files))

    indices = range(args.start_idx, args.end_idx)
    logger.info(f"Processing {len(indices)} images from index {args.start_idx} to {args.end_idx-1}")

    # Storage for results
    results = []
    vib_stats_all, novib_stats_all = [], []
    precisions, recalls, precisions_nv, recalls_nv = [], [], [], []

    # Process images with multiprocessing
    try:
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            futures = [
                executor.submit(process_image, idx, vib_path, novib_path, gt_path, args)
                for idx in indices
            ]

            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)

                    if result['success']:
                        # Extract data for summary statistics (backward compatibility)
                        if result['vib_stats']:
                            vib_stats_all.append(result['vib_stats'])
                        if result['novib_stats']:
                            novib_stats_all.append(result['novib_stats'])
                        if result['vib_pr_metrics']:
                            precisions.append(result['vib_pr_metrics']["Precision"])
                            recalls.append(result['vib_pr_metrics']["Recall"])
                        if result['novib_pr_metrics']:
                            precisions_nv.append(result['novib_pr_metrics']["Precision"])
                            recalls_nv.append(result['novib_pr_metrics']["Recall"])

                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(indices)} images")

                except Exception as e:
                    logger.error(f"Error processing future {i}: {e}")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return
    except Exception as e:
        logger.error(f"Error in multiprocessing: {e}")
        return

    # Save results to CSV
    logger.info("Saving results to CSV...")
    save_results_to_csv(results, args)

    # Generate summaries (existing functionality)
    logger.info("Generating analysis summaries...")

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
            print("No continuity statistics collected.")

    if args.analysis in ("pr", "both"):
        if precisions and recalls and precisions_nv and recalls_nv:
            print("\n" + "="*50)
            print("PRECISION/RECALL ANALYSIS SUMMARY")
            print("="*50)
            print(f"VIB MODE - Precision: {np.mean(precisions):.3f}±{np.std(precisions):.3f}, "
                  f"Recall: {np.mean(recalls):.3f}±{np.std(recalls):.3f}")
            print(f"NO VIB MODE - Precision: {np.mean(precisions_nv):.3f}±{np.std(precisions_nv):.3f}, "
                  f"Recall: {np.mean(recalls_nv):.3f}±{np.std(recalls_nv):.3f}")

            # Plot PR curve
            try:
                plot_pr_curve(precisions, recalls, precisions_nv, recalls_nv)
            except Exception as e:
                logger.error(f"Error plotting PR curve: {e}")
        else:
            print("No precision/recall metrics collected.")

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()