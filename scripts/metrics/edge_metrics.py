#!/usr/bin/env python3
"""
Flexible Image Edge Analysis
 - Configurable paths
 - Adjustable edge detection parameters
 - Continuity & PR metrics
"""

import cv2
import os
import argparse
import numpy as np
from typing import Optional
import logging
from utils import (
    GTImageDataloader, ImageEdgeDataloader,
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
    parser.add_argument("--start_idx", type=int, default=200, help="Start index of images")
    parser.add_argument("--end_idx", type=int, default=250, help="End index of images")

    # Resize factor
    parser.add_argument("--resize", type=float, default=0.5, help="Resize factor for images")

    # Edge detection params
    parser.add_argument("--adaptive_c", type=int, default=2, help="C value for adaptive threshold")
    parser.add_argument("--canny_min", type=int, default=50, help="Min threshold for Canny")
    parser.add_argument("--canny_max", type=int, default=150, help="Max threshold for Canny")

    parser.add_argument("--gaussian_kernel", type=int, default=7, help="Min threshold for Canny")
    parser.add_argument("--block_size", type=int, default=11, help="Max threshold for Canny")

    # Evaluation params
    parser.add_argument("--tolerance", type=float, default=10.0, help="Pixel distance tolerance")

    # Analysis mode
    parser.add_argument(
        "--analysis",
        choices=["continuity", "pr", "both"],
        default="both",
        help="Type of edge analysis to perform"
    )

    parser.add_argument("--visualize", action="store_true", help="Visualize results during analysis")

    return parser.parse_args()


def main():
    args = parse_args()

    vib_path = os.path.join(args.path, "harmeda/img_gray_33333/")
    novib_path = os.path.join(args.path, "ev/img_gray_33333/")

    # Create dataloaders
    vib_loader = ImageEdgeDataloader(
        vib_path, start_index=args.start_idx, end_index=args.end_idx,
        resize_factor=args.resize, adaptive_c=args.adaptive_c,
        gaussian_kernel=args.gaussian_kernel, block_size=args.block_size
    )
    novib_loader = ImageEdgeDataloader(
        novib_path, start_index=args.start_idx, end_index=args.end_idx,
        resize_factor=args.resize, adaptive_c=args.adaptive_c,
        gaussian_kernel=args.gaussian_kernel, block_size=args.block_size
    )

    if args.analysis in ("both", "pr"):
        if not args.gt_path:
            logger.error("Ground truth path is required for Precision-Recall analysis.")
            return
        if not os.path.exists(args.gt_path):
            logger.error(f"Ground truth path {args.gt_path} does not exist.")
            return

        gt_loader = GTImageDataloader(
            args.gt_path, hz=100,
            start_index=args.start_idx + 1, end_index=args.end_idx + 1,
            resize_factor=args.resize,
            canny_min=args.canny_min, canny_max=args.canny_max
        )

    logger.info(f"Vibration dataloader length: {len(vib_loader)}")
    logger.info(f"No-vibration dataloader length: {len(novib_loader)}")

    vib_stats_all, novib_stats_all = [], []
    precisions, recalls = [], []
    precisions_nv, recalls_nv = [], []

    def visualize(vib_orig, novib_orig, vib_edges, novib_edges):
        cv2.imshow(f"VIB", vib_orig)
        cv2.imshow(f"NO VIB", novib_orig)
        cv2.imshow(f"Edges VIB", vib_edges)
        cv2.imshow(f"Edges NO VIB", novib_edges)
        cv2.waitKey(1)

    def continuity_analysis(vib_edges, novib_edges):
        vib_stats_all.append(edge_connectivity_stats(vib_edges))
        novib_stats_all.append(edge_connectivity_stats(novib_edges))

    def pr_analysis(vib_edges, novib_edges, gt_edges):
        metrics_vib, overlay_vib = evaluate_edges(vib_edges, gt_edges, tolerance=args.tolerance)
        metrics_nv, overlay_nv = evaluate_edges(novib_edges, gt_edges, tolerance=args.tolerance)

        logger.info(f"VIB idx {vib_idx}: {metrics_vib}")
        logger.info(f"NO VIB idx {novib_idx}: {metrics_nv}")

        precisions.append(metrics_vib["Precision"])
        recalls.append(metrics_vib["Recall"])
        precisions_nv.append(metrics_nv["Precision"])
        recalls_nv.append(metrics_nv["Recall"])

        cv2.imshow("Overlay VIB", overlay_vib)
        cv2.imshow("Overlay NO VIB", overlay_nv)

    if args.analysis == "both":
        logger.info("Performing both Continuity and Precision-Recall analysis")
        for vib, novib, gt in zip(vib_loader, novib_loader, gt_loader):
            vib_idx, vib_orig, vib_edges = vib
            novib_idx, novib_orig, novib_edges = novib
            _, gt_orig, gt_edges = gt
            continuity_analysis(vib_edges, novib_edges)

            pr_analysis(vib_edges, novib_edges, gt_edges)

            if args.visualize:
                visualize(vib_orig, novib_orig, vib_edges, novib_edges)


    elif args.analysis == "continuity":
        logger.info("Performing Continuity analysis")
        for vib, novib in zip(vib_loader, novib_loader):
            vib_idx, vib_orig, vib_edges = vib
            novib_idx, novib_orig, novib_edges = novib

            # continuity_analysis(vib_edges, novib_edges)

            if args.visualize:
                visualize(vib_orig, novib_orig, vib_edges, novib_edges)

    elif args.analysis == "pr":
        logger.info("Performing Precision-Recall analysis")

        for vib, novib, gt in zip(vib_loader, novib_loader, gt_loader):
            vib_idx, vib_orig, vib_edges = vib
            novib_idx, novib_orig, novib_edges = novib
            _, gt_orig, gt_edges = gt

            pr_analysis(vib_edges, novib_edges, gt_edges)

            if args.visualize:
                visualize(vib_orig, novib_orig, vib_edges, novib_edges)

    else:
        logger.error("Invalid analysis type specified. Use 'continuity', 'pr', or 'both'.")
        return

    # Summary
    if args.analysis in ("continuity", "both"):
        vib_summary = summarize_continuity_stats(vib_stats_all)
        novib_summary = summarize_continuity_stats(novib_stats_all)
        print("\n=== CONTINUITY SUMMARY ===")
        print("VIB MODE:", vib_summary)
        print("NO VIB MODE:", novib_summary)

    if args.analysis in ("pr", "both"):
        print("\n=== PRECISION/RECALL SUMMARY ===")
        plot_pr_curve(precisions, recalls, precisions_nv, recalls_nv)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
