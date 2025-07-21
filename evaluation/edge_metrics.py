#!/usr/bin/env python

# Author: Igor Topcin <topcin@ime.usp.br>
# PTC5892 Processamento de Imagens Medicas
# POLI - University of Sao Paulo

# Implementation of the 

# References:
# [1] W. K. Pratt, Digital Image Processing. New York: Wiley, 1977
# [2] Y. Yu and S. T. Acton, Speckle Reducing Anisotropic Diffusion.
# IEEE Transactions on Image Processing, Vol. 11, No. 11, 2002

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import f1_score
from glob import glob
import cv2 as cv
from tqdm import tqdm



DEFAULT_ALPHA = 1.0 / 9

def fom(edges, edges_gt, alpha = DEFAULT_ALPHA):
    """
    Computes Pratt's Figure of Merit for the given image img, using a gold
    standard image as source of the ideal edge pixels.
    """

    # To avoid oversmoothing, we apply canny edge detection with very low
    # standard deviation of the Gaussian kernel (sigma = 0.1).
    
    # Compute the distance transform for the gold standard image.
    # removes third dimension from each
    #edges = edges[:,:,0]
    #edges_gt = edges_gt[:,:,0]
    dist = distance_transform_edt(np.invert(edges_gt))

    fom = 1.0 / np.maximum(
        np.count_nonzero(edges),
        np.count_nonzero(edges_gt))

    N, M = edges.shape

    for i in range(0, N):
        for j in range(0, M):
            if edges[i, j]:
                fom += 1.0 / ( 1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(
        np.count_nonzero(edges),
        np.count_nonzero(edges_gt))    

    return fom

def f_score(edges,edges_gt):
    #edges = edges[:,:,0]
    #edges_gt = edges_gt[:,:,0]
    return f1_score(edges,edges_gt,average='micro')

def calculate_dataset_metrics(gt_directory,
                              pred_directory,
                              metric = 'fom'):
    """
    Calculates desired metric over a dataset

    Assumes the directories point to a directory with all the given GT/predicted edge images.

    """
    # loads and sorts GT images
    gt_images = sorted(glob(gt_directory + '/*.png'))

    # loads and sorts edge images
    pred_images = sorted(glob(pred_directory + '/*.png'))
    # pred_images = sorted(glob(pred_directory + '/*.png'),key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # "skips" intermediate edges if one is longer than the other
    gt_len = len(gt_images)
    pred_len = len(pred_images)
    if gt_len != pred_len:
        indices = np.linspace(start=0,stop=max(gt_len,pred_len)-1,num=min(gt_len,pred_len)).astype(int)
    if gt_len < pred_len:
        #pred_images = pred_images[indices]
        pred_images = [pred_images[i] for i in indices]


    elif pred_len < gt_len:
        #gt_images = gt_images[indices]
        gt_images = [gt_images[i] for i in indices]

    

    metric_average = 0
    for gt_image_dir, pred_image_dir in tqdm(zip(gt_images,pred_images)):
        # loads images
        gt_image = cv.imread(gt_image_dir, cv.IMREAD_GRAYSCALE) > 100 
        pred_image = cv.imread(pred_image_dir, cv.IMREAD_GRAYSCALE) > 100 
        cv.imshow('Pred', pred_image.astype(np.uint8) * 255)
        cv.waitKey(0)
        metric_val = 0
        if metric == 'fom':
            metric_val += fom(pred_image,gt_image)
        elif metric == 'f_score':
            metric_val += f_score(pred_image,gt_image)
        metric_average += metric_val
    metric_average = metric_average/min(gt_len,pred_len)
    return metric_average


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Calculate edge metrics for a dataset.')
    parser.add_argument('gt_directory', type=str, help='Directory containing ground truth edge images.')
    parser.add_argument('pred_directory', type=str, help='Directory containing predicted edge images.')
    parser.add_argument('--metric', type=str, choices=['fom', 'f_score'], default='fom', help='Metric to calculate.')

    args = parser.parse_args()

    metric_value = calculate_dataset_metrics(args.gt_directory, args.pred_directory, args.metric)
    print(f'Metric ({args.metric}) value: {metric_value}')

if __name__ == '__main__':
    main()