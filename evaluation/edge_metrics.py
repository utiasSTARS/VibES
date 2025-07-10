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
    edges = edges[:,:,0]
    edges_gt = edges_gt[:,:,0]
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
    edges = edges[:,:,0]
    edges_gt = edges_gt[:,:,0]
    return f1_score(edges,edges_gt,average='micro')