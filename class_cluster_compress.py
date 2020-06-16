from __future__ import print_function
import numpy as np
from utils import *

def compute_scale_map(data, class_map, centers):
    data_double = data.astype(float)
    modules = np.sqrt(np.einsum('ijk,ijk->ij', data_double, data_double))

    # compute modules of centers
    center_modules = np.sqrt(np.einsum('ij,ij->i', centers, centers))

    # compute scale factor map
    scale_map = np.empty_like(modules)
    for i in range(scale_map.shape[0]):
        for j in range(scale_map.shape[1]):
            scale_map[i, j] = modules[i, j] / center_modules[class_map[i, j]]

    return scale_map


def reconstruct_image(scale_map, class_map, centers):
    image_shape = class_map.shape + (centers.shape[1],)
    reconstruction = np.empty(image_shape)
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            reconstruction[i, j] = centers[class_map[i, j]] * scale_map[i, j]

    return reconstruction

def get_avg_diff(data, reconstruction):
    diffs = data - reconstruction
    abs_diffs = np.abs(diffs)
    as1darray = abs_diffs.reshape((diffs.shape[0] * diffs.shape[1] * diffs.shape[2],))
    return np.average(as1darray)

def print_diff_stats(data, reconstruction):
    diffs = data - reconstruction
    abs_diffs = np.abs(diffs)

    diffs_norm = np.linalg.norm(diffs)
    print('Norm: ', diffs_norm)
    as1darray = abs_diffs.reshape((diffs.shape[0] * diffs.shape[1] * diffs.shape[2],))
    print('Max: ', np.max(as1darray))
    print('Min: ', np.min(as1darray))
    print('Avg: ', np.average(as1darray))

