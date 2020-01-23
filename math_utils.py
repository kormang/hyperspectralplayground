# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from __future__ import print_function
from spectral import *
import numpy as np

def normalized(a, order=2, axis=-1):
    norms = np.atleast_1d(np.linalg.norm(a, order, axis))
    norms[norms == 0] = 1
    return a / np.expand_dims(norms, axis)

def image_of_modules(image):
    return np.sqrt(np.einsum('ijk,ijk->ij', image, image))

def cluster_histogram(centers, class_map):
    hist = np.empty((centers.shape[0],), dtype=int)
    for i in range(centers.shape[0]):
        hist[i] = np.count_nonzero(class_map == i)
    return hist
