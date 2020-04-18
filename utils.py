# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from spectral import *
import numpy as np

def filter_out_invalid_signatures(image):
    """
        Transforms 3D array into 2D array,
        where each row represents one signature.
        Only signatures that do not contain negative values are left.
        Also type is transformed into float32.
    """
    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    sigs = image.reshape((N, nbands))
    invalid_inds = np.argwhere(np.any(sigs < 0, axis=1))
    sigs = np.delete(sigs, invalid_inds, axis=0).astype(float)
    return sigs

def find_image_coords_of_signature(image, sig):
    (nrows, ncols, nbands) = image.shape
    for i in range(nrows):
        for j in range(ncols):
            if np.array_equal(image[i, j], sig):
                return (i, j)
    
    return None
