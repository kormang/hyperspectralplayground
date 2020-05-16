# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from spectral import *
import numpy as np
import scipy

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

def replace_invalid(signature, value):
    """
        Replaces invalid values in signature, by value.
        Affects original signature, and also returns it.
    """
    signature[signature < 0.0] = value
    return signature

def interpolate_invalid(signature, kind='slinear'):
    full_xs = list(range(len(signature)))
    xs = [x for x, y in zip(full_xs, signature) if y > 0.0]
    ys = [y for y in signature if y > 0.0]
    if xs[0] > full_xs[0]:
        xs.insert(0, full_xs[0])
        ys.insert(0, ys[0])
    if xs[-1] < full_xs[-1]:
        xs.append(full_xs[-1])
        ys.append(ys[-1])
    f = scipy.interpolate.interp1d(xs, ys, kind=kind, assume_sorted=True)
    return f(full_xs)