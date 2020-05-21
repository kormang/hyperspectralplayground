# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from spectral import *
import numpy as np
import scipy

def filter_out_invalid_spectra(image):
    """
        Transforms 3D array into 2D array,
        where each row represents one spectrum.
        Only spectra that do not contain negative values are left.
        Also type is transformed into float32.
    """
    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    spectra = image.reshape((N, nbands))
    invalid_inds = np.argwhere(np.any(spectra < 0, axis=1))
    spectra = np.delete(spectra, invalid_inds, axis=0).astype(float)
    return spectra

def find_image_coords_of_spectrum(image, spectrum):
    (nrows, ncols, nbands) = image.shape
    for i in range(nrows):
        for j in range(ncols):
            if np.array_equal(image[i, j], spectrum):
                return (i, j)

    return None

def replace_invalid(spectrum, value):
    """
        Replaces invalid values in spectrum, by value.
        Affects original spectrum, and also returns it.
    """
    spectrum[spectrum < 0.0] = value
    return spectrum

def interpolate_invalid(spectrum, kind='slinear'):
    full_xs = list(range(len(spectrum)))
    xs = [x for x, y in zip(full_xs, spectrum) if y > 0.0]
    ys = [y for y in spectrum if y > 0.0]
    if xs[0] > full_xs[0]:
        xs.insert(0, full_xs[0])
        ys.insert(0, ys[0])
    if xs[-1] < full_xs[-1]:
        xs.append(full_xs[-1])
        ys.append(ys[-1])
    f = scipy.interpolate.interp1d(xs, ys, kind=kind, assume_sorted=True)
    return f(full_xs)

def lower_bound(array, min_val):
    ret = np.argmax(array >= min_val)
    return ret if array[ret] >= min_val else len(array)

def upper_bound(array, max_val):
    ret = np.argmax(array > max_val)
    return ret if array[ret] > max_val else len(array)