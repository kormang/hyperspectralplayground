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
        Also type is transformed into float.
    """
    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    spectra = image.reshape((N, nbands))
    invalid_inds = np.argwhere(np.any(spectra < 0, axis=1))
    spectra = np.delete(spectra, invalid_inds, axis=0).astype(float)
    return spectra

def filter_out_of_range_spectra(image, min_val, max_val):
    """
        Transforms 3D array into 2D array,
        where each row represents one spectrum.
        Only spectra that do not contain values below min_val or above max_val are left.
        Also type is transformed into float.
    """
    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    spectra = image.reshape((N, nbands))
    invalid_inds = np.argwhere(np.any(np.logical_or(spectra < min_val, spectra > max_val), axis=1))
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


def resample_at(spectrum, src_wls, dest_wls, src_bw = None, dest_bw = None):
    """
        Returns spectrum resampled to different wavelengths and bandwidths.
    """
    resampler = BandResampler(src_wls, dest_wls, src_bw, dest_bw)
    return resampler(spectrum)

def interpolate_at(spectrum, src_wls, dest_wls, kind='quadratic'):
    """
        Returns spectrum interpoleted at specified wavelengths,
        based on wavelengths and reflectances of original.
    """
    f = scipy.interpolate.interp1d(src_wls, spectrum, kind=kind, assume_sorted=True, fill_value='extrapolate')
    return f(dest_wls)

def cut_range(spectrum, src_wls, dst_wls):
    """
        Return wavelengths and spectrum part between min wavelength and max wavelength.
        Min wavelength is max(src_wls[0], dst_wls[0]).
        Max wavelengths is min(src_wls[-1], dst_wls[-1]).
        dst_wls can be ndarray, or simply 2-tuple with max and min wavelengths.
    """
    min_wl = max(src_wls[0], dst_wls[0])
    max_wl = min(src_wls[-1], dst_wls[-1])
    min_index = np.argmax(src_wls >= min_wl)
    max_index = -1 if src_wls[-1] <= max_wl else np.argmax(src_wls > max_wl)
    range_wl = src_wls[min_index:max_index]
    range_spec = spectrum[min_index:max_index]
    return range_wl, range_spec
