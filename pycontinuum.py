# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
import numpy as np
import numba
from numba import prange
import scipy
import math

def continuum_points_original(spectrum, wavelengths, strict_range = None):
    strict_range = len(spectrum) if strict_range is None else strict_range
    indices = [0]
    indices_append = indices.append
    n = len(spectrum)
    i = 0 # Points to last point that belongs to the curve.
    j = 1 # Points to current potential point.
    m = n - 1
    while j < m:
        # Check all points in front of j,
        # to make sure it belongs to the curve.
        wli = wavelengths[i]
        spi = spectrum[i]
        qoef_j = (spectrum[j] - spi) / (wavelengths[j] - wli)
        cont_limit = min(j + 1 + strict_range, n)
        stopped_at_k = np.argmax(((spectrum[j+1: cont_limit] - spi) / (wavelengths[j+1: cont_limit] - wli)) > qoef_j) + j + 1

        # Need to check it because even if none is found, argmax returns 0.
        if ((spectrum[stopped_at_k] - spi) / (wavelengths[stopped_at_k] - wli)) <= qoef_j:
            # j belongs
            indices_append(j)

            # Last point that belongs is not j.
            i = j
            # Next, check j + 1
            j = j + 1
        else:
            # j does not belong.
            # Maybe k does, so it is next potential point.
            # We don't use j + 1, because we skip al the way to k,
            # since, all points between j and k are "below" j,
            # so must be "below" k too. "Below" means, j is above line
            # connecting them and i.
            j = stopped_at_k
    indices_append(j)
    return (wavelengths[indices], spectrum[indices])

def continuum_original(spectrum, wavelengths, strict_range = None):
    points = continuum_points_original(spectrum, wavelengths, strict_range)
    return interpolate_points(points, wavelengths)

def continuum_removed_original(spectrum, cont):
    return spectrum / cont

def interpolate_points(points, wavelengths, kind='linear'):
    """
        Points are 2-tuple, where element 0 is array of x values,
        and element 1 is array of y values.
    """
    f = scipy.interpolate.interp1d(points[0], points[1], kind=kind, assume_sorted=True)
    return f(wavelengths)

@numba.jit('int64(float64[:], float64, float64[:], float64)', nopython=True)
def _argmax_dot_product(wls, naxis_x, spectrum, naxis_y):
    n = len(wls)
    valmax = wls[0] * naxis_x + spectrum[0] * naxis_y
    imax = 0
    for i in prange(1, n):
        val = wls[i] * naxis_x + spectrum[i] * naxis_y
        if val > valmax:
            valmax = val
            imax = i

    return imax

@numba.jit('UniTuple(double[:], 2)(float64[:], float64[:], int64[:])', nopython=True)
def find_continuum_points_iterative(spectrum, wavelengths, indices):
    n = len(spectrum)
    indices[0] = 0
    ind_fill_i = 1
    stack = []
    stack_push = stack.append
    stack_pop = stack.pop

    ibegin = 0
    iend = n

    while True:
        iendi = iend - 1
        # Find normal to new axis. Swap x and y, and negate new x.
        # If we negate x instead of y, normal will always point upward.
        naxis_y = wavelengths[iendi] - wavelengths[ibegin]
        naxis_x = spectrum[ibegin] - spectrum[iendi]
#         imax = np.argmax(wavelengths[ibegin:iendi] * naxis_x + spectrum[ibegin:iendi] * naxis_y) + ibegin
        imax = _argmax_dot_product(wavelengths[ibegin:iendi], naxis_x, spectrum[ibegin:iendi], naxis_y) + ibegin

        if imax == ibegin:
            # In current range all values are below convex hull, so go back where we came from.
            if len(stack) == 0:
                break
            imax, ibegin, iend = stack_pop()
            # Save middle index.
            indices[ind_fill_i] = imax
            ind_fill_i += 1

        elif imax > ibegin + 1:
            # Check left side in next iteration.
            # Remember on stack where to continue.
            stack_push((imax, imax, iend))
            iend = imax + 1
        else:
            # Can't go left any more.
            # Save current middle.
            indices[ind_fill_i] = imax
            ind_fill_i += 1
            if imax < iend - 2:
                # We can still co right, prepare right side in next iteration.
                ibegin = imax
            else:
                # Can't go left, nor right.
                # Pop and go up.
                if len(stack) == 0:
                    break
                imax, ibegin, iend = stack_pop()
                # Save middle index.
                indices[ind_fill_i] = imax
                ind_fill_i += 1


    indices[ind_fill_i] = n-1
    indices = indices[:ind_fill_i+1]

    return (wavelengths[indices], spectrum[indices])

@numba.jit('int64(float64[:], float64[:], int64[:], int64, int64, int64)', nopython=True)
def _find_indices_in_range(spectrum, wavelengths, indices, ind_fill, ibegin, iend):
    iendi = iend - 1
    # Find normal to new axis. Swap x and y, and negate new x.
    # If we negate x instead of y, normal will always point upward.
    naxis_y = wavelengths[iendi] - wavelengths[ibegin]
    naxis_x = spectrum[ibegin] - spectrum[iendi]
#     imax = np.argmax(wavelengths[ibegin:iendi] * naxis_x + spectrum[ibegin:iendi] * naxis_y) + ibegin
    imax = _argmax_dot_product(wavelengths[ibegin:iendi], naxis_x, spectrum[ibegin:iendi], naxis_y) + ibegin

    if imax == ibegin:
        return ind_fill

    # Check left side.
    if imax > ibegin + 1:
        ind_fill = _find_indices_in_range(spectrum, wavelengths, indices, ind_fill, ibegin, imax + 1)

    # Push middle index.
    indices[ind_fill] = imax
    ind_fill += 1

    # Check right side.
    if imax < iend - 2:
        ind_fill =_find_indices_in_range(spectrum, wavelengths, indices, ind_fill, imax, iend)

    return ind_fill

@numba.jit('UniTuple(double[:], 2)(float64[:], float64[:], int64[:])', nopython=True)
def find_continuum_points_recursive(spectrum, wavelengths, indices):
    n = len(spectrum)
    indices[0] = 0
    ind_fill = 1

    ind_fill = _find_indices_in_range(spectrum, wavelengths, indices, ind_fill, 0, n)
    indices[ind_fill] = n - 1
    indices = indices[:ind_fill + 1]

    return (wavelengths[indices], spectrum[indices])

@numba.jit('void(float64[:], float64[:], float64[:])', nopython=True)
def compute_1d_continuum(data, wavelengths, out):
    indices = np.empty(data.shape[-1], np.int64)
    points = find_continuum_points_recursive(data, wavelengths, indices)
    out[:] = np.interp(wavelengths, points[0], points[1])

@numba.jit('void(float64[:,:], float64[:], float64[:,:])', nopython=True)
def compute_2d_continuums(data, wavelengths, out):
    interp = np.interp
    n = data.shape[-1]
    indices = np.empty(n, np.int64)

    for i in prange(data.shape[0]):
        points = find_continuum_points_recursive(data[i], wavelengths, indices)
        out[i, :] = interp(wavelengths, points[0], points[1])

@numba.jit('void(float64[:,:,:], float64[:], float64[:,:,:])', nopython=True, parallel=True)
def compute_3d_continuums(data, wavelengths, out):
    interp = np.interp
    n = data.shape[-1]

    for i in prange(data.shape[0]):
        indices = np.empty(n, np.int64)
        for j in range(data.shape[1]):
            points = find_continuum_points_recursive(data[i, j], wavelengths, indices)
            out[i, j, :] = interp(wavelengths, points[0], points[1])


def continuum_points(spectrum, wavelengths, indices = None):
    indices = np.empty_like(spectrum, dtype='int64') if indices is None else indices
    return find_continuum_points_recursive(spectrum, wavelengths, indices)


def continuum(data, wavelengths, out = None):
    if len(data.shape) == 1:
        out = np.empty_like(data) if out is None else out
        compute_1d_continuum(data, wavelengths, out)
    elif len(data.shape) == 2:
        compute_2d_continuums(data, wavelengths, out)
    elif len(data.shape) == 3:
        compute_3d_continuums(data, wavelengths, out)
    return out


def continuum_removed(spectra, wavelengths, out = None):
    out = continuum(spectra, wavelengths, out)
    np.divide(spectra, out, out=out)
    return out