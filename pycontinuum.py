# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
import numpy as np
import scipy
import math

# In python, we don't compute continuum on many spectra.
# That would be too slow.

def continuum_removed(spectrum, continuum, removed = None):
    """Be careful, it is not perfomant, """
    """use this function only for testing purposes."""
    if removed is None:
        removed = np.empty_like(spectrum)
    np.divide(spectrum, continuum, out=removed)
    return removed

def continuum_points(spectrum, wavelengths, strict_range = None):
    strict_range = len(spectrum) if strict_range is None else strict_range
    indices = [0]
    indices_append = indices.append
    n = len(spectrum)
    i = 0 # Points to last point that belongs to the curve.
    j = 1 # Points to current potential point.
    while j < n:
        # Check all points in front of j,
        # to make sure it belongs to the curve.
        wli = wavelengths[i]
        spi = spectrum[i]
        qoef_j = (spectrum[j] - spi) / (wavelengths[j] - wli)
        stopped_at_k = -1
        cont_limit = min(j + 1 + strict_range, n)
        for k in range(j + 1, cont_limit):
            qoef_k = (spectrum[k] - spi) / (wavelengths[k] - wli)
            if qoef_j < qoef_k:
                stopped_at_k = k
                break

        if stopped_at_k == -1:
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
    return (wavelengths[indices], spectrum[indices])

def interpolate_points(points, wavelengths, kind='linear'):
    """
        Points are 2-tuple, where element 0 is array of x values,
        and element 1 is array of y values.
    """
    f = scipy.interpolate.interp1d(points[0], points[1], kind=kind, assume_sorted=True)
    return f(wavelengths)

def continuum(data, wavelengths, strict_range = None, out_data = None):
    interp = scipy.interpolate.interp1d
    strict_range = data.shape[-1] if strict_range is None else strict_range

    if len(data.shape) == 1:
        points = continuum_points(data, wavelengths, strict_range)
        f = interp(points[0], points[1], assume_sorted=True)
        result = f(wavelengths)
        if out_data is not None:
            out_data[i, :]
        return result
    elif len(data.shape) == 2:
        for i in range(data.shape[0]):
            points = continuum_points(data[i], wavelengths, strict_range)
            f = interp(points[0], points[1], assume_sorted=True)
            out_data[i, :] = f(wavelengths)
    elif len(data.shape) == 3:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                points = continuum_points(data[i, j], wavelengths, strict_range)
                f = interp(points[0], points[1], assume_sorted=True)
                out_data[i, j, :] = f(wavelengths)



def find_continuum_points(wavelengths, spectrum):
    indices = [0]
    indices_append = indices.append

    n = len(spectrum)
    rotated_ys = np.empty(n)

    def find_indices_in_range(ibegin, iend):
        #print('finding between ' + str(ibegin) + ' and ' + str(iend))
        sp = np.array((wavelengths[ibegin], spectrum[ibegin]))
        ep = np.array((wavelengths[iend-1], spectrum[iend-1]))
        new_x_axis = ep - sp
        axis_vector_length = np.sqrt(np.dot(new_x_axis, new_x_axis))
        cos_theta = new_x_axis[0] / axis_vector_length
        # -sin(theta), minus because of clockewise rotation.
        sin_theta = -new_x_axis[1] / axis_vector_length
        rotated_ys[ibegin:iend] = wavelengths[ibegin:iend] * sin_theta + spectrum[ibegin:iend] * cos_theta
        imax = np.argmax(rotated_ys[ibegin+1:iend-1]) + ibegin + 1

        if rotated_ys[imax] - rotated_ys[ibegin] <= 0:
            return

        # Check left side.
        if imax > ibegin + 1:
            find_indices_in_range(ibegin, imax + 1)

        # Push middle index.
        indices_append(imax)

        # Check right side.
        if imax < iend - 2:
            find_indices_in_range(imax, iend)

    find_indices_in_range(0, n)
    indices.append(n-1)

    return (wavelengths[indices], spectrum[indices])
