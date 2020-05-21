# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
import numpy as np
import scipy
import math

# In python, we don't compute continuum on many spectra.
# That would be too slow. And also it is less important.
# In python, we compute single continuum and remove it from all spectra.
# That single continuum should be computed on spectrum that potentially
# represents max values in each channel in respect to atmospheric absorption.
# But how to determine such spectrum remains a mistery.

def pycontinuum_removed(spectrum, continuum, removed = None):
    """Be careful, it is not perfomant, """
    """use this function only for testing purposes."""
    if removed is None:
        removed = np.empty_like(spectrum)
    np.divide(spectrum, continuum, out=removed)
    return removed

def pycontinuum(spectrum, wavelengths, continuum = None):
    if continuum is None:
        continuum = np.empty_like(spectrum)
    continuum[0] = spectrum[0]
    n = len(spectrum)
    i = 0 # Points to last point that belongs to the curve.
    j = 1 # Points to current potential point.
    while j < n:
        # Check all points in front of j,
        # to make sure it belongs to the curve.
        k = j + 1
        while k < n:
            qoef = (spectrum[k] - spectrum[i]) / (wavelengths[k] - wavelengths[i])
            intersection = qoef * (wavelengths[j] - wavelengths[i]) + spectrum[i]
            if spectrum[j] < intersection:
                break # J does not belong.
            k += 1

        if k == n:
            # j belongs.

            # We need to fill the values in continuum between i and j.
            qoef = (spectrum[j] - spectrum[i]) / (wavelengths[j] - wavelengths[i])
            for t in range(i + 1, j):
                continuum[t] = qoef * (wavelengths[t] - wavelengths[i]) + spectrum[i]

            # For j, fill exact value.
            continuum[j] = spectrum[j]

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
            j = k
    return continuum


def pypartial_continuum(spectrum, wavelengths, strict_range, continuum = None):
    if continuum is None:
        continuum = np.empty_like(spectrum)
    continuum[0] = spectrum[0]
    n = len(spectrum)
    i = 0 # Points to last point that belongs to the curve.
    j = 1 # Points to current potential point.
    while j < n:
        # Check all points in front of j,
        # to make sure it belongs to the curve.
        k = j + 1
        cont_limit = min(k + strict_range, n)
        while k < cont_limit:
            qoef = (spectrum[k] - spectrum[i]) / (wavelengths[k] - wavelengths[i])
            intersection = qoef * (wavelengths[j] - wavelengths[i]) + spectrum[i]
            if spectrum[j] < intersection:
                break # J does not belong.
            k += 1

        if k == cont_limit:
            # j belongs.

            # We need to fill the values in continuum between i and j.
            qoef = (spectrum[j] - spectrum[i]) / (wavelengths[j] - wavelengths[i])
            for t in range(i + 1, j):
                continuum[t] = qoef * (wavelengths[t] - wavelengths[i]) + spectrum[i]

            # For j, fill exact value.
            continuum[j] = spectrum[j]

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
            j = k
    return continuum


## NOTE: using points and interpolation, or other methods we can construct any kind of continuum we like.
## Leave above function as reference and for backward compatibility, but further develop only points-based functions.

def pypartial_continuum_points(spectrum, wavelengths, strict_range):
    points = [(wavelengths[0], spectrum[0])]
    n = len(spectrum)
    i = 0 # Points to last point that belongs to the curve.
    j = 1 # Points to current potential point.
    while j < n:
        # Check all points in front of j,
        # to make sure it belongs to the curve.
        k = j + 1
        cont_limit = min(k + strict_range, n)
        while k < cont_limit:
            qoef = (spectrum[k] - spectrum[i]) / (wavelengths[k] - wavelengths[i])
            intersection = qoef * (wavelengths[j] - wavelengths[i]) + spectrum[i]
            if spectrum[j] < intersection:
                break # J does not belong.
            k += 1

        if k == cont_limit:
            # j belongs.
            points.append((wavelengths[j], spectrum[j]))

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
            j = k
    return points


def pycontinuum_points(spectrum, wavelengths):
    """
        Returns list of points (i, spectrum[i]) that belong to continuum.
    """
    return pypartial_continuum_points(spectrum, wavelengths, len(spectrum))

def pypartial_continuum_points_angle(spectrum, wavelengths, tolerance_angle):
    """
        Calculates partial continuum, but checks angles instead of intersection.
        That enables us to use parameter tolerance_angle, to loosen strictness of
        convexity. Parameter tolerance_angle is given in radians, and even if some points goes
        below convexity line, at is still candidate if it is not more than tolerance_angle.
    """
    points = [(wavelengths[0], spectrum[0])]
    n = len(spectrum)
    i = 0 # Points to last point that belongs to the curve.
    j = 1 # Points to current potential point.
    while j < n:
        # Check all points in front of j,
        # to make sure it belongs to the curve.
        k = j + 1
        while k < n:
            qoef_k = (spectrum[k] - spectrum[i]) / (wavelengths[k] - wavelengths[i])
            qoef_j = (spectrum[j] - spectrum[i]) / (wavelengths[j] - wavelengths[i])
            if qoef_k > qoef_j:
                # J might not belong, but lets check if we can tolerate that.
                if math.atan(qoef_k) - math.atan(qoef_j) > tolerance_angle:
                    break # J does not belong for sure.
            k += 1

        if k == n:
            # j belongs.
            points.append((wavelengths[j], spectrum[j]))

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
            j = k
    return points

def interpolate_points(points, wavelengths, kind='linear'):
    xp = [x for x, _ in points]
    yp = [y for _, y in points]
    f = scipy.interpolate.interp1d(xp, yp, kind=kind, assume_sorted=True)
    return f(wavelengths)
