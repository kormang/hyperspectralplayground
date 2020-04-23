# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
import numpy as np

# In python, we don't compute continuum on many signatures.
# That would be too slow. And also it is less important.
# In python, we compute single continuum and remove it from all signatures.
# That single continuum should be computed on signature that potentially
# represents max values in each channel in respect to atmospheric absorption.
# But how to determine such signature remains a mistery.

def pycontinuum(signature, continuum = None):
    if continuum is None:
        continuum = np.empty_like(signature)
    continuum[0] = signature[0]
    n = len(signature)
    i = 0 # Points to last point that belongs to the curve.
    j = 1 # Points to current potential point.
    while j < n:
        # Check all points in front of j,
        # to make sure it belongs to the curve.
        k = j + 1
        while k < n:
            qoef = (signature[k] - signature[i]) / (k - i)
            intersection = qoef * (j - i) + signature[i]
            if signature[j] < intersection:
                break # J does not belong.
            k += 1
        
        if k == n:
            # j belongs.
            
            # We need to fill the values in continuum between i and j.
            qoef = (signature[j] - signature[i]) / (j - i)
            for t in range(i + 1, j):
                continuum[t] = qoef * (t - i) + signature[i]
            
            # For j, fill exact value.
            continuum[j] = signature[j]
            
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

def pycontinuum_removed(signature, continuum, removed = None):
    """Be careful, it is not perfomant, """
    """use this function only for testing purposes."""
    if removed is None:
        removed = np.empty_like(signature)
    np.divide(signature, continuum, out=removed)
    return removed

def pypartial_continuum(signature, strict_range, continuum = None):
    if continuum is None:
        continuum = np.empty_like(signature)
    continuum[0] = signature[0]
    n = len(signature)
    i = 0 # Points to last point that belongs to the curve.
    j = 1 # Points to current potential point.
    while j < n:
        # Check all points in front of j,
        # to make sure it belongs to the curve.
        k = j + 1
        cont_limit = min(k + strict_range, n)
        while k < cont_limit:
            qoef = (signature[k] - signature[i]) / (k - i)
            intersection = qoef * (j - i) + signature[i]
            if signature[j] < intersection:
                break # J does not belong.
            k += 1
        
        if k == cont_limit:
            # j belongs.
            
            # We need to fill the values in continuum between i and j.
            qoef = (signature[j] - signature[i]) / (j - i)
            for t in range(i + 1, j):
                continuum[t] = qoef * (t - i) + signature[i]
            
            # For j, fill exact value.
            continuum[j] = signature[j]
            
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

def pycontinuum_points(signature):
    """
        Returns list of points (i, signature[i]) that belong to continuum.
    """
    points = [(0, signature[0])]
    n = len(signature)
    i = 0 # Points to last point that belongs to the curve.
    j = 1 # Points to current potential point.
    while j < n:
        # Check all points in front of j,
        # to make sure it belongs to the curve.
        k = j + 1
        while k < n:
            qoef = (signature[k] - signature[i]) / (k - i)
            intersection = qoef * (j - i) + signature[i]
            if signature[j] < intersection:
                break # J does not belong.
            k += 1
        
        if k == n:
            # j belongs.
            points.append((j, signature[j]))
            
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