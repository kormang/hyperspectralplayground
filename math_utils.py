# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from __future__ import print_function
from spectral import *
import numpy as np
import math

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

def find_max_diff_l2(sigs, from_sig):
    diff = sigs - from_sig
    distances = np.einsum('ij,ij->i', diff, diff, optimize='optimal')
    index_of_max = np.argmax(distances)
    return sigs[index_of_max]

def find_max_diff_l1(sigs, from_sig):
    distances = np.sum(np.abs(sigs - from_sig), axis=1)
    index_of_max = np.argmax(distances)
    return sigs[index_of_max]

def find_max_diff_angle(sigs, from_sig):
    norm_sigs = normalized(sigs)
    norm_from_sig = from_sig / np.linalg.norm(from_sig)
    cosines = np.dot(norm_sigs, norm_from_sig)
    index_of_max = np.argmin(cosines)
    return sigs[index_of_max]

def find_with_max_norm2(sigs):
    norms = np.linalg.norm(sigs, 2, 1)
    index_of_max = np.argmax(norms)
    return sigs[index_of_max]

def find_with_max_norm1(sigs):
    norms = np.sum(np.abs(sigs), axis=1)
    index_of_max = np.argmax(norms)
    return sigs[index_of_max]

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

#### 2nd degree bsplines

def bspline2t_x(x, p0, p1, p2):
    a = p0[0] - 2*p1[0] + p2[0]
    b = -2*p0[0] + 2*p1[0]
    c = p0[0] + p1[0] - 2*x
    
    detr = b*b - 4*a*c
    
    if detr < 0:
        print(x, p0, p1, p2, a, b, c, detr)
    
    assert (detr >= 0), 'Failed to calculate bspline coordinate - determinant < 0!'
    
    result = (-b + math.sqrt(detr)) / (2*a)
    if result < 0 or result > 1:
        result = (-b - math.sqrt(detr)) / (2*a)
    
    if (result < 0 or result > 1):
        print(x, p0, p1, p2, a, b, c, detr)
        print(((-b + math.sqrt(detr)) / (2*a)), (-b - math.sqrt(detr)) / (2*a))
        return 0.0
    
    assert (result >= 0 and result <= 1), 'Failed to calculate bspline coordinate - all solutions out of [0, 1]'
    return result

def bspline2y_t(t, p0, p1, p2):
    q0 = (1 - t)*(1 - t)
    q1 = -2*t*t + 2*t + 1
    q2 = t*t
    y = (q0*p0[1] + q1*p1[1] + q2*p2[1])/2
    return y

def bspline2y_x(x, p0, p1, p2):
    t = bspline2t_x(x, p0, p1, p2)
    return bspline2y_t(t, p0, p1, p2)

def bspline2x_t(t, p0, p1, p2):
    q0 = (1 - t)*(1 - t)
    q1 = -2*t*t + 2*t + 1
    q2 = t*t
    x = (q0*p0[0] + q1*p1[0] + q2*p2[0])/2
    return x

def bspline2_get_points_on_segment(s, points):
    plen = len(points)
    p0i = clamp(s - 1, 0, plen - 1)
    p1i = clamp(s, 0, plen - 1)
    p2i = clamp(s + 1, 0, plen - 1)
    return points[p0i], points[p1i], points[p2i]

def bspline2_from_points(points):
    """
        Internally highly dependens on result of pycontinuum_points,
        e.g. there should be results[0][0] == 0.
        Also it gives x coordinates always as integers,
        and assumes there is not two stitching points between integers values.
    """
    segments = len(points)
    
    # Calculate points where segments stitch together.
    # Also includes first point and last point.
    x_stitching_points = [0]
    for s in range(segments):
        p0, p1, p2 = bspline2_get_points_on_segment(s, points)
        x_stitching_points.append(bspline2x_t(1.0, p0, p1, p2))
    
    # Fill results with pairs (i, y(i)),
    # where i is x coordinate and also index in signature array,
    # and y(i) is respective  y coordinate of bspline2.
    results = []
    for s in range(segments):
        p0, p1, p2 = bspline2_get_points_on_segment(s, points)
        # We need special case for the first element,
        # as we need integer x in results, but first element could
        # be rounded down to x point that is outside this segment.
        # That leads to errors in equations, where we get complex solutions.
        x = x_stitching_points[s]
        results.append(bspline2y_x(x, p0, p1, p2))
        #results[int(x)] = bspline2y_x(x, p0, p1, p2)
        for x in range(int(x_stitching_points[s]) + 1, int(x_stitching_points[s + 1])):
            #results[x] = bspline2y_x(x, p0, p1, p2)
            results.append(bspline2y_x(x, p0, p1, p2))
    
    # Last stitching (accually not stitching, since it is the last one of stitching points)
    # of the last segment is not included because of non-inclusive ranges.
    # We still need it.
    p0, p1, p2 = bspline2_get_points_on_segment(segments - 1, points)
    x = x_stitching_points[-1]
    #results[int(x)] = bspline2y_x(x, p0, p1, p2)
    results.append(bspline2y_x(x, p0, p1, p2))

    return np.array(results)