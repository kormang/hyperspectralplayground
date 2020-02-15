# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from __future__ import print_function
from spectral import *
import numpy as np
import random
from math_utils import normalized

def find_related_clusters2(image, min_correlation, **kwargs):
    from spectral.algorithms.spymath import has_nan, NaNValueError

    if has_nan(image):
        raise NaNValueError('Image data contains NaN values.')

    start_centers = None

    for (key, val) in list(kwargs.items()):
        if key == 'start_centers':
            start_centers = normalized(val)
        else:
            raise NameError('Unsupported keyword argument.')

    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    image = normalized(image.reshape((N, nbands)))
    clusters = np.zeros((N,), int) - 1
    MAX_CENTERS = 65536
    centers = np.zeros((nbands, MAX_CENTERS))
    num_centers = 0
    if start_centers is not None:
        centers[:, :start_centers.shape[0]] = start_centers.T
        num_centers = start_centers.shape[0]
    else:
        centers[:, 0] = image[0]
        num_centers = 1

    percentage = 0.0
    max_exceeded_warning_printed = False
    for i in range(N):
        match_index = np.argmax(np.matmul(image[i], centers[:, :num_centers]))
        if np.dot(image[i], centers[:, match_index]) < min_correlation:
            if num_centers < MAX_CENTERS:
                clusters[i] = num_centers
                centers[:, num_centers] = image[i]
                num_centers += 1
            else:
                if not max_exceeded_warning_printed:
                    print('Exceeded max number of centers, pixels will be assigned to best existing match. Try with lower coefficient.')
                    max_exceeded_warning_printed = True
                clusters[i] = match_index
        else:
            clusters[i] = match_index

        if float(i)/N >= percentage + 0.01:
            percentage = float(i)/N
            print('\r%d%% completed' % int(percentage * 100), end='')
            sys.stdout.flush()

    print('')
    return (clusters.reshape(nrows, ncols), centers[:, :num_centers].T.copy())


def find_mincorr_centers(values, centers):
    minci = np.argmin(np.sum(np.matmul(values, centers), axis=1))
    return values[minci], minci

def find_mincorr3(values, centers = None, reduce_coef = None):
    if reduce_coef is not None and values.shape[0] > reduce_coef:
        allinds = np.arange(values.shape[0])
        np.random.shuffle(allinds)
        values = values[allinds[:values.shape[0]//reduce_coef], :]
    if centers is None:
        centers = values.T
    index = 0
    minsumcos = float('inf')
    percentage = 0.0
    N = values.shape[0]

    for i in range(N):
        sumcos = np.sum(np.matmul(values[i][np.newaxis], centers))
        if sumcos < minsumcos:
            minsumcos = sumcos
            index = i

        # if float(i)/N >= percentage + 0.01:
        #     percentage = float(i)/N
        #     print('\r%d%% finding min corr completed' % int(percentage * 100), end='')
        #     sys.stdout.flush()

    # print(' ')

    # index = np.argmin(np.sum(np.matmul(values, values.T), axis=1))
    # index = np.argmin(np.einsum('ij,jk->k', values, values.T, optimize='optimal'))

    return values[index]

def find_maxdist_from_center(values, centers):
    avg = np.average(values if centers is None else centers, axis=0)
    diff = values - avg
    distances = np.einsum('ij,ij->i', diff, diff, optimize='optimal')
    index_of_max = np.argmax(distances)
    return values[index_of_max]

def find_mincorr_from_center(values, centers):
    """
        Returns vector that is least correlated from center of centers or values (if centers are None).
        Values are of shape (N, c) where c is number of channel or dimensionality of each vector.
        Values are expected to be normalized.
    """
    avg = np.average(values if centers is None else centers, axis=0)
    corrs = np.matmul(values, avg)
    index_of_mин = np.argmin(corrs)
    return values[index_of_mин]

def find_maxdist_clusters(image, min_correlation):
    from spectral.algorithms.spymath import has_nan, NaNValueError

    if has_nan(image):
        raise NaNValueError('Image data contains NaN values.')

    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    values = normalized(image.reshape((N, nbands)))
    #values = image.copy()
    clusters = np.zeros((N,), int) - 1
    MAX_CENTERS = 65536
    centers = np.zeros((nbands, MAX_CENTERS))
    num_centers = 0
    #avg = np.average(image, axis=0)
    #values -= avg
    #centers[:, 0], minci = find_mincorr2(values)
    #values = np.delete(values, minci, axis=0)
    #newcenter = find_mincorr3(values, None, 10)
    newcenter = find_mincorr_from_center(values)
    centers[:, num_centers] = newcenter
    num_centers += 1
    corrs_with_newcenter = np.matmul(values, newcenter)
    high_corr_inds = np.argwhere(corrs_with_newcenter >= min_correlation)
    values = np.delete(values, high_corr_inds, axis=0)

    while values.size > 0 and num_centers < MAX_CENTERS:
        newcenter = find_mincorr_from_center(values, centers[:, :num_centers])
        centers[:, num_centers] = newcenter
        num_centers += 1
        corrs_with_newcenter = np.matmul(values, newcenter)
        high_corr_inds = np.argwhere(corrs_with_newcenter >= min_correlation)
        values = np.delete(values, high_corr_inds, axis=0)

        print('%d centers found, left %d         ' % (num_centers, values.shape[0]), end='\r')
        sys.stdout.flush()

    print('assigning values...')

    centers = centers[:, :num_centers]
    print(centers)
    clusters = np.argmax(np.matmul(image, centers), 1)
    print('done')

    return (clusters.reshape(nrows, ncols), centers.T.copy())