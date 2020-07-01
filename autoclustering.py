# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from __future__ import print_function
from spectral import *
import numpy as np
import random
from math_utils import normalized

def find_related_clusters2(image, min_correlation, **kwargs):
    """
        Primitive algorithm that scans image left to right, top to buttom,
        and assigns pixels/spectra to clusters. When, during scaning it finds
        pixel that is less than min_correlation correlated with all already found centers,
        algorithm assigns that pixel to new cluster, and that pixel is also new center.
        if kwarg start_centers is not present, then starting center is first pixel.
        Which means, centers are not really centers, but starting pixels of each
        cluster. However, these centers are good potetial starting centers
        to apply K-Means.
        Returns pair of (map of cluster indices, centers).
        Centers have shape (num_centers, spectrum_length).
    """
    from spectral.utilities.errors import has_nan, NaNValueError

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
    """
        Finds value that is least correleted with all the centers,
        that is, value that has smallest sum of cosines between itself
        and each of the centers.
        Values and centers should be normalized.
        Values are of shape (N, spectrum_length), and centers (spec_len, N).
        Returns pair of value and its index.
    """
    minci = np.argmin(np.sum(np.matmul(values, centers), axis=1))
    return values[minci], minci

def find_mincorr3(values, centers = None, reduce_coef = None):
    """
        Like find_mincorr, but with few notable differences.
        If there is reduce_coef and number of values is greater than that,
        values are transformed (locally, does not effect caller),
        so that their number is reduced by the factor of reduce_ceof.
        In other words, there will be reduce_ceof times less values
        than originally provided, and they will be taken at random.
        This is for perfomance reasons.
        If centers are not provided they will be transpose of values.
    """
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
    """
        Returns vector that is furthest away (by L2) from center of centers or values (if centers are None).
        Values are of shape (N, c) where c is number of channel or dimensionality of each vector.
        Values are expected to be normalized.
    """
    avg = np.average(values if centers is None else centers, axis=0)
    diff = values - avg
    distances = np.einsum('ij,ij->i', diff, diff, optimize='optimal')
    index_of_max = np.argmax(distances)
    return values[index_of_max]

def find_mincorr_from_center(values, centers = None):
    """
        Returns value that is least correlated from center of centers or values (if centers are None).
        Values are of shape (N, c) where c is number of channel or dimensionality of each vector.
        Values are expected to be normalized.
    """
    if centers is None:
        centers = values
    avg = np.average(centers, axis=0)
    corrs = np.dot(values, avg)
    index_of_min = np.argmin(corrs)
    return values[index_of_min]

def compute_class_map_by_dot_product(image, centers):
    """
        Image is 3D array, of shape WxHxC (where C is number of channels).
        Centers is 2D array, of shape NxC (where N is number of centers, C channels).
        Returns map as 2D array of shape WxH.
    """
    return np.argmax(np.einsum('ijk,mk->ijm', image, centers), axis=2)

def compute_class_map_by_angle(image, centers):
    """
        Image is 3D array, of shape WxHxC (where C is number of channels).
        Centers is 2D array, of shape NxC (where N is number of centers, C channels).
        Returns map as 2D array of shape WxH.
    """
    return compute_class_map_by_dot_product(normalized(image), normalized(centers))

def find_maxdist_clusters(image, min_correlation):
    """
        First finds spectrum that is least correlated with the average.
        It becomes first cluster center.
        Then finds next cluster center, the one that is least correlated
        with average (center) of existing cluster centers.
        Then the steps are repeated, until there is no more unclassified spectra.
        After each iteration all spectra that have correlation with new center
        equal or greater than min_correlation are clussified to belong to this new cluster,
        and excluded from further classification.
        Returns pair of map of indices of clusters,
        and cluster centers of shape (N, spectrum_length).
    """
    from spectral.utilities.errors import has_nan, NaNValueError

    if has_nan(image):
        raise NaNValueError('Image data contains NaN values.')

    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    values = normalized(image.reshape((N, nbands)))
    #values = image.copy()
    clusters = np.zeros((N,), int) - 1
    MAX_CENTERS = 65536
    centers = np.zeros((MAX_CENTERS, nbands))
    num_centers = 0
    #avg = np.average(image, axis=0)
    #values -= avg
    #centers[:, 0], minci = find_mincorr2(values)
    #values = np.delete(values, minci, axis=0)
    #newcenter = find_mincorr3(values, None, 10)
    newcenter = find_mincorr_from_center(values)
    centers[0] = newcenter
    num_centers += 1
    corrs_with_newcenter = np.dot(values, newcenter)
    high_corr_inds = np.argwhere(corrs_with_newcenter >= min_correlation)
    values = np.delete(values, high_corr_inds, axis=0)

    while values.size > 0 and num_centers < MAX_CENTERS:
        newcenter = find_mincorr_from_center(values, centers[:num_centers])
        centers[num_centers] = newcenter
        num_centers += 1
        corrs_with_newcenter = np.dot(values, newcenter)
        high_corr_inds = np.argwhere(corrs_with_newcenter >= min_correlation)
        values = np.delete(values, high_corr_inds, axis=0)

        print('%d centers found, left %d         ' % (num_centers, values.shape[0]), end='\r')
        sys.stdout.flush()

    print('assigning values...')

    centers = centers[:num_centers]
    clusters = compute_class_map_by_angle(image, centers)
    print('done')

    return (clusters, centers)

def find_ppi_clusters(image, iterations = 3000, threshold = 4):
    purity_map = ppi(image, iterations)
    indices = np.nonzero(purity_map > threshold)
    centers = image[indices]
    print(centers.shape)
    return (compute_class_map_by_angle(image, centers), centers)

def find_smacc_clusters(image, min_endmembers = 120, max_residual_norm = float('Inf')):
    S, F, R = smacc(image, min_endmembers, max_residual_norm)
    return (compute_class_map_by_angle(image, S), S)
