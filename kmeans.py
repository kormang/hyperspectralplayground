# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from __future__ import print_function
from spectral import *
import numpy as np
import pylab
import random
from scipy.spatial.distance import cdist
from math_utils import *

def kmeans_cosine(image, nclusters=10, max_iterations=20, **kwargs):
    from spectral.utilities.errors import has_nan, NaNValueError

    if has_nan(image):
        raise NaNValueError('Image data contains NaN values.')
    #orig_image = image
    # defaults for kwargs
    start_clusters = None
    compare = None
    iterations = None

    for (key, val) in list(kwargs.items()):
        if key == 'start_clusters':
            start_clusters = val
        elif key == 'compare':
            compare = val
        elif key == 'frames':
            if not hasattr(val, 'append'):
                raise TypeError('"frames" keyword argument must have "append"'
                                'attribute.')
            iterations = val
        else:
            raise NameError('Unsupported keyword argument.')

    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    image = normalized(image.reshape((N, nbands)))
    #image = image.reshape((N, nbands))
    clusters = np.zeros((N,), int)
    centers = None
    if start_clusters is not None:
        assert (start_clusters.shape[0] == nclusters), 'There must be \
        nclusters clusters in the start_centers array.'
        centers = np.array(normalized(start_clusters))
    else:
        # print('Initializing clusters along diagonal of N-dimensional bounding box.')
        # boxMin = np.amin(image, 0)
        # boxMax = np.amax(image, 0)
        # delta = (boxMax - boxMin) / (nclusters - 1)
        centers = np.empty((nclusters, nbands), float)
        # for i in range(nclusters):
        #     centers[i] = boxMin + i * delta
        random.seed(4)
        for i in range(nclusters):
            centers[i] = image[random.randrange(N)]

    centers = centers.T
    #distances = np.empty((N, nclusters), float)
    old_centers = np.array(centers)
    clusters = np.zeros((N,), int)
    old_clusters = np.copy(clusters)
    n_changed = 0
    itnum = 1
    while (itnum <= max_iterations):
        try:
            print('\rIteration %d (%d pixels reassinged previously)...' % (itnum, n_changed), end='')

            # Assign all pixels
            #distances[:] = np.matmul(image, centers)
            clusters[:] = np.argmax(np.matmul(image, centers), 1)

            # Update cluster centers
            old_centers[:] = centers
            for i in range(nclusters):
                inds = np.nonzero(clusters == i)[0]
                if len(inds) > 0:
                    centers[:, i] = np.mean(image[inds], 0, float)
                    centers[:, i] /= np.linalg.norm(centers[:, i])

            if iterations is not None:
                iterations.append(clusters.reshape(nrows, ncols))

            if compare and compare(old_clusters, clusters):
                print('done.')
                break
            else:
                n_changed = np.sum(clusters != old_clusters)
                # prin(np.abs(n_changed - old_n_changed)/(n_changed + old_n_changed))
                # if np.abs(float(n_changed - old_n_changed))/(n_changed + old_n_changed) > 0.5 and n_changed != old_n_changed:
                #     print(centers)
                #     viewt = imshow(orig_image, classes=clusters.reshape(nrows, ncols))
                #     viewt.set_display_mode('overlay')
                #     viewt.class_alpha = 0.85
                #     old_n_changed = n_changed
                #     # raw_input("Press Enter to continue...")
                if n_changed == 0:
                    break

            sys.stdout.flush()

            old_clusters[:] = clusters
            old_centers[:] = centers
            itnum += 1

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Returning clusters from previous iteration.")
            return (old_clusters.reshape(nrows, ncols), old_centers.T)

    print('%d pixels reassigned in last iteration.' % (n_changed))


    print('kmeans terminated with', len(set(old_clusters.ravel())), \
        'clusters after', itnum - 1, 'iterations.')
    return (old_clusters.reshape(nrows, ncols), centers.T)

def kmeans_L2(image, nclusters=10, max_iterations=20, **kwargs):
    from spectral.utilities.errors import has_nan, NaNValueError

    if has_nan(image):
        raise NaNValueError('Image data contains NaN values.')
    #orig_image = image
    # defaults for kwargs
    start_clusters = None
    compare = None
    iterations = None

    for (key, val) in list(kwargs.items()):
        if key == 'start_clusters':
            start_clusters = val
        elif key == 'compare':
            compare = val
        elif key == 'frames':
            if not hasattr(val, 'append'):
                raise TypeError('"frames" keyword argument must have "append"'
                                'attribute.')
            iterations = val
        else:
            raise NameError('Unsupported keyword argument.')

    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    image = image.reshape((N, nbands))
    #image = image.reshape((N, nbands))
    clusters = np.zeros((N,), int)
    centers = None
    if start_clusters is not None:
        assert (start_clusters.shape[0] == nclusters), 'There must be \
        nclusters clusters in the start_centers array.'
        centers = np.array(normalized(start_clusters))
    else:
        # print('Initializing clusters along diagonal of N-dimensional bounding box.')
        # boxMin = np.amin(image, 0)
        # boxMax = np.amax(image, 0)
        # delta = (boxMax - boxMin) / (nclusters - 1)
        centers = np.empty((nclusters, nbands), float)
        # for i in range(nclusters):
        #     centers[i] = boxMin + i * delta
        random.seed(4)
        for i in range(nclusters):
            centers[i] = image[random.randrange(N)]

    centers = centers.T
    #distances = np.empty((N, nclusters), float)
    old_centers = np.array(centers)
    clusters = np.zeros((N,), int)
    old_clusters = np.copy(clusters)
    #old_n_changed = 1
    itnum = 1
    while (itnum <= max_iterations):
        try:
            print('Iteration %d...' % itnum)

            # Assign all pixels
            #distances[:] = np.matmul(image, centers)
            clusters[:] = np.argmin(
                np.sum(image**2, axis=1)[:, np.newaxis]
                - 2 * np.matmul(image, centers)
                + np.sum(centers**2, axis=0),
            1)

            # Update cluster centers
            old_centers[:] = centers
            for i in range(nclusters):
                inds = np.nonzero(clusters == i)[0]
                if len(inds) > 0:
                    centers[:, i] = np.mean(image[inds], 0, float)

            if iterations is not None:
                iterations.append(clusters.reshape(nrows, ncols))

            if compare and compare(old_clusters, clusters):
                print('done.')
                break
            else:
                n_changed = np.sum(clusters != old_clusters)
                print('%d pixels reassigned.' % (n_changed))
                if n_changed == 0:
                    break

            old_clusters[:] = clusters
            old_centers[:] = centers
            itnum += 1

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Returning clusters from previous iteration.")
            return (old_clusters.reshape(nrows, ncols), old_centers.T)

    print('kmeans terminated with', len(set(old_clusters.ravel())), \
        'clusters after', itnum - 1, 'iterations.')
    return (old_clusters.reshape(nrows, ncols), centers.T)

def kmeans_cdist(image, nclusters=10, max_iterations=20, **kwargs):
    from spectral.utilities.errors import has_nan, NaNValueError

    if has_nan(image):
        raise NaNValueError('Image data contains NaN values.')
    #orig_image = image
    # defaults for kwargs
    start_clusters = None
    compare = None
    iterations = None

    for (key, val) in list(kwargs.items()):
        if key == 'start_clusters':
            start_clusters = val
        elif key == 'compare':
            compare = val
        elif key == 'frames':
            if not hasattr(val, 'append'):
                raise TypeError('"frames" keyword argument must have "append"'
                                'attribute.')
            iterations = val
        else:
            raise NameError('Unsupported keyword argument.')

    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    image = image.reshape((N, nbands))
    #image = image.reshape((N, nbands))
    clusters = np.zeros((N,), int)
    centers = None
    if start_clusters is not None:
        assert (start_clusters.shape[0] == nclusters), 'There must be \
        nclusters clusters in the start_centers array.'
        centers = np.array(start_clusters)
    else:
        # print('Initializing clusters along diagonal of N-dimensional bounding box.')
        # boxMin = np.amin(image, 0)
        # boxMax = np.amax(image, 0)
        # delta = (boxMax - boxMin) / (nclusters - 1)
        centers = np.empty((nclusters, nbands), float)
        # for i in range(nclusters):
        #     centers[i] = boxMin + i * delta
        random.seed(4)
        for i in range(nclusters):
            centers[i] = image[random.randrange(N)]

    centers = centers.T
    #distances = np.empty((N, nclusters), float)
    old_centers = np.array(centers)
    clusters = np.zeros((N,), int)
    old_clusters = np.copy(clusters)
    n_changed = 0
    itnum = 1
    while (itnum <= max_iterations):
        try:
            print('Iteration %d (%d pixels reassinged previously)...' % (itnum, n_changed))

            # Assign all pixels
            #distances[:] = np.matmul(image, centers)
            clusters[:] = np.argmin(cdist(image, centers.T, metric='sqeuclidean'), axis=1)

            # Update cluster centers
            old_centers[:] = centers
            for i in range(nclusters):
                inds = np.nonzero(clusters == i)[0]
                if len(inds) > 0:
                    centers[:, i] = np.mean(image[inds], 0, float)

            if iterations is not None:
                iterations.append(clusters.reshape(nrows, ncols))

            if compare and compare(old_clusters, clusters):
                print('done.')
                break
            else:
                n_changed = np.sum(clusters != old_clusters)
                if n_changed == 0:
                    break

            old_clusters[:] = clusters
            old_centers[:] = centers
            itnum += 1

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Returning clusters from previous iteration.")
            return (old_clusters.reshape(nrows, ncols), old_centers.T)

    print('%d pixels reassigned in last iteration.' % (n_changed))


    print('kmeans terminated with', len(set(old_clusters.ravel())), \
        'clusters after', itnum - 1, 'iterations.')
    return (old_clusters.reshape(nrows, ncols), centers.T)