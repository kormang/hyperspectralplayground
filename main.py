from __future__ import print_function
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from spectral import *
import numpy as np
import scipy.spatial.distance as distance
# import pylab
import spectral.io.envi as envi
import random
from scipy.spatial.distance import cdist
# from sklearn.cluster import KMeans
from timeit import default_timer as timer

def generate_class_colours(n):
    import colorsys
    HSV_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in RGB_tuples]

def generate_class_and_subclass_colours(nc, nsc):
    pass

def normalized(a, order=2, axis=-1):
    norms = np.atleast_1d(np.linalg.norm(a, order, axis))
    norms[norms == 0] = 1
    return a / np.expand_dims(norms, axis)

def image_of_modules(image):
    return np.sqrt(np.einsum('ijk,ijk->ij', image, image))

# def show_centers(centers, title):
#     if title is None:
#         title = "Centers"
#     pylab.figure()
#     # pylab.hold(1) # default and depricated
#     for i in range(min(centers.shape[0], 30)):
#         pylab.plot(centers[i])
#     pylab.title(title)
#     pylab.show()

# def show_histogram(hist_values, title):
#     pylab.figure()
#     pylab.hist(range(len(hist_values)), len(hist_values), weights=hist_values)
#     pylab.title(title)
#     pylab.show()
#     raw_input("Press Enter to continue...")
#     pylab.close()

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

#     show_centers(centers, "Initial centers")
    #raw_input("Press Enter to continue...")

    centers = centers.T
    #distances = np.empty((N, nclusters), float)
    old_centers = np.array(centers)
    clusters = np.zeros((N,), int)
    old_clusters = np.copy(clusters)
    #old_n_changed = 1
    itnum = 1
    while (itnum <= max_iterations):
        try:
            print('\rIteration %d...' % itnum, end='')

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
                print('%d pixels reassigned.' % (n_changed), end='')
                if n_changed == 0:
                    break

            sys.stdout.flush()

            old_clusters[:] = clusters
            old_centers[:] = centers
            itnum += 1

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Returning clusters from previous iteration.")
            return (old_clusters.reshape(nrows, ncols), old_centers.T)

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

#     show_centers(centers, "Initial centers")
    #raw_input("Press Enter to continue...")

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

#     show_centers(centers, "Initial centers")
    #raw_input("Press Enter to continue...")

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

def find_related_clusters(image, min_correlation, **kwargs):
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
    minci = np.argmin(np.sum(np.matmul(values, centers), axis=1))
    return values[minci], minci

def find_mincorr3(values, centers = None, reduce_coef = None):
    if reduce_coef is not None and values.shape[0] > reduce_coef:
        allinds = np.arange(values.shape[0])
        np.random.shuffle(allinds)
        values = values[allinds[:values.shape[0]/reduce_coef], :]
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

def find_maxdist_clusters(image, min_correlation):
    from spectral.utilities.errors import has_nan, NaNValueError

    if has_nan(image):
        raise NaNValueError('Image data contains NaN values.')

    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    image = normalized(image.reshape((N, nbands)))
    values = image.copy()
    clusters = np.zeros((N,), int) - 1
    MAX_CENTERS = 65536
    centers = np.zeros((nbands, MAX_CENTERS))
    num_centers = 0
    #avg = np.average(image, axis=0)
    #values -= avg
    #centers[:, 0], minci = find_mincorr2(values)
    #values = np.delete(values, minci, axis=0)
    newcenter = find_mincorr3(values, None, 10)
    centers[:, num_centers] = newcenter
    num_centers += 1
    corrs_with_newcenter = np.matmul(values, newcenter)
    high_corr_inds = np.argwhere(corrs_with_newcenter >= min_correlation)
    values = np.delete(values, high_corr_inds, axis=0)

    while values.size > 0 and num_centers < MAX_CENTERS:
        newcenter = find_mincorr3(values, centers[:, :num_centers])
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

def find_cluster_edges(hist, edges):
    cedges = [edges[0]]
    i = 1
    while i < len(hist) - 1:
        if hist[i - 1] > hist[i] and hist[i] < hist[i + 1]:
            cedges.append(edges[i])
        i += 1
    cedges.append(edges[-1])
    return cedges

def subdivide_by_modules(image, class_map, num_classes, unify):
    image = image.astype(float)
    modules = image_of_modules(image)
    class_counter = 0
    # for each class
    subclass_map = np.zeros_like(class_map)
    for i in range(num_classes):
        print('\rsubdividing class %d' % i, end='')
        sys.stdout.flush()
        iclass_modules = modules[class_map == i]
        bins='fd'
        iclass_histogram, bin_edges = np.histogram(iclass_modules, bins=bins)
        cedges = find_cluster_edges(iclass_histogram, bin_edges)
        cedges[-1] += 1e-10
        subclasses = np.digitize(iclass_modules, cedges)
        if unify:
            subclasses += class_counter
            class_counter += 1
        subclass_map[np.where(class_map == i)] = subclasses

        #show_histogram(iclass_histogram, "Histogram of modules of class %d" % i)

    print('')
    return subclass_map


#img = open_image('92AV3C.lan')
# gt = open_image('92AV3GT.GIS').read_band(0)
img = envi.open('f080611t01p00r07rdn_c_sc01_ort_img.hdr')

print(img)

data = img[400:1000, 200:, :]
#data = img[400:700, 200:400, :]
data[data <= 0] = 1
print(data.dtype)

# view = imshow(data, (29, 20, 12), title="Image")
# raw_input("Press Enter to continue...")
# exit()

#print(distance.cosine(data[50,15,:], data[100, 1, :]))
nclusters = 500

# cosine distance based clustering
#(class_map, centers) = kmeans_cosine(data, nclusters=nclusters, max_iterations=300)

# L2 distance based clustering
#(class_map, centers) = kmeans_L2(data, nclusters=nclusters, max_iterations=500)
#(class_map, centers) = kmeans_cdist(data, nclusters=nclusters, max_iterations=900)

#(class_map, centers) = find_related_clusters(data, 0.999)
#(class_map, centers) = find_maxdist_clusters(data, 0.99)
#print('Centers\' shape: ', centers.shape)
#subclass_map = subdivide_by_modules(data, class_map, centers.shape[0], True)

#(class_map, centers) = find_related_clusters(data, 0.99)
# from autoclustering import find_maxdist_clusters
# (class_map, centers) = find_maxdist_clusters(data, 0.99)

#print('Centers\' shape: ', centers.shape)

#(class_map, centers) = kmeans_cosine(data, nclusters=centers.shape[0], max_iterations=10, start_clusters=centers)
(class_map, centers) = kmeans_cosine(data, nclusters=1115, max_iterations=100)

view = imshow(data, (29, 20, 12), title="Image")
raw_input("Press Enter to continue...")

def show_classes(class_map):
    class_colours = generate_class_colours(np.max(class_map))
    view = imshow(data, (29, 20, 12), classes=class_map, colors=class_colours, title="Image with class overlay")
    view.set_display_mode('overlay')
    view.class_alpha = 0.5
    return imshow(classes=class_map, colors=class_colours, title="Classes")

view = show_classes(subclass_map - 1)

water_class = class_map[406, 151]
inds_of_water = np.where(class_map == water_class)
spectra_of_water = data[inds_of_water]
subclasses_of_water = np.unique(subclass_map[inds_of_water])

# spectra = np.empty((subclasses_of_water.shape[0], data.shape[2]))
# for i in range(subclasses_of_water.shape[0]):
#    inds_of_subclass = np.where(subclass_map == subclasses_of_water[i])
#    print( (inds_of_subclass[0][0], inds_of_subclass[1][0]))
#    spectra[i] = data[inds_of_subclass[0][0], inds_of_subclass[1][0]]

show_centers(spectra, u'Spectra of water subclasses')

counts_per_clusters = np.empty((centers.shape[0],), dtype=int)
for i in range(centers.shape[0]):
    counts_per_clusters[i] = np.count_nonzero(class_map == i)

print('counts per clusters: ', counts_per_clusters)
show_histogram(counts_per_clusters, 'Number of spectra per cluster')

show_centers(centers, "Final centers")

compute image of modules
data_double = data.astype(float)
modules = image_of_modules(data_double)

# compute modules of centers
center_modules = np.sqrt(np.einsum('ij,ij->i', centers, centers))

# compute scale factor map
scale_map = np.empty_like(modules)
for i in range(scale_map.shape[0]):
    for j in range(scale_map.shape[1]):
        scale_map[i, j] = modules[i, j] / center_modules[class_map[i, j]]

image_shape = data.shape


from class_cluster_compress import *
scale_map = compute_scale_map(data, class_map, centers)
reconstruction = reconstruct_image(scale_map, class_map, centers)
print_diff_stats(data, reconstruction)

# =====================================================================
# =====================================================================
# we can now save image_shape, scale_map, class_map and centers as they contain lossy compressed image

# we can make loseless compression if we calculate differences and then compress those
# diff can also tell as about how much information is lost with lossy compression

# to calculate diff, we first need to calculate reconstruction
reconstruction = np.empty(image_shape)
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        reconstruction[i, j] = centers[class_map[i, j]] * scale_map[i, j]

diffs = data - reconstruction
abs_diffs = np.abs(diffs)
rel_diffs = abs_diffs / data

diffs_norm = np.linalg.norm(diffs)
print('Norm: ', diffs_norm)
as1darray = abs_diffs.reshape((diffs.shape[0] * diffs.shape[1] * diffs.shape[2],))
print('Max: ', np.max(as1darray))
print('Min: ', np.min(as1darray))
print('Avg: ', np.average(as1darray))

index_of_max = np.argmax(as1darray)
band_of_max = index_of_max % diffs.shape[2]
index_of_max /= diffs.shape[2]
j_of_max = index_of_max % diffs.shape[1]
i_of_max = index_of_max / diffs.shape[1]
(i_of_max, j_of_max, band_of_max) = np.unravel_index(np.argmax(abs_diffs), abs_diffs.shape)
print('Coordinates of max: ', i_of_max, j_of_max, band_of_max)
print('Check the difference: ', data[i_of_max, j_of_max, band_of_max] - reconstruction[i_of_max, j_of_max, band_of_max])
print(data[i_of_max, j_of_max] - reconstruction[i_of_max, j_of_max])

pylab.figure()
# pylab.hold(1) # default and depricated
pylab.plot(data[i_of_max, j_of_max], label = 'original')
pylab.plot(reconstruction[i_of_max, j_of_max], label = 'reconstruction')
pylab.title("Original and reconstruction of pixel with greatest error")
pylab.legend()
pylab.show()

(i_of_max, j_of_max, band_of_max) = np.unravel_index(np.argmax(rel_diffs), rel_diffs.shape)
print('Coordinates of rel max: ', i_of_max, j_of_max, band_of_max)
print('Values: ', data[i_of_max, j_of_max, band_of_max], reconstruction[i_of_max, j_of_max, band_of_max])
pylab.figure()
# pylab.hold(1) # default and depricated
pylab.plot(data[i_of_max, j_of_max], label = 'original')
pylab.plot(reconstruction[i_of_max, j_of_max], label = 'reconstruction')
pylab.title("Original and reconstruction of pixel with greatest relative error")
pylab.legend()
pylab.show()

rel_diffs_norm = np.linalg.norm(rel_diffs)
print('Rel. Norm: ' + str(rel_diffs_norm))
rel_as1darray = rel_diffs.reshape((rel_diffs.shape[0] * rel_diffs.shape[1] * rel_diffs.shape[2],))
print('Rel. Max: ' + str(np.max(rel_as1darray)))
print('Rel. Min: ' + str(np.min(rel_as1darray)))
print('Rel. Avg: ' + str(np.average(rel_as1darray)))

#raw_input("Press Enter to continue...")
