# pylint: disable=invalid-name
from __future__ import print_function
from spectral import *
import numpy as np
import scipy.spatial.distance as distance
import pylab
import spectral.io.envi as envi
import random
from scipy.spatial.distance import cdist
import gc

def generate_class_colours(n):
    import colorsys
    HSV_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in RGB_tuples]

def normalized(a, order=2, axis=-1):
    norms = np.atleast_1d(np.linalg.norm(a, order, axis))
    norms[norms == 0] = 1
    return a / np.expand_dims(norms, axis)

def show_centers(centers, title):
    if title is None:
        title = "Centers"
    pylab.figure()
    # pylab.hold(1) # default and depricated
    for i in range(min(centers.shape[0], 30)):
        pylab.plot(centers[i])
    pylab.title(title)
    pylab.show()

def kmeans_cosine(image, nclusters=10, max_iterations=20, **kwargs):
    from spectral.algorithms.spymath import has_nan, NaNValueError

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

    #show_centers(centers, "Initial centers")
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
            if itnum % 10 == 0:
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
                # print(np.abs(n_changed - old_n_changed)/(n_changed + old_n_changed))
                # if np.abs(float(n_changed - old_n_changed))/(n_changed + old_n_changed) > 0.5 and n_changed != old_n_changed:
                #     print(centers)
                #     viewt = imshow(orig_image, classes=clusters.reshape(nrows, ncols))
                #     viewt.set_display_mode('overlay')
                #     viewt.class_alpha = 0.85
                #     old_n_changed = n_changed
                #     # raw_input("Press Enter to continue...")
                if itnum % 10 == 0:
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
    from spectral.algorithms.spymath import has_nan, NaNValueError

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

    #show_centers(centers, "Initial centers")
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
            if itnum % 10 == 0:
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
                if itnum % 10 == 0:
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

        if float(i)/N >= percentage + 0.2:
            percentage = float(i)/N
            print('\r%d%% completed' % int(percentage * 100), end='\r')

    return (clusters.reshape(nrows, ncols), centers[:, :num_centers].T.copy())


#img = open_image('92AV3C.lan')
# gt = open_image('92AV3GT.GIS').read_band(0)
img = envi.open('f080611t01p00r07rdn_c_sc01_ort_img.hdr')

print(img)

data = img[400:1000, 200:, :]
data[data <= 0] = 1
print(data.dtype)

view = imshow(data, (29, 20, 12), title="Image")
#raw_input("Press Enter to continue...")

#print(distance.cosine(data[50,15,:], data[100, 1, :]))
nclusters = 500

# cosine distance based clustering
#(class_map, centers) = kmeans_cosine(data, nclusters=nclusters, max_iterations=300)

# L2 distance based clustering
#(class_map, centers) = kmeans_L2(data, nclusters=nclusters, max_iterations=500)
#(class_map, centers) = kmeans_cdist(data, nclusters=nclusters, max_iterations=900)

(class_map, centers) = find_related_clusters(data, 0.99)
#(class_map, centers) = find_related_clusters(data, 0.99, start_centers=centers)

# compute image of modules
data_double = data.astype(float)
modules = np.sqrt(np.einsum('ijk,ijk->ij', data_double, data_double))
scale_map = np.empty_like(modules)
image_shape = data.shape
reconstruction = np.empty(image_shape)
diffs = np.empty(image_shape)
abs_diffs = np.empty(image_shape)
# rel_diffs = np.empty(image_shape)

accumulated_iter_count = 0
for iter in [1, 3, 6, 10, 30, 60, 100]:
    gc.collect()
    print('>>>>>>>> starting iteration ' + str(accumulated_iter_count  + 1) + " and batch of " + str(iter) + " iterations :")
    accumulated_iter_count += iter
    (class_map, centers) = kmeans_cosine(data, nclusters=centers.shape[0], max_iterations=iter, start_clusters=centers)

    print('Centers\' shape: ', centers.shape)
    class_colours = generate_class_colours(centers.shape[0])

    view = imshow(classes=class_map, colors=class_colours, title="Classes after " + str(accumulated_iter_count) + " iterations")

    # compute modules of centers
    center_modules = np.sqrt(np.einsum('ij,ij->i', centers, centers))

    # compute scale factor map
    for i in range(scale_map.shape[0]):
        for j in range(scale_map.shape[1]):
            scale_map[i, j] = modules[i, j] / center_modules[class_map[i, j]]

    # =====================================================================
    # =====================================================================
    # we can now save image_shape, scale_map, class_map and centers as they contain lossy compressed image

    # we can make loseless compression if we calculate differences and then compress those
    # diff can also tell as about how much information is lost with lossy compression

    # to calculate diff, we first need to calculate reconstruction
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            reconstruction[i, j] = centers[class_map[i, j]] * scale_map[i, j]

    diffs[:] = data - reconstruction
    abs_diffs[:] = np.abs(diffs)
    #rel_diffs[:] = abs_diffs / data

    diffs_norm = np.linalg.norm(diffs)
    print('Norm: ' + str(diffs_norm))
    as1darray = abs_diffs.reshape((diffs.shape[0] * diffs.shape[1] * diffs.shape[2],))
    print('Max: ' + str(np.max(as1darray)))
    print('Min: ' + str(np.min(as1darray)))
    print('Avg: ' + str(np.average(as1darray))

    #index_of_max = np.argmax(as1darray)
    #band_of_max = index_of_max % diffs.shape[2]
    #index_of_max /= diffs.shape[2]
    #j_of_max = index_of_max % diffs.shape[1]
    #i_of_max = index_of_max / diffs.shape[1]
    (i_of_max, j_of_max, band_of_max) = np.unravel_index(np.argmax(abs_diffs), abs_diffs.shape)
    print('Coordinates of max: ', i_of_max, j_of_max, band_of_max)
    print('Check the difference: ', data[i_of_max, j_of_max, band_of_max] - reconstruction[i_of_max, j_of_max, band_of_max])
    # print(data[i_of_max, j_of_max] - reconstruction[i_of_max, j_of_max])

    # pylab.figure()
    # # pylab.hold(1) # default and depricated
    # pylab.plot(data[i_of_max, j_of_max], label = 'original')
    # pylab.plot(reconstruction[i_of_max, j_of_max], label = 'reconstruction')
    # pylab.title("Original and reconstruction of pixel with greatest error")
    # pylab.legend()
    # pylab.show()

    # (i_of_max, j_of_max, band_of_max) = np.unravel_index(np.argmax(rel_diffs), rel_diffs.shape)
    # print('Coordinates of rel max: ', i_of_max, j_of_max, band_of_max)
    # print('Values: ', data[i_of_max, j_of_max, band_of_max], reconstruction[i_of_max, j_of_max, band_of_max])
    # pylab.figure()
    # # pylab.hold(1) # default and depricated
    # pylab.plot(data[i_of_max, j_of_max], label = 'original')
    # pylab.plot(reconstruction[i_of_max, j_of_max], label = 'reconstruction')
    # pylab.title("Original and reconstruction of pixel with greatest relative error")
    # pylab.legend()
    # pylab.show()

    # rel_diffs_norm = np.linalg.norm(rel_diffs)
    # print('Rel. Norm: ' + str(rel_diffs_norm))
    # rel_as1darray = rel_diffs.reshape((rel_diffs.shape[0] * rel_diffs.shape[1] * rel_diffs.shape[2],))
    # print('Rel. Max: ' + str(np.max(rel_as1darray))
    # print('Rel. Min: ' + str(np.min(rel_as1darray))
    # print('Rel. Avg: ' + str(np.average(rel_as1darray))
    print('============================================ ' + str(accumulated_iter_count) + " iters completed ===================================")
    gc.collect()

raw_input("Press Enter to continue...")
