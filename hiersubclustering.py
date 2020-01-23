# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from __future__ import print_function
from spectral import *
import numpy as np

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