# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from __future__ import print_function
from spectral import *
import numpy as np
import pylab

def generate_class_colours(n):
    import colorsys
    HSV_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in RGB_tuples]

def generate_class_and_subclass_colours(nc, nsc):
    pass

def show_centers(centers, title):
    if title is None:
        title = "Centers"
    pylab.figure()
    # pylab.hold(1) # default and depricated
    for i in range(min(centers.shape[0], 30)):
        pylab.plot(centers[i])
    pylab.title(title)
    pylab.show()
    
def show_histogram(hist_values, title):
    pylab.figure()
    pylab.hist(range(len(hist_values)), len(hist_values), weights=hist_values)
    pylab.title(title)
    pylab.show()
    raw_input("Press Enter to continue...")
    pylab.close()

def show_classes(class_map):
    class_colours = generate_class_colours(np.max(class_map))
    view = imshow(data, (29, 20, 12), classes=class_map, colors=class_colours, title="Image with class overlay")
    view.set_display_mode('overlay')
    view.class_alpha = 0.5
    return imshow(classes=class_map, colors=class_colours, title="Classes")


def draw_common_2cols_graphs(graphs):
    fig = pylab.figure()
    fig.set_size_inches(3*4, 2*6)

    for r in range(len(graphs)):
        for c in range(len(graphs[r])):
            pylab.subplot(3, 2, (r*2)+1+c)
            for g in range(len(graphs[r][c])):
                pylab.plot(graphs[r][c][g])

    pylab.show()