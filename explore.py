# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from spectral import *
import numpy as np
import scipy.spatial.distance as distance
import pylab
import spectral.io.envi as envi
import random
from scipy.spatial.distance import cdist

def generate_class_colours(n):
    import colorsys
    HSV_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in RGB_tuples]

def normalized(a, order=2, axis=-1):
    norms = np.atleast_1d(np.linalg.norm(a, order, axis))
    norms[norms == 0] = 1
    return a / np.expand_dims(norms, axis)

def show_centers(centers, title, labels):
    if title is None:
        title = "Centers"
    pylab.figure()
    # pylab.hold(1) # default and depricated
    for i in range(min(centers.shape[0], 30)):
        pylab.plot(centers[i], label=(labels[i] if labels is not None else None))
    pylab.title(title)
    pylab.legend()
    pylab.show()

#img = open_image('92AV3C.lan')
# gt = open_image('92AV3GT.GIS').read_band(0)
img = envi.open('f080611t01p00r07rdn_c_sc01_ort_img.hdr')

print img

#data = img[1100:1400, 300:600, :]
data = img.load()
#data[data <= 0] = 1
print data.dtype

#view = imshow(data, title="Image")
#raw_input("Press Enter to continue...")

#for i in range(data.shape[2]):
#    save_rgb('./aerodrom_channels/' + str(i+1) + '.png', data[:, :, i])

spectra = np.empty((5, data.shape[2]))
labels = [u"вода", u"загрязнённая вода", u"бетон", u"болото", u"почва"]
positions = [(863, 369), (999, 519), (1199, 422), (1056, 352), (357, 585)]
for i in range(len(positions)):
    spectra[i] = data[positions[i][0], positions[i][1]]

show_centers(spectra, u'Сигнатуры некоторых точек изображения', labels)

