# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from __future__ import print_function
from spectral import *
import numpy as np
import math
from scipy.fft import fft, ifft
from scipy.signal import convolve

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

def find_max_diff_l2(spectra, from_spectrum):
    diff = spectra - from_spectrum
    distances = np.einsum('ij,ij->i', diff, diff, optimize='optimal')
    index_of_max = np.argmax(distances)
    return spectra[index_of_max]

def find_max_diff_l1(spectra, from_spectrum):
    distances = np.sum(np.abs(spectra - from_spectrum), axis=1)
    index_of_max = np.argmax(distances)
    return spectra[index_of_max]

def find_max_diff_angle(spectra, from_spectrum):
    norm_spectra = normalized(spectra)
    norm_from_spectrum = from_spectrum / np.linalg.norm(from_spectrum)
    cosines = np.dot(norm_spectra, norm_from_spectrum)
    index_of_max = np.argmin(cosines)
    return spectra[index_of_max]

def find_with_max_norm2(spectra):
    norms = np.linalg.norm(spectra, 2, 1)
    index_of_max = np.argmax(norms)
    return spectra[index_of_max]

def find_with_max_norm1(spectra):
    norms = np.sum(np.abs(spectra), axis=1)
    index_of_max = np.argmax(norms)
    return spectra[index_of_max]

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def step_at(length, start_at):
    if start_at >= length:
        raise ValueError('start_at must be less than length')
    res = np.zeros(length, dtype=int)
    res[start_at:] = 1
    return res

def square_signals(length, ones_intervals):
    res = np.zeros(length, dtype=int)
    for start, end in ones_intervals:
        res[start:end] = 1

    return res

def freq_square_filter(signal, square_intervals):
    return np.abs(ifft(fft(signal) * square_signals(len(signal), square_intervals)))

def moving_average_filter(signal, window):
    return convolve(signal, np.ones(window)/window, mode='same', method='direct')