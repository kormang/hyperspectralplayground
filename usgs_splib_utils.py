import numpy as np
import os
import re
import scipy
from spectral.io.spyfile import find_file_path
from spectral import *
from utils import *

class SpectralData:
    def __init__(self, spectrum = None, libname = None,
                 record = None, description = None, spectrometer = None,
                 purity = None, measurement_type = None, spectrometer_data = None):
        self.spectrum = spectrum
        self.libname = libname
        self.record = record
        self.description = description
        self.spectrometer = spectrometer
        self.purity = purity
        self.measurement_type = measurement_type
        self.spectrometer_data = spectrometer_data
        self.fixed = False

    def header(self):
        return '{} Record={}: {} {}{} {}'.format(self.libname, self.record,
                                               self.description, self.spectrometer,
                                               self.purity, self.measurement_type)

    def __repr__(self):
        spectra_lines = ['{:.5E}'.format(s) for s in self.spectrum]
        return self.header() + '\n' + ('\n'.join(spectra_lines))

    def __str__(self):
        return self.header() + ' \t {} channels'.format(len(self.spectrum))

    @staticmethod
    def parse_header(header_line):
        elements = header_line.split()

        libname = elements[0]

        # From 'Record=1234:' extract 1234.
        record = int(elements[1].split('=')[1][:-1])

        # Join everything between record and spectrometer into description.
        description = ' '.join(elements[2:-2])

        # Split 'AVIRIS13aa' into ['', 'AVIRIS13', 'aa', ''].
        smpurity = re.split('([A-Z12]+)([a-z]+)', elements[-2])
        spectrometer = smpurity[1]
        purity = smpurity[2]

        measurement_type = elements[-1]

        return libname, record, description, spectrometer, purity, measurement_type

    @staticmethod
    def read_from_file(filename):
        path = find_file_path(filename)

        with open(path) as f:
            header_line = f.readline()
            libname, record, description, spectrometer, purity, measurment_type = \
                SpectralData.parse_header(header_line.strip())

            spectrum = []
            for line in f:
                try:
                    spectrum.append(float(line.strip()))
                except:
                    pass

            spectrometer_data = SpectrometerData.get_by_name(spectrometer)

            return SpectralData(np.array(spectrum),
                                libname, record, description,
                                spectrometer, purity, measurment_type,
                                spectrometer_data)

    def replace_invalid(self, value):
        self.spectrum[self.spectrum < 0.0] = value
        return self

    def interpolate_invalid(self, kind='slinear'):
        self.spectrum = interpolate_invalid(self.spectrum, kind)
        return self

    def resample_as(self, spectrometer_name, with_fixed_dest = False):
        """
            Returns spectrum resampled to different spectrometer.
        """
        return resample_as(self.spectrum, self.wavelengths(), spectrometer_name, with_fixed_dest, self.spectrometer_data.bandwidths)

    def resample_at(self, dest_wls, dest_bw = None):
        """
            Returns spectrum resampled to specified wavelengths and bandwidths.
        """
        return resample_at(self.spectrum, self.wavelengths(), dest_wls, self.spectrometer_data.bandwidths, dest_bw)


    def interpolate_as(self, spectrometer_name, with_fixed_dest = True, kind='quadratic'):
        """
            Returns spectrum interpoleted at wavelengths of different spectrometer,
            based on wavelengths and reflectances of original.
        """
        return interpolate_as(self.spectrum, self.wavelengths(), spectrometer_name, with_fixed_dest, kind)

    def interpolate_at(self, dest_wls, kind='quadratic'):
        """
            Returns spectrum interpoleted at specified wavelengths,
            based on wavelengths and reflectances of original.
        """
        return interpolate_at(self.spectrum, self.wavelengths(), dest_wls)

    def in_range(self, dst_wls):
        """
            Return wavelengths and spectrum part between min wavelength and max wavelength.
            Min wavelength is max(src_wls[0], dst_wls[0]).
            Max wavelengths is min(src_wls[-1], dst_wls[-1]).
            dst_wls can be ndarray, or simply 2-tuple with max and min wavelengths.
        """
        return cut_range(self.spectrum, self.wavelengths(), dst_wls)

    def in_range_of(self, spectrometer_name):
        """
            Return wavelengths and spectrum part that overlaps with other spectrometer.
        """
        return cut_range_of(self.spectrum, self.wavelengths(), spectrometer_name)

    def fix(self):
        self.interpolate_invalid()
        self.spectrum = self.resample_as(self.spectrometer, True)
        self.fixed = True
        self._wavelengths = np.sort(self.spectrometer_data.wavelengths)
        return self

    def wavelengths(self):
        return self._wavelengths if self.fixed else self.spectrometer_data.wavelengths


class SpectrometerData:
    def __init__(self, libname, record, measurement, spectrometer_name, description, wavelengths, bandwidths):
        self.libname = libname
        self.record = record
        self.measurement = measurement
        self.spectrometer_name = spectrometer_name
        self.description = description
        self.wavelengths = wavelengths
        self.bandwidths = bandwidths

    def header(self):
        return '{} Record={}: {} {} {}'.format(self.libname, self.record,  self.measurement,
                                                self.spectrometer_name, self.description)

    def __repr__(self):
        spectra_lines = ['{:.5E}'.format(s) for s in self.spectrum]
        return self.header() + '\n' + ('\n'.join(wavelengths))

    def __str__(self):
        return self.header() + ' \t wavelengths from {} to {}'.format(np.min(self.wavelengths), np.max(self.wavelengths))

    @staticmethod
    def parse_header(header_line):
        elements = header_line.split()

        libname = elements[0]

        # From 'Record=1234:' extract 1234.
        record = int(elements[1].split('=')[1][:-1])

        measurement = elements[2]
        spectrometer_name = elements[3]

        description = ' '.join(elements[4:])

        return libname, record, measurement, spectrometer_name, description


    _name2bandpassfilename = {
        'ASDFR': 'gsus_spectrometer_data/splib07a_Bandpass_(FWHM)_ASDFR_StandardResolution.txt',
        'ASDHR': 'gsus_spectrometer_data/splib07a_Bandpass_(FWHM)_ASDHR_High-Resolution.txt',
        'ASDNG': 'gsus_spectrometer_data/splib07a_Bandpass_(FWHM)_ASDNG_High-Res_NextGen.txt',
        'AVIRIS': 'gsus_spectrometer_data/splib07a_Bandpass_(FWHM)_AVIRIS_1996_in_microns.txt',
        'BECK': 'gsus_spectrometer_data/splib07a_Bandpass_(FWHM)_BECK_Beckman_in_microns.txt',
        'NIC': 'gsus_spectrometer_data/splib07a_Bandpass_(FWHM)_NIC4_Nicolet_in_microns.txt'
    }

    _name2wavelengthfilename = {
        'ASDFR': 'gsus_spectrometer_data/splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt',
        'ASDHR': 'gsus_spectrometer_data/splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt',
        'ASDNG': 'gsus_spectrometer_data/splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt',
        'AVIRIS': 'gsus_spectrometer_data/splib07a_Wavelengths_AVIRIS_1996_0.37-2.5_microns.txt',
        'BECK': 'gsus_spectrometer_data/splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt',
        'NIC': 'gsus_spectrometer_data/splib07a_Wavelengths_NIC4_Nicolet_1.12-216microns.txt'
    }

    @staticmethod
    def read_data_from_file(filename):
        path = find_file_path(filename)
        with open(path) as f:
            header_line = f.readline()
            libname, record, measurement, spectrometer_name, description = \
                SpectrometerData.parse_header(header_line.strip())

            data = []
            for line in f:
                try:
                    data.append(float(line.strip()))
                except:
                    pass

            data = np.array(data)

            return libname, record, measurement, spectrometer_name, description, data

    @staticmethod
    def read_from_file_by_name(name):
        assert(name in SpectrometerData._name2wavelengthfilename), \
            'Spectrometer with name ' + name + ' is not supported.'
        wlfilepath = SpectrometerData._name2wavelengthfilename[name]
        libname, record, measurement, spectrometer_name, description, wavelengths = \
            SpectrometerData.read_data_from_file(wlfilepath)
        bpfilepath = SpectrometerData._name2bandpassfilename[name]
        libname, record, measurement, spetrometer_name, description, bandpass = \
            SpectrometerData.read_data_from_file(bpfilepath)
        return SpectrometerData(libname, record, measurement, spectrometer_name, description, wavelengths, bandpass)


    _name2specdata = {}

    @staticmethod
    def get_pure_name(name):
        # Split 'AVIRIS13' into ['', 'AVIRIS', '13', ''].
        # Split 'BECK' into ['BECK']
        splitted = re.split('([A-Z]+)([0-9]+)', name)
        return splitted[1] if len(splitted) > 1 else splitted[0]

    @staticmethod
    def get_by_name(name):
        name = SpectrometerData.get_pure_name(name)
        #if name in SpectrometerData._name2specdata:
        #    return SpectrometerData._name2specdata

        sd = SpectrometerData.read_from_file_by_name(name)
        SpectrometerData._name2specdata[name] = sd
        return sd

##### Free utility functions: ########

def get_bands_of(spectrometer_name, with_fixed_wls = True):
    wls = SpectrometerData.get_by_name(spectrometer_name).wavelengths
    if with_fixed_wls:
        wls = np.sort(wls)
    return wls

def resample_as(spectrum, src_wls, spectrometer_name, with_fixed_dest = False, src_bw = None):
    """
        Returns spectrum resampled to different spectrometer.
    """
    dest = SpectrometerData.get_by_name(spectrometer_name)
    dest_wls = dest.wavelengths
    dest_bw = dest.bandwidths
    if with_fixed_dest:
        dest_wls = np.sort(dest_wls)
        dest_bw = None

    return resample_at(spectrum, src_wls, dest_wls, src_bw, dest_bw)

def interpolate_as(spectrum, src_wls, spectrometer_name, with_fixed_dest = True, kind='quadratic'):
    """
        Returns spectrum interpoleted at wavelengths of different spectrometer,
        based on wavelengths and reflectances of original.
    """
    dest = SpectrometerData.get_by_name(spectrometer_name)
    dest_wls = dest.wavelengths
    if with_fixed_dest:
        dest_wls = np.sort(dest_wls)

    return interpolate_at(spectrum, src_wls, dest_wls, kind)


def cut_range_of(spectrum, wls, spectrometer_name):
    """
        Return wavelengths and spectrum part that overlaps with other spectrometer.
    """
    dest_wls = SpectrometerData.get_by_name(spectrometer_name).wavelengths
    x, y = cut_range(spectrum, wls, dest_wls)
    return x, y