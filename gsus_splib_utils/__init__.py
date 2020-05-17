import numpy as np
import os
import re
import scipy
from spectral.io.spyfile import find_file_path
from spectral import *

class SpectralData:
    def __init__(self, signature = None, libname = None,
                 record = None, description = None, spectrometer = None,
                 purity = None, measurement_type = None, spectrometer_data = None):
        self.signature = signature
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
        sig_lines = ['{:.5E}'.format(s) for s in self.signature]
        return self.header() + '\n' + ('\n'.join(sig_lines))

    def __str__(self):
        return self.header() + ' \t {} channels'.format(len(self.signature))

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

            signature = []
            for line in f:
                try:
                    signature.append(float(line.strip()))
                except:
                    pass

            spectrometer_data = SpectrometerData.get_by_name(spectrometer)

            return SpectralData(np.array(signature),
                                libname, record, description,
                                spectrometer, purity, measurment_type,
                                spectrometer_data)

    def replace_invalid(self, value):
        self.signature[self.signature < 0.0] = value
        return self

    def interpolate_invalid(self, kind='slinear'):
        full_xs = list(range(len(self.signature)))
        xs = [x for x, y in zip(full_xs, self.signature) if y > 0.0]
        ys = [y for y in self.signature if y > 0.0]
        if xs[0] > full_xs[0]:
            xs.insert(0, full_xs[0])
            ys.insert(0, ys[0])
        if xs[-1] < full_xs[-1]:
            xs.append(full_xs[-1])
            ys.append(ys[-1])
        f = scipy.interpolate.interp1d(xs, ys, kind=kind, assume_sorted=True)
        self.signature = f(full_xs)
        return self

    def resample_as(self, spectrometer_name, with_fixed_dest = False):
        """
            Returns signature resampled to different spectrometer.
        """
        dest = SpectrometerData.get_by_name(spectrometer_name)
        dest_wl = dest.wavelengths
        dest_bw = dest.bandwidths
        if with_fixed_dest:
            dest_wl = np.sort(dest_wl)
            dest_bw = None

        return self.resample_at(dest_wl, dest_bw)

    def resample_at(self, dest_wl, dest_bw = None):
        """
            Returns signature resampled to different spectrometer.
        """
        resampler = BandResampler(self.spectrometer_data.wavelengths, dest_wl,
                                  self.spectrometer_data.bandwidths, dest_bw)
        return resampler(self.signature)


    def interpolate_as(self, spectrometer_name, with_fixed_dest = True, kind='quadratic'):
        """
            Returns signature interpoleted at wavelengths of different spectrometer,
            based on wavelengths and reflectances of original.
        """
        dest = SpectrometerData.get_by_name(spectrometer_name)
        dest_wl = dest.wavelengths
        if with_fixed_dest:
            dest_wl = np.sort(dest_wl)

        return self.interpolate_at(dest_wl, kind)

    def interpolate_at(self, dest_wl, kind='quadratic'):
        """
            Returns signature interpoleted at specified wavelengths,
            based on wavelengths and reflectances of original.
        """
        xs = self.wavelengths()
        ys = self.signature
        f = scipy.interpolate.interp1d(xs, ys, kind=kind, assume_sorted=True, fill_value='extrapolate')
        return f(dest_wl)

    def in_range(self, min_wl, max_wl):
        """
            Return wavelengths and signature part between min wavelength and max wavelength.
        """
        src_wavelengths = self.spectrometer_data.wavelengths
        min_wl = max(src_wavelengths[0], min_wl)
        max_wl = min(src_wavelengths[-1], max_wl)
        min_index = np.argmax(src_wavelengths >= min_wl)
        max_index = np.argmax(src_wavelengths > max_wl)
        range_wl = src_wavelengths[min_index:max_index]
        range_sig = self.signature[min_index:max_index]
        return range_wl, range_sig

    def in_range_of(self, spectrometer_name):
        """
            Return wavelengths and signature part that overlaps with other spectrometer.
        """
        dest_wavelengths = SpectrometerData.get_by_name(spectrometer_name).wavelengths
        return self.in_range(dest_wavelengths[0], dest_wavelengths[-1])

    def fix(self):
        self.interpolate_invalid()
        self.signature = self.resample_as(self.spectrometer, True)
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
        sig_lines = ['{:.5E}'.format(s) for s in self.signature]
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
        'ASDFR': 'splib07a_Bandpass_(FWHM)_ASDFR_StandardResolution.txt',
        'ASDHR': 'splib07a_Bandpass_(FWHM)_ASDHR_High-Resolution.txt',
        'ASDNG': 'splib07a_Bandpass_(FWHM)_ASDNG_High-Res_NextGen.txt',
        'AVIRIS': 'splib07a_Bandpass_(FWHM)_AVIRIS_1996_in_microns.txt',
        'BECK': 'splib07a_Bandpass_(FWHM)_BECK_Beckman_in_microns.txt',
        'NIC': 'splib07a_Bandpass_(FWHM)_NIC4_Nicolet_in_microns.txt'
    }

    _name2wavelengthfilename = {
        'ASDFR': 'splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt',
        'ASDHR': 'splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt',
        'ASDNG': 'splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt',
        'AVIRIS': 'splib07a_Wavelengths_AVIRIS_1996_0.37-2.5_microns.txt',
        'BECK': 'splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt',
        'NIC': 'splib07a_Wavelengths_NIC4_Nicolet_1.12-216microns.txt'
    }

    @staticmethod
    def read_data_from_file(filename):
        with open(filename) as f:
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
        wlfilepath = os.path.join(os.path.dirname(__file__),
                                  SpectrometerData._name2wavelengthfilename[name])
        libname, record, measurement, spectrometer_name, description, wavelengths = \
            SpectrometerData.read_data_from_file(wlfilepath)
        bpfilepath = os.path.join(os.path.dirname(__file__),
                                  SpectrometerData._name2bandpassfilename[name])
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
