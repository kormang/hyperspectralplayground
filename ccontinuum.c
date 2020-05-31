#include "ccontinuum.h"

#include <stdio.h>
#include <math.h>

// Used to represent wavelengths in case we switch
// to algorithm that includes wavelengths.
#define WL(x) (wavelengths[x])

void continuum_old(double input[], double output[], double wavelengths[], size_t n) {
	output[0] = input[0];
	// i points to the last point that belongs to the curve.
	// j points to the current potential point.
	// k points to the current point that could eliminate j as potential.
	for (size_t i = 0, j = 1; j < n;) {
		// Check if j belongs to curve.
        double qoef_j = (input[j] - input[i]) / (WL(j) - WL(i));
		size_t k = j + 1;
		for (; k < n; ++k) {
			double qoef_k = (input[k] - input[i]) / (WL(k) - WL(i));
			if (qoef_j < qoef_k) {
				break;
			}
		}
		if (k == n) {
			// j belongs to the curve if we reached the end,
			// and haven't found that input[j] is below
			// line connecting input[i] and input[k].
			// Fill the gap using qoef_j.
			for (size_t t = i + 1; t < j; ++t) {
				output[t] = qoef_j * (WL(t) - WL(i)) + input[i];
			}
			// Fill exact value.
			output[j] = input[j];
			// Start searching for next point on the continuum curve.
			i = j;
			j = i + 1;
		} else {
			// We have found that line between input[i] and input[k]
			// goes above input[j] so, j is not on the curve,
			// but maybe k is.
			j = k;
		}
	}
}

void continuum_removed(double input[], double output[], double wavelengths[], size_t indices[], size_t n) {
	continuum(input, output, wavelengths, indices, n);
	for (size_t i = 0; i < n; ++i) {
		output[i] = input[i] / output[i];
	}
}

struct data_t {
	double* spectrum;
	double* wavelengths;
	size_t* indices;
	size_t n;
	size_t ind_fill;
};

static void find_indices(struct data_t* data, size_t ibegin, size_t iend) {
	double* spectrum = data->spectrum;
	double* wavelengths = data->wavelengths;
	size_t iendi = iend - 1;
	double naxis_y = wavelengths[iendi] - wavelengths[ibegin];
	double naxis_x = spectrum[ibegin] - spectrum[iendi];
	double maxval = -INFINITY;
	double imax = ibegin;

	for(size_t i = ibegin; i < iendi; ++i) {
		double newval = wavelengths[i] * naxis_x + spectrum[i] * naxis_y;
		if (newval > maxval) {
			maxval = newval;
			imax = i;
		}
	}

	if (imax == ibegin) {
		return;
	}

	if (imax > ibegin + 1) {
		find_indices(data, ibegin, imax + 1);
	}

	data->indices[data->ind_fill++] = imax;

	if (imax < iend - 2) {
		find_indices(data, imax, iend);
	}
}

void continuum(double spectrum[], double output[], double wavelengths[], size_t indices[], size_t n) {
	struct data_t data = { .spectrum = spectrum, .wavelengths = wavelengths, .indices = indices, .n = n, .ind_fill = 1 };

	// Find indices of points that belong to convex hull.
	indices[0] = 0;
	find_indices(&data, 0, n);
	indices[data.ind_fill] = n - 1; // Didn't increase ind_fill on purpose.

	// Linear interpolation of points.
	for (size_t i = 0; i < data.ind_fill; ++i) {
		size_t ibegin = indices[i];
		size_t iend = indices[i + 1];
		// Put exact values where possible.
		output[ibegin] = spectrum[ibegin];
		// Calculate line parameters.
		double a = (spectrum[iend] - spectrum[ibegin]) / (wavelengths[iend] - wavelengths[ibegin]);
		double b = spectrum[ibegin] - a * wavelengths[ibegin];
		// Fill.
		for (size_t j = ibegin + 1; j < iend; ++j) {
			output[j] = a * wavelengths[j] + b;
		}
	}
	output[n - 1] = spectrum[n - 1];
}
