#include "continuum.h"

#include <stdio.h>

// Used to represent wavelengths in case we switch
// to algorithm that includes wavelengths.
#define WL(x) (wavelengths[x])

void continuum(double input[], double output[], double wavelengths[], size_t n) {

	output[0] = input[0];
	// i points to the last point that belongs to the curve.
	// j points to the current potential point.
	// k points to the current point that could eliminate j as potential.
	for (size_t i = 0, j = 1; j < n;) {
		// Check if j belongs to curve.
		size_t k = j + 1;
		for (; k < n; ++k) {
			double qoef = (input[k] - input[i]) / (WL(k) - WL(i));
			double intersection = qoef * (WL(j) - WL(i)) + input[i];
			if (input[j] < intersection) {
				break;
			}
		}
		if (k == n) {
			// j belongs to the curve if we reached the end,
			// and haven't found that input[j] is below
			// line connecting input[i] and input[k].
			// Fill the gap using qoef.
			double qoef = (input[j] - input[i]) / (WL(j) - WL(i));
			for (size_t t = i + 1; t < j; ++t) {
				output[t] = qoef * (WL(t) - WL(i)) + input[i];
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

void continuum_removed(double input[], double output[], double wavelengths[], size_t n) {
	continuum(input, output, wavelengths, n);
	for (size_t i = 0; i < n; ++i) {
		output[i] = input[i] / output[i];
	}
}
