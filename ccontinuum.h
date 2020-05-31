
#include <stddef.h>

void continuum_old(double input[], double output[], double wavelength[], size_t n);
void continuum_removed(double input[], double output[], double wavelength[], size_t indices[], size_t n);
void continuum(double spectrum[], double output[], double wavelengths[], size_t indices[], size_t n);
