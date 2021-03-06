
/* Use this file as a template to start implementing a module that
   also declares object types. All occurrences of 'Xxo' should be changed
   to something reasonable for your objects. After that, all other
   occurrences of 'xx' should be changed to something reasonable for your
   module. If your module is named foo your sourcefile should be named
   foomodule.c.

   You will probably want to delete all references to 'x_attr' and add
   your own types of attributes instead.  Maybe you want to name your
   local variables other than 'self'.  If your object type is needed in
   other files, you'll have to create a file "foobarobject.h"; see
   floatobject.h for an example. */

/* Xxo objects */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include "Python.h"
#include "numpy/arrayobject.h"
#include "ccontinuum.h"

static PyObject *ErrorObject;

/*
  Check that PyArrayObject is a double (Float) type and a 1d, or 2d, or 3d array.
  Return 1 if an error and raise exception.
*/
static int check_type(PyArrayObject* a, int min_dim, int max_dim)  {
  Py_ssize_t ndim = PyArray_NDIM(a);
	if (PyArray_TYPE(a) != NPY_DOUBLE || (ndim < min_dim || ndim > max_dim))  {
		PyErr_SetString(PyExc_ValueError,
			"In check_type: array has incorrect number of dimensions");
		return 1;
  }
	return 0;
}

/*
* Return 1 if shapes are not same.
*/
static int check_same_shapes(PyArrayObject* a, PyArrayObject* b) {
  if (PyArray_NDIM(a) != PyArray_NDIM(b)) {
    PyErr_SetString(PyExc_ValueError,
	"In check_same_shapes: arrays must have same number of dimensions.");
    return 1;
  }
  Py_ssize_t ndims = PyArray_NDIM(a);
  for (int i = 0; i < ndims; ++i) {
    if (PyArray_SHAPE(a)[i] != PyArray_SHAPE(b)[i]) {
      PyErr_SetString(PyExc_ValueError,
        "In check_same_shapes: arrays must have same number of dimensions.");
      return 1;
    }
  }
  return 0;
}


static PyObject *
continuum_generic_impl(PyObject *self, PyObject *args,
  void (*continuum_processing_f)(double*, double*, double*, size_t*, size_t))
{
    PyArrayObject* ain;
    PyArrayObject* aout;
    PyArrayObject* awl;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &ain,
        &PyArray_Type, &aout, &PyArray_Type, &awl)) {
        return Py_None;
    }

    if (NULL == ain)  return Py_None;
    if (NULL == aout)  return Py_None;
    if (NULL == awl)  return Py_None;

    if (check_type(ain, 1, 3)) return Py_None;
    if (check_type(aout, 1, 3)) return Py_None;
    if (check_type(awl, 1, 1)) return Py_None;

    if (check_same_shapes(ain, aout)) return Py_None;

    double* dataawl = (double*)PyArray_DATA(awl);

    if (PyArray_NDIM(ain) == 1) {
      double* datain = (double*)PyArray_DATA(ain);
      double* dataout = (double*)PyArray_DATA(aout);
      Py_ssize_t spectrum_length = PyArray_SHAPE(ain)[0];
      if (spectrum_length != PyArray_SHAPE(awl)[0]) {
          PyErr_SetString(PyExc_ValueError,
              "In continuum_generic_impl: wavelengths array has incorrect length.");
          return Py_None;
      }
      size_t* indices = malloc(sizeof(size_t) * spectrum_length);
      continuum_processing_f(datain, dataout, dataawl, indices, spectrum_length);
      free(indices);
    } else if (PyArray_NDIM(ain) == 2) {
      Py_ssize_t num_spectra = PyArray_SHAPE(ain)[0];
      Py_ssize_t spectrum_length = PyArray_SHAPE(ain)[1];
      if (spectrum_length != PyArray_SHAPE(awl)[0]) {
          PyErr_SetString(PyExc_ValueError,
              "In continuum_generic_impl: wavelengths array has incorrect length.");
          return Py_None;
      }

      #pragma omp parallel
      {
      size_t* indices = malloc(sizeof(size_t) * spectrum_length);

      #pragma omp for
      for (Py_ssize_t i = 0; i < num_spectra; ++i) {
        double* datain = (double*)(PyArray_DATA(ain)
          + i * PyArray_STRIDES(ain)[0]);
        double* dataout = (double*)(PyArray_DATA(aout)
          + i * PyArray_STRIDES(aout)[0]);
        continuum_processing_f(datain, dataout, dataawl, indices, spectrum_length);
      }

      free(indices);
      } // pragma omp parallel
    } else {
      Py_ssize_t num_rows = PyArray_SHAPE(ain)[0];
      Py_ssize_t num_cols = PyArray_SHAPE(ain)[1];
      Py_ssize_t spectrum_length = PyArray_SHAPE(ain)[2];
      if (spectrum_length != PyArray_SHAPE(awl)[0]) {
          PyErr_SetString(PyExc_ValueError,
              "In continuum_generic_impl: wavelengths array has incorrect length.");
          return Py_None;
      }

      //TODO: This could be optimized for better work sharing.
      #pragma omp parallel
      {
      size_t* indices = malloc(sizeof(size_t) * spectrum_length);

      #pragma omp for
      for (Py_ssize_t i = 0; i < num_rows; ++i) {
        for (Py_ssize_t j = 0; j < num_cols; ++j) {
	  double* datain = (double*)(PyArray_DATA(ain)
            + i * PyArray_STRIDES(ain)[0] + j * PyArray_STRIDES(ain)[1]);
          double* dataout = (double*)(PyArray_DATA(aout)
            + i * PyArray_STRIDES(aout)[0] + j * PyArray_STRIDES(aout)[1]);
          continuum_processing_f(datain, dataout, dataawl, indices, spectrum_length);
	}
      }

      free(indices);
      } // pragma omp parallel
    }

    return Py_None;
}

/* Function accepting 1d array and returning continuum spectrum. */

PyDoc_STRVAR(ccontinuum_continuum_doc,
"continuum(spectrum)\n\
\n\
Return continuum of the spectrum or each spectrum in image.");

static PyObject *
ccontinuum_continuum(PyObject *self, PyObject *args)
{
    return continuum_generic_impl(self, args, &continuum);
}

/* Function accepting 1d array and returning continuum removed spectrum. */

PyDoc_STRVAR(ccontinuum_continuum_removed_doc,
"continuum_removed(spectrum)\n\
\n\
Return continuum removed spectrum or image.");

static PyObject *
ccontinuum_continuum_removed(PyObject *self, PyObject *args)
{
  return continuum_generic_impl(self, args, &continuum_removed);
}

/* List of functions defined in the module */

static PyMethodDef ccontinuum_methods[] = {
    {"continuum_removed", ccontinuum_continuum_removed, METH_VARARGS,
      ccontinuum_continuum_removed_doc},
    {"continuum", ccontinuum_continuum, METH_VARARGS, ccontinuum_continuum_doc},
    {NULL,              NULL}           /* sentinel */
};

PyDoc_STRVAR(module_doc,
"This is a template module just for instruction.");


static int
ccontinuum_exec(PyObject *m)
{
    /* Slot initialization is subject to the rules of initializing globals.
       C99 requires the initializers to be "address constants".  Function
       designators like 'PyType_GenericNew', with implicit conversion to
       a pointer, are valid C99 address constants.

       However, the unary '&' operator applied to a non-static variable
       like 'PyBaseObject_Type' is not required to produce an address
       constant.  Compilers may support this (gcc does), MSVC does not.

       Both compilers are strictly standard conforming in this particular
       behavior.
    */
    // Null_Type.tp_base = &PyBaseObject_Type;
    // Str_Type.tp_base = &PyUnicode_Type;

    /* Finalize the type object including setting type of the new type
     * object; doing it here is required for portability, too. */
    // if (PyType_Ready(&Xxo_Type) < 0)
    //     goto fail;

    /* Add some symbolic constants to the module */
    if (ErrorObject == NULL) {
        ErrorObject = PyErr_NewException("continuum.error", NULL, NULL);
        if (ErrorObject == NULL)
            goto fail;
    }
    Py_INCREF(ErrorObject);
    PyModule_AddObject(m, "error", ErrorObject);

    /* Add Str */
    // if (PyType_Ready(&Str_Type) < 0)
    //     goto fail;
    // PyModule_AddObject(m, "Str", (PyObject *)&Str_Type);

    /* Add Null */
    // if (PyType_Ready(&Null_Type) < 0)
    //     goto fail;
    // PyModule_AddObject(m, "Null", (PyObject *)&Null_Type);
    return 0;
 fail:
    Py_XDECREF(m);
    return -1;
}

static struct PyModuleDef_Slot ccontinuum_slots[] = {
    {Py_mod_exec, ccontinuum_exec},
    {0, NULL},
};

static struct PyModuleDef ccontinuummodule = {
    PyModuleDef_HEAD_INIT,
    "ccontinuum",
    module_doc,
    0,
    ccontinuum_methods,
    ccontinuum_slots,
    NULL,
    NULL,
    NULL
};

/* Export function for the module (*must* be called PyInit_ccontinuum) */

PyMODINIT_FUNC
PyInit_ccontinuum(void)
{
    import_array();
    return PyModuleDef_Init(&ccontinuummodule);
}
