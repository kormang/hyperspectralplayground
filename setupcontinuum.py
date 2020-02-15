from distutils.core import setup, Extension
import numpy as np

module1 = Extension('continuum',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = [
                        '/usr/local/include',
                        np.get_include()
                    ],
                    libraries = ['gomp'],
                    library_dirs = ['/usr/local/lib'],
                    extra_compile_args = ['-fopenmp'],
                    sources = ['continuummodule.c', 'continuum.c'])

setup (name = 'Continuum',
       version = '1.0',
       description = 'Continuum processing for HSI',
       author = 'Marko Ivanovic',
       author_email = 'ivanovic.marko@yandex.com',
       url = 'https://docs.python.org/extending/building',
       long_description = '''
Continuum processing for HSI.
''',
       ext_modules = [module1])
