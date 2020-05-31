from distutils.core import setup, Extension
import numpy as np

module1 = Extension('ccontinuum',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = [
                        '/usr/local/include',
                        np.get_include()
                    ],
                    libraries = ['gomp'],
                    library_dirs = ['/usr/local/lib'],
                    extra_compile_args = ['-fopenmp', '-O3'],
                    sources = ['ccontinuummodule.c', 'ccontinuum.c'])

setup (name = 'CContinuum',
       version = '1.0',
       description = 'C continuum processing for HSI',
       author = 'Marko Ivanovic',
       author_email = 'ivanovic.marko@yandex.com',
       url = 'https://docs.python.org/extending/building',
       long_description = '''
C continuum processing for HSI.
''',
       ext_modules = [module1])
