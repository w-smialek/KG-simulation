# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# from Cython.Build import cythonize
# import numpy

# libs = ['m', 'fftw3']
# args = ['-std=c99', '-O3']
# sources = ['test.pyx', 'src/fft_stuff.c']
# include = [numpy.get_include()]
# linkerargs = ['-Wl,-rpath,lib']
# libdirs = ['lib']


# extensions = [
#     Extension("test",
#               sources=sources,
#               include_dirs=include,
#               libraries=libs,
#               library_dirs=libdirs,
#               extra_compile_args=args,
#               extra_link_args=linkerargs)
# ]

# setup(name='test',
#       packages=['test'],
#       ext_modules=cythonize(extensions),
#       )

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy

libs = ['m', 'fftw3']
args = ['-std=c99']
sources = ['test.pyx', 'src/fft_stuff.cpp']
include = [numpy.get_include()]
linkerargs = ['-Wl,-rpath,lib']
libdirs = ['lib']
lang = 'c++'


extensions = [
    Extension("test",
              sources=sources,
              include_dirs=include,
              libraries=libs,
              library_dirs=libdirs,
              extra_compile_args=args,
              extra_link_args=linkerargs,
              language=lang)
]

setup(
    name='test',
    packages=['test'],
    ext_modules=cythonize(extensions)
)
