from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy

# extensions = [
#     Extension("cy_solver_field", sources=["./cy_solver_field.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=['-std=c++2a'], language="c++")
# ]

# setup(
#     name="cy_solver_field",
#     ext_modules=cythonize(extensions, annotate=False, language_level=3)
# )

libs = ['m', 'fftw3']
args = ['-std=c++2a']
sources = ['cy_solver_field.pyx', 'fft_conv/src/fft_stuff.cpp']
include = [numpy.get_include()]
linkerargs = ['-Wl,-rpath,lib']
libdirs = ['lib']
lang = 'c++'


extensions = [
    Extension("cy_solver_field",
              sources=sources,
              include_dirs=include,
              libraries=libs,
              library_dirs=libdirs,
              extra_compile_args=args,
              extra_link_args=linkerargs,
              language=lang)
]

setup(
    name='cy_solver_field',
    packages=['cy_solver_field'],
    ext_modules=cythonize(extensions)
)
