from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy

extensions = [
    Extension("ftinterp", sources=["./ftinterp.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=['/std:c++20'], language="c++")
]

setup(
    name="ftinterp",
    ext_modules=cythonize(extensions, annotate=False)
)