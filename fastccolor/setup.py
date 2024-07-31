from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy

extensions = [
    Extension("colorize", sources=["./colorize.pyx"], include_dirs=[numpy.get_include()], language="c")
]

setup(
    name="colorize",
    ext_modules=cythonize(extensions, annotate=False)
)