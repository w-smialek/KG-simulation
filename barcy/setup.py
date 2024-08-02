from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy

extensions = [
    Extension("barcy", sources=["./barcy.pyx"], include_dirs=[numpy.get_include()], language="c")
]

setup(
    name="barcy",
    ext_modules=cythonize(extensions, annotate=False)
)