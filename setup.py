from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy

extensions = [
    Extension("cy_solver", sources=["./cy_solver.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=['/std:c++20'], language="c++")
]

setup(
    name="cy_solver",
    ext_modules=cythonize(extensions, annotate=False)
)