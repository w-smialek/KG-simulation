from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
from Cython.Distutils import build_ext

libs = ['m', 'fftw3']
args = ['-std=c++2a']
sources = ['cy_solver_field.pyx', 'fft_conv/src/fft_stuff.cpp']
include = [numpy.get_include()]
linkerargs = ['-Wl,-rpath,lib']
libdirs = ['lib']
lang = 'c++'


extensions = [
    Extension("cy_solver_field",
              sources=['cy_solver_field.pyx', 'fft_conv/src/fft_stuff.cpp'],
              include_dirs=[numpy.get_include()],
              libraries=['m', 'fftw3'],
              library_dirs=['lib'],
              extra_compile_args=['-std=c++2a'],
              extra_link_args=['-Wl,-rpath,lib'],
              language='c++'),
    Extension("fftlib",
              sources=['fft_conv/src/fftlib.pyx', 'fft_conv/src/fft_stuff.cpp'],
              include_dirs=[numpy.get_include()],
              libraries=['m', 'fftw3'],
              library_dirs=['lib'],
              extra_compile_args=['-std=c99', '-O3'],
              extra_link_args=['-Wl,-rpath,lib'],
              language='c++'),
    Extension("test",
              sources=['fft_conv/test.pyx', 'fft_conv/src/fft_stuff.cpp'],
              include_dirs=[numpy.get_include()],
              libraries=['m', 'fftw3'],
              library_dirs=['lib'],
              extra_compile_args=['-std=c99'],
              extra_link_args=['-Wl,-rpath,lib'],
              language='c++'),
    Extension("colorize", 
              sources=["fastccolor/colorize.pyx"], 
              include_dirs=[numpy.get_include()], 
              language="c"),
    Extension("cysolverNew", 
              sources=['mycyrk/cy/cysolverNew.pyx'],
            #   ,'mycyrk/cy/baseline_func.cpp',
            #            'mycyrk/cy/common.c','mycyrk/cy/common.cpp','mycyrk/cy/cysolverbase_class.cpp',
            #            'mycyrk/cy/cysolverresult_class.cpp','mycyrk/cy/dense.cpp','mycyrk/cy/pysolver_cyhook.cpp','mycyrk/cy/rk.cpp',
            #            'mycyrk/cy/rk_step.c'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-std=c++2a'],
              language="c++")
]

setup(
    ext_modules=cythonize('mycyrk/cy/cysolverNew.pyx'),
)

setup(
    name='kgsim',
    packages=["cy_solver_field","fftlib","test","colorize","cysolverNew","cython"],
    cmdclass = {'build_ext': build_ext},
    ext_modules=cythonize(extensions),
compiler_directives={'language_level' : "3str"}
)

import shutil

shutil.move("fftlib.cpython-38-x86_64-linux-gnu.so", "fft_conv/src/fftlib.cpython-38-x86_64-linux-gnu.so")
shutil.move("test.cpython-38-x86_64-linux-gnu.so", "fft_conv/test.cpython-38-x86_64-linux-gnu.so")
shutil.move("colorize.cpython-38-x86_64-linux-gnu.so", "fastccolor/colorize.cpython-38-x86_64-linux-gnu.so")
shutil.move("cysolverNew.cpython-38-x86_64-linux-gnu.so", "mycyrk/cy/cysolverNew.cpython-38-x86_64-linux-gnu.so")
