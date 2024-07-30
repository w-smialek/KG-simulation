from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy
import cython
from Cython.Distutils import build_ext

# file_name = "cysolverNew"

# extensions = [
#     Extension(file_name, sources=["./cy/%s.pyx"%file_name], include_dirs=[numpy.get_include()], extra_compile_args=['/std:c++20'], language="c++")
# ]

# setup(
#     name=file_name,
#     ext_modules=cythonize(extensions, annotate=True)
# )

filesstr = ["cysolverNew"]

extensions = [
    Extension(fstr, [".\cy\%s.pyx"%fstr], include_dirs=[numpy.get_include(),"../array","../nb","../rk","../utils","../cy"], extra_compile_args=['/std:c++20'], language="c++")
    for fstr in filesstr
]

setup(
    name="mycyrk",
    cmdclass = {'build_ext': build_ext},
    ext_modules=cythonize(extensions, annotate=False)#, build_dir="./cy")
)

import shutil
shutil.move('./cysolverNew.cp310-win_amd64.pyd','./cy/cysolverNew.cp310-win_amd64.pyd')