from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(name="cythext.fast.core", sources=["./cythext/fast/core.pyx"],
              include_dirs=['/usr/include/openblas', 'cythext/fast/', numpy.get_include()],
              libraries=['openblas'],
              extra_compile_args=["-O3", "-funroll-loops", "-std=c++11"],
              language="c++")
]

setup(
    name="cythext",
    ext_modules=cythonize(extensions, nthreads=8, language="c++"),
    packages=find_packages()
)