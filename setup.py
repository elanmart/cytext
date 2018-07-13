import glob
import os
import sys
from codecs import open
from distutils.command.clean import clean
from os import path

import numpy
from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension

HERE  = path.abspath(path.dirname(__file__))
DEBUG = bool(int(os.environ.get('CYTEXT_DEBUG', '0')))
CLEAN = sys.argv[1] == 'clean'


with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def cython_ext_modules():

    if CLEAN:
        return []

    compiler_directives = {'language_level': 3}

    if not DEBUG:
        compiler_directives = {**compiler_directives, **dict(
            boundscheck=False, wraparound=False, nonecheck=False, overflowcheck=False,
            initializedcheck=False, cdivision=True, embedsignature=True,
        )}

    ext = Extension(
        name="cytext.*",
        sources=["cytext/*.pyx"],
        include_dirs=[
            '/usr/include/openblas',
            './cytext',
            numpy.get_include(),
        ],
        libraries=[
            'openblas'
        ],
        extra_compile_args=[
            "-O3",
            "-funroll-loops",
            "-std=c++11"
        ],
        language="c++"
    )

    ext_module = cythonize(
        module_list=[ext],
        compiler_directives=compiler_directives,
        nthreads=8,
        language="c++",
        force=True,
    )

    return ext_module


class CythonClean(clean):
    def run(self):
        clean.run(self)

        msg = "{} does not exist -- can't clean it"

        cpp = 'cytext/*.cpp'
        so  = 'cytext/*.so'

        for pattern in [cpp, so]:
            files = list(glob.glob(pattern))
            if not files:
                print(msg.format(pattern), file=sys.stderr)

            for f in files:
                os.remove(f)


setup(
    name='cytext',
    version='0.0.1',
    url='https://github.com/elanmart/py-zsl',
    author='Marcin Elantkowski',
    author_email='marcin.elantkowski@gmail.com',
    license='BSD 3-clause',
    keywords='',
    packages=find_packages(
        exclude=[
            'notebooks', 'scripts', 'storage', 'test', 'writeup'
        ]
    ),
    install_requires=[],
    package_data={},
    entry_points={
        'console_scripts': [],
    },
    ext_modules=cython_ext_modules() + [],
    cmdclass={
        'clean': CythonClean
    }
)
