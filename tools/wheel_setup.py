import os
from setuptools import setup
from setuptools.dist import Distribution
import sys

NAME = "heyoka"
VERSION = "@heyoka.py_VERSION@"
DESCRIPTION = "Python library for ODE integration via Taylor's method and LLVM"
LONG_DESCRIPTION = "heyoka is a Python library for the integration of ordinary differential equations (ODEs) via Taylor's method, based on automatic differentiation techniques and aggressive just-in-time compilation via LLVM."
URL = "https://github.com/bluescarni/heyoka.py"
AUTHOR = "Francesco Biscani, Dario Izzo"
AUTHOR_EMAIL = "bluescarni@gmail.com"
LICENSE = "MPL-2.0"
CLASSIFIERS = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 5 - Production/Stable",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
    "Programming Language :: Python :: 3",
]
KEYWORDS = "science math physics ode"
INSTALL_REQUIRES = ["numpy==@_HEYOKA_PY_NPY_MAJOR_MINOR@.*", "cloudpickle"]
PLATFORMS = ["Unix"]


class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    platforms=PLATFORMS,
    install_requires=INSTALL_REQUIRES,
    packages=["heyoka", "heyoka.model", "heyoka.callback"],
    # Include pre-compiled extension
    package_data={"heyoka": [f for f in os.listdir("heyoka/") if f.endswith(".so")]},
    distclass=BinaryDistribution,
)
