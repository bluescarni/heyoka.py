#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Install wget.
sudo yum -y install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-ppc64le.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
mamba create -y -q -p $deps_dir cxx-compiler c-compiler cmake llvmdev tbb-devel tbb astroquery boost-cpp sleef xtensor xtensor-blas blas blas-devel fmt spdlog python=3.8 pybind11 numpy mpmath sympy cloudpickle mppp git make
source activate $deps_dir

# Checkout, build and install heyoka's HEAD.
git clone https://github.com/bluescarni/heyoka.git heyoka_cpp
cd heyoka_cpp
mkdir build
cd build

# GCC build.
cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Release -DHEYOKA_WITH_SLEEF=yes -DHEYOKA_WITH_MPPP=yes -DBoost_NO_BOOST_CMAKE=ON -DHEYOKA_PY_SETUP_DOCS=no
make -j2 VERBOSE=1 install

cd ../../

# Create the build dir and cd into it.
mkdir build
cd build

cmake ../heyoka.py -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Release -DBoost_NO_BOOST_CMAKE=ON
make -j2 VERBOSE=1 install

cd

python -c "from heyoka import test; test.run_test_suite()"

set +e
set +x
