#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Install conda+deps.
curl -L -o miniconda.sh https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-ppc64le.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda create -y -q -p $deps_dir cxx-compiler c-compiler cmake llvmdev tbb-devel tbb astroquery libboost-devel sleef xtensor xtensor-blas blas blas-devel fmt skyfield spdlog python pybind11 'numpy>=2' mpmath sympy cloudpickle mppp git make
source activate $deps_dir

# Checkout, build and install heyoka's HEAD.
git clone --depth 1 https://github.com/bluescarni/heyoka.git heyoka_cpp
cd heyoka_cpp
mkdir build
cd build

# GCC build.
cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Release -DHEYOKA_WITH_SLEEF=yes -DHEYOKA_WITH_MPPP=yes -DHEYOKA_INSTALL_LIBDIR=lib
make VERBOSE=1 install

cd ../../

# Create the build dir and cd into it.
mkdir build
cd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Release
make VERBOSE=1 install

echo "INSTALL DONE"

cd ../tools

echo "MOVED OUT"

$deps_dir/bin/python ci_test_runner.py

echo "PYTHON RUN"

set +e
set +x
