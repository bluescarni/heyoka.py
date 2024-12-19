#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -y -q -p $deps_dir python=3.12 git pybind11 'numpy>=2' mpmath cmake llvmdev \
    ninja tbb-devel tbb libboost-devel 'mppp=2.*' sleef fmt skyfield spdlog sympy \
    cloudpickle c-compiler cxx-compiler numba zlib
source activate $deps_dir

# Clear the compilation flags set up by conda.
unset CXXFLAGS
unset CFLAGS

# Checkout, build and install heyoka's HEAD.
git clone --depth 1 https://github.com/bluescarni/heyoka.git heyoka_cpp
cd heyoka_cpp
mkdir build
cd build

cmake ../ -G Ninja \
    -DCMAKE_INSTALL_PREFIX=$deps_dir \
    -DCMAKE_PREFIX_PATH=$deps_dir \
    -DHEYOKA_WITH_MPPP=yes \
    -DHEYOKA_WITH_SLEEF=yes \
    -DHEYOKA_FORCE_STATIC_LLVM=yes \
    -DHEYOKA_HIDE_LLVM_SYMBOLS=yes

ninja -j2 -v install

cd ../../

# Create the build dir and cd into it.
mkdir build
cd build

cmake ../ -G Ninja \
    -DCMAKE_INSTALL_PREFIX=$deps_dir \
    -DCMAKE_PREFIX_PATH=$deps_dir

ninja -j2 -v install

cd ../tools

python ci_test_runner.py --with-numba

set +e
set +x
