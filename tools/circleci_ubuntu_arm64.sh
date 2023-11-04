#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniforge/bin:$PATH"
bash miniforge.sh -b -p $HOME/miniforge
mamba create -y -q -p $deps_dir cxx-compiler c-compiler cmake llvmdev tbb-devel tbb astroquery boost-cpp sleef fmt spdlog python=3.10 pybind11 numpy mpmath sympy scipy cloudpickle sphinx myst-nb matplotlib sphinx-book-theme mppp
source activate $deps_dir

# Checkout, build and install heyoka's HEAD.
git clone https://github.com/bluescarni/heyoka.git heyoka_cpp
cd heyoka_cpp
mkdir build
cd build

# GCC build.
cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_WITH_SLEEF=yes -DBoost_NO_BOOST_CMAKE=ON -DHEYOKA_WITH_MPPP=yes
make -j2 VERBOSE=1 install

cd ../../

# Create the build dir and cd into it.
mkdir build
cd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_PY_ENABLE_IPO=yes -DBoost_NO_BOOST_CMAKE=ON
make -j2 VERBOSE=1 install

cd ../tools

python ci_test_runner.py

set +e
set +x
