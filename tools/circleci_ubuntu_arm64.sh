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
conda create -y -q -p $deps_dir cxx-compiler c-compiler cmake llvmdev tbb-devel tbb astroquery libboost-devel 'mppp=2.*' sleef fmt skyfield spdlog python=3.10 pybind11 'numpy>=2' mpmath sympy scipy cloudpickle myst-nb matplotlib 'sphinx=7.*' 'sphinx-book-theme=1.*'
source activate $deps_dir

# Checkout, build and install heyoka's HEAD.
git clone --depth 1 https://github.com/bluescarni/heyoka.git heyoka_cpp
cd heyoka_cpp
mkdir build
cd build

# GCC build.
cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_WITH_SLEEF=yes -DHEYOKA_WITH_MPPP=yes
make -j4 VERBOSE=1 install

cd ../../

# Create the build dir and cd into it.
mkdir build
cd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_PY_ENABLE_IPO=yes
make -j4 VERBOSE=1 install

cd ../tools

python ci_test_runner.py

set +e
set +x
