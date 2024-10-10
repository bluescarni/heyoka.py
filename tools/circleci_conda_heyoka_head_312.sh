#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniforge/bin:$PATH"
bash miniforge.sh -b -p $HOME/miniforge
conda create -y -p $deps_dir python=3.12 c-compiler cxx-compiler git pybind11 'numpy<2' mpmath cmake llvmdev tbb-devel tbb libboost-devel 'mppp=1.*' sleef fmt skyfield spdlog sympy cloudpickle
source activate $deps_dir

# Checkout, build and install heyoka's HEAD.
git clone --depth 1 https://github.com/bluescarni/heyoka.git heyoka_cpp
cd heyoka_cpp
mkdir build
cd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_WITH_MPPP=yes -DHEYOKA_WITH_SLEEF=yes
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
