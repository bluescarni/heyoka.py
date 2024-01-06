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
mamba create -y -p $deps_dir python=3.9 c-compiler cxx-compiler git pybind11 numpy mpmath cmake llvmdev tbb-devel tbb astroquery libboost-devel 'mppp=1.*' sleef fmt spdlog myst-nb matplotlib sympy scipy pykep cloudpickle 'sphinx=7.*' 'sphinx-book-theme=1.*'
source activate $deps_dir

export HEYOKA_PY_PROJECT_DIR=`pwd`

# Checkout, build and install heyoka's HEAD.
git clone --depth 1 https://github.com/bluescarni/heyoka.git heyoka_cpp
cd heyoka_cpp
mkdir build
cd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_WITH_MPPP=yes -DHEYOKA_WITH_SLEEF=yes -DBoost_NO_BOOST_CMAKE=ON
make -j4 VERBOSE=1 install

cd ../../

# Create the build dir and cd into it.
mkdir build
cd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_PY_ENABLE_IPO=yes -DBoost_NO_BOOST_CMAKE=ON
make -j4 VERBOSE=1 install

cd ../tools

python ci_test_runner.py

cd $HEYOKA_PY_PROJECT_DIR

cd doc

make html linkcheck

set +e
set +x
