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
# NOTE: the scipy pin is necessary otherwise the
# notebooks do not execute correctly due to pykep
# using deprecated scipy functions.
conda create -y -p $deps_dir c-compiler cxx-compiler python=3.12 git pybind11 \
    ninja 'numpy<2' mpmath cmake llvmdev tbb-devel tbb astroquery libboost-devel \
    'mppp=2.*' sleef fmt skyfield spdlog myst-nb matplotlib sympy 'scipy<1.14' pykep cloudpickle \
    'sphinx=7.*' 'sphinx-book-theme=1.*'
source activate $deps_dir

export HEYOKA_PY_PROJECT_DIR=`pwd`

# Checkout, build and install heyoka's HEAD.
git clone --depth 1 https://github.com/bluescarni/heyoka.git heyoka_cpp
cd heyoka_cpp
mkdir build
cd build

cmake -G Ninja ../ \
    -DCMAKE_INSTALL_PREFIX=$deps_dir \
    -DCMAKE_PREFIX_PATH=$deps_dir \
    -DHEYOKA_WITH_MPPP=yes \
    -DHEYOKA_WITH_SLEEF=yes

ninja -v install

cd ../../

mkdir build
cd build

cmake -G Ninja ../ \
    -DCMAKE_INSTALL_PREFIX=$deps_dir \
    -DCMAKE_PREFIX_PATH=$deps_dir \
    -DHEYOKA_PY_ENABLE_IPO=yes

ninja -v install

cd ../tools

python ci_test_runner.py

cd $HEYOKA_PY_PROJECT_DIR

cd doc

make html doctest

set +e
set +x
