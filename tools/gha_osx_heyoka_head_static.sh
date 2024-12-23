#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${HEYOKA_PY_CONDA_ARCH}.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda create -y -p $deps_dir python=${HEYOKA_PY_PY_VERSION} c-compiler cxx-compiler git pybind11 'numpy>=2' \
    ninja cmake llvmdev tbb-devel tbb astroquery libboost-devel sleef fmt skyfield \
    spdlog sympy cloudpickle 'mppp=2.*' numba
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
    -DHEYOKA_WITH_SLEEF=yes \
    -DHEYOKA_WITH_MPPP=yes \
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
