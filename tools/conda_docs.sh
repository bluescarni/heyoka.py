#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${HEYOKA_PY_CONDA_ARCH}.sh -O miniforge.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniforge/bin:$PATH"
bash miniforge.sh -b -p $HOME/miniforge
conda create -y -p $deps_dir c-compiler cxx-compiler python=${HEYOKA_PY_PY_VERSION} git pybind11 \
    ninja 'numpy>=2' mpmath cmake llvmdev tbb-devel tbb astroquery libboost-devel \
    'mppp=2.*' sleef fmt skyfield spdlog myst-nb matplotlib sympy scipy cartopy cloudpickle \
    'sphinx=8.*' 'sphinx-book-theme=1.*' 'sphinxcontrib-bibtex=2.6.*'
source activate $deps_dir

# NOTE: pykep not on linux arm64 yet.
if [ "$HEYOKA_PY_CONDA_ARCH" == "Linux-x86_64" ]; then
    conda install -y pykep
fi

export HEYOKA_PY_PROJECT_DIR=`pwd`

# Clear the compilation flags set up by conda.
unset CXXFLAGS
unset CFLAGS

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

ninja -j2 -v install

cd ../../

mkdir build
cd build

cmake -G Ninja ../ \
    -DCMAKE_INSTALL_PREFIX=$deps_dir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH=$deps_dir \
    -DCMAKE_CXX_FLAGS_DEBUG="-g -Og"

ninja -j2 -v install

cd ../tools

python ci_test_runner.py

if [ "$HEYOKA_PY_CONDA_ARCH" == "Linux-x86_64" ]; then
    cd $HEYOKA_PY_PROJECT_DIR
    cd doc
    make html doctest
fi

set +e
set +x
