#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Install conda+deps.
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -y -q -p $deps_dir python=3.8 git pybind11 numpy cmake llvmdev tbb-devel tbb astroquery boost-cpp sleef 'fmt=8.0.*' 'spdlog=1.9.2' sympy cloudpickle
source activate $deps_dir

# Checkout, build and install heyoka's HEAD.
git clone https://github.com/bluescarni/heyoka.git heyoka_cpp
cd heyoka_cpp
mkdir build
cd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_WITH_SLEEF=yes -DBoost_NO_BOOST_CMAKE=ON
make -j2 VERBOSE=1 install

cd ../../

# Create the build dir and cd into it.
mkdir build
cd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_PY_ENABLE_IPO=yes -DBoost_NO_BOOST_CMAKE=ON
make -j2 VERBOSE=1 install

cd

python -c "from heyoka import test; test.run_test_suite()"

set +e
set +x
