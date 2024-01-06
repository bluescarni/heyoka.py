#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install build-essential wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniforge/bin:$PATH"
bash miniforge.sh -b -p $HOME/miniforge
mamba create -y -q -p $deps_dir python=3.10 git pybind11 numpy mpmath cmake llvmdev tbb-devel tbb astroquery libboost-devel 'mppp=1.*' sleef fmt spdlog myst-nb matplotlib sympy scipy pykep cloudpickle 'sphinx=7.*' 'sphinx-book-theme=1.*'
source activate $deps_dir

export HEYOKA_PY_PROJECT_DIR=`pwd`

# Checkout, build and install heyoka's HEAD.
git clone --depth 1 https://github.com/bluescarni/heyoka.git heyoka_cpp
cd heyoka_cpp
mkdir build
cd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DHEYOKA_WITH_MPPP=yes -DHEYOKA_WITH_SLEEF=yes -DBoost_NO_BOOST_CMAKE=ON
make -j4 VERBOSE=1 install

cd ../../

# Create the build dir and cd into it.
mkdir build
cd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DHEYOKA_PY_ENABLE_IPO=yes -DBoost_NO_BOOST_CMAKE=ON
make -j4 VERBOSE=1 install

cd ../tools

python ci_test_runner.py

cd $HEYOKA_PY_PROJECT_DIR

cd doc

# NOTE: do not run linkcheck on the main branch,
# as we don't want to prevent the docs upload
# in case of transient network errors.
if [[ "${CIRCLE_BRANCH}" == "main" ]]; then
    make html
else
    make html linkcheck
fi

if [[ ! -z "${CI_PULL_REQUEST}" ]]; then
    echo "Testing a pull request, the generated documentation will not be uploaded.";
    exit 0;
fi

if [[ "${CIRCLE_BRANCH}" != "main" ]]; then
    echo "Branch is not main, the generated documentation will not be uploaded.";
    exit 0;
fi

# Check out the gh_pages branch in a separate dir.
cd ../
git config --global push.default simple
git config --global user.name "CircleCI"
git config --global user.email "bluescarni@gmail.com"
set +x
git clone "https://${GH_TOKEN}@github.com/bluescarni/heyoka.py.git" heyoka_py_gh_pages -q
set -x
cd heyoka_py_gh_pages
git checkout -b gh-pages --track origin/gh-pages;
git rm -fr *;
mv ../doc/_build/html/* .;
git add *;
# We assume here that a failure in commit means that there's nothing
# to commit.
git commit -m "Update Sphinx documentation, commit ${CIRCLE_SHA1} [skip ci]." || exit 0
PUSH_COUNTER=0
until git push -q
do
    git pull -q
    PUSH_COUNTER=$((PUSH_COUNTER + 1))
    if [ "$PUSH_COUNTER" -gt 3 ]; then
        echo "Push failed, aborting.";
        exit 1;
    fi
done

set +e
set +x
