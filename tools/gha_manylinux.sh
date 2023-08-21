#!/usr/bin/env bash

# Echo each command.
set -x

# Exit on error.
set -e

# Report on the environrnt variables used for this build.
echo "HEYOKA_PY_BUILD_TYPE: ${HEYOKA_PY_BUILD_TYPE}"
echo "GITHUB_REF: ${GITHUB_REF}"
echo "GITHUB_WORKSPACE: ${GITHUB_WORKSPACE}"
# No idea why but this following line seems to be necessary (added: 18/01/2023)
git config --global --add safe.directory ${GITHUB_WORKSPACE}
BRANCH_NAME=`git rev-parse --abbrev-ref HEAD`
echo "BRANCH_NAME: ${BRANCH_NAME}"

# Read for what python wheels have to be built.
if [[ ${HEYOKA_PY_BUILD_TYPE} == *38* ]]; then
	PYTHON_DIR="cp38-cp38"
elif [[ ${HEYOKA_PY_BUILD_TYPE} == *39* ]]; then
	PYTHON_DIR="cp39-cp39"
elif [[ ${HEYOKA_PY_BUILD_TYPE} == *310* ]]; then
	PYTHON_DIR="cp310-cp310"
elif [[ ${HEYOKA_PY_BUILD_TYPE} == *311* ]]; then
	PYTHON_DIR="cp311-cp311"
elif [[ ${HEYOKA_PY_BUILD_TYPE} == *312* ]]; then
	PYTHON_DIR="cp312-cp312"
else
	echo "Invalid build type: ${HEYOKA_PY_BUILD_TYPE}"
	exit 1
fi

# Report the inferred directory whwere python is found.
echo "PYTHON_DIR: ${PYTHON_DIR}"

# The numpy version heyoka.py will be built against.
export NUMPY_VERSION="1.24.*"

# The heyoka version to be used for releases.
export HEYOKA_VERSION_RELEASE="1.1.0"

# Check if this is a release build.
if [[ "${GITHUB_REF}" == "refs/tags/v"* ]]; then
    echo "Tag build detected"
	export HEYOKA_PY_RELEASE_BUILD="yes"
else
	echo "Non-tag build detected"
fi

# Python mandatory deps.
/opt/python/${PYTHON_DIR}/bin/pip install numpy==${NUMPY_VERSION} cloudpickle
# Python optional deps.
/opt/python/${PYTHON_DIR}/bin/pip install sympy mpmath

# In the pagmo2/manylinux228_x86_64_with_deps:latest image in dockerhub
# the working directory is /root/install, we will install heyoka there.
cd /root/install

# Install heyoka.
if [[ "${HEYOKA_PY_RELEASE_BUILD}" == "yes" ]]; then
	curl -L -o heyoka.tar.gz https://github.com/bluescarni/heyoka/archive/refs/tags/v${HEYOKA_VERSION_RELEASE}.tar.gz
	tar xzf heyoka.tar.gz
	cd heyoka-${HEYOKA_VERSION_RELEASE}
else
	git clone https://github.com/bluescarni/heyoka.git
	cd heyoka
fi

mkdir build
cd build
cmake -DBoost_NO_BOOST_CMAKE=ON \
    -DHEYOKA_WITH_MPPP=yes \
    -DHEYOKA_WITH_SLEEF=yes \
    -DHEYOKA_ENABLE_IPO=ON \
    -DHEYOKA_FORCE_STATIC_LLVM=yes \
    -DHEYOKA_HIDE_LLVM_SYMBOLS=yes \
    -DCMAKE_BUILD_TYPE=Release ../;
make -j4 install

# Install heyoka.py.
cd ${GITHUB_WORKSPACE}
mkdir build
cd build
cmake -DBoost_NO_BOOST_CMAKE=ON \
	-DCMAKE_BUILD_TYPE=Release \
	-DHEYOKA_PY_ENABLE_IPO=ON \
	-DPython3_EXECUTABLE=/opt/python/${PYTHON_DIR}/bin/python ../;
make -j4 install

# Making the wheel and installing it
cd wheel
# Move the installed heyoka.py files into the current dir.
mv `/opt/python/${PYTHON_DIR}/bin/python -c 'import site; print(site.getsitepackages()[0])'`/heyoka ./
# Create the wheel and repair it.
# NOTE: this is temporary because some libraries in the docker
# image are installed in lib64 rather than lib and they are
# not picked up properly by the linker.
export LD_LIBRARY_PATH="/usr/local/lib64:/usr/local/lib"
/opt/python/${PYTHON_DIR}/bin/python setup.py bdist_wheel
auditwheel repair dist/heyoka* -w ./dist2
# Try to install it and run the tests.
unset LD_LIBRARY_PATH
cd /
/opt/python/${PYTHON_DIR}/bin/pip install ${GITHUB_WORKSPACE}/build/wheel/dist2/heyoka*
cd ${GITHUB_WORKSPACE}/tools
/opt/python/${PYTHON_DIR}/bin/python ci_test_runner.py
cd /

# Upload to PyPI.
if [[ "${HEYOKA_PY_RELEASE_BUILD}" == "yes" ]]; then
	/opt/python/${PYTHON_DIR}/bin/pip install twine
	/opt/python/${PYTHON_DIR}/bin/twine upload -u __token__ ${GITHUB_WORKSPACE}/build/wheel/dist2/heyoka*
fi

set +e
set +x
