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


# 1 - We read for what python wheels have to be built.
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

# Python mandatory deps.
/opt/python/${PYTHON_DIR}/bin/pip install numpy==${NUMPY_VERSION} cloudpickle
# Python optional deps.
/opt/python/${PYTHON_DIR}/bin/pip install sympy mpmath

# In the pagmo2/manylinux228_x86_64_with_deps:latest image in dockerhub
# the working directory is /root/install, we will install heyoka there.
cd /root/install

# Install heyoka.
git clone https://github.com/bluescarni/heyoka.git
cd heyoka

mkdir build
cd build
cmake -DBoost_NO_BOOST_CMAKE=ON \
    -DHEYOKA_WITH_MPPP=yes \
    -DHEYOKA_WITH_SLEEF=yes \
    -DHEYOKA_ENABLE_IPO=ON \
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
make -j2 install

# Making the wheel and installing it
cd wheel
# Copy the installed heyoka.py files into the current dir.
cp -r `/opt/python/${PYTHON_DIR}/bin/python -c 'import site; print(site.getsitepackages()[0])'`/heyoka ./
# Create the wheel and repair it.
/opt/python/${PYTHON_DIR}/bin/python setup.py bdist_wheel
auditwheel repair dist/heyoka* -w ./dist2
# Try to install it and run the tests.
cd /
/opt/python/${PYTHON_DIR}/bin/pip install ${GITHUB_WORKSPACE}/build/wheel/dist2/heyoka*
/opt/python/${PYTHON_DIR}/bin/python -c "import heyoka; heyoka.test.run_test_suite();"

# Upload to pypi. This variable will contain something if this is a tagged build (vx.y.z), otherwise it will be empty.
# if [[ "${PYGMO_RELEASE_VERSION}" != "" ]]; then
# 	echo "Release build detected, creating the source code archive."
# 	cd ${GITHUB_WORKSPACE}
# 	TARBALL_NAME=${GITHUB_WORKSPACE}/build/wheel/dist2/pygmo-${PYGMO_RELEASE_VERSION}.tar
# 	git archive --format=tar --prefix=pygmo2/ -o ${TARBALL_NAME} ${BRANCH_NAME}
# 	tar -rf ${TARBALL_NAME} --transform "s,^build/wheel/pygmo.egg-info,pygmo2," build/wheel/pygmo.egg-info/PKG-INFO
# 	gzip -9 ${TARBALL_NAME}
# 	echo "... uploading all to PyPi."
# 	/opt/python/${PYTHON_DIR}/bin/pip install twine
# 	/opt/python/${PYTHON_DIR}/bin/twine upload -u ci4esa ${GITHUB_WORKSPACE}/build/wheel/dist2/pygmo*
# fi

set +e
set +x

