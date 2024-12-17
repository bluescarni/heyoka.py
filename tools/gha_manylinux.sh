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

# Detect the Python version.
if [[ ${HEYOKA_PY_BUILD_TYPE} == *39* ]]; then
	PYTHON_DIR="cp39-cp39"
elif [[ ${HEYOKA_PY_BUILD_TYPE} == *310* ]]; then
	PYTHON_DIR="cp310-cp310"
elif [[ ${HEYOKA_PY_BUILD_TYPE} == *311* ]]; then
	PYTHON_DIR="cp311-cp311"
elif [[ ${HEYOKA_PY_BUILD_TYPE} == *312* ]]; then
	PYTHON_DIR="cp312-cp312"
elif [[ ${HEYOKA_PY_BUILD_TYPE} == *313* ]]; then
	PYTHON_DIR="cp313-cp313"
else
	echo "Invalid build type: ${HEYOKA_PY_BUILD_TYPE}"
	exit 1
fi

# Report the inferred directory where python is found.
echo "PYTHON_DIR: ${PYTHON_DIR}"

# The heyoka version to be used for releases.
export HEYOKA_VERSION_RELEASE="7.0.0"

# Check if this is a release build.
if [[ "${GITHUB_REF}" == "refs/tags/v"* ]]; then
    echo "Tag build detected"
	export HEYOKA_PY_RELEASE_BUILD="yes"
else
	echo "Non-tag build detected"
fi

# In the manylinux image in dockerhub the working directory is /root/install, we will install heyoka there.
cd /root/install

# Install heyoka.
if [[ "${HEYOKA_PY_RELEASE_BUILD}" == "yes" ]]; then
	curl -L -o heyoka.tar.gz https://github.com/bluescarni/heyoka/archive/refs/tags/v${HEYOKA_VERSION_RELEASE}.tar.gz
	tar xzf heyoka.tar.gz
	cd heyoka-${HEYOKA_VERSION_RELEASE}
else
	git clone --depth 1 https://github.com/bluescarni/heyoka.git
	cd heyoka
fi

mkdir build
cd build
cmake -DHEYOKA_WITH_MPPP=yes \
    -DHEYOKA_WITH_SLEEF=yes \
    -DHEYOKA_ENABLE_IPO=ON \
    -DHEYOKA_FORCE_STATIC_LLVM=yes \
    -DHEYOKA_HIDE_LLVM_SYMBOLS=yes \
    -DCMAKE_BUILD_TYPE=Release ../;
make -j4 install

cd ${GITHUB_WORKSPACE}

# NOTE: this is temporary because some libraries in the docker
# image are installed in lib64 rather than lib and they are
# not picked up properly by the linker.
export LD_LIBRARY_PATH="/usr/local/lib64:/usr/local/lib"

if [[ "${HEYOKA_PY_BUILD_SDIST}" == "yes" ]]; then
	# Build the heyoka.py sdist.
	/opt/python/${PYTHON_DIR}/bin/python -m build . --sdist
	# Try to install it and run the tests.
	/opt/python/${PYTHON_DIR}/bin/pip install dist/heyoka*
	cd ${GITHUB_WORKSPACE}/tools
	/opt/python/${PYTHON_DIR}/bin/python ci_test_runner.py
	cd /

	# Upload to PyPI.
	if [[ "${HEYOKA_PY_RELEASE_BUILD}" == "yes" ]]; then
		/opt/python/${PYTHON_DIR}/bin/pip install twine
		/opt/python/${PYTHON_DIR}/bin/twine upload -u __token__ ${GITHUB_WORKSPACE}/dist/heyoka*
	fi
else
	# Build the heyoka.py wheel.
	/opt/python/${PYTHON_DIR}/bin/pip wheel . -v
	# Repair it.
	auditwheel repair ./heyoka*.whl -w ./repaired_wheel
	# Try to install it and run the tests.
	unset LD_LIBRARY_PATH
	cd /
	/opt/python/${PYTHON_DIR}/bin/pip install ${GITHUB_WORKSPACE}/repaired_wheel/heyoka*
	cd ${GITHUB_WORKSPACE}/tools
	/opt/python/${PYTHON_DIR}/bin/python ci_test_runner.py
	cd /

	# Upload to PyPI.
	if [[ "${HEYOKA_PY_RELEASE_BUILD}" == "yes" ]]; then
		/opt/python/${PYTHON_DIR}/bin/pip install twine
		/opt/python/${PYTHON_DIR}/bin/twine upload -u __token__ ${GITHUB_WORKSPACE}/repaired_wheel/heyoka*
	fi
fi

set +e
set +x
