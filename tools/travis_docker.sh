#!/usr/bin/env bash

# Echo each command.
set -x

# Exit on error.
set -e

docker run --rm -v `pwd`:/home/conda/heyoka.py quay.io/condaforge/linux-anvil-ppc64le bash /home/conda/heyoka.py/tools/travis_ubuntu_ppc64.sh

set +e
set +x
