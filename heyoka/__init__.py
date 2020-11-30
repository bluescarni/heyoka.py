# Copyright 2019-2020 Francesco Biscani (bluescarni@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import llvmlite

# Version setup.
from ._version import __version__
# We import the sub-modules into the root namespace
from .core import *
