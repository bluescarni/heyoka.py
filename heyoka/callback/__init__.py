# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from .. import core as _core

_lst = list(filter(lambda name: name.startswith("_callback_"), dir(_core)))

for _name in _lst:
    exec(f"from ..core import {_name} as {_name[10:]}")

del _core
del _lst
del _name
