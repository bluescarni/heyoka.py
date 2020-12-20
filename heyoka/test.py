# Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

def run_test_suite():
    from . import make_nbody_sys, taylor_adaptive

    sys = make_nbody_sys(2, masses=[1.1,2.1], Gconst=1)
    ta = taylor_adaptive(sys, [1,2,3,4,5,6,7,8,9,10,11,12])
