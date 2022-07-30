// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_CUSTOM_CASTERS_HPP
#define HEYOKA_PY_CUSTOM_CASTERS_HPP

#include <pybind11/pybind11.h>

namespace pybind11::detail
{

template <>
struct type_caster<long double> {
    PYBIND11_TYPE_CASTER(long double, _("long double"));
    bool load(handle, bool);
    static handle cast(const long double &, return_value_policy, handle);
};

} // namespace pybind11::detail

#endif
