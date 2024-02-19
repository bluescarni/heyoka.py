// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_CUSTOM_CASTERS_HPP
#define HEYOKA_PY_CUSTOM_CASTERS_HPP

#include <heyoka/config.hpp>

#include <pybind11/pybind11.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

namespace pybind11::detail
{

template <>
struct type_caster<float> {
    PYBIND11_TYPE_CASTER(float, _("numpy.float32"));
    bool load(handle, bool);
    static handle cast(const float &, return_value_policy, handle);
};

template <>
struct type_caster<long double> {
    PYBIND11_TYPE_CASTER(long double, _("numpy.longdouble"));
    bool load(handle, bool);
    static handle cast(const long double &, return_value_policy, handle);
};

#if defined(HEYOKA_HAVE_REAL128)

template <>
struct type_caster<mppp::real128> {
    PYBIND11_TYPE_CASTER(mppp::real128, _("heyoka.core.real128"));
    bool load(handle, bool);
    static handle cast(const mppp::real128 &, return_value_policy, handle);
};

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
struct type_caster<mppp::real> {
    PYBIND11_TYPE_CASTER(mppp::real, _("heyoka.core.real"));
    bool load(handle, bool);
    static handle cast(const mppp::real &, return_value_policy, handle);
};

#endif

} // namespace pybind11::detail

#endif
