// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <type_traits>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include "dtypes.hpp"

#if defined(HEYOKA_HAVE_REAL128)

#include "expose_real128.hpp"

#endif

namespace heyoka_py
{

template <typename T>
int get_dtype()
{
    if constexpr (std::is_same_v<T, double>) {
        return NPY_DOUBLE;
    } else if constexpr (std::is_same_v<T, long double>) {
        return NPY_LONGDOUBLE;
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return npy_registered_py_real128;
#endif
    } else {
        assert(false);
        throw;
    }
}

// Explicit instantiations.
template int get_dtype<double>();
template int get_dtype<long double>();

#if defined(HEYOKA_HAVE_REAL128)

template int get_dtype<mppp::real128>();

#endif

} // namespace heyoka_py
