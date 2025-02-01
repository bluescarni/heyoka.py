// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include "dtypes.hpp"

#if defined(HEYOKA_HAVE_REAL128)

#include "expose_real128.hpp"

#endif

#if defined(HEYOKA_HAVE_REAL)

#include "expose_real.hpp"

#endif

namespace heyoka_py
{

namespace detail
{

namespace
{

// Machinery to associate C++ builtin types to a NumPy dtype.
template <typename>
struct cpp_to_numpy_t {
};

#define HEYOKA_PY_ASSOC_TY(cpp_tp, npy_tp)                                                                             \
    template <>                                                                                                        \
    struct cpp_to_numpy_t<cpp_tp> {                                                                                    \
        static constexpr int value = npy_tp;                                                                           \
    }

HEYOKA_PY_ASSOC_TY(float, NPY_FLOAT);
HEYOKA_PY_ASSOC_TY(double, NPY_DOUBLE);
HEYOKA_PY_ASSOC_TY(long double, NPY_LONGDOUBLE);
HEYOKA_PY_ASSOC_TY(npy_int8, NPY_INT8);
HEYOKA_PY_ASSOC_TY(npy_int16, NPY_INT16);
HEYOKA_PY_ASSOC_TY(npy_int32, NPY_INT32);
HEYOKA_PY_ASSOC_TY(npy_int64, NPY_INT64);
HEYOKA_PY_ASSOC_TY(npy_uint8, NPY_UINT8);
HEYOKA_PY_ASSOC_TY(npy_uint16, NPY_UINT16);
HEYOKA_PY_ASSOC_TY(npy_uint32, NPY_UINT32);
HEYOKA_PY_ASSOC_TY(npy_uint64, NPY_UINT64);

#undef HEYOKA_PY_ASSOC_TY

} // namespace

} // namespace detail

template <typename T>
int get_dtype()
{
#if defined(HEYOKA_HAVE_REAL128)

    if constexpr (std::is_same_v<T, mppp::real128>) {
        return npy_registered_py_real128;
    }

#endif

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        return npy_registered_py_real;
    }

#endif

    if constexpr (std::is_arithmetic_v<T>) {
        return detail::cpp_to_numpy_t<T>::value;
    }

    assert(false);
    throw;
}

// Explicit instantiations.
template int get_dtype<float>();
template int get_dtype<double>();
template int get_dtype<long double>();
template int get_dtype<npy_int8>();
template int get_dtype<npy_int16>();
template int get_dtype<npy_int32>();
template int get_dtype<npy_int64>();
template int get_dtype<npy_uint8>();
template int get_dtype<npy_uint16>();
template int get_dtype<npy_uint32>();
template int get_dtype<npy_uint64>();

#if defined(HEYOKA_HAVE_REAL128)

template int get_dtype<mppp::real128>();

#endif

#if defined(HEYOKA_HAVE_REAL)

template int get_dtype<mppp::real>();

#endif

} // namespace heyoka_py
