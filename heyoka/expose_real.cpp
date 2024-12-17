// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(HEYOKA_HAVE_REAL)

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include <mp++/integer.hpp>
#include <mp++/real.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include "common_utils.hpp"

#endif

#include "custom_casters.hpp"
#include "dtypes.hpp"
#include "expose_real.hpp"
#include "numpy_memory.hpp"

#if defined(HEYOKA_HAVE_REAL128)

#include "expose_real128.hpp"

#endif

namespace heyoka_py
{

namespace py = pybind11;

#if defined(HEYOKA_HAVE_REAL)

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-align"

#if !defined(__clang__)

#pragma GCC diagnostic ignored "-Wcast-function-type"

#endif

#endif

// NOTE: more type properties will be filled in when initing the module.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyTypeObject py_real_type = {PyVarObject_HEAD_INIT(nullptr, 0)};

// NOTE: this is an integer used to represent the
// real type *after* it has been registered in the
// NumPy dtype system. This is needed to expose
// ufuncs and it will be set up during module
// initialisation.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int npy_registered_py_real = 0;

namespace detail
{

namespace
{

// Double check that malloc() aligns memory suitably
// for py_real. See:
// https://en.cppreference.com/w/cpp/types/max_align_t
static_assert(alignof(py_real) <= alignof(std::max_align_t));

// A few function object to implement basic mathematical
// operations in a generic fashion.
const auto identity_func = [](const mppp::real &x) { return x; };
const auto identity_func2 = [](mppp::real &ret, const mppp::real &x) { ret = x; };

const auto negation_func = [](const mppp::real &x) { return -x; };
const auto negation_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::neg(ret, x); };

const auto abs_func = [](const mppp::real &x) { return mppp::abs(x); };
const auto abs_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::abs(ret, x); };

const auto floor_divide_func = [](const mppp::real &x, const mppp::real &y) { return mppp::floor(x / y); };
const auto floor_divide_func3 = [](mppp::real &ret, const mppp::real &x, const mppp::real &y) {
    mppp::div(ret, x, y);
    mppp::floor(ret, ret);
};

const auto pow_func = [](const mppp::real &x, const mppp::real &y) { return mppp::pow(x, y); };
const auto pow_func3 = [](mppp::real &ret, const mppp::real &x, const mppp::real &y) { mppp::pow(ret, x, y); };

const auto sqrt_func = [](const mppp::real &x) { return mppp::sqrt(x); };
const auto sqrt_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::sqrt(ret, x); };

const auto cbrt_func = [](const mppp::real &x) { return mppp::cbrt(x); };
const auto cbrt_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::cbrt(ret, x); };

const auto sin_func = [](const mppp::real &x) { return mppp::sin(x); };
const auto sin_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::sin(ret, x); };

const auto cos_func = [](const mppp::real &x) { return mppp::cos(x); };
const auto cos_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::cos(ret, x); };

const auto tan_func = [](const mppp::real &x) { return mppp::tan(x); };
const auto tan_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::tan(ret, x); };

const auto asin_func = [](const mppp::real &x) { return mppp::asin(x); };
const auto asin_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::asin(ret, x); };

const auto acos_func = [](const mppp::real &x) { return mppp::acos(x); };
const auto acos_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::acos(ret, x); };

const auto atan_func = [](const mppp::real &x) { return mppp::atan(x); };
const auto atan_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::atan(ret, x); };

const auto atan2_func = [](const mppp::real &x, const mppp::real &y) { return mppp::atan2(x, y); };
const auto atan2_func3 = [](mppp::real &ret, const mppp::real &x, const mppp::real &y) { mppp::atan2(ret, x, y); };

const auto sinh_func = [](const mppp::real &x) { return mppp::sinh(x); };
const auto sinh_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::sinh(ret, x); };

const auto cosh_func = [](const mppp::real &x) { return mppp::cosh(x); };
const auto cosh_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::cosh(ret, x); };

const auto tanh_func = [](const mppp::real &x) { return mppp::tanh(x); };
const auto tanh_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::tanh(ret, x); };

const auto asinh_func = [](const mppp::real &x) { return mppp::asinh(x); };
const auto asinh_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::asinh(ret, x); };

const auto acosh_func = [](const mppp::real &x) { return mppp::acosh(x); };
const auto acosh_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::acosh(ret, x); };

const auto atanh_func = [](const mppp::real &x) { return mppp::atanh(x); };
const auto atanh_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::atanh(ret, x); };

const auto deg2rad_func = [](const mppp::real &x) {
    using safe_mpfr_prec_t = boost::safe_numerics::safe<mpfr_prec_t>;

    // Compute 2*pi/360 with a few extra bits of precition wrt x.
    const mpfr_prec_t prec = safe_mpfr_prec_t(x.get_prec()) + 10;
    auto fact = mppp::real_pi(prec);
    fact /= 180;
    fact.prec_round(x.get_prec());

    return x * std::move(fact);
};

const auto deg2rad_func2 = [](mppp::real &ret, const mppp::real &x) {
    using safe_mpfr_prec_t = boost::safe_numerics::safe<mpfr_prec_t>;

    // Compute 2*pi/360 with a few extra bits of precition wrt x.
    const mpfr_prec_t prec = safe_mpfr_prec_t(x.get_prec()) + 10;
    ret.set_prec(prec);
    mppp::real_pi(ret);
    ret /= 180;
    ret.prec_round(x.get_prec());

    ret *= x;
};

const auto rad2deg_func = [](const mppp::real &x) {
    using safe_mpfr_prec_t = boost::safe_numerics::safe<mpfr_prec_t>;

    // Compute 360/(2*pi) with a few extra bits of precition wrt x.
    static const mppp::real c180(180);
    const mpfr_prec_t prec = safe_mpfr_prec_t(x.get_prec()) + 10;
    auto fact = mppp::real_pi(prec);
    mppp::div(fact, c180, fact);
    fact.prec_round(x.get_prec());

    return x * std::move(fact);
};

const auto rad2deg_func2 = [](mppp::real &ret, const mppp::real &x) {
    using safe_mpfr_prec_t = boost::safe_numerics::safe<mpfr_prec_t>;

    // Compute 360/(2*pi) with a few extra bits of precition wrt x.
    static const mppp::real c180(180);
    const mpfr_prec_t prec = safe_mpfr_prec_t(x.get_prec()) + 10;
    ret.set_prec(prec);
    mppp::real_pi(ret);
    mppp::div(ret, c180, ret);
    ret.prec_round(x.get_prec());

    ret *= x;
};

const auto exp_func = [](const mppp::real &x) { return mppp::exp(x); };
const auto exp_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::exp(ret, x); };

const auto exp2_func = [](const mppp::real &x) { return mppp::exp2(x); };
const auto exp2_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::exp2(ret, x); };

const auto expm1_func = [](const mppp::real &x) { return mppp::expm1(x); };
const auto expm1_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::expm1(ret, x); };

const auto log_func = [](const mppp::real &x) { return mppp::log(x); };
const auto log_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::log(ret, x); };

const auto log2_func = [](const mppp::real &x) { return mppp::log2(x); };
const auto log2_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::log2(ret, x); };

const auto log10_func = [](const mppp::real &x) { return mppp::log10(x); };
const auto log10_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::log10(ret, x); };

const auto log1p_func = [](const mppp::real &x) { return mppp::log1p(x); };
const auto log1p_func2 = [](mppp::real &ret, const mppp::real &x) { mppp::log1p(ret, x); };

const auto sign_func = [](const mppp::real &x) {
    if (x.nan_p()) {
        return x;
    }

    const auto sgn = x.sgn();

    if (sgn == 0) {
        return mppp::real(0, x.get_prec());
    }

    if (sgn < 0) {
        return mppp::real(-1, x.get_prec());
    }

    return mppp::real(1, x.get_prec());
};

const auto sign_func2 = [](mppp::real &ret, const mppp::real &x) {
    // NOTE: this already sets to nan.
    ret.set_prec(x.get_prec());

    if (!x.nan_p()) {
        const auto sgn = x.sgn();

        if (sgn == 0) {
            ret.set_zero();
        } else if (sgn == -1) {
            ret.set(-1);
        } else {
            ret.set(1);
        }
    }
};

const auto min_func = [](const mppp::real &x, const mppp::real &y) { return std::min(x, y); };
const auto min_func3 = [](mppp::real &ret, const mppp::real &x, const mppp::real &y) { ret = std::min(x, y); };

const auto max_func = [](const mppp::real &x, const mppp::real &y) { return std::max(x, y); };
const auto max_func3 = [](mppp::real &ret, const mppp::real &x, const mppp::real &y) { ret = std::max(x, y); };

const auto isnan_func = [](const mppp::real &x) { return mppp::isnan(x); };
const auto isinf_func = [](const mppp::real &x) { return mppp::isinf(x); };
const auto isfinite_func = [](const mppp::real &x) { return mppp::isfinite(x); };

const auto square_func = [](const mppp::real &x) { return mppp::sqr(x); };
const auto square2_func = [](mppp::real &ret, const mppp::real &x) { mppp::sqr(ret, x); };

// Ternary arithmetic primitives.
const auto add3_func = [](mppp::real &ret, const mppp::real &a, const mppp::real &b) { mppp::add(ret, a, b); };
const auto sub3_func = [](mppp::real &ret, const mppp::real &a, const mppp::real &b) { mppp::sub(ret, a, b); };
const auto mul3_func = [](mppp::real &ret, const mppp::real &a, const mppp::real &b) { mppp::mul(ret, a, b); };
const auto div3_func = [](mppp::real &ret, const mppp::real &a, const mppp::real &b) { mppp::div(ret, a, b); };

// Methods for the number protocol.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyNumberMethods py_real_as_number = {};

// Create a py_real containing an mppp::real
// constructed from the input set of arguments.
template <typename... Args>
PyObject *py_real_from_args(Args &&...args)
{
    // Acquire the storage for a py_real.
    void *pv = py_real_type.tp_alloc(&py_real_type, 0);
    if (pv == nullptr) {
        return nullptr;
    }

    // Construct the py_real instance.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto *p = ::new (pv) py_real;

    if (with_pybind11_eh([&]() {
            // Setup its internal data.
            ::new (p->m_storage) mppp::real(std::forward<Args>(args)...);
        })) {

        // Clean up.
        py_real_type.tp_free(pv);

        return nullptr;
    }

    return reinterpret_cast<PyObject *>(p);
}

// __new__() implementation.
PyObject *py_real_new([[maybe_unused]] PyTypeObject *type, PyObject *, PyObject *)
{
    assert(type == &py_real_type);

    return py_real_from_args();
}

// Small helper to fetch the number of digits and the sign
// of a Python integer. The integer must be nonzero.
auto py_int_size_sign(PyLongObject *nptr)
{

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION <= 11

    // Fetch the signed size.
    const auto ob_size = nptr->ob_base.ob_size;
    assert(ob_size != 0);

    // Is it negative?
    const auto neg = ob_size < 0;

    // Compute the unsigned size.
    using size_type = std::make_unsigned_t<std::remove_const_t<decltype(ob_size)>>;
    static_assert(std::is_same_v<size_type, decltype(static_cast<size_type>(0) + static_cast<size_type>(0))>);
    auto abs_ob_size = neg ? -static_cast<size_type>(ob_size) : static_cast<size_type>(ob_size);

#else

    // NOTE: these shifts and mask values come from here:
    // https://github.com/python/cpython/blob/main/Include/internal/pycore_long.h
    // Not sure if we can rely on this moving on, probably needs to be checked
    // at every new Python release. Also, note that lv_tag is unsigned, so
    // here we are always getting directly the absolute value of the size,
    // unlike in Python<3.12 where we get out a signed size.
    const auto abs_ob_size = nptr->long_value.lv_tag >> 3;
    assert(abs_ob_size != 0u);

    // Is it negative?
    const auto neg = (nptr->long_value.lv_tag & 3) == 2;

#endif

    return std::make_pair(abs_ob_size, neg);
}

// Helper to convert a Python integer to an mp++ real.
// The precision of the result is inferred from the
// bit size of the integer.
// NOTE: for better performance if needed, it would be better
// to avoid the construction of an intermediate mppp::integer
// (perhaps via determining the bit length of arg beforehand?).
std::optional<mppp::real> py_int_to_real(PyObject *arg)
{
    assert(PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyLong_Type)));

    // Prepare the return value.
    std::optional<mppp::real> retval;

    with_pybind11_eh([&]() {
        // Try to see if arg fits in a long or a long long.
        int overflow = 0;
        const auto cand_l = PyLong_AsLongAndOverflow(arg, &overflow);

        // NOTE: PyLong_AsLongAndOverflow() can raise exceptions in principle.
        if (PyErr_Occurred() != nullptr) {
            return;
        }

        if (overflow == 0) {
            retval.emplace(cand_l);
            return;
        }

        overflow = 0;
        const auto cand_ll = PyLong_AsLongLongAndOverflow(arg, &overflow);

        // NOTE: PyLong_AsLongLongAndOverflow() can raise exceptions in principle.
        if (PyErr_Occurred() != nullptr) {
            return;
        }

        if (overflow == 0) {
            retval.emplace(cand_ll);
            return;
        }

        // Need to construct a multiprecision integer from the limb array.
        auto *nptr = reinterpret_cast<PyLongObject *>(arg);

        // Get the size and sign of nptr.
        auto [abs_ob_size, neg] = py_int_size_sign(nptr);

        // Get the limbs array.
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION <= 11
        const auto *ob_digit = nptr->ob_digit;
#else
        const auto *ob_digit = nptr->long_value.ob_digit;
#endif

        // Init the retval with the first (most significant) limb. The Python integer is nonzero, so this is safe.
        mppp::integer<1> retval_int = ob_digit[--abs_ob_size];

        // Keep on reading limbs until we run out of limbs.
        while (abs_ob_size != 0u) {
            retval_int <<= PyLong_SHIFT;
            retval_int += ob_digit[--abs_ob_size];
        }

        // Turn into an mppp::real.
        mppp::real ret(retval_int);

        // Negate if necessary.
        if (neg) {
            retval.emplace(-std::move(ret));
        } else {
            retval.emplace(std::move(ret));
        }
    });

    // NOTE: if exceptions are raised in the C++ code,
    // retval will remain empty.
    return retval;
}

// Helper to convert a Python integer to an mp++ real.
// The precision of the result is provided explicitly.
// NOTE: this currently uses a string conversion for
// large integers.
std::optional<mppp::real> py_int_to_real_with_prec(PyObject *arg, mpfr_prec_t prec)
{
    assert(PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyLong_Type)));

    // Prepare the return value.
    std::optional<mppp::real> retval;

    with_pybind11_eh([&]() {
        // Try to see if arg fits in a long long.
        int overflow = 0;
        const auto cand_ll = PyLong_AsLongLongAndOverflow(arg, &overflow);

        // NOTE: PyLong_AsLongLongAndOverflow() can raise exceptions in principle.
        if (PyErr_Occurred() != nullptr) {
            return;
        }

        if (overflow == 0) {
            retval.emplace(cand_ll, prec);
            return;
        }

        // Go through a string conversion.
        auto *str_rep = PyObject_Str(arg);
        if (str_rep == nullptr) {
            return;
        }
        assert(PyUnicode_Check(str_rep) != 0);

        // NOTE: str remains valid as long as str_rep is alive.
        const auto *str = PyUnicode_AsUTF8(str_rep);
        if (str == nullptr) {
            Py_DECREF(str_rep);
            return;
        }

        try {
            retval.emplace(str, prec);
        } catch (...) {
            // Ensure proper cleanup of str_rep
            // before continuing.
            Py_DECREF(str_rep);
            throw;
        }

        Py_DECREF(str_rep);
    });

    // NOTE: if exceptions are raised in the C++ code,
    // retval will remain empty.
    return retval;
}

// __init__() implementation.
int py_real_init(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *arg = nullptr;
    PyObject *prec_arg = nullptr;

    const char *kwlist[] = {"value", "prec", nullptr};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", const_cast<char **>(&kwlist[0]), &arg, &prec_arg) == 0) {
        return -1;
    }

    // If no arguments are passed, leave the object as constructed
    // by py_real_new() - that is, the value is zero and the precision
    // is the minimum possible.
    if (arg == nullptr) {
        if (prec_arg != nullptr) {
            PyErr_SetString(PyExc_ValueError,
                            "Cannot construct a real from a precision only - a value must also be supplied");

            return -1;
        }

        return 0;
    }

    // Init the precision argument, if provided.
    std::optional<mpfr_prec_t> prec;
    if (prec_arg != nullptr) {
        // Attempt to turn the precision supplied by the
        // user into a long long.
        const auto prec_arg_val = PyLong_AsLongLong(prec_arg);
        if (PyErr_Occurred() != nullptr) {
            return -1;
        }

        // Check it.
        if (prec_arg_val < std::numeric_limits<mpfr_prec_t>::min()
            || prec_arg_val > std::numeric_limits<mpfr_prec_t>::max()) {
            PyErr_Format(PyExc_OverflowError, "A precision of %lli is too large in magnitude and results in overflow",
                         prec_arg_val);

            return -1;
        }

        prec.emplace(static_cast<mpfr_prec_t>(prec_arg_val));
    }

    // Cache a real pointer to self.
    auto *rval = get_real_val(self);

    // Handle the supported types for the input value.
    if (PyFloat_Check(arg)) {
        // Double.
        const auto d_val = PyFloat_AsDouble(arg);

        if (PyErr_Occurred() == nullptr) {
            const auto err = with_pybind11_eh([&]() {
                if (prec) {
                    rval->set_prec(*prec);
                    mppp::set(*rval, d_val);
                } else {
                    *rval = d_val;
                }
            });

            if (err) {
                return -1;
            }
        } else {
            return -1;
        }
    } else if (PyLong_Check(arg)) {
        // Int.
        if (auto opt = prec ? py_int_to_real_with_prec(arg, *prec) : py_int_to_real(arg)) {
            // NOTE: it's important to move here, so that we don't have to bother
            // with exception handling.
            *rval = std::move(*opt);
        } else {
            return -1;
        }
    } else if (PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyFloat32ArrType_Type)) != 0) {
        const auto f32_val = reinterpret_cast<PyFloat32ScalarObject *>(arg)->obval;

        const auto err = with_pybind11_eh([&]() {
            if (prec) {
                rval->set_prec(*prec);
                mppp::set(*rval, f32_val);
            } else {
                *rval = f32_val;
            }
        });

        if (err) {
            return -1;
        }
    } else if (PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyLongDoubleArrType_Type)) != 0) {
        const auto ld_val = reinterpret_cast<PyLongDoubleScalarObject *>(arg)->obval;

        const auto err = with_pybind11_eh([&]() {
            if (prec) {
                rval->set_prec(*prec);
                mppp::set(*rval, ld_val);
            } else {
                *rval = ld_val;
            }
        });

        if (err) {
            return -1;
        }
#if defined(HEYOKA_HAVE_REAL128)
    } else if (py_real128_check(arg)) {
        const auto f128_val = *get_real128_val(arg);

        const auto err = with_pybind11_eh([&]() {
            if (prec) {
                rval->set_prec(*prec);
                mppp::set(*rval, f128_val);
            } else {
                *rval = f128_val;
            }
        });

        if (err) {
            return -1;
        }
#endif
    } else if (py_real_check(arg)) {
        auto *rval2 = get_real_val(arg);

        const auto err = with_pybind11_eh([&]() {
            if (prec) {
                rval->set_prec(*prec);
                mppp::set(*rval, *rval2);
            } else {
                *rval = *rval2;
            }
        });

        if (err) {
            return -1;
        }
    } else if (PyUnicode_Check(arg)) {
        if (!prec) {
            PyErr_SetString(PyExc_ValueError, "Cannot construct a real from a string without a precision value");

            return -1;
        }

        const auto *str = PyUnicode_AsUTF8(arg);

        if (str == nullptr) {
            return -1;
        }

        const auto err = with_pybind11_eh([&]() {
            rval->set_prec(*prec);
            mppp::set(*rval, str);
        });

        if (err) {
            return -1;
        }
    } else {
        PyErr_Format(PyExc_TypeError, "Cannot construct a real from an object of type \"%s\"", Py_TYPE(arg)->tp_name);

        return -1;
    }

    return 0;
}

// __repr__().
PyObject *py_real_repr(PyObject *self)
{
    PyObject *retval = nullptr;

    // NOTE: C++ exceptions here can be thrown only *before*
    // the invocation of PyUnicode_FromString(). Hence, in case of
    // exceptions, retval will remain nullptr.
    with_pybind11_eh([&]() { retval = PyUnicode_FromString(get_real_val(self)->to_string().c_str()); });

    return retval;
}

// Deallocation.
void py_real_dealloc(PyObject *self)
{
    assert(py_real_check(self));

    // Invoke the destructor.
    get_real_val(self)->~real();

    // Free the memory.
    Py_TYPE(self)->tp_free(self);
}

// Helper to construct a real from one of the
// supported Pythonic numerical types:
// - int,
// - float32,
// - float,
// - long double,
// - real128.
// If the conversion is successful, returns {x, true}. If some error
// is encountered, returns {<empty>, false}. If the type of arg is not
// supported, returns {<empty>, true}.
std::pair<std::optional<mppp::real>, bool> real_from_ob(PyObject *arg)
{
    std::pair<std::optional<mppp::real>, bool> ret;

    with_pybind11_eh([&]() {
        if (PyFloat_Check(arg)) {
            auto fp_arg = PyFloat_AsDouble(arg);

            if (PyErr_Occurred() == nullptr) {
                ret.first.emplace(fp_arg);
                ret.second = true;
            } else {
                ret.second = false;
            }
        } else if (PyLong_Check(arg)) {
            if (auto opt = py_int_to_real(arg)) {
                ret.first.emplace(std::move(*opt));
                ret.second = true;
            } else {
                ret.second = false;
            }
        } else if (PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyFloat32ArrType_Type)) != 0) {
            ret.first.emplace(reinterpret_cast<PyFloat32ScalarObject *>(arg)->obval);
            ret.second = true;
        } else if (PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyLongDoubleArrType_Type)) != 0) {
            ret.first.emplace(reinterpret_cast<PyLongDoubleScalarObject *>(arg)->obval);
            ret.second = true;
#if defined(HEYOKA_HAVE_REAL128)
        } else if (py_real128_check(arg)) {
            ret.first.emplace(*get_real128_val(arg));
            ret.second = true;
#endif
        } else {
            ret.second = true;
        }
    });

    return ret;
}

// The precision getter.
PyObject *py_real_prec_getter(PyObject *self, void *)
{
    assert(py_real_check(self));

    return PyLong_FromLongLong(static_cast<long long>(get_real_val(self)->get_prec()));
}

// The limb address getter.
PyObject *py_real_limb_address_getter(PyObject *self, void *)
{
    assert(py_real_check(self));

    return PyLong_FromUnsignedLongLong(
        static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(get_real_val(self)->get_mpfr_t()->_mpfr_d)));
}

// The array for computed attribute instances. See:
// https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_getset
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyGetSetDef py_real_get_set[] = {{"prec", py_real_prec_getter, nullptr, nullptr, nullptr},
                                 {"_limb_address", py_real_limb_address_getter, nullptr, nullptr, nullptr},
                                 {nullptr}};

// Implementation of several methods for real.
PyObject *py_real_set_prec(PyObject *self, PyObject *args)
{
    long long prec = 0;

    if (PyArg_ParseTuple(args, "L", &prec) == 0) {
        return nullptr;
    }

    const auto err = with_pybind11_eh([&]() { get_real_val(self)->set_prec(boost::numeric_cast<mpfr_prec_t>(prec)); });

    if (err) {
        return nullptr;
    }

    Py_RETURN_NONE;
}

PyObject *py_real_prec_round(PyObject *self, PyObject *args)
{
    long long prec = 0;

    if (PyArg_ParseTuple(args, "L", &prec) == 0) {
        return nullptr;
    }

    const auto err
        = with_pybind11_eh([&]() { get_real_val(self)->prec_round(boost::numeric_cast<mpfr_prec_t>(prec)); });

    if (err) {
        return nullptr;
    }

    Py_RETURN_NONE;
}

PyObject *py_real_copy(PyObject *self, [[maybe_unused]] PyObject *args)
{
    assert(args == nullptr);

    PyObject *retval = nullptr;

    with_pybind11_eh([&]() { retval = py_real_from_args(*get_real_val(self)); });

    return retval;
}

PyObject *py_real_deepcopy(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *memo_arg = nullptr;

    const char *kwlist[] = {"memo", nullptr};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "|O", const_cast<char **>(&kwlist[0]), &memo_arg) == 0) {
        return nullptr;
    }

    PyObject *retval = nullptr;

    with_pybind11_eh([&]() { retval = py_real_from_args(*get_real_val(self)); });

    return retval;
}

// Get the internal state of a real for serialisation purposes. The internal
// state is returned as a binary blob.
PyObject *py_real_getstate(PyObject *self, [[maybe_unused]] PyObject *args)
{
    assert(args == nullptr);

    PyObject *retval = nullptr;

    with_pybind11_eh([&]() {
        std::ostringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);
            oa << *get_real_val(self);
        }

        // NOTE: we might be able to avoid a copy here if we manage to operate
        // directly on the stringstream.
        const auto str = ss.str();
        retval = PyBytes_FromStringAndSize(str.c_str(), boost::numeric_cast<Py_ssize_t>(str.size()));
    });

    return retval;
}

// Set the internal state of a real from a binary blob.
PyObject *py_real_setstate(PyObject *self, PyObject *args)
{
    PyBytesObject *bytes_obj = nullptr;

    if (PyArg_ParseTuple(args, "S", &bytes_obj) == 0) {
        return nullptr;
    }

    assert(bytes_obj != nullptr);
    auto *ob_ptr = reinterpret_cast<PyObject *>(bytes_obj);

    const auto err = with_pybind11_eh([&]() {
        std::stringstream ss;

        std::copy(PyBytes_AsString(ob_ptr), PyBytes_AsString(ob_ptr) + PyBytes_Size(ob_ptr),
                  std::ostreambuf_iterator<char>(ss));

        boost::archive::binary_iarchive ia(ss);
        ia >> *get_real_val(self);
    });

    if (err) {
        return nullptr;
    }

    Py_RETURN_NONE;
}

// Implementation of the reduce protocol. See:
// https://docs.python.org/3/library/pickle.html#object.__reduce__
PyObject *py_real_reduce(PyObject *self, [[maybe_unused]] PyObject *args)
{
    assert(args == nullptr);

    // Fetch the factory function.
    auto *hy_mod = PyImport_ImportModule("heyoka");
    if (hy_mod == nullptr) {
        return nullptr;
    }

    auto *fact = PyObject_GetAttrString(hy_mod, "_real_reduce_factory");
    Py_DECREF(hy_mod);
    if (fact == nullptr) {
        return nullptr;
    }

    // The factory function requires no arguments.
    auto *fact_args = PyTuple_New(0);
    if (fact_args == nullptr) {
        Py_DECREF(fact);
        return nullptr;
    }

    // Get the state of self.
    auto *state = py_real_getstate(self, nullptr);
    if (state == nullptr) {
        Py_DECREF(fact);
        Py_DECREF(fact_args);
        return nullptr;
    }

    // Assemble the return value.
    auto *ret = PyTuple_New(3);
    if (ret == nullptr) {
        Py_DECREF(fact);
        Py_DECREF(fact_args);
        Py_DECREF(state);
        return nullptr;
    }

    PyTuple_SetItem(ret, 0, fact);
    PyTuple_SetItem(ret, 1, fact_args);
    PyTuple_SetItem(ret, 2, state);

    return ret;
}

// Same as py_real_reduce(), but takes in input a protocol version
// number which we ignore.
PyObject *py_real_reduce_ex(PyObject *self, PyObject *args)
{
    int version = 0;

    if (PyArg_ParseTuple(args, "i", &version) == 0) {
        return nullptr;
    }

    return py_real_reduce(self, nullptr);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyMethodDef py_real_methods[]
    = {{"set_prec", py_real_set_prec, METH_VARARGS, nullptr},
       {"prec_round", py_real_prec_round, METH_VARARGS, nullptr},
       {"__copy__", py_real_copy, METH_NOARGS, nullptr},
       {"__deepcopy__", reinterpret_cast<PyCFunction>(reinterpret_cast<void *>(py_real_deepcopy)),
        METH_VARARGS | METH_KEYWORDS, nullptr},
       // NOTE: for pickling support we need to override the reduce/reduce_ex functions, as they
       // are implemented in the base NumPy class and they take the precedence over get/set state.
       {"__setstate__", py_real_setstate, METH_VARARGS, nullptr},
       {"__reduce__", py_real_reduce, METH_NOARGS, nullptr},
       {"__reduce_ex__", py_real_reduce_ex, METH_VARARGS, nullptr},
       {nullptr}};

// Generic implementation of unary operations.
template <typename F>
PyObject *py_real_unop(PyObject *a, const F &op)
{
    if (py_real_check(a)) {
        PyObject *ret = nullptr;
        const auto *x = get_real_val(a);

        // NOTE: if any exception is thrown in the C++ code, ret
        // will stay (and the function will return null). This is in line
        // with the requirements of the number protocol functions:
        // https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods
        with_pybind11_eh([&]() { ret = py_real_from_args(op(*x)); });

        return ret;
    } else {
        Py_RETURN_NOTIMPLEMENTED;
    }
}

// Generic implementation of binary operations.
template <typename F>
PyObject *py_real_binop(PyObject *a, PyObject *b, const F &op)
{
    const auto a_is_real = py_real_check(a);
    const auto b_is_real = py_real_check(b);

    if (a_is_real && b_is_real) {
        // Both operands are real.
        const auto *x = get_real_val(a);
        const auto *y = get_real_val(b);

        PyObject *ret = nullptr;

        with_pybind11_eh([&]() { ret = py_real_from_args(op(*x, *y)); });

        return ret;
    }

    if (a_is_real) {
        // a is a real, b is not. Try to convert
        // b to a real.
        auto p = real_from_ob(b);
        auto &r = p.first;
        auto &flag = p.second;

        if (r) {
            // The conversion was successful, do the op.
            PyObject *ret = nullptr;
            with_pybind11_eh([&]() { ret = py_real_from_args(op(*get_real_val(a), *r)); });
            return ret;
        }

        if (flag) {
            // b's type is not supported.
            Py_RETURN_NOTIMPLEMENTED;
        }

        // Attempting to convert b to a real generated an error.
        return nullptr;
    }

    if (b_is_real) {
        // The mirror of the previous case.
        auto p = real_from_ob(a);
        auto &r = p.first;
        auto &flag = p.second;

        if (r) {
            PyObject *ret = nullptr;
            with_pybind11_eh([&]() { ret = py_real_from_args(op(*r, *get_real_val(b))); });
            return ret;
        }

        if (flag) {
            Py_RETURN_NOTIMPLEMENTED;
        }

        return nullptr;
    }

    // Neither a nor b are real.
    Py_RETURN_NOTIMPLEMENTED;
}

// Rich comparison operator.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyObject *py_real_rcmp(PyObject *a, PyObject *b, int op)
{
    auto impl = [a, b](const auto &func) -> PyObject * {
        const auto a_is_real = py_real_check(a);
        const auto b_is_real = py_real_check(b);

        if (a_is_real && b_is_real) {
            // Both operands are real.
            const auto *x = get_real_val(a);
            const auto *y = get_real_val(b);

            // NOTE: the standard comparison operators for mppp::real never throw,
            // thus we never need to wrap the invocation of func() in the exception
            // handling mechanism.
            if (func(*x, *y)) {
                Py_RETURN_TRUE;
            } else {
                Py_RETURN_FALSE;
            }
        }

        if (a_is_real) {
            // a is a real, b is not. Try to convert
            // b to a real.
            auto [r, flag] = real_from_ob(b);

            if (r) {
                // The conversion was successful, do the op.
                if (func(*get_real_val(a), *r)) {
                    Py_RETURN_TRUE;
                } else {
                    Py_RETURN_FALSE;
                }
            }

            if (flag) {
                // b's type is not supported.
                Py_RETURN_NOTIMPLEMENTED;
            }

            // Attempting to convert b to a real generated an error.
            return nullptr;
        }

        if (b_is_real) {
            // The mirror of the previous case.
            auto [r, flag] = real_from_ob(a);

            if (r) {
                if (func(*r, *get_real_val(b))) {
                    Py_RETURN_TRUE;
                } else {
                    Py_RETURN_FALSE;
                }
            }

            if (flag) {
                Py_RETURN_NOTIMPLEMENTED;
            }

            return nullptr;
        }

        Py_RETURN_NOTIMPLEMENTED;
    };

    switch (op) {
        case Py_LT:
            return impl(std::less{});
        case Py_LE:
            return impl(std::less_equal{});
        case Py_EQ:
            return impl(std::equal_to{});
        case Py_NE:
            return impl(std::not_equal_to{});
        case Py_GT:
            return impl(std::greater{});
        default:
            assert(op == Py_GE);
            return impl(std::greater_equal{});
    }
}

// NumPy array function.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyArray_ArrFuncs npy_py_real_arr_funcs = {};

// NumPy type descriptor proto.
// NOTE: this has been switched from PyArray_Descr to PyArray_DescrProto in NumPy 2 for backwards compat mode
// with NumPy 1. See:
//
// https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_RegisterDataType.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyArray_DescrProto npy_py_real_descr_proto = {PyObject_HEAD_INIT(0) & py_real_type};

// The actual type descriptor. This will be set at module init via PyArray_DescrFromType().
PyArray_Descr *npy_py_real_descr = nullptr;

// Small helper to check if the input pointer is suitably
// aligned for mppp::real.
bool ptr_real_aligned(void *ptr)
{
    assert(ptr != nullptr);

    auto space = sizeof(mppp::real);

    return std::align(alignof(mppp::real), sizeof(mppp::real), ptr, space) != nullptr;
}

// Helper to access a global default-constructed real instance.
// This is used in several places when trying to access
// a not-yet-constructed real in an array.
// NOTE: mark it noexcept as this is equivalent to being unable
// to construct a global variable, a situation which
// we don't really need to be able to recover from.
const auto &get_zero_real() noexcept
{
    static const mppp::real zr;

    return zr;
}

// This helper checks if a valid mppp::real exists at the address ptr by
// reading from ptr the first member of an MPFR struct, that is,
// the precision value, which cannot be zero. Coupled to the fact that we
// specify NPY_NEEDS_INIT when registering the NumPy type (which zeroes out
// allocated memory buffers before using them within NumPy), this allows
// us to detect at runtime if NumPy tries to use memory buffers NOT allocated
// via the NEP 49 functions to store mppp::real instances.
bool is_valid_real(const void *ptr) noexcept
{
    mpfr_prec_t prec = 0;
    std::memcpy(&prec, ptr, sizeof(mpfr_prec_t));

    return prec != 0;
}

// This function will check if a valid real exists at the memory address ptr.
//
// ptr may or may not be located in a memory buffer managed by NEP 49:
//
// - if it is, both base_ptr and ct_flags are non-null,
//   base_ptr is assumed to point to beginning of a NEP 49 memory buffer and ct_flags is the
//   construction flags metadata associated to the memory buffer;
// - otherwise, both base_ptr and ct_flags are nullptr. If ptr contains a valid real
//   (as detected by is_valid_real()), then true will be returned, false otherwise.
bool check_cted_real(const unsigned char *base_ptr, const bool *ct_flags, const void *ptr) noexcept
{
    assert((base_ptr == nullptr) == (ct_flags == nullptr));

    if (base_ptr == nullptr) {
        // The memory buffer is not managed by NEP 49, check if a real
        // has been constructed in ptr by someone else.
        return is_valid_real(ptr);
    }

    // The memory buffer is managed by NEP 49. Compute the position of ptr in the buffer.
    const auto bytes_idx = reinterpret_cast<const unsigned char *>(ptr) - base_ptr;
    assert(bytes_idx >= 0);

    // Compute the position in the ct_flags array.
    auto idx = static_cast<std::size_t>(bytes_idx);
    assert(idx % sizeof(mppp::real) == 0u);
    idx /= sizeof(mppp::real);

    return ct_flags[idx];
}

// This function will return either a const pointer to an mppp::real stored
// at the memory address ptr, if it exists, or the fallback value.
//
// ptr may or may not be located in a memory buffer managed by NEP 49:
//
// - if it is, both base_ptr and ct_flags are non-null,
//   base_ptr is assumed to point to beginning of a NEP 49 memory buffer and ct_flags is the
//   construction flags metadata associated to the memory buffer. If a constructed mppp::real exists
//   in ptr, then its address will be returned, otherwise fallback will be returned instead;
// - otherwise, both base_ptr and ct_flags are nullptr. If ptr contains a valid real
//   (as detected by is_valid_real()) then it is returned, otherwise fallback will be returned instead.
const mppp::real *fetch_cted_real(const unsigned char *base_ptr, const bool *ct_flags, const void *ptr,
                                  const mppp::real *fallback) noexcept
{
    assert((base_ptr == nullptr) == (ct_flags == nullptr));
    assert(fallback != nullptr);

    if (base_ptr == nullptr) {
        // The memory buffer is not managed by NEP 49, check if a real
        // has been constructed in ptr by someone else.
        if (is_valid_real(ptr)) {
            // A valid real exists in ptr, return it.
            return std::launder(reinterpret_cast<const mppp::real *>(ptr));
        } else {
            // No valid real exists in ptr, return the
            // fallback pointer.
            // NOTE: I am not sure this can actually happen - that is,
            // NumPy allocating buffers outside NEP 49 and then reading
            // from them without having written anything into them. But
            // better safe than sorry...
            return fallback;
        }
    }

    // The memory buffer is managed by NEP 49. Compute the position of ptr in the buffer.
    const auto bytes_idx = reinterpret_cast<const unsigned char *>(ptr) - base_ptr;
    assert(bytes_idx >= 0);

    // Compute the position in the ct_flags array.
    auto idx = static_cast<std::size_t>(bytes_idx);
    assert(idx % sizeof(mppp::real) == 0u);
    idx /= sizeof(mppp::real);

    if (ct_flags[idx]) {
        // A constructed mppp::real exists, fetch it.
        return std::launder(reinterpret_cast<const mppp::real *>(ptr));
    } else {
        // No constructed mppp::real exists, return the fallback value.
        return fallback;
    }
}

// This function will return a pointer to the mppp::real stored at the address
// ptr, if it exists. If it does not, a new mppp::real will
// be constructed in-place at ptr using the result of the supplied nullary function f.
//
// ptr may or may not be located in a memory buffer managed by NEP 49:
//
// - if it is, both base_ptr and ct_flags are non-null,
//   base_ptr is assumed to point to the beginning of a NEP 49 memory buffer and ct_flags is the
//   construction flags metadata associated to that memory buffer. If a constructed mppp::real exists
//   in ptr, then its address will be returned, otherwise a new mppp::real will be
//   constructed in ptr, and its address will be returned;
// - otherwise, both base_ptr and ct_flags are nullptr. If ptr contains a valid real
//   (as detected by is_valid_real()) then its address is returned, otherwise a new mppp::real will be
//   constructed in ptr, and its address will be returned.
//
// If the initialisation of a new mppp::real throws, the returned optional will be empty.
// The second member of the return value is a flag signalling whether the returned pointer is a
// newly-constructed real (true) or a pointer to an existing real (false).
template <typename F>
std::pair<std::optional<mppp::real *>, bool> ensure_cted_real(const unsigned char *base_ptr, bool *ct_flags, void *ptr,
                                                              const F &f) noexcept
{
    assert((base_ptr == nullptr) == (ct_flags == nullptr));

    if (base_ptr == nullptr) {
        // The memory buffer is not managed by NEP 49, check if a real
        // has been constructed in ptr by someone else.
        if (!is_valid_real(ptr)) {
            // No valid real exists in ptr, construct one.
            // This should happen only in case of NumPy buffers
            // not managed by NEP 49 (e.g., within the casting logic).
            // NOTE: this will lead to memory leaks because
            // nobody will be destroying this, but I see no other
            // solution.
            const auto err = with_pybind11_eh([&]() { ::new (ptr) mppp::real(f()); });

            if (err) {
                return {};
            } else {
                return {std::launder(reinterpret_cast<mppp::real *>(ptr)), true};
            }
        } else {
            return {std::launder(reinterpret_cast<mppp::real *>(ptr)), false};
        }
    }

    // The memory buffer is managed by NEP 49. Compute the position of ptr in the buffer.
    const auto bytes_idx = reinterpret_cast<const unsigned char *>(ptr) - base_ptr;
    assert(bytes_idx >= 0);

    // Compute the position in the ct_flags array.
    auto idx = static_cast<std::size_t>(bytes_idx);
    assert(idx % sizeof(mppp::real) == 0u);
    idx /= sizeof(mppp::real);

    if (ct_flags[idx]) {
        // An mppp::real exists, return it.
        return {std::launder(reinterpret_cast<mppp::real *>(ptr)), false};
    } else {
        // No mppp::real exists, construct it.
        const auto err = with_pybind11_eh([&]() { ::new (ptr) mppp::real(f()); });

        if (err) {
            return {};
        } else {
            // Signal that a new mppp::real was constructed.
            ct_flags[idx] = true;

            return {std::launder(reinterpret_cast<mppp::real *>(ptr)), true};
        }
    }
}

// Array getitem.
PyObject *npy_py_real_getitem(void *data, [[maybe_unused]] void *arr)
{
    // NOTE: getitem could be invoked with misaligned data.
    // Detect such occurrence and error out.
    if (!ptr_real_aligned(data)) {
        PyErr_SetString(PyExc_ValueError, "Cannot invoke __getitem__() on misaligned real data");
        return nullptr;
    }

    PyObject *ret = nullptr;

    with_pybind11_eh([&]() {
        // Try to locate data in the memory map.
        const auto [base_ptr, meta_ptr] = get_memory_metadata(data);
        const auto *ct_flags = meta_ptr == nullptr ? nullptr : meta_ptr->ensure_ct_flags_inited<mppp::real>();

        ret = py_real_from_args(*fetch_cted_real(base_ptr, ct_flags, data, &get_zero_real()));
    });

    return ret;
}

// Array setitem.
int npy_py_real_setitem(PyObject *item, void *data, [[maybe_unused]] void *arr)
{
    // NOTE: getitem could be invoked with misaligned data.
    // Detect such occurrence and error out.
    if (!ptr_real_aligned(data)) {
        PyErr_SetString(PyExc_ValueError, "Cannot invoke __setitem__() on misaligned real data");
        return -1;
    }

    // Try to locate data in the memory map.
    const auto [base_ptr, meta_ptr] = get_memory_metadata(data);
    auto *ct_flags = (meta_ptr == nullptr) ? nullptr : meta_ptr->ensure_ct_flags_inited<mppp::real>();

    if (py_real_check(item)) {
        const auto &src_val = *get_real_val(item);

        auto [ptr_opt, new_real]
            = ensure_cted_real(base_ptr, ct_flags, data, [&]() -> const mppp::real & { return src_val; });
        if (!ptr_opt) {
            return -1;
        }

        if (new_real) {
            return 0;
        } else {
            const auto err = with_pybind11_eh([&src_val, &ptr = *ptr_opt]() { *ptr = src_val; });

            return err ? -1 : 0;
        }
    }

    // item is not a real, but it might be convertible to a real.
    auto [r, flag] = real_from_ob(item);

    if (r) {
        // Conversion successful.

        // Make sure we have a real in data.
        auto [ptr_opt, new_real] = ensure_cted_real(base_ptr, ct_flags, data,
                                                    [&, &r_ref = *r]() -> mppp::real && { return std::move(r_ref); });
        if (!ptr_opt) {
            return -1;
        }

        if (!new_real) {
            // NOTE: move assignment, no need for exception handling.
            **ptr_opt = std::move(*r);
        }

        return 0;
    }

    if (flag) {
        // Cannot convert item to a real because the type of item is not supported.
        PyErr_Format(PyExc_TypeError, "Cannot invoke __setitem__() on a real array with an input value of type \"%s\"",
                     Py_TYPE(item)->tp_name);
        return -1;
    }

    // Attempting a conversion to real generated an error.
    // The error flag has already been set by real_from_ob(),
    // just return -1.
    return -1;
}

// Copyswap primitive.
// NOTE: apparently there's no mechanism to directly report an error from this
// function, as it returns void. As a consequence, in case of errors Python throws
// a generic SystemError (rather than the exception of our choice) and complains that
// the function "returned a result with an exception set". This is not optimal,
// but better than just crashing I guess?
// NOTE: implementing copyswapn() would allow us to avoid peeking in the memory
// map for every element that is being copied.
void npy_py_real_copyswap(void *dst, void *src, int swap, [[maybe_unused]] void *arr)
{
    if (swap != 0) {
        PyErr_SetString(PyExc_ValueError, "Cannot byteswap real arrays");
        return;
    }

    if (src == nullptr) {
        return;
    }

    assert(dst != nullptr);

    // NOTE: copyswap could be invoked with misaligned data.
    // Detect such occurrence and error out.
    if (!ptr_real_aligned(src)) {
        PyErr_SetString(PyExc_ValueError, "Cannot copyswap() misaligned real data");
        return;
    }
    if (!ptr_real_aligned(dst)) {
        PyErr_SetString(PyExc_ValueError, "Cannot copyswap() misaligned real data");
        return;
    }

    // Try to locate src and dst data in the memory map.
    const auto [base_ptr_dst, meta_ptr_dst] = get_memory_metadata(dst);
    auto *ct_flags_dst = (meta_ptr_dst == nullptr) ? nullptr : meta_ptr_dst->ensure_ct_flags_inited<mppp::real>();

    const auto [base_ptr_src, meta_ptr_src] = get_memory_metadata(src);
    const auto *ct_flags_src = (meta_ptr_src == nullptr) ? nullptr : meta_ptr_src->ensure_ct_flags_inited<mppp::real>();

    // Fetch a constructed real from src.
    const auto *src_x = fetch_cted_real(base_ptr_src, ct_flags_src, src, &get_zero_real());

    // Write into the destination.
    auto [dst_x_opt, new_real]
        = ensure_cted_real(base_ptr_dst, ct_flags_dst, dst, [&]() -> const mppp::real & { return *src_x; });
    if (!dst_x_opt) {
        // ensure_cted_real() generated an error, exit.
        return;
    }

    if (!new_real) {
        // NOTE: if a C++ exception is thrown here, dst is not modified
        // and an error code will have been set.
        with_pybind11_eh([&src_x, &dst_ptr = *dst_x_opt]() { *dst_ptr = *src_x; });
    }
}

// Nonzero primitive.
npy_bool npy_py_real_nonzero(void *data, void *)
{
    if (!ptr_real_aligned(data)) {
        PyErr_SetString(PyExc_ValueError, "Cannot detect nonzero elements in an array of misaligned real data");
        return NPY_FALSE;
    }

    // Try to locate data in the memory map.
    const auto [base_ptr, meta_ptr] = get_memory_metadata(data);
    const auto *ct_flags = (meta_ptr == nullptr) ? nullptr : meta_ptr->ensure_ct_flags_inited<mppp::real>();

    // Fetch a real from data.
    const auto *x_ptr = fetch_cted_real(base_ptr, ct_flags, data, &get_zero_real());

    // NOTE: zero_p() does not throw.
    return x_ptr->zero_p() ? NPY_FALSE : NPY_TRUE;
}

// Comparison primitive, used for sorting.
// NOTE: use special comparisons to ensure NaNs are put at the end
// of an array when sorting.
// NOTE: this has pretty horrible performance because it needs to
// look into the memory metadata element by element. We could maybe
// implement the specialised sorting primitives from here:
// https://numpy.org/doc/stable/reference/c-api/types-and-structures.html#c.PyArray_ArrFuncs.sort
// This perhaps allows to fetch the metadata only once.
int npy_py_real_compare(const void *d0, const void *d1, void *)
{
    // Try to locate d0 and d1 in the memory map.
    // NOTE: probably we could assume that d0 and d1 are coming from the same
    // array/buffer, but better safe than sorry.
    const auto [base_ptr_0, meta_ptr_0] = get_memory_metadata(d0);
    const auto *ct_flags_0 = (meta_ptr_0 == nullptr) ? nullptr : meta_ptr_0->ensure_ct_flags_inited<mppp::real>();

    const auto [base_ptr_1, meta_ptr_1] = get_memory_metadata(d1);
    const auto *ct_flags_1 = (meta_ptr_1 == nullptr) ? nullptr : meta_ptr_1->ensure_ct_flags_inited<mppp::real>();

    // Fetch the zero constant.
    const auto &zr = get_zero_real();

    // Fetch the real arguments.
    const auto &x = *fetch_cted_real(base_ptr_0, ct_flags_0, d0, &zr);
    const auto &y = *fetch_cted_real(base_ptr_1, ct_flags_1, d1, &zr);

    // NOTE: the comparison functions do not throw.
    if (mppp::real_lt(x, y)) {
        return -1;
    }

    if (mppp::real_equal_to(x, y)) {
        return 0;
    }

    return 1;
}

// argmin/argmax implementation.
template <typename F>
int npy_py_real_argminmax(void *data, npy_intp n, npy_intp *max_ind, const F &cmp)
{
    if (n == 0) {
        return 0;
    }

    // Try to locate data in the memory map.
    const auto [base_ptr, meta_ptr] = get_memory_metadata(data);
    const auto *ct_flags = (base_ptr == nullptr) ? nullptr : meta_ptr->ensure_ct_flags_inited<mppp::real>();

    // Fetch a pointer to the global zero constant.
    const auto *zr = &get_zero_real();

    // Fetch the char array version of the memory segment.
    const auto *cdata = reinterpret_cast<const char *>(data);

    // Init the best index.
    npy_intp best_i = 0;

    // Init the best value.
    const auto *best_r = fetch_cted_real(base_ptr, ct_flags, cdata, zr);

    for (npy_intp i = 1; i < n; ++i) {
        const auto *cur_r
            = fetch_cted_real(base_ptr, ct_flags, cdata + static_cast<std::size_t>(i) * sizeof(mppp::real), zr);

        // NOTE: the comparison function does not throw.
        if (cmp(*cur_r, *best_r)) {
            best_i = i;
            best_r = cur_r;
        }
    }

    *max_ind = best_i;

    return 0;
}

// Fill primitive (e.g., used for arange()).
int npy_py_real_fill(void *data, npy_intp length, void *)
{
    // NOTE: not sure if this is possible, let's stay on the safe side.
    if (length < 2) {
        return 0;
    }

    const auto err = with_pybind11_eh([&]() {
        // Try to locate data in the memory map.
        const auto [base_ptr, meta_ptr] = get_memory_metadata(data);
        auto *ct_flags = (base_ptr == nullptr) ? nullptr : meta_ptr->ensure_ct_flags_inited<mppp::real>();

        // Fetch a pointer to the global zero const.
        const auto *zr = &get_zero_real();

        // Fetch the char array version of the memory segment.
        auto *cdata = reinterpret_cast<char *>(data);

        // Fetch pointers to the first two elements.
        const auto *el0 = fetch_cted_real(base_ptr, ct_flags, cdata, zr);
        const auto *el1 = fetch_cted_real(base_ptr, ct_flags, cdata + sizeof(mppp::real), zr);

        // Compute the delta and init r.
        const auto delta = *el1 - *el0;
        auto r = *el1;

        for (npy_intp i = 2; i < length; ++i) {
            // Update r.
            r += delta;

            // Fetch a pointer to the destination value.
            auto *dst_ptr = cdata + static_cast<std::size_t>(i) * sizeof(mppp::real);
            auto [opt_ptr, new_real]
                = ensure_cted_real(base_ptr, ct_flags, dst_ptr, [&]() -> const mppp::real & { return r; });

            if (!opt_ptr) {
                // NOTE: if ensure_cted_real() fails, it will have set the Python
                // exception already. Thus we throw the special error_already_set
                // C++ exception so that the error message is propagated outside this scope.
                throw py::error_already_set();
            }

            if (!new_real) {
                **opt_ptr = r;
            }
        }
    });

    return err ? -1 : 0;
}

// Fill with scalar primitive.
int npy_py_real_fillwithscalar(void *buffer, npy_intp length, void *value, void *)
{
    // Fetch the fill value.
    const auto &r = *static_cast<mppp::real *>(value);

    const auto err = with_pybind11_eh([&]() {
        // Try to locate buffer in the memory map.
        const auto [base_ptr, meta_ptr] = get_memory_metadata(buffer);
        auto *ct_flags = (base_ptr == nullptr) ? nullptr : meta_ptr->ensure_ct_flags_inited<mppp::real>();

        // Fetch the char array version of the memory segment.
        auto *cdata = reinterpret_cast<char *>(buffer);

        for (npy_intp i = 0; i < length; ++i) {
            // Fetch a pointer to the destination value.
            auto *dst_ptr = cdata + static_cast<std::size_t>(i) * sizeof(mppp::real);
            auto [opt_ptr, new_real]
                = ensure_cted_real(base_ptr, ct_flags, dst_ptr, [&]() -> const mppp::real & { return r; });

            if (!opt_ptr) {
                throw py::error_already_set();
            }

            if (!new_real) {
                **opt_ptr = r;
            }
        }
    });

    return err ? -1 : 0;
}

// Generic NumPy conversion function to real.
template <typename From>
void npy_cast_to_real(void *from, void *to_data, npy_intp n, void *, void *)
{
    // Sanity check, we don't want to implement real -> real conversion.
    static_assert(!std::is_same_v<From, mppp::real>);

    const auto *typed_from = static_cast<const From *>(from);

    with_pybind11_eh([&]() {
        // Try to locate to_data in the memory map.
        const auto [base_ptr, meta_ptr] = get_memory_metadata(to_data);
        auto *ct_flags = (base_ptr == nullptr) ? nullptr : meta_ptr->ensure_ct_flags_inited<mppp::real>();

        // Fetch the char array version of the memory segment.
        auto *c_to_data = reinterpret_cast<char *>(to_data);

        // NOTE: we have no way of signalling failure in the casting functions,
        // but at least we will stop and just return if some exception is raised.
        for (npy_intp i = 0; i < n; ++i) {
            auto *dst_ptr = c_to_data + static_cast<std::size_t>(i) * sizeof(mppp::real);

            auto [opt_ptr, new_real] = ensure_cted_real(base_ptr, ct_flags, dst_ptr, [&]() { return typed_from[i]; });

            if (!opt_ptr) {
                throw py::error_already_set();
            }

            if (!new_real) {
                **opt_ptr = typed_from[i];
            }
        }
    });
}

// Generic NumPy conversion function from real.
template <typename To>
void npy_cast_from_real(void *from_data, void *to, npy_intp n, void *, void *)
{
    // Sanity check, we don't want to implement real -> real conversion.
    static_assert(!std::is_same_v<To, mppp::real>);

    auto *typed_to = static_cast<To *>(to);

    with_pybind11_eh([&]() {
        // Try to locate from_data in the memory map.
        const auto [base_ptr, meta_ptr] = get_memory_metadata(from_data);
        const auto *ct_flags = (base_ptr == nullptr) ? nullptr : meta_ptr->ensure_ct_flags_inited<mppp::real>();

        // Fetch a pointer to the global zero const.
        const auto *zr = &get_zero_real();

        // Fetch the char array version of the memory segment.
        const auto *c_from_data = reinterpret_cast<const char *>(from_data);

        for (npy_intp i = 0; i < n; ++i) {
            const auto *src_ptr = c_from_data + static_cast<std::size_t>(i) * sizeof(mppp::real);
            typed_to[i] = static_cast<To>(*fetch_cted_real(base_ptr, ct_flags, src_ptr, zr));
        }
    });
}

// Helper to register NumPy casting functions to/from T.
template <typename T>
void npy_register_cast_functions()
{
    if (PyArray_RegisterCastFunc(PyArray_DescrFromType(get_dtype<T>()), npy_registered_py_real, &npy_cast_to_real<T>)
        < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

    // NOTE: this is to signal that conversion of any scalar type to real is safe.
    if (PyArray_RegisterCanCast(PyArray_DescrFromType(get_dtype<T>()), npy_registered_py_real, NPY_NOSCALAR) < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

    if (PyArray_RegisterCastFunc(npy_py_real_descr, get_dtype<T>(), &npy_cast_from_real<T>) < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }
}

// Generic NumPy unary operation with real operand and real output.
template <typename F1, typename F2>
void py_real_ufunc_unary(char **args, const npy_intp *dimensions, const npy_intp *steps, void *, const F1 &f1,
                         const F2 &f2)
{
    npy_intp is1 = steps[0], os1 = steps[1], n = dimensions[0];
    char *ip1 = args[0], *op1 = args[1];

    with_pybind11_eh([&]() {
        // Try to locate the input/output buffers in the memory map.
        const auto [base_ptr_i1, meta_ptr_i1] = get_memory_metadata(ip1);
        const auto *ct_flags_i1
            = (base_ptr_i1 == nullptr) ? nullptr : meta_ptr_i1->ensure_ct_flags_inited<mppp::real>();

        const auto [base_ptr_o, meta_ptr_o] = get_memory_metadata(op1);
        auto *ct_flags_o = (base_ptr_o == nullptr) ? nullptr : meta_ptr_o->ensure_ct_flags_inited<mppp::real>();

        // Fetch a pointer to the global zero const. This will be used in place
        // of non-constructed input real arguments.
        const auto *zr = &get_zero_real();

        for (npy_intp i = 0; i < n; ++i, ip1 += is1, op1 += os1) {
            // Fetch the input operand.
            const auto *a = fetch_cted_real(base_ptr_i1, ct_flags_i1, ip1, zr);

            // Fetch the output operand, passing in f1 to construct
            // the result if needed.
            auto [opt_ptr, new_real] = ensure_cted_real(base_ptr_o, ct_flags_o, op1, [&]() { return f1(*a); });

            if (!opt_ptr) {
                throw py::error_already_set();
            }

            // If no new real was constructed, perform the binary operation.
            if (!new_real) {
                f2(**opt_ptr, *a);
            }
        };
    });
}

// Generic function for unary comparisons.
template <typename F>
void py_real_ufunc_unary_cmp(char **args, const npy_intp *dimensions, const npy_intp *steps, void *, const F &f)
{
    npy_intp is1 = steps[0], os1 = steps[1], n = dimensions[0];
    char *ip1 = args[0], *op1 = args[1];

    with_pybind11_eh([&]() {
        // Try to locate the input buffer in the memory map.
        const auto [base_ptr_i1, meta_ptr_i1] = get_memory_metadata(ip1);
        const auto *ct_flags_i1
            = (base_ptr_i1 == nullptr) ? nullptr : meta_ptr_i1->ensure_ct_flags_inited<mppp::real>();

        // Fetch a pointer to the global zero const. This will be used in place
        // of non-constructed input real arguments.
        const auto *zr = &get_zero_real();

        for (npy_intp i = 0; i < n; ++i, ip1 += is1, op1 += os1) {
            // Fetch the input operand.
            const auto *a = fetch_cted_real(base_ptr_i1, ct_flags_i1, ip1, zr);

            // Run the comparison and write into the output.
            *reinterpret_cast<npy_bool *>(op1) = static_cast<npy_bool>(f(*a));
        };
    });
}

// Generic NumPy binary operation with real operands and real output.
template <typename F2, typename F3>
void py_real_ufunc_binary(char **args, const npy_intp *dimensions, const npy_intp *steps, void *, const F2 &f2,
                          const F3 &f3)
{
    npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions;
    char *i0 = args[0], *i1 = args[1], *o = args[2];

    with_pybind11_eh([&]() {
        // Try to locate the input/output buffers in the memory map.
        const auto [base_ptr_i0, meta_ptr_i0] = get_memory_metadata(i0);
        const auto *ct_flags_i0
            = (base_ptr_i0 == nullptr) ? nullptr : meta_ptr_i0->ensure_ct_flags_inited<mppp::real>();

        const auto [base_ptr_i1, meta_ptr_i1] = get_memory_metadata(i1);
        const auto *ct_flags_i1
            = (base_ptr_i1 == nullptr) ? nullptr : meta_ptr_i1->ensure_ct_flags_inited<mppp::real>();

        const auto [base_ptr_o, meta_ptr_o] = get_memory_metadata(o);
        auto *ct_flags_o = (base_ptr_o == nullptr) ? nullptr : meta_ptr_o->ensure_ct_flags_inited<mppp::real>();

        // Fetch a pointer to the global zero const. This will be used in place
        // of non-constructed input real arguments.
        const auto *zr = &get_zero_real();

        for (npy_intp k = 0; k < n; ++k) {
            // Fetch the input operands.
            const auto *a = fetch_cted_real(base_ptr_i0, ct_flags_i0, i0, zr);
            const auto *b = fetch_cted_real(base_ptr_i1, ct_flags_i1, i1, zr);

            // Fetch the output operand, passing in f2 to construct
            // the result if needed.
            auto [opt_ptr, new_real] = ensure_cted_real(base_ptr_o, ct_flags_o, o, [&]() { return f2(*a, *b); });

            if (!opt_ptr) {
                throw py::error_already_set();
            }

            // If no new real was constructed, perform the ternary operation.
            if (!new_real) {
                f3(**opt_ptr, *a, *b);
            }

            i0 += is0;
            i1 += is1;
            o += os;
        }
    });
}

// Generic NumPy binary operation for real comparisons.
template <typename F>
void py_real_ufunc_binary_cmp(char **args, const npy_intp *dimensions, const npy_intp *steps, void *, const F &f)
{
    npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions;
    char *i0 = args[0], *i1 = args[1], *o = args[2];

    with_pybind11_eh([&]() {
        // Try to locate the input buffers in the memory map.
        const auto [base_ptr_i0, meta_ptr_i0] = get_memory_metadata(i0);
        const auto *ct_flags_i0
            = (base_ptr_i0 == nullptr) ? nullptr : meta_ptr_i0->ensure_ct_flags_inited<mppp::real>();

        const auto [base_ptr_i1, meta_ptr_i1] = get_memory_metadata(i1);
        const auto *ct_flags_i1
            = (base_ptr_i1 == nullptr) ? nullptr : meta_ptr_i1->ensure_ct_flags_inited<mppp::real>();

        // Fetch a pointer to the global zero const. This will be used in place
        // of non-constructed input real arguments.
        const auto *zr = &get_zero_real();

        for (npy_intp k = 0; k < n; ++k) {
            // Fetch the input operands.
            const auto *a = fetch_cted_real(base_ptr_i0, ct_flags_i0, i0, zr);
            const auto *b = fetch_cted_real(base_ptr_i1, ct_flags_i1, i1, zr);

            // Run the comparison and write into the output.
            *reinterpret_cast<npy_bool *>(o) = static_cast<npy_bool>(f(*a, *b));

            i0 += is0;
            i1 += is1;
            o += os;
        }
    });
}

// Helper to register a NumPy ufunc.
template <typename... Types>
void npy_register_ufunc(py::module_ &numpy, const char *name, PyUFuncGenericFunction func, Types... types_)
{
    py::object ufunc_ob = numpy.attr(name);
    auto *ufunc_ = ufunc_ob.ptr();
    if (PyObject_IsInstance(ufunc_, reinterpret_cast<PyObject *>(&PyUFunc_Type)) == 0) {
        py_throw(PyExc_TypeError, fmt::format("The name '{}' in the NumPy module is not a ufunc", name).c_str());
    }
    auto *ufunc = reinterpret_cast<PyUFuncObject *>(ufunc_);

    int types[] = {types_...};
    if (std::size(types) != boost::numeric_cast<std::size_t>(ufunc->nargs)) {
        py_throw(PyExc_TypeError, fmt::format("Invalid arity for the ufunc '{}': the NumPy function expects {} "
                                              "arguments, but {} arguments were provided instead",
                                              name, ufunc->nargs, std::size(types))
                                      .c_str());
    }

    if (PyUFunc_RegisterLoopForType(ufunc, npy_registered_py_real, func, types, nullptr) < 0) {
        py_throw(PyExc_TypeError, fmt::format("The registration of the ufunc '{}' failed", name).c_str());
    }
}

// Dot product.
void npy_py_real_dot_impl(void *ip0_, npy_intp is0, void *ip1_, npy_intp is1, void *op, npy_intp n,
                          const unsigned char *base_ptr_i0, const bool *ct_flags_i0, const unsigned char *base_ptr_i1,
                          const bool *ct_flags_i1, const unsigned char *base_ptr_o, bool *ct_flags_o,
                          const mppp::real *zr)
{
    with_pybind11_eh([&]() {
        const auto *ip0 = static_cast<const char *>(ip0_), *ip1 = static_cast<const char *>(ip1_);

        // Return value and temporary storage
        // for the multiplication.
        mppp::real ret, tmp;

        for (npy_intp i = 0; i < n; ++i) {
            // Fetch the input operands.
            const auto *a = fetch_cted_real(base_ptr_i0, ct_flags_i0, ip0, zr);
            const auto *b = fetch_cted_real(base_ptr_i1, ct_flags_i1, ip1, zr);

            // NOTE: use fma from MPFR 4 onwards?
            mppp::mul(tmp, *a, *b);
            ret += tmp;

            ip0 += is0;
            ip1 += is1;
        }

        // Write out the result.
        auto [opt_ptr, new_real]
            = ensure_cted_real(base_ptr_o, ct_flags_o, op, [&]() -> mppp::real && { return std::move(ret); });

        if (!opt_ptr) {
            throw py::error_already_set();
        }

        if (!new_real) {
            **opt_ptr = std::move(ret);
        }
    });
}

void npy_py_real_dot(void *ip0, npy_intp is0, void *ip1, npy_intp is1, void *op, npy_intp n, void *)
{
    // Try to locate the input/output buffers in the memory map.
    const auto [base_ptr_i0, meta_ptr_i0] = get_memory_metadata(ip0);
    const auto *ct_flags_i0 = (base_ptr_i0 == nullptr) ? nullptr : meta_ptr_i0->ensure_ct_flags_inited<mppp::real>();

    const auto [base_ptr_i1, meta_ptr_i1] = get_memory_metadata(ip1);
    const auto *ct_flags_i1 = (base_ptr_i1 == nullptr) ? nullptr : meta_ptr_i1->ensure_ct_flags_inited<mppp::real>();

    const auto [base_ptr_o, meta_ptr_o] = get_memory_metadata(op);
    auto *ct_flags_o = (base_ptr_o == nullptr) ? nullptr : meta_ptr_o->ensure_ct_flags_inited<mppp::real>();

    // Fetch a pointer to the global zero const. This will be used in place
    // of non-constructed input real arguments.
    const auto *zr = &get_zero_real();

    npy_py_real_dot_impl(ip0, is0, ip1, is1, op, n, base_ptr_i0, ct_flags_i0, base_ptr_i1, ct_flags_i1, base_ptr_o,
                         ct_flags_o, zr);
}

// Implementation of matrix multiplication.
void npy_py_real_matrix_multiply(char **args, const npy_intp *dimensions, const npy_intp *steps,
                                 const unsigned char *base_ptr_i0, const bool *ct_flags_i0,
                                 const unsigned char *base_ptr_i1, const bool *ct_flags_i1,
                                 const unsigned char *base_ptr_o, bool *ct_flags_o, const mppp::real *zr)
{
    // Pointers to data for input and output arrays.
    char *ip1 = args[0];
    char *ip2 = args[1];
    char *op = args[2];

    // Lengths of core dimensions.
    npy_intp dm = dimensions[0];
    npy_intp dn = dimensions[1];
    npy_intp dp = dimensions[2];

    // Striding over core dimensions.
    npy_intp is1_m = steps[0];
    npy_intp is1_n = steps[1];
    npy_intp is2_n = steps[2];
    npy_intp is2_p = steps[3];
    npy_intp os_m = steps[4];
    npy_intp os_p = steps[5];

    // Calculate dot product for each row/column vector pair.
    for (npy_intp m = 0; m < dm; ++m) {
        npy_intp p = 0;
        for (; p < dp; ++p) {
            npy_py_real_dot_impl(ip1, is1_n, ip2, is2_n, op, dn, base_ptr_i0, ct_flags_i0, base_ptr_i1, ct_flags_i1,
                                 base_ptr_o, ct_flags_o, zr);

            // Advance to next column of 2nd input array and output array.
            ip2 += is2_p;
            op += os_p;
        }

        // Reset to first column of 2nd input array and output array.
        ip2 -= is2_p * p;
        op -= os_p * p;

        // Advance to next row of 1st input array and output array.
        ip1 += is1_m;
        op += os_m;
    }
}

void npy_py_real_gufunc_matrix_multiply(char **args, const npy_intp *dimensions, const npy_intp *steps, void *)
{
    // Length of flattened outer dimensions.
    npy_intp dN = dimensions[0];

    // Striding over flattened outer dimensions for input and output arrays.
    npy_intp s0 = steps[0];
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];

    // Try to locate the input/output buffers in the memory map.
    const auto [base_ptr_i0, meta_ptr_i0] = get_memory_metadata(args[0]);
    const auto *ct_flags_i0 = (base_ptr_i0 == nullptr) ? nullptr : meta_ptr_i0->ensure_ct_flags_inited<mppp::real>();

    const auto [base_ptr_i1, meta_ptr_i1] = get_memory_metadata(args[1]);
    const auto *ct_flags_i1 = (base_ptr_i1 == nullptr) ? nullptr : meta_ptr_i1->ensure_ct_flags_inited<mppp::real>();

    const auto [base_ptr_o, meta_ptr_o] = get_memory_metadata(args[2]);
    auto *ct_flags_o = (base_ptr_o == nullptr) ? nullptr : meta_ptr_o->ensure_ct_flags_inited<mppp::real>();

    // Fetch a pointer to the global zero const. This will be used in place
    // of non-constructed input real arguments.
    const auto *zr = &get_zero_real();

    // Loop through outer dimensions, performing matrix multiply on core dimensions for each loop.
    for (npy_intp N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2) {
        npy_py_real_matrix_multiply(args, dimensions + 1, steps + 3, base_ptr_i0, ct_flags_i0, base_ptr_i1, ct_flags_i1,
                                    base_ptr_o, ct_flags_o, zr);
    }
}

} // namespace

} // namespace detail

// Check if the input object is a py_real.
bool py_real_check(PyObject *ob)
{
    return PyObject_IsInstance(ob, reinterpret_cast<PyObject *>(&py_real_type)) != 0;
}

// Helper to reach the mppp::real stored inside a py_real.
mppp::real *get_real_val(PyObject *self)
{
    assert(py_real_check(self));

    return std::launder(reinterpret_cast<mppp::real *>(reinterpret_cast<py_real *>(self)->m_storage));
}

// Helpers to create a pyreal from a real.
PyObject *pyreal_from_real(const mppp::real &src)
{
    return detail::py_real_from_args(src);
}

PyObject *pyreal_from_real(mppp::real &&src)
{
    return detail::py_real_from_args(std::move(src));
}

namespace detail
{

namespace
{

template <std::size_t NDim, std::size_t CurDim = 0>
void pyreal_check_array_impl(std::array<py::ssize_t, NDim> &idxs, const py::array &arr, const unsigned char *base_ptr,
                             const bool *ct_flags, mpfr_prec_t prec)
{
    static_assert(CurDim < NDim);

    const auto cur_size = arr.shape(static_cast<py::ssize_t>(CurDim));

    if constexpr (CurDim == NDim - 1u) {
        for (idxs[CurDim] = 0; idxs[CurDim] < cur_size; ++idxs[CurDim]) {
            const auto *ptr = std::apply([&arr](auto... js) { return arr.data(js...); }, idxs);

            if (!check_cted_real(base_ptr, ct_flags, ptr)) {
                py_throw(
                    PyExc_ValueError,
                    fmt::format("A non-constructed/invalid real was detected in a NumPy array at the indices {}", idxs)
                        .c_str());
            }

            if (prec != 0) {
                const auto cur_prec = std::launder(reinterpret_cast<const mppp::real *>(ptr))->get_prec();

                if (cur_prec != prec) {
                    py_throw(PyExc_ValueError,
                             fmt::format("A real with precision {} was detected at the indices {} in an array which "
                                         "should instead contain elements with a precision of {}",
                                         cur_prec, idxs, prec)
                                 .c_str());
                }
            }
        }
    } else {
        for (idxs[CurDim] = 0; idxs[CurDim] < cur_size; ++idxs[CurDim]) {
            pyreal_check_array_impl<NDim, CurDim + 1u>(idxs, arr, base_ptr, ct_flags, prec);
        }
    }
}

} // namespace

} // namespace detail

// Helper to check if all the real values in arr have been
// properly constructed with a precision of prec. If prec == 0,
// then no check on the precision is performed (i.e., only the
// check for proper construction is performed).
void pyreal_check_array(const py::array &arr, mpfr_prec_t prec)
{
    assert(arr.dtype().num() == npy_registered_py_real);

    // Exit immediately if the array does not contain any data.
    if (arr.size() == 0 || arr.ndim() == 0) {
        return;
    }

    // Fetch the base pointer and the metadata.
    const auto [base_ptr, meta_ptr] = get_memory_metadata(arr.data());
    const auto *ct_flags = (base_ptr == nullptr) ? nullptr : meta_ptr->ensure_ct_flags_inited<mppp::real>();

    // NOTE: handle 1 and 2 dimensions for now, that's all we need.
    switch (arr.ndim()) {
        case 1: {
            std::array<py::ssize_t, 1> idxs{};
            detail::pyreal_check_array_impl(idxs, arr, base_ptr, ct_flags, prec);
            break;
        }
        case 2: {
            std::array<py::ssize_t, 2> idxs{};
            detail::pyreal_check_array_impl(idxs, arr, base_ptr, ct_flags, prec);
            break;
        }
        default:
            py_throw(
                PyExc_ValueError,
                fmt::format("Cannot call pyreal_check_array() on an array with {} dimensions", arr.ndim()).c_str());
    }
}

namespace detail
{

namespace
{

template <std::size_t NDim, std::size_t CurDim = 0>
void pyreal_ensure_array_impl(std::array<py::ssize_t, NDim> &idxs, py::array &arr, const unsigned char *base_ptr,
                              bool *ct_flags, mpfr_prec_t prec)
{
    static_assert(CurDim < NDim);

    const auto cur_size = arr.shape(static_cast<py::ssize_t>(CurDim));

    if constexpr (CurDim == NDim - 1u) {
        for (idxs[CurDim] = 0; idxs[CurDim] < cur_size; ++idxs[CurDim]) {
            auto *ptr = std::apply([&arr](auto... js) { return arr.mutable_data(js...); }, idxs);

            // Ensure that ptr contains a constructed real.
            auto [ptr_opt, new_real] = ensure_cted_real(base_ptr, ct_flags, ptr,
                                                        [prec]() { return mppp::real(mppp::real_kind::zero, prec); });

            if (!ptr_opt) {
                // The construction of a new real was attempted but it failed.
                // In this case, the Python error flag has already been set.
                throw py::error_already_set();
            }

            if (!new_real) {
                // No new real was created, round the existing one.
                (*ptr_opt)->prec_round(prec);
            }
        }
    } else {
        for (idxs[CurDim] = 0; idxs[CurDim] < cur_size; ++idxs[CurDim]) {
            pyreal_ensure_array_impl<NDim, CurDim + 1u>(idxs, arr, base_ptr, ct_flags, prec);
        }
    }
}

} // namespace

} // namespace detail

// Helper to ensure that arr contains properly constructed
// real values with precision prec. If real values already
// exist in arr, they will be rounded to precision prec. Otherwise,
// new reals with value 0 and precision prec will be constructed.
void pyreal_ensure_array(py::array &arr, mpfr_prec_t prec)
{
    assert(arr.dtype().num() == npy_registered_py_real);

    // Exit immediately if the array does not contain any data.
    if (arr.size() == 0 || arr.ndim() == 0) {
        return;
    }

    // Fetch the base pointer and the metadata.
    const auto [base_ptr, meta_ptr] = get_memory_metadata(arr.data());
    auto *ct_flags = (base_ptr == nullptr) ? nullptr : meta_ptr->ensure_ct_flags_inited<mppp::real>();

    // NOTE: handle 1 and 2 dimensions for now, that's all we need.
    switch (arr.ndim()) {
        case 1: {
            std::array<py::ssize_t, 1> idxs{};
            detail::pyreal_ensure_array_impl(idxs, arr, base_ptr, ct_flags, prec);
            break;
        }
        case 2: {
            std::array<py::ssize_t, 2> idxs{};
            detail::pyreal_ensure_array_impl(idxs, arr, base_ptr, ct_flags, prec);
            break;
        }
        default:
            py_throw(
                PyExc_ValueError,
                fmt::format("Cannot call pyreal_ensure_array() on an array with {} dimensions", arr.ndim()).c_str());
    }
}

void expose_real(py::module_ &m)
{
    // Setup the custom NumPy memory management functions.
    setup_custom_numpy_mem_handler(m);

    // Fill out the entries of py_real_type.
    py_real_type.tp_base = &PyGenericArrType_Type;
    py_real_type.tp_name = "heyoka.core.real";
    py_real_type.tp_basicsize = sizeof(py_real);
    py_real_type.tp_flags = Py_TPFLAGS_DEFAULT;
    py_real_type.tp_doc = PyDoc_STR("");
    py_real_type.tp_new = detail::py_real_new;
    py_real_type.tp_init = detail::py_real_init;
    py_real_type.tp_dealloc = detail::py_real_dealloc;
    py_real_type.tp_repr = detail::py_real_repr;
    py_real_type.tp_as_number = &detail::py_real_as_number;
    py_real_type.tp_richcompare = &detail::py_real_rcmp;
    py_real_type.tp_getset = detail::py_real_get_set;
    py_real_type.tp_methods = detail::py_real_methods;

    // Fill out the functions for the number protocol. See:
    // https://docs.python.org/3/c-api/number.html
    detail::py_real_as_number.nb_negative = [](PyObject *a) { return detail::py_real_unop(a, detail::negation_func); };
    detail::py_real_as_number.nb_positive = [](PyObject *a) { return detail::py_real_unop(a, detail::identity_func); };
    detail::py_real_as_number.nb_absolute = [](PyObject *a) { return detail::py_real_unop(a, detail::abs_func); };
    detail::py_real_as_number.nb_add
        = [](PyObject *a, PyObject *b) { return detail::py_real_binop(a, b, std::plus{}); };
    detail::py_real_as_number.nb_subtract
        = [](PyObject *a, PyObject *b) { return detail::py_real_binop(a, b, std::minus{}); };
    detail::py_real_as_number.nb_multiply
        = [](PyObject *a, PyObject *b) { return detail::py_real_binop(a, b, std::multiplies{}); };
    detail::py_real_as_number.nb_true_divide
        = [](PyObject *a, PyObject *b) { return detail::py_real_binop(a, b, std::divides{}); };
    detail::py_real_as_number.nb_floor_divide
        = [](PyObject *a, PyObject *b) { return detail::py_real_binop(a, b, detail::floor_divide_func); };
    detail::py_real_as_number.nb_power = [](PyObject *a, PyObject *b, PyObject *mod) -> PyObject * {
        if (mod != Py_None) {
            PyErr_SetString(PyExc_ValueError, "Modular exponentiation is not supported for real");

            return nullptr;
        }

        return detail::py_real_binop(a, b, detail::pow_func);
    };
    // NOTE: these conversions can never throw.
    detail::py_real_as_number.nb_bool
        = [](PyObject *arg) { return static_cast<int>(static_cast<bool>(*get_real_val(arg))); };
    detail::py_real_as_number.nb_float
        = [](PyObject *arg) { return PyFloat_FromDouble(static_cast<double>(*get_real_val(arg))); };
    // NOTE: for large integers, this goes through a string conversion.
    // Of course, this can be implemented much faster.
    detail::py_real_as_number.nb_int = [](PyObject *arg) -> PyObject * {
        const auto &val = *get_real_val(arg);

        if (mppp::isnan(val)) {
            PyErr_SetString(PyExc_ValueError, "Cannot convert real NaN to integer");
            return nullptr;
        }

        if (!mppp::isfinite(val)) {
            PyErr_SetString(PyExc_OverflowError, "Cannot convert real infinity to integer");
            return nullptr;
        }

        PyObject *retval = nullptr;

        with_pybind11_eh([&]() {
            const auto val_int = mppp::integer<1>{val};
            long long candidate = 0;
            if (val_int.get(candidate)) {
                retval = PyLong_FromLongLong(candidate);
            } else {
                retval = PyLong_FromString(val_int.to_string().c_str(), nullptr, 10);
            }
        });

        return retval;
    };

    // Finalize py_real_type.
    if (PyType_Ready(&py_real_type) < 0) {
        // NOTE: PyType_Ready() already sets the exception flag.
        throw py::error_already_set();
    }

    // Fill out the NumPy descriptor.
    detail::npy_py_real_descr_proto.kind = 'f';
    detail::npy_py_real_descr_proto.type = 'r';
    detail::npy_py_real_descr_proto.byteorder = '=';
    // NOTE: NPY_NEEDS_INIT is important because it gives us a way
    // to detect the use by NumPy of memory buffers not managed
    // via NEP 49.
    detail::npy_py_real_descr_proto.flags
        = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM | NPY_NEEDS_INIT | NPY_LIST_PICKLE;
    detail::npy_py_real_descr_proto.elsize = sizeof(mppp::real);
    detail::npy_py_real_descr_proto.alignment = alignof(mppp::real);
    detail::npy_py_real_descr_proto.f = &detail::npy_py_real_arr_funcs;

    // Setup the basic NumPy array functions.
    // NOTE: not 100% sure PyArray_InitArrFuncs() is needed, as npy_py_real_arr_funcs
    // is already cleared out on creation.
    PyArray_InitArrFuncs(&detail::npy_py_real_arr_funcs);
    // NOTE: get/set item, copyswap and nonzero are the functions that
    // need to be able to deal with "misbehaved" array:
    // https://numpy.org/doc/stable/reference/c-api/types-and-structures.html
    // Hence, we group the together at the beginning.
    detail::npy_py_real_arr_funcs.getitem = detail::npy_py_real_getitem;
    detail::npy_py_real_arr_funcs.setitem = detail::npy_py_real_setitem;
    detail::npy_py_real_arr_funcs.copyswap = detail::npy_py_real_copyswap;
    // NOTE: in recent NumPy versions, there's apparently no need to provide
    // an implementation for this, as NumPy is capable of synthesising an
    // implementation based on copyswap. If needed, we could consider
    // a custom implementation to improve performance.
    // detail::npy_py_real_arr_funcs.copyswapn = detail::npy_py_real_copyswapn;
    detail::npy_py_real_arr_funcs.nonzero = detail::npy_py_real_nonzero;
    detail::npy_py_real_arr_funcs.compare = detail::npy_py_real_compare;
    detail::npy_py_real_arr_funcs.argmin = [](void *data, npy_intp n, npy_intp *max_ind, void *) {
        return detail::npy_py_real_argminmax(data, n, max_ind, std::less{});
    };
    detail::npy_py_real_arr_funcs.argmax = [](void *data, npy_intp n, npy_intp *max_ind, void *) {
        return detail::npy_py_real_argminmax(data, n, max_ind, std::greater{});
    };
    detail::npy_py_real_arr_funcs.fill = detail::npy_py_real_fill;
    detail::npy_py_real_arr_funcs.fillwithscalar = detail::npy_py_real_fillwithscalar;
    detail::npy_py_real_arr_funcs.dotfunc = detail::npy_py_real_dot;
    // NOTE: not sure if this is needed - it does not seem to have
    // any effect and the online examples of user dtypes do not set it.
    // Let's leave it commented out at this time.
    // detail::npy_py_real_arr_funcs.scalarkind = [](void *) -> int { return NPY_FLOAT_SCALAR; };

    // Register the NumPy data type.
    Py_SET_TYPE(&detail::npy_py_real_descr_proto, &PyArrayDescr_Type);
    npy_registered_py_real = PyArray_RegisterDataType(&detail::npy_py_real_descr_proto);
    if (npy_registered_py_real < 0) {
        // NOTE: PyArray_RegisterDataType() already sets the error flag.
        throw py::error_already_set();
    }
    // Set the actual descriptor. See:
    //
    // https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_RegisterDataType.
    detail::npy_py_real_descr = PyArray_DescrFromType(npy_registered_py_real);

    // Support the dtype(real) syntax.
    if (PyDict_SetItemString(py_real_type.tp_dict, "dtype", reinterpret_cast<PyObject *>(detail::npy_py_real_descr))
        < 0) {
        py_throw(PyExc_TypeError, "Cannot add the 'dtype' field to the real class");
    }

    // NOTE: need access to the numpy module to register ufuncs.
    auto numpy_mod = py::module_::import("numpy");

    // Arithmetics.
    detail::npy_register_ufunc(
        numpy_mod, "add",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary(args, dimensions, steps, data, std::plus{}, detail::add3_func);
        },
        npy_registered_py_real, npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "subtract",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary(args, dimensions, steps, data, std::minus{}, detail::sub3_func);
        },
        npy_registered_py_real, npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "multiply",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary(args, dimensions, steps, data, std::multiplies{}, detail::mul3_func);
        },
        npy_registered_py_real, npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "divide",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary(args, dimensions, steps, data, std::divides{}, detail::div3_func);
        },
        npy_registered_py_real, npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "square",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::square_func, detail::square2_func);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "floor_divide",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary(args, dimensions, steps, data, detail::floor_divide_func,
                                         detail::floor_divide_func3);
        },
        npy_registered_py_real, npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "absolute",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::abs_func, detail::abs_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "fabs",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::abs_func, detail::abs_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "positive",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::identity_func, detail::identity_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "negative",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::negation_func, detail::negation_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    // Power/roots.
    detail::npy_register_ufunc(
        numpy_mod, "power",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary(args, dimensions, steps, data, detail::pow_func, detail::pow_func3);
        },
        npy_registered_py_real, npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "sqrt",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::sqrt_func, detail::sqrt_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "cbrt",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::cbrt_func, detail::cbrt_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    // Trigonometry.
    detail::npy_register_ufunc(
        numpy_mod, "sin",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::sin_func, detail::sin_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "cos",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::cos_func, detail::cos_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "tan",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::tan_func, detail::tan_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "arcsin",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::asin_func, detail::asin_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "arccos",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::acos_func, detail::acos_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "arctan",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::atan_func, detail::atan_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "arctan2",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary(args, dimensions, steps, data, detail::atan2_func, detail::atan2_func3);
        },
        npy_registered_py_real, npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "sinh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::sinh_func, detail::sinh_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "cosh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::cosh_func, detail::cosh_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "tanh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::tanh_func, detail::tanh_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "arcsinh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::asinh_func, detail::asinh_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "arccosh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::acosh_func, detail::acosh_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "arctanh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::atanh_func, detail::atanh_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "deg2rad",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::deg2rad_func, detail::deg2rad_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "radians",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::deg2rad_func, detail::deg2rad_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "rad2deg",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::rad2deg_func, detail::rad2deg_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "degrees",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::rad2deg_func, detail::rad2deg_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    // Exponentials and logarithms.
    detail::npy_register_ufunc(
        numpy_mod, "exp",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::exp_func, detail::exp_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "exp2",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::exp2_func, detail::exp2_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "expm1",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::expm1_func, detail::expm1_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "log",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::log_func, detail::log_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "log2",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::log2_func, detail::log2_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "log10",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::log10_func, detail::log10_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "log1p",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::log1p_func, detail::log1p_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    // Comparisons.
    detail::npy_register_ufunc(
        numpy_mod, "equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary_cmp(args, dimensions, steps, data, std::equal_to{});
        },
        npy_registered_py_real, npy_registered_py_real, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "not_equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary_cmp(args, dimensions, steps, data, std::not_equal_to{});
        },
        npy_registered_py_real, npy_registered_py_real, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "less",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary_cmp(args, dimensions, steps, data, std::less{});
        },
        npy_registered_py_real, npy_registered_py_real, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "less_equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary_cmp(args, dimensions, steps, data, std::less_equal{});
        },
        npy_registered_py_real, npy_registered_py_real, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "greater",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary_cmp(args, dimensions, steps, data, std::greater{});
        },
        npy_registered_py_real, npy_registered_py_real, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "greater_equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary_cmp(args, dimensions, steps, data, std::greater_equal{});
        },
        npy_registered_py_real, npy_registered_py_real, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "sign",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary(args, dimensions, steps, data, detail::sign_func, detail::sign_func2);
        },
        npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "maximum",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary(args, dimensions, steps, data, detail::max_func, detail::max_func3);
        },
        npy_registered_py_real, npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "minimum",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_binary(args, dimensions, steps, data, detail::min_func, detail::min_func3);
        },
        npy_registered_py_real, npy_registered_py_real, npy_registered_py_real);
    detail::npy_register_ufunc(
        numpy_mod, "isnan",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary_cmp(args, dimensions, steps, data, detail::isnan_func);
        },
        npy_registered_py_real, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "isinf",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary_cmp(args, dimensions, steps, data, detail::isinf_func);
        },
        npy_registered_py_real, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "isfinite",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real_ufunc_unary_cmp(args, dimensions, steps, data, detail::isfinite_func);
        },
        npy_registered_py_real, NPY_BOOL);
    // Matrix multiplication.
    detail::npy_register_ufunc(numpy_mod, "matmul", detail::npy_py_real_gufunc_matrix_multiply, npy_registered_py_real,
                               npy_registered_py_real, npy_registered_py_real);

    // Casting.
    detail::npy_register_cast_functions<float>();
    detail::npy_register_cast_functions<double>();
    // NOTE: registering conversions to/from long double has several
    // adverse effects on the casting rules. Unclear at this time if
    // such issues are from our code or NumPy's. Let's leave it commented
    // at this time.
    // detail::npy_register_cast_functions<long double>();
#if defined(HEYOKA_HAVE_REAL128)
    detail::npy_register_cast_functions<mppp::real128>();
#endif

    detail::npy_register_cast_functions<npy_int8>();
    detail::npy_register_cast_functions<npy_int16>();
    detail::npy_register_cast_functions<npy_int32>();
    detail::npy_register_cast_functions<npy_int64>();

    detail::npy_register_cast_functions<npy_uint8>();
    detail::npy_register_cast_functions<npy_uint16>();
    detail::npy_register_cast_functions<npy_uint32>();
    detail::npy_register_cast_functions<npy_uint64>();

    // NOTE: need to do bool manually as it typically overlaps
    // with another C++ type (e.g., uint8).
    if (PyArray_RegisterCastFunc(PyArray_DescrFromType(NPY_BOOL), npy_registered_py_real,
                                 &detail::npy_cast_to_real<npy_bool>)
        < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

    // NOTE: this is to signal that conversion of bool to real is safe.
    if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_BOOL), npy_registered_py_real, NPY_NOSCALAR) < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

    if (PyArray_RegisterCastFunc(detail::npy_py_real_descr, NPY_BOOL, &detail::npy_cast_from_real<npy_bool>) < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

    // Add py_real_type to the module.
    Py_INCREF(&py_real_type);
    if (PyModule_AddObject(m.ptr(), "real", reinterpret_cast<PyObject *>(&py_real_type)) < 0) {
        Py_DECREF(&py_real_type);
        py_throw(PyExc_TypeError, "Could not add the real type to the module");
    }

    // Min/max prec getters.
    m.def("real_prec_min", &mppp::real_prec_min);
    m.def("real_prec_max", &mppp::real_prec_max);

    // Expose functions for testing.
    m.def("_make_no_real_array", [](std::vector<mppp::real> vec) {
        auto vec_ptr = std::make_unique<std::vector<mppp::real>>(std::move(vec));

        py::capsule vec_caps(vec_ptr.get(), [](void *ptr) {
            std::unique_ptr<std::vector<mppp::real>> vptr(static_cast<std::vector<mppp::real> *>(ptr));
        });

        // NOTE: at this point, the capsule has been created successfully (including
        // the registration of the destructor). We can thus release ownership from vec_ptr,
        // as now the capsule is responsible for destroying its contents. If the capsule constructor
        // throws, the destructor function is not registered/invoked, and the destructor
        // of vec_ptr will take care of cleaning up.
        auto *ptr = vec_ptr.release();

        return py::array(py::dtype(npy_registered_py_real),
                         py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ptr->size())}, ptr->data(),
                         std::move(vec_caps));
    });

    // Function to test the custom caster.
    m.def("_copy_real", [](const mppp::real &x) { return x; });

    // Function to test the pyreal_check_array() helper.
    using namespace pybind11::literals;
    m.def("_real_check_array", &pyreal_check_array, "arr"_a, "prec"_a.noconvert() = 0);

    // Function to test the pyreal_ensure_array() helper.
    m.def("_real_ensure_array", &pyreal_ensure_array);
}

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#else

void expose_real(py::module_ &) {}

#endif

} // namespace heyoka_py
