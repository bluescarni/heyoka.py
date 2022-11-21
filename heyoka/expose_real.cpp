// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <pybind11/pybind11.h>

#if defined(HEYOKA_HAVE_REAL)

#include <cassert>
#include <cstddef>
#include <limits>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include <boost/safe_numerics/safe_integer.hpp>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include <mp++/integer.hpp>
#include <mp++/real.hpp>

#include "common_utils.hpp"

#endif

#include "custom_casters.hpp"
#include "expose_real.hpp"

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

#endif

// NOTE: more type properties will be filled in when initing the module.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyTypeObject py_real_type = {PyVarObject_HEAD_INIT(nullptr, 0)};

namespace detail
{

namespace
{

// Double check that malloc() aligns memory suitably
// for py_real. See:
// https://en.cppreference.com/w/cpp/types/max_align_t
static_assert(alignof(py_real) <= alignof(std::max_align_t));

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

// Helper to convert a Python integer to an mp++ real.
// The precision of the result is inferred from the
// bit size of the integer.
// NOTE: for better performance if needed, it would be better
// to avoid the construction of an intermediate mppp::integer:
// determine the bit length of arg, and construct directly
// an mppp::real using the same idea as in py_int_to_real_with_prec().
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

        // Get the signed size of nptr.
        const auto ob_size = nptr->ob_base.ob_size;
        assert(ob_size != 0);

        // Get the limbs array.
        const auto *ob_digit = nptr->ob_digit;

        // Is it negative?
        const auto neg = ob_size < 0;

        // Compute the unsigned size.
        using size_type = std::make_unsigned_t<std::remove_const_t<decltype(ob_size)>>;
        static_assert(std::is_same_v<size_type, decltype(static_cast<size_type>(0) + static_cast<size_type>(0))>);
        auto abs_ob_size = neg ? -static_cast<size_type>(ob_size) : static_cast<size_type>(ob_size);

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

        // Need to construct a multiprecision integer from the limb array.
        auto *nptr = reinterpret_cast<PyLongObject *>(arg);

        // Get the signed size of nptr.
        const auto ob_size = nptr->ob_base.ob_size;
        assert(ob_size != 0);

        // Get the limbs array.
        const auto *ob_digit = nptr->ob_digit;

        // Is it negative?
        const auto neg = ob_size < 0;

        // Compute the unsigned size.
        using size_type = std::make_unsigned_t<std::remove_const_t<decltype(ob_size)>>;
        static_assert(std::is_same_v<size_type, decltype(static_cast<size_type>(0) + static_cast<size_type>(0))>);
        auto abs_ob_size = neg ? -static_cast<size_type>(ob_size) : static_cast<size_type>(ob_size);

        // Init the retval with the first (most significant) limb. The Python integer is nonzero, so this is safe.
        mppp::real ret{ob_digit[--abs_ob_size], prec};

        // Init the number of binary digits consumed so far.
        // NOTE: this is of course not zero, as we read *some* bits from
        // the most significant limb. However we don't know exactly how
        // many bits we read from the top limb, so we err on the safe
        // side in order to ensure that, when ncdigits reaches prec,
        // we have indeed consumed *at least* that many bits.
        mpfr_prec_t ncdigits = 0;

        // Keep on reading limbs until either:
        // - we run out of limbs, or
        // - we have read at least prec bits.
        while (ncdigits < prec && abs_ob_size != 0u) {
            using safe_mpfr_prec_t = boost::safe_numerics::safe<mpfr_prec_t>;

            mppp::mul_2si(ret, ret, PyLong_SHIFT);
            // NOTE: if prec is small, this addition may increase
            // the precision of ret. This is ok as long as we run
            // a final rounding before returning.
            ret += ob_digit[--abs_ob_size];
            ncdigits = ncdigits + safe_mpfr_prec_t(PyLong_SHIFT);
        }

        if (abs_ob_size != 0u) {
            using safe_long = boost::safe_numerics::safe<long>;

            // We have filled up the mantissa, we just need to adjust the exponent.
            // We still have abs_ob_size * PyLong_SHIFT bits remaining, and we
            // need to shift that much.
            mppp::mul_2si(ret, ret, safe_long(abs_ob_size) * safe_long(PyLong_SHIFT));
        }

        // Do a final rounding. This is necessary if prec is very low,
        // in which case operations on ret might have increased its precision.
        ret.prec_round(prec);

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

// The precision getter.
PyObject *py_real_prec_getter(PyObject *self, void *)
{
    assert(py_real_check(self));

    return PyLong_FromLongLong(static_cast<long long>(get_real_val(self)->get_prec()));
}

// The array for computed attribute instances. See:
// https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_getset
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyGetSetDef py_real_get_set[] = {{"prec", py_real_prec_getter, nullptr, nullptr, nullptr}, {nullptr}};

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

void expose_real(py::module_ &m)
{
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
    // py_real_type.tp_as_number = &detail::py_real_as_number;
    // py_real_type.tp_richcompare = &detail::py_real_rcmp;
    py_real_type.tp_getset = detail::py_real_get_set;

    // Finalize py_real_type.
    if (PyType_Ready(&py_real_type) < 0) {
        py_throw(PyExc_TypeError, "Could not finalise the real type");
    }

    // Add py_real_type to the module.
    Py_INCREF(&py_real_type);
    if (PyModule_AddObject(m.ptr(), "real", reinterpret_cast<PyObject *>(&py_real_type)) < 0) {
        Py_DECREF(&py_real_type);
        py_throw(PyExc_TypeError, "Could not add the real type to the module");
    }
}

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#else

void expose_real(py::module_ &) {}

#endif

} // namespace heyoka_py
