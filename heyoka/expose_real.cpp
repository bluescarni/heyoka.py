// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
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
const auto identity_func = [](const auto &x) { return x; };

const auto negation_func = [](const auto &x) { return -x; };

const auto abs_func = [](const auto &x) {
    using std::abs;

    return abs(x);
};

const auto floor_divide_func = [](const auto &x, const auto &y) {
    using std::floor;

    return floor(x / y);
};

const auto pow_func = [](const auto &x, const auto &y) {
    using std::pow;

    return pow(x, y);
};

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

// Helper to construct a real from one of the
// supported Pythonic numerical types:
// - int,
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

// The array for computed attribute instances. See:
// https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_getset
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyGetSetDef py_real_get_set[] = {{"prec", py_real_prec_getter, nullptr, nullptr, nullptr}, {nullptr}};

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

// NumPy type descriptor.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyArray_Descr npy_py_real_descr = {PyObject_HEAD_INIT(0) & py_real_type};

// Small helper to check if the input pointer is suitably
// aligned for mppp::real.
bool ptr_real_aligned(void *ptr)
{
    assert(ptr != nullptr);

    auto space = sizeof(mppp::real);

    return std::align(alignof(mppp::real), sizeof(mppp::real), ptr, space) != nullptr;
}

// Array getitem.
PyObject *npy_py_real_getitem(void *data, void *arr)
{
    assert(PyArray_Check(reinterpret_cast<PyObject *>(arr)) != 0);

    // NOTE: getitem could be invoked with misaligned data.
    // Detect such occurrence and error out.
    if (!ptr_real_aligned(data)) {
        PyErr_SetString(PyExc_ValueError, "Cannot invoke __getitem__() on misaligned real data");
        return nullptr;
    }

    PyObject *ret = nullptr;

    with_pybind11_eh([&]() {
        if (auto *rptr = numpy_check_cted<mppp::real>(data)) {
            ret = py_real_from_args(*rptr);
        } else {
            ret = py_real_from_args();
        }
    });

    return ret;
}

// Array setitem.
int npy_py_real_setitem(PyObject *item, void *data, void *arr)
{
    assert(PyArray_Check(reinterpret_cast<PyObject *>(arr)) != 0);

    // NOTE: getitem could be invoked with misaligned data.
    // Detect such occurrence and error out.
    if (!ptr_real_aligned(data)) {
        PyErr_SetString(PyExc_ValueError, "Cannot invoke __setitem__() on misaligned real data");
        return -1;
    }

    // TODO handle other types.
    if (!py_real_check(item)) {
        PyErr_Format(PyExc_TypeError, "Cannot invoke __setitem__() on a real array with an input value of type \"%s\"",
                     Py_TYPE(item)->tp_name);
        return -1;
    }

    const auto &src_val = *get_real_val(item);

    auto ptr_opt = numpy_ensure_cted<mppp::real>(data);
    if (!ptr_opt) {
        return -1;
    }

    const auto err = with_pybind11_eh([&]() { **ptr_opt = src_val; });

    return err ? -1 : 0;
}

// Copyswap primitive.
// NOTE: apparently there's no mechanism to directly report an error from this
// function, as it returns void. As a consequence, in case of errors Python throws
// a generic SystemError (rather than the exception of our choice) and complains that
// the function "returned a result with an exception set". This is not optimal,
// but better than just crashing I guess?
void npy_py_real_copyswap(void *dst, void *src, int swap, void *arr)
{
    assert(PyArray_Check(reinterpret_cast<PyObject *>(arr)) != 0);

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

    // Ensure that the destination contains a real, first and foremost.
    auto dst_x_opt = numpy_ensure_cted<mppp::real>(dst);
    if (!dst_x_opt) {
        // numpy_ensure_cted generated an error, exit.
        return;
    }

    // Check if the source contains a real.
    const auto *src_x = numpy_check_cted<mppp::real>(src);

    // NOTE: if a C++ exception is thrown here, dst is not modified
    // and an error code will have been set.
    with_pybind11_eh([&]() {
        if (src_x != nullptr) {
            **dst_x_opt = *src_x;
        } else {
            **dst_x_opt = mppp::real{};
        }
    });
}

// Nonzero primitive.
npy_bool npy_py_real_nonzero(void *data, void *)
{
    if (!ptr_real_aligned(data)) {
        PyErr_SetString(PyExc_ValueError, "Cannot detect nonzero elements in an array of misaligned real data");
        return NPY_FALSE;
    }

    // Check if the source contains a real.
    const auto *x_ptr = numpy_check_cted<mppp::real>(data);

    if (x_ptr == nullptr) {
        return NPY_FALSE;
    } else {
        // NOTE: no exceptions are possible here.
        return x_ptr->zero_p() ? NPY_FALSE : NPY_TRUE;
    }
}

// Helper to access a global default-constructed real instance.
// This is used in the NmuPy helpers below when trying to access
// a not-yet-constructed real in an array.
// NOTE: mark it noexcept as this is equivalent to being unable
// to construct a global variable, a situation which
// we don't really need to be able to recover from.
const auto &get_zero_real() noexcept
{
    static const mppp::real zr;

    return zr;
}

// Helper that, given a memory location in a NumPy array, returns a const
// reference either to the real value stored in that location, if it exists,
// or to the value returned by get_zero_real() otherwise.
const auto &get_real_or_zero(const void *data)
{
    const auto *x_ptr = numpy_check_cted<mppp::real>(data);

    if (x_ptr == nullptr) {
        return get_zero_real();
    } else {
        return *x_ptr;
    }
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
    const auto &x = get_real_or_zero(d0);
    const auto &y = get_real_or_zero(d1);

    // NOTE: no exceptions are possible in these comparisons.
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

    // Build/fetch the array of construction flags, if possible.
    const auto *ct_flags = (base_ptr == nullptr) ? nullptr : meta_ptr->ensure_ct_flags_inited<mppp::real>();

    // Fetch a pointer to the global zero const.
    const auto *zr = &get_zero_real();

    // Fetch the char array version of the memory segment.
    const auto *cdata = reinterpret_cast<const char *>(data);

    // Init the best index/value with the first index/value.
    npy_intp best_i = 0;
    const auto *best_r
        = (ct_flags == nullptr || ct_flags[0]) ? std::launder(reinterpret_cast<const mppp::real *>(cdata)) : zr;

    for (npy_intp i = 1; i < n; ++i) {
        const auto *cur_r = (ct_flags == nullptr || ct_flags[i]) ? std::launder(reinterpret_cast<const mppp::real *>(
                                cdata + static_cast<std::size_t>(i) * sizeof(mppp::real)))
                                                                 : zr;

        // NOTE: real comparisons never throw, no need
        // for exception handling.
        if (cmp(*cur_r, *best_r)) {
            best_i = i;
            best_r = cur_r;
        }
    }

    *max_ind = best_i;

    return 0;
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

// Helper to create a pyreal from a real.
PyObject *pyreal_from_real(const mppp::real &src)
{
    return detail::py_real_from_args(src);
}

void expose_real(py::module_ &m)
{
    // Install the custom NumPy memory management functions.
    install_custom_numpy_mem_handler();

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
        using std::isfinite;
        using std::isnan;

        const auto &val = *get_real_val(arg);

        if (isnan(val)) {
            PyErr_SetString(PyExc_ValueError, "Cannot convert real NaN to integer");
            return nullptr;
        }

        if (!isfinite(val)) {
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
        py_throw(PyExc_TypeError, "Could not finalise the real type");
    }

    // Fill out the NumPy descriptor.
    detail::npy_py_real_descr.kind = 'f';
    detail::npy_py_real_descr.type = 'r';
    detail::npy_py_real_descr.byteorder = '=';
    detail::npy_py_real_descr.flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
    detail::npy_py_real_descr.elsize = sizeof(mppp::real);
    detail::npy_py_real_descr.alignment = alignof(mppp::real);
    detail::npy_py_real_descr.f = &detail::npy_py_real_arr_funcs;

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
#if 0
    detail::npy_py_real128_arr_funcs.fill = detail::npy_py_real128_fill;
    detail::npy_py_real128_arr_funcs.fillwithscalar = detail::npy_py_real128_fillwithscalar;
    detail::npy_py_real128_arr_funcs.dotfunc = detail::npy_py_real128_dot;
    // NOTE: not sure if this is needed - it does not seem to have
    // any effect and the online examples of user dtypes do not set it.
    // Let's leave it commented out at this time.
    // detail::npy_py_real128_arr_funcs.scalarkind = [](void *) -> int { return NPY_FLOAT_SCALAR; };
#endif

    // Register the NumPy data type.
    Py_SET_TYPE(&detail::npy_py_real_descr, &PyArrayDescr_Type);
    npy_registered_py_real = PyArray_RegisterDataType(&detail::npy_py_real_descr);
    if (npy_registered_py_real < 0) {
        py_throw(PyExc_TypeError, "Could not register the real type in NumPy");
    }

    // Support the dtype(real) syntax.
    if (PyDict_SetItemString(py_real_type.tp_dict, "dtype", reinterpret_cast<PyObject *>(&detail::npy_py_real_descr))
        < 0) {
        py_throw(PyExc_TypeError, "Cannot add the 'dtype' field to the real class");
    }

    // Add py_real_type to the module.
    Py_INCREF(&py_real_type);
    if (PyModule_AddObject(m.ptr(), "real", reinterpret_cast<PyObject *>(&py_real_type)) < 0) {
        Py_DECREF(&py_real_type);
        py_throw(PyExc_TypeError, "Could not add the real type to the module");
    }

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
}

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#else

void expose_real(py::module_ &) {}

#endif

} // namespace heyoka_py
