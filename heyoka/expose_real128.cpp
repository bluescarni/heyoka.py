// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <pybind11/pybind11.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <functional>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>

#include <mp++/config.hpp>
#include <mp++/real128.hpp>

#include "common_utils.hpp"

#endif

#include "expose_real128.hpp"

namespace heyoka_py
{

namespace py = pybind11;

#if defined(HEYOKA_HAVE_REAL128)

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

// NOTE: more type properties will be filled in when initing the module.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyTypeObject py_real128_type = {PyVarObject_HEAD_INIT(nullptr, 0)};

// NOTE: this is an integer used to represent the
// real128 type *after* it has been registered in the
// NumPy dtype system. This is needed to expose
// ufuncs and it will be set up during module
// initialisation.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int npy_registered_py_real128 = 0;

namespace detail
{

namespace
{

// Double check that malloc() aligns memory suitably
// for py_real128. See:
// https://en.cppreference.com/w/cpp/types/max_align_t
static_assert(alignof(py_real128) <= alignof(std::max_align_t));

// A few function object to implement basic mathematical
// operations in a generic fashion.
const auto identity_func = [](auto x) { return x; };

const auto negation_func = [](auto x) { return -x; };

const auto abs_func = [](auto x) {
    using std::abs;

    return abs(x);
};

const auto sqrt_func = [](auto x) {
    using std::sqrt;

    return sqrt(x);
};

const auto sin_func = [](auto x) {
    using std::sin;

    return sin(x);
};

const auto cos_func = [](auto x) {
    using std::cos;

    return cos(x);
};

const auto isfinite_func = [](auto x) {
    using std::isfinite;

    return isfinite(x);
};

const auto floor_divide_func = [](auto x, auto y) {
    using std::floor;

    auto ret = x / y;
    return floor(ret);
};

const auto pow_func = [](auto x, auto y) {
    using std::pow;

    return pow(x, y);
};

// Methods for the number protocol.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyNumberMethods py_real128_as_number = {};

// Create a py_real128 containing an mppp::real128
// constructed from the input set of arguments.
template <typename... Args>
PyObject *py_real128_from_args(Args &&...args)
{
    // Acquire the storage for a py_real128.
    void *pv = py_real128_type.tp_alloc(&py_real128_type, 0);
    if (pv == nullptr) {
        return nullptr;
    }

    // Construct the py_real128 instance.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto *p = ::new (pv) py_real128;

    // Setup its internal data.
    ::new (p->m_storage) mppp::real128(std::forward<Args>(args)...);

    return reinterpret_cast<PyObject *>(p);
}

// __new__() implementation.
PyObject *py_real128_new([[maybe_unused]] PyTypeObject *type, PyObject *, PyObject *)
{
    assert(type == &py_real128_type);

    return py_real128_from_args();
}

// Helper to convert a Python integer to real128.
std::optional<mppp::real128> py_int_to_real128(PyObject *arg)
{
    assert(PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyLong_Type)));

    // Try to see if nptr fits in a long long.
    int overflow = 0;
    const auto candidate = PyLong_AsLongLongAndOverflow(arg, &overflow);

    if (overflow == 0) {
        return candidate;
    }

    // Need to construct a real128 from the limb array.
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
    mppp::real128 retval = ob_digit[--abs_ob_size];

    // Init the number of binary digits consumed so far.
    // NOTE: this is of course not zero, as we read *some* bits from
    // the most significant limb. However we don't know exactly how
    // many bits we read from the top limb, so we err on the safe
    // side in order to ensure that, when ncdigits reaches the bit width
    // of real128, we have indeed consumed *at least* that many bits.
    auto ncdigits = 0;

    // Keep on reading limbs until either:
    // - we run out of limbs, or
    // - we have read as many bits as the bit width of real128.
    static_assert(std::numeric_limits<mppp::real128>::digits < std::numeric_limits<int>::max() - PyLong_SHIFT);
    while (ncdigits < std::numeric_limits<mppp::real128>::digits && abs_ob_size != 0u) {
        retval = scalbn(retval, PyLong_SHIFT);
        retval += ob_digit[--abs_ob_size];
        ncdigits += PyLong_SHIFT;
    }

    if (abs_ob_size != 0u) {
        // We have filled up the mantissa, we just need to adjust the exponent.
        // We still have abs_ob_size * PyLong_SHIFT bits remaining, and we
        // need to shift that much.
        if (abs_ob_size
            > static_cast<unsigned long>(std::numeric_limits<long>::max()) / static_cast<unsigned>(PyLong_SHIFT)) {
            PyErr_SetString(PyExc_OverflowError,
                            "An overflow condition was detected while converting a Python integer to a real128");

            return {};
        }
        retval = scalbln(retval, static_cast<long>(abs_ob_size * static_cast<unsigned>(PyLong_SHIFT)));
    }

    return neg ? -retval : retval;
}

// __init__() implementation.
int py_real128_init(PyObject *self, PyObject *args, PyObject *)
{
    PyObject *arg = nullptr;

    if (PyArg_ParseTuple(args, "|O", &arg) == 0) {
        return -1;
    }

    if (arg == nullptr) {
        return 0;
    }

    if (PyFloat_Check(arg)) {
        auto fp_arg = PyFloat_AsDouble(arg);

        if (PyErr_Occurred() == nullptr) {
            *get_val(self) = fp_arg;
        } else {
            return -1;
        }
    } else if (PyLong_Check(arg)) {
        if (auto opt = py_int_to_real128(arg)) {
            *get_val(self) = *opt;
        } else {
            return -1;
        }
#if defined(MPPP_FLOAT128_WITH_LONG_DOUBLE)
    } else if (PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyLongDoubleArrType_Type)) != 0) {
        *get_val(self) = reinterpret_cast<PyLongDoubleScalarObject *>(arg)->obval;
#endif
    } else if (py_real128_check(arg)) {
        *get_val(self) = *get_val(arg);
    } else if (PyUnicode_Check(arg)) {
        const auto *str = PyUnicode_AsUTF8(arg);

        if (str == nullptr) {
            return -1;
        }

        try {
            *get_val(self) = mppp::real128(str);
        } catch (const std::invalid_argument &ia) {
            PyErr_SetString(PyExc_ValueError, ia.what());

            return -1;
        } catch (const std::exception &ex) {
            PyErr_SetString(PyExc_RuntimeError, ex.what());

            return -1;
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "Unknown exception caught while trying to build a real128 from string");

            return -1;
        }
    } else {
        PyErr_Format(PyExc_TypeError, "Cannot construct a real128 from an object of type \"%s\"",
                     Py_TYPE(arg)->tp_name);

        return -1;
    }

    return 0;
}

// Deallocation.
void py_real128_dealloc(PyObject *self)
{
    // Invoke the destructor.
    get_val(self)->~real128();

    // Free the memory.
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// Helper to construct a real128 from one of the
// supported Pythonic numerical types:
// - int,
// - float,
// - long double.
// If the conversion is successful, returns {x, true}. If some error
// is encountered, returns {<empty>, false}. If the type of arg is not
// supported, returns {<empty>, true}.
std::pair<std::optional<mppp::real128>, bool> real128_from_ob(PyObject *arg)
{
    if (PyFloat_Check(arg)) {
        auto fp_arg = PyFloat_AsDouble(arg);

        if (PyErr_Occurred() == nullptr) {
            return {fp_arg, true};
        } else {
            return {{}, false};
        }
    } else if (PyLong_Check(arg)) {
        if (auto opt = py_int_to_real128(arg)) {
            return {*opt, true};
        } else {
            return {{}, false};
        }
#if defined(MPPP_FLOAT128_WITH_LONG_DOUBLE)
    } else if (PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyLongDoubleArrType_Type)) != 0) {
        return {reinterpret_cast<PyLongDoubleScalarObject *>(arg)->obval, true};
#endif
    } else {
        return {{}, true};
    }
}

// __repr__().
PyObject *py_real128_repr(PyObject *self)
{
    try {
        return PyUnicode_FromString(get_val(self)->to_string().c_str());
    } catch (const std::exception &ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "An unknown exception was caught while trying to obtain the representation of a real128");
    }

    return nullptr;
}

// Generic implementation of unary operations.
template <typename F>
PyObject *py_real128_unop(PyObject *a, const F &op)
{
    if (py_real128_check(a)) {
        const auto *x = get_val(a);
        return py_real128_from_args(op(*x));
    } else {
        Py_RETURN_NOTIMPLEMENTED;
    }
}

// Generic implementation of binary operations.
template <typename F>
PyObject *py_real128_binop(PyObject *a, PyObject *b, const F &op)
{
    const auto a_is_real128 = py_real128_check(a);
    const auto b_is_real128 = py_real128_check(b);

    if (a_is_real128 && b_is_real128) {
        // Both operands are real128.
        const auto *x = get_val(a);
        const auto *y = get_val(b);

        return py_real128_from_args(op(*x, *y));
    }

    if (a_is_real128) {
        // a is a real128, b is not. Try to convert
        // b to a real128.
        auto [r, flag] = real128_from_ob(b);

        if (r) {
            // The conversion was successful, do the op.
            return py_real128_from_args(op(*get_val(a), *r));
        }

        if (flag) {
            // b's type is not supported.
            Py_RETURN_NOTIMPLEMENTED;
        }

        // Attempting to convert b to a real128 generated an error.
        return nullptr;
    }

    if (b_is_real128) {
        // The mirror of the previous case.
        auto [r, flag] = real128_from_ob(a);

        if (r) {
            return py_real128_from_args(op(*r, *get_val(b)));
        }

        if (flag) {
            Py_RETURN_NOTIMPLEMENTED;
        }

        return nullptr;
    }

    // Neither a nor b are real128.
    Py_RETURN_NOTIMPLEMENTED;
}

// Rich comparison operator.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyObject *py_real128_rcmp(PyObject *a, PyObject *b, int op)
{
    auto impl = [a, b](const auto &func) -> PyObject * {
        const auto a_is_real128 = py_real128_check(a);
        const auto b_is_real128 = py_real128_check(b);

        if (a_is_real128 && b_is_real128) {
            // Both operands are real128.
            const auto *x = get_val(a);
            const auto *y = get_val(b);

            if (func(*x, *y)) {
                Py_RETURN_TRUE;
            } else {
                Py_RETURN_FALSE;
            }
        }

        if (a_is_real128) {
            // a is a real128, b is not. Try to convert
            // b to a real128.
            auto [r, flag] = real128_from_ob(b);

            if (r) {
                // The conversion was successful, do the op.
                if (func(*get_val(a), *r)) {
                    Py_RETURN_TRUE;
                } else {
                    Py_RETURN_FALSE;
                }
            }

            if (flag) {
                // b's type is not supported.
                Py_RETURN_NOTIMPLEMENTED;
            }

            // Attempting to convert b to a real128 generated an error.
            return nullptr;
        }

        if (b_is_real128) {
            // The mirror of the previous case.
            auto [r, flag] = real128_from_ob(a);

            if (r) {
                if (func(*r, *get_val(b))) {
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
            return impl(std::less<>{});
        case Py_LE:
            return impl(std::less_equal<>{});
        case Py_EQ:
            return impl(std::equal_to<>{});
        case Py_NE:
            return impl(std::not_equal_to<>{});
        case Py_GT:
            return impl(std::greater<>{});
        default:
            assert(op == Py_GE);
            return impl(std::greater_equal<>{});
    }
}

// Helper to import the NumPy API bits.
PyObject *import_numpy(PyObject *m)
{
    import_array();

    import_umath();

    return m;
}

// NumPy array function.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyArray_ArrFuncs npy_py_real128_arr_funcs = {};

// NumPy type descriptor.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyArray_Descr npy_py_real128_descr = {PyObject_HEAD_INIT(0) & py_real128_type};

// Array getitem.
PyObject *npy_py_real128_getitem(void *data, void *)
{
    mppp::real128 q;
    std::memcpy(&q, data, sizeof(q));
    return py_real128_from_args(q);
}

// Array setitem.
int npy_py_real128_setitem(PyObject *item, void *data, void *)
{
    if (py_real128_check(item)) {
        const auto *q = get_val(item);
        std::memcpy(data, q, sizeof(mppp::real128));
        return 0;
    } else {
        PyErr_Format(PyExc_TypeError,
                     "Cannot invoke __setitem__() on a real128 array with an input value of type \"%s\"",
                     Py_TYPE(item)->tp_name);
        return -1;
    }
}

// Byteswap a real128 in place.
void byteswap(mppp::real128 *x)
{
    static_assert(sizeof(mppp::real128) == sizeof(x->m_value));

    const auto bs = sizeof(mppp::real128);

    auto *cptr = reinterpret_cast<char *>(x);
    for (std::size_t i = 0; i < bs / 2u; ++i) {
        const auto j = bs - i - 1u;
        const auto t = cptr[i];
        cptr[i] = cptr[j];
        cptr[j] = t;
    }
}

// Copyswap primitive.
void npy_py_real128_copyswap(void *dst, void *src, int swap, void *)
{
    assert(dst != nullptr);

    auto *r = static_cast<mppp::real128 *>(dst);

    if (src != nullptr) {
        // NOTE: not sure if src and dst can overlap here,
        // use memmove() just in case.
        std::memmove(r, src, sizeof(mppp::real128));
    }

    if (swap != 0) {
        byteswap(r);
    }
}

// Nonzero primitive.
npy_bool npy_py_real128_nonzero(void *data, void *)
{
    mppp::real128 q;
    std::memcpy(&q, data, sizeof(mppp::real128));

    return q != 0 ? NPY_TRUE : NPY_FALSE;
}

// Generic NumPy conversion function to real128.
template <typename From>
void npy_cast_to_real128(void *from, void *to, npy_intp n, void *, void *)
{
    const auto *typed_from = static_cast<const From *>(from);
    auto *typed_to = static_cast<mppp::real128 *>(to);

    for (npy_intp i = 0; i < n; ++i) {
        typed_to[i] = typed_from[i];
    }
}

// Generic NumPy conversion function from real128.
template <typename To>
void npy_cast_from_real128(void *from, void *to, npy_intp n, void *, void *)
{
    const auto *typed_from = static_cast<const mppp::real128 *>(from);
    auto *typed_to = static_cast<To *>(to);

    for (npy_intp i = 0; i < n; ++i) {
        typed_to[i] = static_cast<To>(typed_from[i]);
    }
}

// Machinery to associate a C++ type to a NumPy type.
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

#if defined(MPPP_FLOAT128_WITH_LONG_DOUBLE)

HEYOKA_PY_ASSOC_TY(long double, NPY_LONGDOUBLE);

#endif

HEYOKA_PY_ASSOC_TY(npy_int8, NPY_INT8);
HEYOKA_PY_ASSOC_TY(npy_int16, NPY_INT16);
HEYOKA_PY_ASSOC_TY(npy_int32, NPY_INT32);
HEYOKA_PY_ASSOC_TY(npy_int64, NPY_INT64);

#undef HEYOKA_PY_ASSOC_TY

// Shortcut.
template <typename T>
constexpr auto npy_type = cpp_to_numpy_t<T>::value;

// Helper to register NumPy casting functions to/from T.
template <typename T>
void npy_register_cast_functions()
{
    if (PyArray_RegisterCastFunc(PyArray_DescrFromType(npy_type<T>), npy_registered_py_real128, &npy_cast_to_real128<T>)
        < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

    // NOTE: this is to signal that conversion of any scalar type to real128 is safe.
    if (PyArray_RegisterCanCast(PyArray_DescrFromType(npy_type<T>), npy_registered_py_real128, NPY_NOSCALAR) < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

    if (PyArray_RegisterCastFunc(&npy_py_real128_descr, npy_type<T>, &npy_cast_from_real128<T>) < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }
}

// Generic NumPy unary operation.
template <typename T, typename F>
void py_real128_ufunc_unary(char **args, const npy_intp *dimensions, const npy_intp *steps, void *, const F &f)
{
    npy_intp is1 = steps[0], os1 = steps[1], n = dimensions[0];
    char *ip1 = args[0], *op1 = args[1];

    for (npy_intp i = 0; i < n; ++i, ip1 += is1, op1 += os1) {
        const auto &x = *reinterpret_cast<mppp::real128 *>(ip1);
        *reinterpret_cast<T *>(op1) = static_cast<T>(f(x));
    };
}

template <typename F>
void py_real128_ufunc_unary(char **args, const npy_intp *dimensions, const npy_intp *steps, void *p, const F &f)
{
    py_real128_ufunc_unary<mppp::real128>(args, dimensions, steps, p, f);
}

// Generic NumPy binary operation.
template <typename T, typename F>
void py_real128_ufunc_binary(char **args, const npy_intp *dimensions, const npy_intp *steps, void *, const F &f)
{
    npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions;
    char *i0 = args[0], *i1 = args[1], *o = args[2];

    for (npy_intp k = 0; k < n; ++k) {
        const auto &x = *reinterpret_cast<mppp::real128 *>(i0);
        const auto &y = *reinterpret_cast<mppp::real128 *>(i1);
        *reinterpret_cast<T *>(o) = static_cast<T>(f(x, y));
        i0 += is0;
        i1 += is1;
        o += os;
    }
}

template <typename F>
void py_real128_ufunc_binary(char **args, const npy_intp *dimensions, const npy_intp *steps, void *p, const F &f)
{
    py_real128_ufunc_binary<mppp::real128>(args, dimensions, steps, p, f);
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

    if (PyUFunc_RegisterLoopForType(ufunc, npy_registered_py_real128, func, types, nullptr) < 0) {
        py_throw(PyExc_TypeError, fmt::format("The registration of the ufunc '{}' failed", name).c_str());
    }
}

} // namespace

} // namespace detail

// Check if the input object is a py_real128.
bool py_real128_check(PyObject *ob)
{
    return PyObject_IsInstance(ob, reinterpret_cast<PyObject *>(&py_real128_type)) != 0;
}

// Helpers to reach the mppp::real128 stored inside a py_real128.
mppp::real128 *get_val(py_real128 *self)
{
    return std::launder(reinterpret_cast<mppp::real128 *>(self->m_storage));
}

mppp::real128 *get_val(PyObject *self)
{
    assert(py_real128_check(self));

    return get_val(reinterpret_cast<py_real128 *>(self));
}

void expose_real128(py::module_ &m)
{
    if (detail::import_numpy(m.ptr()) == nullptr) {
        // NOTE: on failure, the NumPy macros already set
        // the error indicator. Thus, all it is left to do
        // is to throw the pybind11 exception.
        throw py::error_already_set();
    }

    // Fill out the entries of py_real128_type.
    py_real128_type.tp_base = &PyGenericArrType_Type;
    py_real128_type.tp_name = "heyoka.core.real128";
    py_real128_type.tp_basicsize = sizeof(py_real128);
    py_real128_type.tp_flags = Py_TPFLAGS_DEFAULT;
    py_real128_type.tp_doc = PyDoc_STR("");
    py_real128_type.tp_new = detail::py_real128_new;
    py_real128_type.tp_init = detail::py_real128_init;
    py_real128_type.tp_dealloc = detail::py_real128_dealloc;
    py_real128_type.tp_repr = detail::py_real128_repr;
    py_real128_type.tp_as_number = &detail::py_real128_as_number;
    py_real128_type.tp_richcompare = &detail::py_real128_rcmp;

    // Fill out the functions for the number protocol. See:
    // https://docs.python.org/3/c-api/number.html
    detail::py_real128_as_number.nb_negative
        = [](PyObject *a) { return detail::py_real128_unop(a, detail::negation_func); };
    detail::py_real128_as_number.nb_positive
        = [](PyObject *a) { return detail::py_real128_unop(a, detail::identity_func); };
    detail::py_real128_as_number.nb_absolute = [](PyObject *a) { return detail::py_real128_unop(a, detail::abs_func); };
    detail::py_real128_as_number.nb_add
        = [](PyObject *a, PyObject *b) { return detail::py_real128_binop(a, b, std::plus<>{}); };
    detail::py_real128_as_number.nb_subtract
        = [](PyObject *a, PyObject *b) { return detail::py_real128_binop(a, b, std::minus<>{}); };
    detail::py_real128_as_number.nb_multiply
        = [](PyObject *a, PyObject *b) { return detail::py_real128_binop(a, b, std::multiplies<>{}); };
    detail::py_real128_as_number.nb_true_divide
        = [](PyObject *a, PyObject *b) { return detail::py_real128_binop(a, b, std::divides<>{}); };
    detail::py_real128_as_number.nb_floor_divide
        = [](PyObject *a, PyObject *b) { return detail::py_real128_binop(a, b, detail::floor_divide_func); };
    detail::py_real128_as_number.nb_power = [](PyObject *a, PyObject *b, PyObject *mod) -> PyObject * {
        if (mod != Py_None) {
            PyErr_SetString(PyExc_ValueError, "Modular exponentiation is not supported for real128");

            return nullptr;
        }

        return detail::py_real128_binop(a, b, detail::pow_func);
    };
    detail::py_real128_as_number.nb_bool
        = [](PyObject *arg) { return static_cast<int>(static_cast<bool>(*get_val(arg))); };

    // Finalize py_real128_type.
    if (PyType_Ready(&py_real128_type) < 0) {
        py_throw(PyExc_TypeError, "Could not finalise the real128 type");
    }

    // Fill out the NumPy descriptor.
    detail::npy_py_real128_descr.kind = 'f';
    detail::npy_py_real128_descr.type = 'q';
    detail::npy_py_real128_descr.byteorder = '=';
    detail::npy_py_real128_descr.flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
    detail::npy_py_real128_descr.elsize = sizeof(mppp::real128);
    detail::npy_py_real128_descr.alignment = alignof(mppp::real128);
    detail::npy_py_real128_descr.f = &detail::npy_py_real128_arr_funcs;

    // Setup the basic NumPy array functions.
    // NOTE: not 100% sure PyArray_InitArrFuncs() is needed, as npy_py_real128_arr_funcs
    // is already cleared out on creation.
    PyArray_InitArrFuncs(&detail::npy_py_real128_arr_funcs);
    detail::npy_py_real128_arr_funcs.getitem = detail::npy_py_real128_getitem;
    detail::npy_py_real128_arr_funcs.setitem = detail::npy_py_real128_setitem;
    detail::npy_py_real128_arr_funcs.copyswap = detail::npy_py_real128_copyswap;
    detail::npy_py_real128_arr_funcs.nonzero = detail::npy_py_real128_nonzero;

    // Register the NumPy data type.
    Py_TYPE(&detail::npy_py_real128_descr) = &PyArrayDescr_Type;
    npy_registered_py_real128 = PyArray_RegisterDataType(&detail::npy_py_real128_descr);
    if (npy_registered_py_real128 < 0) {
        py_throw(PyExc_TypeError, "Could not register the real128 type in NumPy");
    }

    // Support the dtype(real128) syntax.
    if (PyDict_SetItemString(py_real128_type.tp_dict, "dtype",
                             reinterpret_cast<PyObject *>(&detail::npy_py_real128_descr))
        < 0) {
        py_throw(PyExc_TypeError, "Cannot add the 'dtype' field to the real128 class");
    }

    // NOTE: need access to the numpy module to register ufuncs.
    auto numpy_mod = py::module_::import("numpy");

    detail::npy_register_ufunc(
        numpy_mod, "add",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, std::plus<>{});
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "subtract",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, std::minus<>{});
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "multiply",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, std::multiplies<>{});
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "divide",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, std::divides<>{});
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "floor_divide",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, detail::floor_divide_func);
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "power",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, detail::pow_func);
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "absolute",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::abs_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "positive",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::identity_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "negative",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::negation_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "sqrt",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::sqrt_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "sin",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::sin_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "cos",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::cos_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "less",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::less<>{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "less_equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::less_equal<>{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::equal_to<>{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "not_equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::not_equal_to<>{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "greater",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::greater<>{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "greater_equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::greater_equal<>{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "isfinite",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary<npy_bool>(args, dimensions, steps, data, detail::isfinite_func);
        },
        npy_registered_py_real128, NPY_BOOL);

    // Casting.
    detail::npy_register_cast_functions<float>();
    detail::npy_register_cast_functions<double>();

#if defined(MPPP_FLOAT128_WITH_LONG_DOUBLE)

    detail::npy_register_cast_functions<long double>();

#endif

    detail::npy_register_cast_functions<npy_int8>();
    detail::npy_register_cast_functions<npy_int16>();
    detail::npy_register_cast_functions<npy_int32>();
    detail::npy_register_cast_functions<npy_int64>();

    // Add py_real128_type to the module.
    Py_INCREF(&py_real128_type);
    if (PyModule_AddObject(m.ptr(), "real128", reinterpret_cast<PyObject *>(&py_real128_type)) < 0) {
        Py_DECREF(&py_real128_type);
        py_throw(PyExc_TypeError, "Could not add the real128 type to the module");
    }

    // Helper to return the epsilon of real128.
    // TODO fix with automatic conversion.
    m.def("_get_real128_eps", []() {
        return py::reinterpret_steal<py::object>(
            detail::py_real128_from_args(std::numeric_limits<mppp::real128>::epsilon()));
    });
}

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#else

void expose_real128(py::module_ &) {}

#endif

} // namespace heyoka_py
