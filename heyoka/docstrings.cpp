// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>

#include "docstrings.hpp"

namespace heyoka_py::docstrings
{

std::string expression()
{
    return R"(Class to represent symbolic expressions.

.. note::

   A :ref:`tutorial <ex_system>` explaining the use of this
   class is available.

This is the main class used to represent mathematical expressions in heyoka.
It is a union of several *node types*:

- symbolic variables,
- mathematical constants,
- runtime parameters,
- n-ary functions.

Expressions can be created in a variety of ways, and after construction
they can be combined via arithmetic operators and mathematical functions
to form new expressions of arbitrary complexity.

)";
}

std::string expression_init()
{
    return R"(__init__(self, x: int | numpy.single | float | numpy.longdouble | str = 0.)

Constructor.

This constructor will initialise the expression either as a numerical
constant (if *x* is of a numerical type) or as a variable (if *x* is
a :py:class:`str`).

:param x: the construction argument.

)";
}

std::string diff_args()
{
    return R"(Enum for selecting differentiation arguments.

Values of this enum are used in :func:`~heyoka.diff_tensors()` to select
all variables and/or parameters as differentiation arguments.

)";
}

std::string diff_args_vars()
{
    return R"(Differentiate with respect to all variables. The variables are considered in alphabetical order.

)";
}

std::string diff_args_pars()
{
    return R"(Differentiate with respect to all parameters. The parameters are considered in ascending index order.

)";
}

std::string diff_args_all()
{
    return R"(Differentiate with respect to all variables and parameters. The variables, in alphabetical order, are considered before the parameters, in ascending index order.

)";
}

std::string make_vars()
{
    return R"(make_vars(*args: str) -> expression | list[expression]

Create variable expressions from strings.

This function will return one or more :py:class:`~heyoka.expression`
instances containing variables constructed from the input arguments.
If a single argument is supplied, a single expression is returned.
Otherwise, a list of expressions (one for each argument) is returned.

:param args: the input string(s).

:returns: one or more expressions constructed from *args*.

:raises ValueError: if the number of *args* is zero.

Examples:
  >>> from heyoka import make_vars
  >>> x = make_vars("x")
  >>> y, z = make_vars("y", "z")

)";
}

std::string dtens()
{
    return R"(Derivative tensors.

.. note::

   A :ref:`tutorial <computing_derivatives>` explaining the use of this
   class is available.

This class is used to store the derivative tensors computed by
:func:`~heyoka.diff_tensors()`.

)";
}

std::string dtens_init()
{
    return R"(__init__(self)

Default constructor.

This constructor will initialise the object into an empty state.

Note that usually :class:`~heyoka.dtens` object are returned by the
invocation of :func:`~heyoka.diff_tensors()`, rather than being constructed
directly by the user.

)";
}

std::string dtens_get_derivatives()
{
    return R"(get_derivatives(order: int, component: Optional[int] = None) -> list[tuple[list[int], expression]]

Get the derivatives for the specified order and component.

The derivatives are returned as a sorted list mapping vectors of indices to the
corresponding derivatives, as explained in detail in the :ref:`tutorial <computing_derivatives>`.

If an invalid *order* or *component* is provided, an empty list will be returned.

:param order: the differentiation order.
:param component: the function component. If not provided, the derivatives for all components
   will be returned.

:returns: the list of derivatives for the desired order and component(s).

Examples:
  >>> import heyoka as hy
  >>> x, y = hy.make_vars("x", "y")
  >>> dt = hy.diff_tensors([x**2, x*y], [x, y])
  >>> dt.get_derivatives(diff_order=1) # Fetch the Jacobian.
  [([0, 1, 0], {(2.0000000000000000 * x)}), ([0, 0, 1], 0.0000000000000000), ([1, 1, 0], {y}), ([1, 0, 1], {x})]
  >>> dt.get_derivatives(diff_order=1, component=0) # Fetch the gradient of the first component.
  [([0, 1, 0], {(2.0000000000000000 * x)}), ([0, 0, 1], 0.0000000000000000)]

)";
}

std::string dtens_order()
{
    return R"(The maximum differentiation order.

:rtype: int

)";
}

std::string dtens_nouts()
{
    return R"(The number of function components.

:rtype: int

)";
}

std::string dtens_nargs()
{
    return R"(The number of arguments with respect to which the derivatives are computed.

:rtype: int

)";
}

std::string dtens_index_of()
{
    return R"(index_of(vidx: list[int] | tuple[int, list[tuple[int, int]]]) -> int

Get the position corresponding to the input indices vector.

This method will return the positional index of the derivative corresponding
to the vector of indices *vidx*, which can be supplied in dense or sparse format.

If *vidx* is invalid, the length of *self* will be returned.

:param vidx: the input indices vector.

:return: the positional index of *vidx* in *self*.

)";
}

std::string dtens_args()
{
    return R"(The list of arguments with respect to which the derivatives are computed.

:rtype: list[expression]

)";
}

std::string dtens_gradient()
{
    return R"(The gradient of the function.

:rtype: list[expression]

:raises ValueError: if the function has not exactly 1 component or if the maximum derivative order is zero.

)";
}

std::string dtens_jacobian()
{
    return R"(The Jacobian of the function as a 2D array.

:rtype: numpy.ndarray[expression]

:raises ValueError: if the function has zero components or if the maximum derivative order is zero.

)";
}

std::string diff_tensors()
{
    return R"(diff_tensors(func: list[expression], diff_args: list[expression] | diff_args, diff_order: int = 1) -> dtens

Compute the tensors of derivatives.

.. note::

   A :ref:`tutorial <computing_derivatives>` explaining the use of this
   function is available.

This function will compute the tensors of derivatives of the vector function *func* with
respect to the arguments *diff_args* up to the derivative order *diff_order*. The derivatives
will be returned as a :class:`~heyoka.dtens` object.

Several checks are run on the input arguments. Specifically:

- the number of function components (i.e., the length of *func*) cannot be zero,
- *diff_args* cannot be empty,
- *diff_args* must consist only of variable and/or parameter expressions,
- *diff_args* cannot contain duplicates.

:param func: the vector function whose derivatives will be computed.
:param diff_args: the arguments with respect to which the derivatives will be computed.
:param diff_order: the maximum derivative order.

:returns: the tensor of derivatives of *func* with respect to *diff_args* up to order *diff_order*.

:raises ValueError: if one or more input arguments are malformed, as explained above.

)";
}

std::string lagrangian()
{
    return R"(lagrangian(L: expression, qs: list[expression], qdots: list[expression], D: expression = expression(0.)) -> list[tuple[expression, expression]]
    
Formulate the Euler-Lagrange equations for a Lagrangian.

.. note::

   A :ref:`tutorial <lagham_tut>` illustrating the use of this function is available.

This function will formulate the differential equations for the Lagrangian *L*.
The lists of generalised coordinates and velocities are given in *qs* and *qdots*
respectively. *D* is an optional Rayleigh dissipation function - a quadratic form in the
generalised velocities that can be used to add dissipative forces to the
dynamical system.

An error will be raised if one or more input arguments are malformed. Specifically:

- *qs* and *qdots* must be non-empty and have the same length,
- all expressions in *qs* and *qdots* must be variables,
- *qs* and *qdots* must not contain duplicates, and a variable appearing
  in one list cannot appear in the other,
- *L* must depend only on the variables listed in *qs* and *qdots*,
- *D* must depend only on the variables listed in *qdots*.

:param L: the Lagrangian.
:param qs: the generalised coordinates.
:param qdots: the generalised velocities.
:param D: the Rayleigh dissipation function.

:returns: the Euler-Lagrange equations in explicit form for the Lagrangian *L*.

:raises ValueError: if one or more input arguments are malformed, as explained above.

)";
}

std::string hamiltonian()
{
    return R"(hamiltonian(H: expression, qs: list[expression], ps: list[expression]) -> list[tuple[expression, expression]]
    
Formulate Hamilton's equations for a Hamiltonian.

.. note::

   A :ref:`tutorial <lagham_tut>` illustrating the use of this function is available.

This function will formulate the differential equations for the Hamiltonian *H*.
The lists of generalised coordinates and momenta are given in *qs* and *ps*
respectively.

An error will be raised if one or more input arguments are malformed. Specifically:

- *qs* and *ps* must be non-empty and have the same length,
- all expressions in *qs* and *ps* must be variables,
- *qs* and *ps* must not contain duplicates, and a variable appearing
  in one list cannot appear in the other,
- *H* must depend only on the variables listed in *qs* and *ps*.

:param H: the Hamiltonian.
:param qs: the generalised coordinates.
:param ps: the generalised momenta.

:returns: Hamilton's equations for the Hamiltonian *H*.

:raises ValueError: if one or more input arguments are malformed, as explained above.

)";
}

} // namespace heyoka_py::docstrings
