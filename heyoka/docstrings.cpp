// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>

#include <boost/algorithm/string.hpp>

#include <fmt/core.h>

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
    return R"(Class to store derivative tensors.

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
    return R"(get_derivatives(self, order: int, component: Optional[int] = None) -> list[tuple[list[int], expression]]

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
  [([0, 1, 0], (2.0000000000000000 * x)), ([0, 0, 1], 0.0000000000000000), ([1, 1, 0], y), ([1, 0, 1], x)]
  >>> dt.get_derivatives(diff_order=1, component=0) # Fetch the gradient of the first component.
  [([0, 1, 0], (2.0000000000000000 * x)), ([0, 0, 1], 0.0000000000000000)]

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
    return R"(index_of(self, vidx: list[int] | tuple[int, list[tuple[int, int]]]) -> int

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

std::string dtens_hessian()
{
    return R"(hessian(self, component: int) -> numpy.ndarray[expression]

Hessian of a component.

.. versionadded:: 4.0.0

This method will return the Hessian of the selected function component as a 2D array.

:param component: the index of the function component whose Hessian will be returned.

:return: the Hessian of the selected component.

:raises ValueError: if *component* is invalid or if the derivative order is not at least 2.

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

.. versionadded:: 4.0.0

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

.. versionadded:: 4.0.0

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

std::string subs()
{
    return R"(subs(arg: expression | list[expression], smap: dict[str | expression, expression]) -> expression | list[expression]

Substitution.

This function will traverse the input expression(s) *arg* and substitute occurrences of the
keys in *smap* with the corresponding values. String keys in *smap* are interpreted
as variable names.

:param arg: the input expression(s).
:param smap: the substitution dictionary.

:returns: the result of the substitution.

)";
}

std::string sum()
{
    return R"(sum(terms: list[expression]) -> expression

Multivariate summation.

This function will create a multivariate summation containing the arguments in *terms*.
If *terms* is empty, zero will be returned.

:param terms: the input term(s).

:returns: the expression representing the summation.

Examples:
  >>> from heyoka import make_vars, sum
  >>> x, y, z = make_vars("x", "y", "z")
  >>> sum([x, y, z])
  (x + y + z)

)";
}

std::string prod()
{
    return R"(prod(terms: list[expression]) -> expression

Multivariate product.

This function will create a multivariate product containing the arguments in *terms*.
If *terms* is empty, one will be returned.

:param terms: the input term(s).

:returns: the expression representing the product.

Examples:
  >>> from heyoka import make_vars, prod
  >>> x, y, z = make_vars("x", "y", "z")
  >>> prod([x, y, z])
  (x * y * z)

)";
}

// Models
std::string cart2geo()
{
    return R"(cart2geo(xyz: typing.Iterable[expression], ecc2: float = 0.006694379990197619, R_eq: float = 6378137.0, n_iters: int = 4) -> list[expression]

Transform Cartesian coordinates into geodetic coordinates.

.. versionadded:: 4.0.0

.. note::

   A :ref:`tutorial <Thermonets>` showcasing also the use of this
   function is available.

This function will compute the expressions of the geodetic coordinates as functions of the input Cartesian coordinates using
the Hirvonen and Moritz iterations (see "Physical Geodesy" by Heiskanen and Moritz, pp.181-183). The *n_iters* parameter
selects the number of iterations - a higher number will produce a more accurate result, at a higher computational cost.
The default value ensures an accuracy at the centimetre level on the Earth's surface.

A few checks are run on the input arguments. Specifically:

- the number of Cartesian variable (i.e., the length of *xyz*) must be three,
- *ecc2* must be finite and non-negative,
- *R_eq* must be finite and positive,
- *n_iters* must be positive.

The default values for *R_eq* and *ecc2* are taken from the `WGS84 <https://en.wikipedia.org/wiki/World_Geodetic_System>`__ model.

:param xyz: expressions for the Cartesian components [units consistent with *R_eq*].
:param ecc2: the reference ellipsoid eccentricity squared.
:param R_eq: the reference ellipsoid equatorial radius [units consistent with *xyz*].
:param n_iters: number of Hirvonen and Moritz iterations of the inversion algorithm.

:returns: the expressions for the geodetic coordinates [alt, lat, lon]. *alt* in the same units as *xyz* and *R_eq*,
  *lat* in :math:`\left[ -\frac{\pi}{2}, \frac{\pi}{2} \right]` and *lon* in :math:`\left[ -\pi, \pi \right]`.

:raises ValueError: if one or more input arguments are malformed, as explained above.

)";
}

std::string geo2cart()
{
    return R"(geo2cart(geo: typing.Iterable[expression], ecc2: float = 0.006694379990197619, R_eq: float = 6378137.0) -> list[expression]

Transform geodetic coordinates into Cartesian coordinates.

.. versionadded:: 7.3.0

This function will convert the input geodetic coordinates (height, latitude, longitude) into Cartesian
coordinates. The input height is expected in the same units as *R_eq*, while latitude and longitude are
expected in radians.

A few checks are run on the input arguments. Specifically:

- the number of geodetic variables (i.e., the length of *geo*) must be three,
- *ecc2* must be finite and non-negative,
- *R_eq* must be finite and positive.

The default values for *R_eq* and *ecc2* are taken from the `WGS84 <https://en.wikipedia.org/wiki/World_Geodetic_System>`__ model.

:param geo: expressions for the geodetic components.
:param ecc2: the reference ellipsoid eccentricity squared.
:param R_eq: the reference ellipsoid equatorial radius.

:returns: the expressions for the Cartesian coordinates [x, y, z] in the same units as *R_eq*.

:raises ValueError: if one or more input arguments are malformed, as explained above.

)";
}

std::string nrlmsise00_tn()
{
    return R"(nrlmsise00_tn(geodetic: typing.Iterable[expression], f107: expression, f107a: expression, ap: expression, time_expr: expression) -> expression

Produces the expression of the thermospheric density as a function of geodetic coordinates and weather indices.

.. versionadded:: 4.0.0

The expression is approximated by an artificial neural network (a thermoNET) trained over NRLMSISE00 data. 

.. note::

   The thermoNET parameters are published in the work:
   Izzo, Dario, Giacomo Acciarini, and Francesco Biscani.
   "NeuralODEs for VLEO simulations: Introducing thermoNET for Thermosphere Modeling"
   (`arXiv preprint <https://arxiv.org/abs/2405.19384>`__).

.. note::

   A :ref:`tutorial <Thermonets>` showcasing the use of this
   function is available.

A few checks are run on the input arguments. Specifically, the number of geodetic variables (i.e., the length of *geodetic*)
must be three.

:param geodetic: expressions for the Geodetic components. [h, lat, lon] with h in km and lat in :math:`\left[ -\frac{\pi}{2}, \frac{\pi}{2} \right]`.
:param f107: the F10.7 index for the day before *time_expr*.
:param f107a: the 81-day average of the F10.7 index centred on the day of *time_expr*.
:param ap: the AP index on the day of *time_expr*.
:param time_expr: number of fractional days passed since 1st of January.

:returns: the thermospheric density in [kg / m^3] as predicted by the NRLMSISE00 thermoNET model.

:raises ValueError: if one or more input arguments are malformed, as explained above.

)";
}

std::string jb08_tn()
{
    return R"(jb08_tn(geodetic: typing.Iterable[expression], f107: expression, f107a: expression, s107: expression, s107a: expression, m107: expression, m107a: expression, y107: expression, y107a: expression, dDstdT: expression, time_expr: expression) -> expression

Produces the expression of the thermospheric density as a function of geodetic coordinates and weather indices.

.. versionadded:: 4.0.0

The expression is approximated by an artificial neural network (a thermoNET) trained over JB08 data. 

.. note::

   The thermoNET parameters are published in the work:
   Izzo, Dario, Giacomo Acciarini, and Francesco Biscani.
   "NeuralODEs for VLEO simulations: Introducing thermoNET for Thermosphere Modeling"
   (`arXiv preprint <https://arxiv.org/abs/2405.19384>`__).
   
.. note::

   A :ref:`tutorial <Thermonets>` showcasing the use of this
   function is available.

A few checks are run on the input arguments. Specifically, the number of geodetic variables (i.e., the length of *geodetic*) must be three.

:param geodetic: expressions for the Geodetic components. [h, lat, lon] with h in km and lat in :math:`\left[ -\frac{\pi}{2}, \frac{\pi}{2} \right]`.
:param f107: the F10.7 index.
:param f107a: the F10.7 averaged index.
:param s107: the S10.7 index.
:param s107a: the S10.7 averaged index.
:param m107: the M10.7 index.
:param m107a: the M10.7 averaged index.
:param y107: the Y10.7 index.
:param y107a: the Y10.7 averaged index.
:param dDstdT: the dDstdT index.
:param time_expr: number of fractional days passed since 1st of January.

:returns: the thermospheric density in [kg / m^3] as predicted by the JB08 thermoNET model.

:raises ValueError: if one or more input arguments are malformed, as explained above.

)";
}

std::string var_ode_sys()
{
    return R"(Class to represent variational ODE systems.

.. versionadded:: 5.0.0

.. note::

   A :ref:`tutorial <var_ode_sys>` explaining the use of this
   class is available.

)";
}

std::string var_ode_sys_init()
{
    return R"(__init__(self, sys: list[tuple[expression, expression]], args: var_args | list[expression], order: int = 1)

Constructor.

A variational ODE system is constructed from two mandatory arguments: the original ODE system
*sys* and an *args* parameter representing the quantities with respect to which the variational
equations will be formulated.

If *args* is of type :class:`~heyoka.var_args`, then the variational equations will be constructed
with respect to arguments deduced from *sys*. E.g., if *sys* contains the two state variables
:math:`x` and :math:`y` and *args* is the *vars* enumerator of :class:`~heyoka.var_args`, then
the variational equations will be formulated with respect to the initial conditions of
:math:`x` and :math:`y`. Similarly, if *sys* contains two parameters ``par[0]`` and ``par[1]``
and *args* is the *params* enumerator of :class:`~heyoka.var_args`, then
the variational equations will be formulated with respect to the two parameters.

If *args* is a list of :class:`~heyoka.expression`, then the variational equations will be formulated
with respect to the quantities contained in the list. Specifically:

- variable expression are used to request derivatives with respect to the intial conditions
  for that state variable;
- parameter expressions are used to request derivatives with respect to those parameters;
- :attr:`heyoka.time` is used to request the derivative with respect to the initial integration time.

Several checks are run on the input arguments. Specifically:

- *sys* must be a valid ODE system;
- if *args* is of type :class:`~heyoka.var_args`, it must be either one of the valid enumerators
  or a combination of the valid enumerators;
- if *args* is a list of expressions:

  - it must not be empty,
  - it must consists only of variables, parameters or the :attr:`heyoka.time` placeholder,
  - it must not contain duplicates,
  - any variable expression must refer to a state variable in *sys*.

Additionally, the differentiation order *order* must be at least 1.

:param sys: the input ODE system.
:param args: the variational arguments.
:param order: the differentiation order.

:raises ValueError: if one or more input arguments are malformed, as explained above.

)";
}

std::string var_ode_sys_sys()
{
    return R"(The full system of equations (including partials).

:rtype: list[tuple[expression, expression]]

)";
}

std::string var_ode_sys_vargs()
{
    return R"(The list of variational arguments.

:rtype: list[expression]

)";
}

std::string var_ode_sys_n_orig_sv()
{
    return R"(The number of original state variables.

:rtype: int

)";
}

std::string var_ode_sys_order()
{
    return R"(The differentitation order.

:rtype: int

)";
}

std::string var_args()
{
    return R"(Enum for selecting variational arguments.

.. versionadded:: 5.0.0

Values of this enum can be used in the constructor of :class:`~heyoka.var_ode_sys()` to select
the arguments with respect to which the variational equations will be formulated.

The enumerators can be combined with the logical OR ``|`` operator.

Examples:
  >>> from heyoka import var_args
  >>> va = var_args.vars | var_args.params # Select differentiation wrt all initial conditions and parameters
  >>> var_args.vars | var_args.params | var_args.time == var_args.all
  True

)";
}

std::string var_args_vars()
{
    return R"(Differentiate with respect to the initial conditions of all state variables.

)";
}

std::string var_args_params()
{
    return R"(Differentiate with respect to all runtime parameters.

)";
}

std::string var_args_time()
{
    return R"(Differentiate with respect to the initial integration time.

)";
}

std::string var_args_all()
{
    return R"(Differentiate with respect to the initial conditions of all state variables, all runtime parameters and the initial integration time.

)";
}

std::string fixed_centres()
{
    return R"(fixed_centres(Gconst: expression = 1., masses:  list[expression] = [], positions: typing.Iterable = numpy.empty((0, 3), dtype=float)) -> list[tuple[expression, expression]]

Produces the expressions for the dynamics in a fixed-centres problem.

In the fixed-centres problem, a test particle moves in the Newtonian gravitational field generated
by a number of massive particles whose positions are fixed in space. The test particle's Cartesian position and
velocity are represented by the variables ``[x, y, z]`` and ``[vx, vy, vz]`` respectively.

Several checks are run on the input arguments:

- *positions* must be (convertible into) an ``N x 3`` array, with each row containing
  the Cartesian position vector of a mass,
- the number of elements in *masses* must be equal to the number of three-dimensional
  position vectors in *positions*.

:param Gconst: the gravitational constant.
:param masses: the list of mass values (one for each particle).
:param positions: the positions of the particles.

:returns: the dynamics of the Newtonian fixed centres problem.

:raises ValueError: if one or more input arguments are malformed, as explained above.

)";
}

std::string pendulum()
{
    return R"(pendulum(gconst: expression = 1., length: expression = 1.) -> list[tuple[expression, expression]]

Produces the expression for the dynamics of the simple pendulum.

The gravitational constant is *gconst*, while the length of the pendulum is *length*.
In the return value, the angle with respect to the downwards vertical is represented by
the state variable ``x``, while its time derivative is represented by the state
variable ``v``.

:param gconst: the gravitational constant.
:param length: the length of the pendulum.

:returns: the dynamics of the simple pendulum.

Examples:
  >>> from heyoka import model
  >>> model.pendulum()
  [(x, v), (v, -sin(x))]

)";
}

std::string sgp4()
{
    return R"(sgp4(inputs: list[expression] | None = None) -> list[expression]

Produces the expressions for the SGP4 propagator.

.. versionadded:: 5.1.0

.. versionadded:: 7.0.0

   This function now optionally accepts a list of input expressions.

.. note::

   This is a low-level function for advanced use cases. If you are looking for a fast,
   high-level SGP4 propagator supporting parallel and batched operations, please refer
   to :func:`~heyoka.model.sgp4_propagator()`.

SGP4 is a widely-used analytical propagator for the dynamics of Earth-orbiting satellites,
described in detail in the `spacetrack report #3 <https://celestrak.org/NORAD/documentation/spacetrk.pdf>`__
(see also the `update from 2006 <https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev3.pdf>`__,
on which this implementation is based).

SGP4 takes in input a general perturbations element set (GPE), for instance in the form of
a `two-line element set (aka TLE) <https://en.wikipedia.org/wiki/Two-line_element_set>`__, and
a time delta relative to the epoch in the GPE. It returns the Cartesian state vector
(position and velocity) of the spacecraft at the specified time in the True Equator Mean Equinox
(TEME) reference frame.

If *inputs* is provided and it has nonzero length, it must be a list of 8 expressions,
which must represent, respectively:

- ``n0``: the mean motion from the GPE (in [rad / min]),
- ``e0``: the eccentricity from the GPE,
- ``i0``: the inclination from the GPE (in [rad]),
- ``node0``: the right ascension of the ascending node from the GPE (in [rad]),
- ``omega0``: the argument of perigee from the GPE (in [rad]),
- ``m0``: the mean anomaly from the GPE (in [rad]),
- ``bstar``: the `BSTAR <https://en.wikipedia.org/wiki/BSTAR>`__ drag term from
  the GPE (in the same unit as given in the GPE),
- ``tsince``: the time elapsed from the GPE epoch (in [min]).

If *inputs* is not provided or it has a length of zero, 8 variable expressions named
``["n0", "e0", "i0", "node0", "omega0", "m0", "bstar", "tsince"]`` will be used as inputs.

This function will return 7 expressions: the first 6 correspond to the Cartesian state (position and
velocity respectively) of the spacecraft according to the SGP4 algorithm, while the last expression
represents an error code which, if nonzero, signals the occurrence of an error in the SGP4 propagation
routine. The Cartesian coordinates ``x, y, z`` of the satellite are returned in [km], while the velocities
``vx, vy, vz`` are returned in [km / s]. When nonzero, the error code can assume the following values:

- 1: the mean eccentricity is outside the range [0.0, 1.0],
- 2: the mean mean motion is less than zero,
- 3: the perturbed eccentricity is outside the range [0.0, 1.0],
- 4: the semilatus rectum is less than zero,
- 5: the satellite was underground (**NOTE**: this error code is no longer in use),
- 6: the satellite has decayed.

.. note::

   Currently this function does not implement the deep-space part of the
   SGP4 algorithm (aka SDP4), and consequently it should not be used with satellites
   whose orbital period is greater than 225 minutes. You can use the
   :py:func:`~heyoka.model.gpe_is_deep_space()` function to check whether
   a GPE is deep-space or not.

.. seealso::

   `NORAD Two-Line Element Set Format <https://celestrak.org/NORAD/documentation/tle-fmt.php>`_

:returns: a list of 7 expressions representing the Cartesian state vector of an Earth-orbiting
   satellite and an error code, as functions of the *inputs*.

:raises ValueError: if the list of inputs has a length other than 0 or 8.

)";
}

std::string gpe_is_deep_space()
{
    return R"(gpe_is_deep_space(n0: float, e0: float, i0: float) -> bool

Check whether a GPE is deep-space.

.. versionadded:: 7.0.0

This function takes in input the mean motion, eccentricity and inclination from a general
perturbations element set (GPE), and determines whether or not the propagation of the GPE
requires the SDP4 deep-space algorithm.

:param n0: the mean motion from the GPE (in [rad / min]).
:param e0: the eccentricity from the GPE.
:param i0: the inclination from the GPE (in [rad]).

:returns: a flag signalling whether or not the input GPE requires deep-space propagation.

)";
}

std::string sgp4_propagator(const std::string &p)
{
    return fmt::format(R"(SGP4 propagator ({} precision).

.. versionadded:: 5.1.0

.. note::

   A :ref:`tutorial <tut_sgp4_propagator>` explaining the use of this class
   is available.

.. note::

   Consider using the factory function :py:func:`~heyoka.model.sgp4_propagator()`, rather
   that constructing instances of this class by hand.

)",
                       p);
}

std::string sgp4_propagator_init(const std::string &tp)
{
    return fmt::format(R"(__init__(self, sat_list: list | numpy.ndarray[{0}], diff_order: int = 0, **kwargs)

Constructor.

.. versionadded:: 7.0.0

   This function now also accepts *sat_list* as a NumPy array.

The constructor will initialise the propagator from *sat_list*, which must be either a list
of general perturbations element sets (GPEs) represented as ``Satrec`` objects from the
`sgp4 Python module <https://pypi.org/project/sgp4/>`__, or a 2D array.

In the former case, the GPE data is taken directly from the ``Satrec`` objects.
In the latter case, *sat_list* is expected to be a 9 x ``n`` C-style contiguous
array, where ``n`` is the total number of satellites and the rows contain the following
GPE data:

0. the mean motion (in [rad / min]),
1. the eccentricity,
2. the inclination (in [rad]),
3. the right ascension of the ascending node (in [rad]),
4. the argument of perigee (in [rad]),
5. the mean anomaly (in [rad]),
6. the `BSTAR <https://en.wikipedia.org/wiki/BSTAR>`__ drag term (in the same unit as given in the GPE),
7. the reference epoch (as a Julian date),
8. a fractional correction to the epoch (in Julian days).

Epochs must be provided in the UTC scale of time, following the convention that days in which
leap seconds are added/removed are 1 second longer/shorter than one SI day.

The *diff_order* argument indicates the desired differentiation order. If equal to 0, then
derivatives are disabled.

*kwargs* can optionally contain keyword arguments from the :ref:`api_common_kwargs_llvm` set
and the :ref:`api_common_kwargs_cfunc` set.

:param sat_list: the GPE data.
:param diff_order: the derivatives order.

:raises ImportError: if *sat_list* is a list and the sgp4 Python module is not available.
:raises TypeError: if *sat_list* is a list and one or more of its elements is not a ``Satrec`` object.
:raises ValueError: if *sat_list* is an invalid array, as explained above.

)",
                       tp);
}

std::string sgp4_propagator_jdtype(const std::string &tp)
{
    return fmt::format(R"(Data type representing Julian dates with a fractional component.

This is a :ref:`structured NumPy datatype<numpy:defining-structured-types>` consisting of
two fields of type :py:class:`{}`, the first one called ``jd`` and representing a Julian date,
the second one called ``frac`` representing a fractional correction to ``jd`` (so that the full
Julian date is ``jd + frac``).

:rtype: numpy.dtype

)",
                       tp);
}

std::string sgp4_propagator_nsats()
{
    return R"(The total number of satellites.

:rtype: int

)";
}

std::string sgp4_propagator_nouts()
{
    return R"(The total number of outputs, including the derivatives.

:rtype: int

)";
}

std::string sgp4_propagator_diff_args()
{
    return R"(The list of differentiation arguments.

.. note::

   This property is available only if derivatives were requested on construction.

:rtype: list[expression]

)";
}

std::string sgp4_propagator_diff_order()
{
    return R"(The differentiation order.

:rtype: int

)";
}

std::string sgp4_propagator_sat_data(const std::string &suffix, const std::string &tp)
{
    return fmt::format(R"(The GPE data.

A 9 x :py:attr:`~heyoka.model.sgp4_propagator_{}.nsats` array containing the general
perturbations element sets (GPEs) of each satellite.

The rows contain the following quantities:

0. the mean motion (in [rad / min]),
1. the eccentricity,
2. the inclination (in [rad]),
3. the right ascension of the ascending node (in [rad]),
4. the argument of perigee (in [rad]),
5. the mean anomaly (in [rad]),
6. the `BSTAR <https://en.wikipedia.org/wiki/BSTAR>`__ drag term (in the same unit as given in the GPE),
7. the reference epoch (as a Julian date),
8. a fractional correction to the epoch (in Julian days).

:rtype: numpy.ndarray[{}]

)",
                       suffix, tp);
}

std::string sgp4_propagator_get_dslice()
{
    return R"(get_dslice(self, order: int, component: int | None = None) -> slice

Fetch a slice of derivatives.

.. note::

   This method is available only if derivatives were requested on construction.

This method will return a slice representing the range of indices containing the derivatives
of order *order* in the result of a propagation. If *component* is :py:data:`None`, then the output
range encompasses the derivatives of all the propagated quantities. Otherwise, the output range
includes only the derivatives of the *component*-th propagated quantity.

:param order: the differentiation order.
:param component: the component to consider.

:returns: a range of indices into the output of a propagation containing the requested derivatives.

)";
}

std::string sgp4_propagator_get_mindex(const std::string &suffix)
{
    return fmt::format(R"(get_mindex(self, i: int) -> list[int]

Fetch a differentiation multiindex.

.. note::

   This method is available only if derivatives were requested on construction.

This method will return the differentiation multiindex corresponding to the *i*-th row
of the propagation output of a differentiable SGP4 propagator.

The multiindex begins with the component index (that is, the index of the output quantity
whose derivatives have been computed). The remaining indices are the differentiation
orders with respect to the quantities listed in :py:attr:`~heyoka.model.sgp4_propagator_{}.diff_args`.

For instance, if the return value is ``[2, 0, 1, 0, 0, 0, 0, 0]``, the multiindex refers
to the first-order derivative of the output quantity at index 2 (i.e., the Cartesian :math:`z` coordinate)
with respect to the second differentiation argument.

:param i: the input index.

:returns: a differentiation multiindex.

)",
                       suffix);
}

std::string sgp4_propagator_call(const std::string &suffix, const std::string &tp)
{
    return fmt::format(
        R"(__call__(self, times: numpy.ndarray, out: numpy.ndarray[{1}] | None = None) -> numpy.ndarray[{1}]

Propagation.

The call operator will propagate the states of all the satellites (and, if requested, their derivatives)
up to the specified *times*.

The *times* array can contain either floating-point values (of type :py:class:`{1}`),
or Julian dates (represented via the :py:attr:`~heyoka.model.sgp4_propagator_{0}.jdtype` type). In the former case,
the input *times* will be interpreted as minutes elapsed since the GPE reference epochs (which in general differ
from satellite to satellite). In the latter case, the states will be propagated up to the specified Julian dates,
which must be provided in the UTC scale of time.

*times* can be either a one-dimensional array, or a two-dimensional one. In the former case (scalar propagation),
its length must be exactly :py:attr:`~heyoka.model.sgp4_propagator_{0}.nsats` (i.e., one time/date per satellite).
In the latter case (batch-mode propagation), the number of columns must be exactly
:py:attr:`~heyoka.model.sgp4_propagator_{0}.nsats`, while the number of rows represents how many propagations
per satellite will be performed.

The return value is either a two-dimensional (scalar propagation) or three-dimensional (batch-mode propagation)
array. In the former case, the number of rows will be equal to :py:attr:`~heyoka.model.sgp4_propagator_{0}.nouts`, while the
number of columns will be equal to :py:attr:`~heyoka.model.sgp4_propagator_{0}.nsats`. In the latter case,
the first dimension will be equal to the number of propagations performed per satellite, the second dimension
will be equal to :py:attr:`~heyoka.model.sgp4_propagator_{0}.nouts`, and the third dimension will be equal to
:py:attr:`~heyoka.model.sgp4_propagator_{0}.nsats`.

If an *out* array with the correct data type and shape is provided, it will be used as the return
value. Otherwise, a new array will be returned.

All input arguments must be C-style contiguous arrays, with no memory overlap between *times* and *out*.

:param times: the propagation times/dates.
:param out: the output array.

:returns: the result of the propagation.

:raises ValueError: if an invalid input array is detected, as explained above. 

)",
        suffix, tp);
}

std::string sgp4_propagator_replace_sat_data(const std::string &suffix, const std::string &tp)
{
    return fmt::format(R"(replace_sat_data(self, sat_list: list | numpy.ndarray[{1}]) -> None

Replace the GPE data.

.. versionadded:: 7.0.0

   This function now also accepts *sat_list* as a NumPy array.

This method will replace the GPE data in the propagator with the data from *sat_list*.
*sat_list* must be either a list of GPEs represented as ``Satrec`` objects
from the `sgp4 Python module <https://pypi.org/project/sgp4/>`__, or a 2D NumPy array.
See the documentation of the :py:meth:`constructor <heyoka.model.sgp4_propagator_{0}.__init__>`
for more information on *sat_list*.

The number of satellites in *sat_list* must be equal to the number of satellites
in the propagator - that is, it is not possible to change the total number of satellites
in the propagator via this method.

:param sat_list: the new GPE data.

:raises TypeError: if *sat_list* is a list and one or more of its elements is not a ``Satrec`` object.
:raises ValueError: if *sat_list* is an invalid array or if the number of satellites in *sat_list* differs
   from the number of satellites in the propagator.

)",
                       suffix, tp);
}

std::string code_model()
{
    return R"(Enum for selecting the LLVM code model used during JIT compilation.

.. versionadded:: 6.0.0

The default code model used by heyoka.py is ``small``. Large computational graphs may require the
use of the ``large`` code model. Note that currently only the ``small`` and ``large``
code models are supported on all platforms.

)";
}

std::string code_model_tiny()
{
    return R"(Tiny code model (corresponds to ``llvm::CodeModel::Model::Tiny``).

)";
}

std::string code_model_small()
{
    return R"(Small code model (corresponds to ``llvm::CodeModel::Model::Small``).

)";
}

std::string code_model_kernel()
{
    return R"(Kernel code model (corresponds to ``llvm::CodeModel::Model::Kernel``).

)";
}

std::string code_model_medium()
{
    return R"(Medium code model (corresponds to ``llvm::CodeModel::Model::Medium``).

)";
}

std::string code_model_large()
{
    return R"(Large code model (corresponds to ``llvm::CodeModel::Model::Large``).

)";
}

std::string delta_tdb_tt()
{
    return R"(delta_tdb_tt(time_expr: expression = heyoka.time) -> expression

Difference between TDB and TT.

.. versionadded:: 7.3.0

This function will return the difference (in seconds) between
`barycentric dynamical time (TDB) <https://en.wikipedia.org/wiki/Barycentric_Dynamical_Time>`__
and `terrestrial time (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__ as a function
of the input TDB time (expressed in seconds elapsed from the epoch of J2000).

This function is implemented following the simplified approach described in the
`NASA SPICE toolkit <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/time.html#The%20Relationship%20between%20TT%20and%20TDB>`__.
Specifically, we assume a Keplerian orbit for the motion of the Earth, so that the TDB-TT difference
is a periodic function. This approach is accurate to approximately 0.000030 seconds.

.. note::

   Although this function nominally takes in input a TDB time, the corresponding TT time can be used in its place
   with no practical effects on the accuracy of the computation.

:param time_expr: the number of TDB seconds elapsed from the epoch of J2000.

:returns: the difference (in seconds) between TDB and TT.

)";
}

std::string eop_data()
{
    return R"(EOP data class.

.. versionadded:: 7.3.0

This class is used to manage and access Earth orientation parameters data.

.. note::

   A :ref:`tutorial <tut_eop_data>` illustrating the use of this class is available.

)";
}

std::string eop_data_init()
{
    return R"(__init__(self)

Default constructor.

The default constructor initialises the EOP data with a builtin copy of the ``finals2000A.all``
rapid data file from the `IERS datacenter <https://datacenter.iers.org/eop.php>`__.

Note that the builtin EOP data is likely to be outdated. You can use functions such as
:py:func:`~heyoka.eop_data.fetch_latest_iers_rapid()` to fetch up-to-date data from the internet.

)";
}

std::string eop_data_table()
{
    return R"(EOP data table.

This is a :ref:`structured NumPy array<numpy:defining-structured-types>` containing the raw EOP data.
The dtype of the returned array is :py:attr:`~heyoka.eop_data_row`.

:rtype: numpy.ndarray

)";
}

std::string eop_data_timestamp()
{
    return R"(EOP data timestamp.

A timestamp in string format which can be used to disambiguate between different versions of
the same dataset.

The timestamp is inferred from the timestamp of the files on the remote data servers.

:rtype: str

)";
}

std::string eop_data_identifier()
{
    return R"(EOP data identifier.

A string uniquely identifying the source of EOP data.

:rtype: str

)";
}

std::string eop_data_fetch_latest_iers_rapid()
{
    return R"(fetch_latest_iers_rapid(filename: str = 'finals2000A.all') -> eop_data

Fetch the latest IERS EOP rapid data.

This function will download from the `IERS datacenter <https://datacenter.iers.org/eop.php>`__
one the latest EOP rapid data files, from which it will construct and return an :py:class:`~heyoka.eop_data` instance.

The *filename* argument specifies which EOP data file will be downloaded, and it can be one of:

* ``"finals2000A.all"``,
* ``"finals2000A.daily"``,
* ``"finals2000A.daily.extended"``,
* ``"finals2000A.data"``.

These datafiles are updated frequently and they contain predictions for the future. They differ from each other mainly
in the timespans they provide data for. For instance, ``finals2000A.all`` contains several decades worth of data,
while ``finals2000A.daily`` contains only the most recent data.

Please refer to the documentation on the `IERS datacenter website <https://datacenter.iers.org/eop.php>`__
for more information about the content of these files.

.. note::

   This function will release the `global interpreter lock (GIL) <https://docs.python.org/3/glossary.html#term-global-interpreter-lock>`__
   while downloading.

:param filename: the file to be downloaded.

:returns: an :py:class:`~heyoka.eop_data` instance constructed from the remote file.

:raises ValueError: if *filename* is invalid.

)";
}

std::string eop_data_fetch_latest_iers_long_term()
{
    return R"(fetch_latest_iers_long_term() -> eop_data

Fetch the latest IERS EOP long-term data.

This function will download from the `IERS datacenter <https://datacenter.iers.org/eop.php>`__
the latest EOP long-term datafile, from which it will construct and return an :py:class:`~heyoka.eop_data` instance.

The file downloaded by this function is ``eopc04_20.1962-now.csv``, which contains the IAU2000A EOP data
from 1962 up to (roughly) the present time. Note that long-term data does **not** contain predictions for the future.

.. note::

   This function will release the `global interpreter lock (GIL) <https://docs.python.org/3/glossary.html#term-global-interpreter-lock>`__
   while downloading.

:returns: an :py:class:`~heyoka.eop_data` instance constructed from the remote file.

)";
}

std::string rot_fk5j2000_icrs()
{
    return R"(rot_fk5j2000_icrs(xyz: typing.Iterable[expression]) -> list[expression]

Rotation from FK5 to ICRS.

.. versionadded:: 7.3.0

This function will rotate the input Cartesian coordinates *xyz* from the axes of the FK5 frame at J2000
to the axes of the `ICRS <https://en.wikipedia.org/wiki/International_Celestial_Reference_System_and_its_realizations>`__
frame.

:param xyz: the input Cartesian coordinates.

:returns: the rotated Cartesian coordinates.

)";
}

std::string rot_icrs_fk5j2000()
{
    return R"(rot_icrs_fk5j2000(xyz: typing.Iterable[expression]) -> list[expression]

Rotation from ICRS to FK5.

.. versionadded:: 7.3.0

This function will rotate the input Cartesian coordinates *xyz* from the axes of the
`ICRS <https://en.wikipedia.org/wiki/International_Celestial_Reference_System_and_its_realizations>`__ frame to the 
axes of the FK5 frame at J2000.

:param xyz: the input Cartesian coordinates.

:returns: the rotated Cartesian coordinates.

)";
}

std::string rot_itrs_icrs(double thresh)
{
    return fmt::format(
        R"(rot_itrs_icrs(xyz: typing.Iterable[expression], time_expr: expression = heyoka.time, thresh: float = {}, eop_data: eop_data = eop_data()) -> list[expression]

Rotation from ITRS to ICRS.

.. versionadded:: 7.3.0

This function will rotate the input Cartesian coordinates *xyz* from the axes of the
`ITRS <https://en.wikipedia.org/wiki/International_Terrestrial_Reference_System_and_Frame>`__ frame to the axes of the
`ICRS <https://en.wikipedia.org/wiki/International_Celestial_Reference_System_and_its_realizations>`__ frame.

While the orientation of the ICRS is fixed in inertial space, the orientation of the ITRS is fixed relative to the Earth,
and thus the rotation depends on the input time *time_expr* (which is expected to represent the number of Julian centuries elapsed
since the epoch of J2000 in the `terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__).

The rotation also requires in input an Earth orientation parameters dataset *eop_data* and a truncation threshold *thresh*
for the IAU2000/2006 precession-nutation theory. Please see the documentation of :py:class:`~heyoka.eop_data` and
:py:func:`~heyoka.model.iau2006()` for more information.

:param xyz: the input Cartesian coordinates.
:param time_expr: the input time expression.
:param thresh: the truncation threshold for the IAU2000/2006 theory.
:param eop_data: the EOP data to be used for the computation.

:returns: the rotated Cartesian coordinates.

)",
        thresh);
}

std::string rot_icrs_itrs(double thresh)
{
    return fmt::format(
        R"(rot_icrs_itrs(xyz: typing.Iterable[expression], time_expr: expression = heyoka.time, thresh: float = {}, eop_data: eop_data = eop_data()) -> list[expression]

Rotation from ICRS to ITRS.

.. versionadded:: 7.3.0

This function will rotate the input Cartesian coordinates *xyz* from the axes of the
`ICRS <https://en.wikipedia.org/wiki/International_Celestial_Reference_System_and_its_realizations>`__ frame to the axes of the
`ITRS <https://en.wikipedia.org/wiki/International_Terrestrial_Reference_System_and_Frame>`__ frame.

While the orientation of the ICRS is fixed in inertial space, the orientation of the ITRS is fixed relative to the Earth,
and thus the rotation depends on the input time *time_expr* (which is expected to represent the number of Julian centuries elapsed
since the epoch of J2000 in the `terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__).

The rotation also requires in input an Earth orientation parameters dataset *eop_data* and a truncation threshold *thresh*
for the IAU2000/2006 precession-nutation theory. Please see the documentation of :py:class:`~heyoka.eop_data` and
:py:func:`~heyoka.model.iau2006()` for more information.

:param xyz: the input Cartesian coordinates.
:param time_expr: the input time expression.
:param thresh: the truncation threshold for the IAU2000/2006 theory.
:param eop_data: the EOP data to be used for the computation.

:returns: the rotated Cartesian coordinates.

)",
        thresh);
}

std::string era()
{
    return R"(era(time_expr: expression = heyoka.time, eop_data: eop_data = eop_data()) -> expression

Earth rotation angle.

.. versionadded:: 7.3.0

This function will return an expression representing the `Earth rotation angle (ERA) <https://en.wikipedia.org/wiki/Sidereal_time#ERA>`__
as a function of the input time expression *time_expr*. *time_expr* is expected to represent the number of Julian centuries elapsed
since the epoch of J2000 in the `terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__. *eop_data* is
the Earth orientation parameters dataset to be used for the computation.

The ERA is modelled as a piecewise linear function of time, where the switch points are given by the dates in *eop_data*. Evaluation
of the ERA outside the dates range of *eop_data* will produce a value of ``NaN``.

The ERA is returned in radians, reduced to the :math:`\left[0, 2\pi\right]` range.

:param time_expr: the input time expression.
:param eop_data: the EOP data to be used for the computation.

:returns: an expression for the ERA as a function of time.

)";
}

std::string erap()
{
    return R"(erap(time_expr: expression = heyoka.time, eop_data: eop_data = eop_data()) -> expression

Derivative of the Earth rotation angle.

.. versionadded:: 7.3.0

This function will return an expression representing the first-order derivative of :py:func:`~heyoka.model.era()`
as a function of the input time expression *time_expr*. *time_expr* is expected to represent the number of Julian centuries elapsed
since the epoch of J2000 in the `terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__. *eop_data* is
the Earth orientation parameters dataset to be used for the computation.

The derivative of the Earth rotation angle (ERA) is modelled as a piecewise constant function of time, where the switch
points are given by the dates in *eop_data*. Evaluation outside the dates range of *eop_data* will produce a value of ``NaN``.

The derivative of the ERA is returned in radians per Julian century (TT).

:param time_expr: the input time expression.
:param eop_data: the EOP data to be used for the computation.

:returns: an expression for the derivative of the ERA as a function of time.

)";
}

namespace detail
{

namespace
{

std::string pm_x_y_impl(const std::string &xory)
{
    return fmt::format(R"(pm_{0}(time_expr: expression = heyoka.time, eop_data: eop_data = eop_data()) -> expression

Polar motion ({0} component).

.. versionadded:: 7.3.0

This function will return an expression representing the :math:`{0}` coordinate of the `polar motion <https://en.wikipedia.org/wiki/Polar_motion>`__
as a function of the input time expression *time_expr*. *time_expr* is expected to represent the number of Julian centuries elapsed
since the epoch of J2000 in the `terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__. *eop_data* is
the Earth orientation parameters dataset to be used for the computation.

This quantity is modelled as a piecewise linear function of time, where the switch points are given by the dates in *eop_data*. Evaluation
outside the dates range of *eop_data* will produce a value of ``NaN``.

The return value is expressed in radians.

:param time_expr: the input time expression.
:param eop_data: the EOP data to be used for the computation.

:returns: an expression for the {0} component of the polar motion as a function of time.

)",
                       xory);
}

std::string pm_xp_yp_impl(const std::string &xory)
{
    return fmt::format(R"(pm_{0}p(time_expr: expression = heyoka.time, eop_data: eop_data = eop_data()) -> expression

Derivative of the polar motion ({0} component).

.. versionadded:: 7.3.0

This function will return an expression representing the first-order derivative of the :math:`{0}` coordinate of the
`polar motion <https://en.wikipedia.org/wiki/Polar_motion>`__ as a function of the input time expression *time_expr*.
*time_expr* is expected to represent the number of Julian centuries elapsed
since the epoch of J2000 in the `terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__. *eop_data* is
the Earth orientation parameters dataset to be used for the computation.

This quantity is modelled as a piecewise constant function of time, where the switch
points are given by the dates in *eop_data*. Evaluation outside the dates range of *eop_data* will produce a value of ``NaN``.

The return value is expressed in radians per Julian century (TT).

:param time_expr: the input time expression.
:param eop_data: the EOP data to be used for the computation.

:returns: an expression for the derivative of the {0} component of the polar motion as a function of time.

)",
                       xory);
}

} // namespace

} // namespace detail

std::string pm_x()
{
    return detail::pm_x_y_impl("x");
}

std::string pm_xp()
{
    return detail::pm_xp_yp_impl("x");
}

std::string pm_y()
{
    return detail::pm_x_y_impl("y");
}

std::string pm_yp()
{
    return detail::pm_xp_yp_impl("y");
}

namespace detail
{

namespace
{

std::string dX_dY_impl(const std::string &XorY)
{
    return fmt::format(R"(d{0}(time_expr: expression = heyoka.time, eop_data: eop_data = eop_data()) -> expression

Correction to the Earth's precession/nutation model ({1} component).

.. versionadded:: 7.3.0

This function will return an expression representing the :math:`{1}` coordinate of the correction to the Earth's
precession-nutation model (IAU 2000/2006) as a function of the input time expression *time_expr*. *time_expr* is
expected to represent the number of Julian centuries elapsed since the epoch of J2000 in the
`terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__. *eop_data* is
the Earth orientation parameters dataset to be used for the computation.

This quantity is modelled as a piecewise linear function of time, where the switch points are given by the dates in *eop_data*. Evaluation
outside the dates range of *eop_data* will produce a value of ``NaN``.

The return value is expressed in radians.

:param time_expr: the input time expression.
:param eop_data: the EOP data to be used for the computation.

:returns: an expression for the {1} component of the correction to the Earth's precession/nutation model.

)",
                       XorY, boost::algorithm::to_lower_copy(XorY));
}

std::string dXp_dYp_impl(const std::string &XorY)
{
    return fmt::format(R"(d{0}p(time_expr: expression = heyoka.time, eop_data: eop_data = eop_data()) -> expression

Derivative of the correction to the Earth's precession/nutation model ({1} component).

.. versionadded:: 7.3.0

This function will return an expression representing the first-order derivative of the :math:`{1}` coordinate of
the correction to the Earth's precession-nutation model (IAU 2000/2006) as a function of the input time expression *time_expr*.
*time_expr* is expected to represent the number of Julian centuries elapsed
since the epoch of J2000 in the `terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__. *eop_data* is
the Earth orientation parameters dataset to be used for the computation.

This quantity is modelled as a piecewise constant function of time, where the switch
points are given by the dates in *eop_data*. Evaluation outside the dates range of *eop_data* will produce a value of ``NaN``.

The return value is expressed in radians per Julian century (TT).

:param time_expr: the input time expression.
:param eop_data: the EOP data to be used for the computation.

:returns: an expression for the derivative of the {1} component of the correction to the Earth's precession/nutation model.

)",
                       XorY, boost::algorithm::to_lower_copy(XorY));
}

} // namespace

} // namespace detail

std::string dX()
{
    return detail::dX_dY_impl("X");
}

std::string dXp()
{
    return detail::dXp_dYp_impl("X");
}

std::string dY()
{
    return detail::dX_dY_impl("Y");
}

std::string dYp()
{
    return detail::dXp_dYp_impl("Y");
}

std::string iau2006(double thresh)
{
    return fmt::format(R"(iau2006(time_expr: expression = heyoka.time, thresh: float = {}) -> list[expression]

IAU2000/2006 precession-nutation theory.

.. versionadded:: 7.3.0

This function will return a set of three expressions representing the :math:`X`, :math:`Y` and :math:`s` angles
from the IAU2000/2006 precession-nutation theory as a function of the input time expression *time_expr*.

:math:`X` and :math:`Y` are the celestial intermediate pole (CIP) coordinates in the International Celestial
Reference System (ICRS). They describe the position of the CIP relative to the Geocentric Celestial Reference Frame
(GCRF), which is aligned with the ICRS. :math:`s` is the Celestial Intermediate Origin (CIO) locator. It represents
an additional rotation in the transformation between celestial and terrestrial
reference frames. For more information about these quantities, please consult standard references such as Vallado's
"Fundamentals of Astrodynamics" (Chapter 3) and Chapter 5 of the
`IERS conventions <https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36.pdf?__blob=publicationFile&v=1>`__.

The angles are returned in radians. *time_expr* is expected to represent the number of Julian centuries elapsed since the epoch
of J2000 in the `terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__.

*thresh* represents the truncation threshold: trigonometric terms in the theory whose coefficients are less than *thresh* in magnitude
will be discarded. In order to formulate the full theory without truncation, use a *thresh* value of zero.

.. note::

   A :ref:`tutorial <tut_iau2006>` is available showcasing the use of this function, including accuracy comparisons for
   several values of the *thresh* argument.

:param time_expr: the input time expression.
:param thresh: the truncation threshold for the coefficients of the trigonometric series (in arcseconds).

:returns: expressions for the :math:`X`, :math:`Y` and :math:`s` angles from the IAU2000/2006 precession-nutation theory.

:raises ValueError: it *thresh* is negative or non-finite.

)",
                       thresh);
}

std::string egm2008_pot(double def_mu, double def_a)
{
    return fmt::format(
        R"(egm2008_pot(xyz: typing.Iterable[expression], n: int, m: int, mu: expression = {}, a: expression = {}) -> expression

Geopotential (EGM2008).

.. versionadded:: 7.3.0

This function will return the value of the Earth's gravitational potential at the input Cartesian position
*xyz* according to the `EGM2008 model <https://en.wikipedia.org/wiki/Earth_Gravitational_Model#EGM2008>`__.
*xyz* is expected to represent the position vector with respect to the Earth-centred Earth-fixed
`WGS84 frame <https://en.wikipedia.org/wiki/World_Geodetic_System>`__ (which itself can be considered coincident
with the `ITRF <https://en.wikipedia.org/wiki/International_Terrestrial_Reference_System_and_Frame>`__).

*n* and *m* are, respectively, the maximum harmonic degree and order to be considered in the computation. Higher degrees
and orders will produce more accurate values, at the cost of increased computational complexity.

*mu* and *a* are, respectively, the gravitational parameter and reference Earth radius to be used in the computation.
Both are expected to be provided in units consistent with each other and with *xyz*. The default values (in SI units) are
those specified in the documentation of the EGM2008 model.

.. note::

   Currently this function implements a version of EGM2008 in which the maximum degree and order
   are capped below those of the full model.

:param xyz: the position at which the potential will be evaluated.
:param n: the maximum harmonic degree to be used in the computation.
:param m: the maximum harmonic order to be used in the computation.
:param mu: the Earth's gravitational parameter.
:param a: the reference Earth radius.

:returns: an expression for the geopotential at the position *xyz*.

:raises ValueError: it *m* > *n* or if *n* is larger than an implementation-defined limit.

)",
        def_mu, def_a);
}

std::string egm2008_acc(double def_mu, double def_a)
{
    return fmt::format(
        R"(egm2008_acc(xyz: typing.Iterable[expression], n: int, m: int, mu: expression = {}, a: expression = {}) -> list[expression]

Gravitational acceleration (EGM2008).

.. versionadded:: 7.3.0

This function will return the value of the Earth's gravitational acceleration at the input Cartesian position
*xyz* according to the `EGM2008 model <https://en.wikipedia.org/wiki/Earth_Gravitational_Model#EGM2008>`__.
*xyz* is expected to represent the position vector with respect to the Earth-centred Earth-fixed
`WGS84 frame <https://en.wikipedia.org/wiki/World_Geodetic_System>`__ (which itself can be considered coincident
with the `ITRF <https://en.wikipedia.org/wiki/International_Terrestrial_Reference_System_and_Frame>`__).

*n* and *m* are, respectively, the maximum harmonic degree and order to be considered in the computation. Higher degrees
and orders will produce more accurate values, at the cost of increased computational complexity.

*mu* and *a* are, respectively, the gravitational parameter and reference Earth radius to be used in the computation.
Both are expected to be provided in units consistent with each other and with *xyz*. The default values (in SI units) are
those specified in the documentation of the EGM2008 model.

.. note::

   Currently this function implements a version of EGM2008 in which the maximum degree and order
   are capped below those of the full model.

:param xyz: the position at which the acceleration will be evaluated.
:param n: the maximum harmonic degree to be used in the computation.
:param m: the maximum harmonic order to be used in the computation.
:param mu: the Earth's gravitational parameter.
:param a: the reference Earth radius.

:returns: an expression for the Cartesian acceleration vector due to the geopotential at the position *xyz*.

:raises ValueError: it *m* > *n* or if *n* is larger than an implementation-defined limit.

)",
        def_mu, def_a);
}

std::string sw_data()
{
    return R"(Space weather data class.

.. versionadded:: 7.3.0

This class is used to manage and access space weather (SW) data.

.. note::

   A :ref:`tutorial <tut_sw_data>` illustrating the use of this class is available.

)";
}

std::string sw_data_init()
{
    return R"(__init__(self)

Default constructor.

The default constructor initialises the SW data with a builtin copy of the ``SW-All.csv``
data file from `celestrak <https://celestrak.org/SpaceData/>`__.

Note that the builtin SW data is likely to be outdated. You can use functions such as
:py:func:`~heyoka.sw_data.fetch_latest_celestrak()` to fetch up-to-date data from the internet.

)";
}

std::string sw_data_table()
{
    return R"(SW data table.

This is a :ref:`structured NumPy array<numpy:defining-structured-types>` containing the raw SW data.
The dtype of the returned array is :py:attr:`~heyoka.sw_data_row`.

:rtype: numpy.ndarray

)";
}

std::string sw_data_timestamp()
{
    return R"(SW data timestamp.

A timestamp in string format which can be used to disambiguate between different versions of
the same dataset.

The timestamp is inferred from the timestamp of the files on the remote data servers.

:rtype: str

)";
}

std::string sw_data_identifier()
{
    return R"(SW data identifier.

A string uniquely identifying the source of SW data.

:rtype: str

)";
}

std::string sw_data_fetch_latest_celestrak()
{
    return R"(fetch_latest_celestrak(long_term: bool = False) -> sw_data

Fetch the latest SW data from celestrak.

This function will download from `celestrak <https://celestrak.org/SpaceData/>`__
one the latest SW data files, from which it will construct and return an :py:class:`~heyoka.sw_data` instance.

The *long_term* argument indicates which SW data file will be donwloaded:

* if ``True``, then the full historical dataset from 1957 up to the present time will be
  downloaded, otherwise
* the dataset for the last 5 years will be downloaded.

Both datasets contain predictions for the near future.

Please refer to the documentation on the `celestrak website <https://celestrak.org/SpaceData/>`__
for more information about the content of these files.

.. note::

   This function will release the `global interpreter lock (GIL) <https://docs.python.org/3/glossary.html#term-global-interpreter-lock>`__
   while downloading.

:param long_term: flag selecting which file to be downloaded.

:returns: an :py:class:`~heyoka.sw_data` instance constructed from the remote file.

)";
}

std::string Ap_avg()
{
    return R"(Ap_avg(time_expr: expression = heyoka.time, sw_data: sw_data = sw_data()) -> expression

Average of the Ap indices.

.. versionadded:: 7.3.0

This function will return an expression representing the average of the 8 `Ap indices <https://en.wikipedia.org/wiki/K-index>`__
as a function of the input time expression *time_expr*. *time_expr* is
expected to represent the number of Julian centuries elapsed since the epoch of J2000 in the
`terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__. *sw_data* is
the space weather dataset to be used for the computation.

This quantity is modelled as a piecewise constant function of time, where the switch points are given by the dates in *sw_data*.
Evaluation outside the dates range of *sw_data* will produce a value of ``NaN``.

:param time_expr: the input time expression.
:param sw_data: the SW data to be used for the computation.

:returns: an expression representing the average of the Ap indices.

)";
}

std::string f107()
{
    return R"(f107(time_expr: expression = heyoka.time, sw_data: sw_data = sw_data()) -> expression

Observed 10.7-cm solar radio flux.

.. versionadded:: 7.3.0

This function will return an expression representing the observed 10.7-cm `solar radio flux <https://en.wikipedia.org/wiki/Solar_flux_unit>`__
as a function of the input time expression *time_expr*. *time_expr* is
expected to represent the number of Julian centuries elapsed since the epoch of J2000 in the
`terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__. *sw_data* is
the space weather dataset to be used for the computation.

This quantity is modelled as a piecewise constant function of time, where the switch points are given by the dates in *sw_data*.
Evaluation outside the dates range of *sw_data* will produce a value of ``NaN``.

:param time_expr: the input time expression.
:param sw_data: the SW data to be used for the computation.

:returns: an expression representing the observed 10.7-cm solar radio flux.

)";
}

std::string f107a_center81()
{
    return R"(f107a_center81(time_expr: expression = heyoka.time, sw_data: sw_data = sw_data()) -> expression

Average of the 10.7-cm solar radio flux.

.. versionadded:: 7.3.0

This function will return an expression representing the 81-day arithmetic average of
the observed `solar radio flux <https://en.wikipedia.org/wiki/Solar_flux_unit>`__ centred
on the input time expression *time_expr*. *time_expr* is
expected to represent the number of Julian centuries elapsed since the epoch of J2000 in the
`terrestrial time scale (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__. *sw_data* is
the space weather dataset to be used for the computation.

This quantity is modelled as a piecewise constant function of time, where the switch points are given by the dates in *sw_data*.
Evaluation outside the dates range of *sw_data* will produce a value of ``NaN``.

:param time_expr: the input time expression.
:param sw_data: the SW data to be used for the computation.

:returns: an expression representing the 81-day arithmetic average of the observed 10.7-cm solar radio flux.

)";
}

} // namespace heyoka_py::docstrings
