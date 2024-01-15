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
