.. _changelog:

Changelog
=========

7.3.2 (2025-04-22)
------------------

Fix
~~~

- Fix uploading of wheels to PyPI.

7.3.0 (2025-04-21)
------------------

New
~~~

- New models: time conversions, frame transformations, Earth orientation
  parameters (including ERA, polar motion, etc.), space weather
  (`#227 <https://github.com/bluescarni/heyoka.py/pull/227>`__,
  `#225 <https://github.com/bluescarni/heyoka.py/pull/225>`__,
  `#223 <https://github.com/bluescarni/heyoka.py/pull/223>`__,
  `#222 <https://github.com/bluescarni/heyoka.py/pull/222>`__).
- Introduce class to represent, manage and update space weather data
  (`#227 <https://github.com/bluescarni/heyoka.py/pull/227>`__).
- Add function to convert from geodetic to Cartesian coordinates
  (`#226 <https://github.com/bluescarni/heyoka.py/pull/226>`__).
- Add precompiled wheels for free-threaded Python 3.13
  (`#226 <https://github.com/bluescarni/heyoka.py/pull/226>`__).
- Add support for free-threaded Python
  (`#226 <https://github.com/bluescarni/heyoka.py/pull/226>`__).
- Add an implementation of the EGM2008 geopotential model
  (`#226 <https://github.com/bluescarni/heyoka.py/pull/226>`__).
- Add an implementation of the IAU2000/2006 precession-nutation theory
  (`#224 <https://github.com/bluescarni/heyoka.py/pull/224>`__).
- Introduce class to represent, manage and update EOP data
  (`#222 <https://github.com/bluescarni/heyoka.py/pull/222>`__).

Changes
~~~~~~~

- heyoka.py now requires version 7.3.0 of the
  heyoka C++ library
  (`#222 <https://github.com/bluescarni/heyoka.py/pull/222>`__).

7.2.2 (2025-02-01)
------------------

Changes
~~~~~~~

- Add Linux aarch64 wheels
  (`#219 <https://github.com/bluescarni/heyoka.py/pull/219>`__).
- Remove wheels for Python 3.9
  (`#219 <https://github.com/bluescarni/heyoka.py/pull/219>`__).

7.2.1 (2025-01-07)
------------------

Fix
~~~

- Work around a thread-safety issue in the erfa routines used
  to convert between UTC and TAI Julian dates.

7.2.0 (2025-01-02)
------------------

Changes
~~~~~~~

- Clarify that epochs and dates have to be provided to the SGP4
  propagator in the UTC scale of time
  (`#213 <https://github.com/bluescarni/heyoka.py/pull/213>`__).
- heyoka.py now requires version 7.2.0 of the
  heyoka C++ library
  (`#213 <https://github.com/bluescarni/heyoka.py/pull/213>`__).

7.0.1 (2024-12-29)
------------------

Fix
~~~

- Exclude the docs from the sdist build in order to limit the size
  of the tarball
  (`#210 <https://github.com/bluescarni/heyoka.py/pull/210>`__).
- Fix unit test failure in Python 3.13
  (`#210 <https://github.com/bluescarni/heyoka.py/pull/210>`__).

7.0.0 (2024-12-28)
------------------

New
~~~

- Expose the ``get_params()`` function, which returns a list of
  parameters contained in an expression or a list of expressions
  (`#207 <https://github.com/bluescarni/heyoka.py/pull/207>`__).
- New function :func:`~heyoka.model.gpe_is_deep_space()` to detect
  deep-space GPEs
  (`#207 <https://github.com/bluescarni/heyoka.py/pull/207>`__).
- Upload the source distribution to PyPI
  (`#203 <https://github.com/bluescarni/heyoka.py/pull/203>`__).

Changes
~~~~~~~

- Small tweaks to the behaviour of the SGP4 propagator: non-normalised double-length
  Julian dates are now accepted, deep-space GPEs do not result in exceptions any more,
  performance improvements for the
  :func:`~heyoka.model.sgp4_propagator_dbl.replace_sat_data()` function,
  the satellite data can now be passed as a NumPy array
  (`#208 <https://github.com/bluescarni/heyoka.py/pull/208>`__,
  `#207 <https://github.com/bluescarni/heyoka.py/pull/207>`__).
- The parallel compilation feature has been temporarily disabled due to several LLVM bugs
  (`#206 <https://github.com/bluescarni/heyoka.py/pull/206>`__).
- heyoka.py now requires version 7.0.0 of the
  heyoka C++ library
  (`#206 <https://github.com/bluescarni/heyoka.py/pull/206>`__).
- **BREAKING**: heyoka.py now requires Python >= 3.9
  (`#206 <https://github.com/bluescarni/heyoka.py/pull/206>`__).
  This is a :ref:`breaking change <bchanges_7_0_0>`.
- **BREAKING**: heyoka.py now requires NumPy >= 2
  (`#206 <https://github.com/bluescarni/heyoka.py/pull/206>`__).
  This is a :ref:`breaking change <bchanges_7_0_0>`.

6.1.2 (2024-10-10)
------------------

Fix
~~~

- Fix PyPI metadata.

6.1.1 (2024-10-10)
------------------

Fix
~~~

- Fix upload of binary wheels.

6.1.0 (2024-10-10)
------------------

New
~~~

- Add a proper ``pyproject.toml`` file and use it to produce
  the binary wheels
  (`#195 <https://github.com/bluescarni/heyoka.py/pull/195>`__).

Fix
~~~

- Do not open the heyoka.py compiled module with ``RTLD_GLOBAL``
  (`#197 <https://github.com/bluescarni/heyoka.py/pull/197>`__).
- Workaround for a clang 17 issue that would result in
  runtime exceptions during (de)serialisation
  (`#196 <https://github.com/bluescarni/heyoka.py/pull/196>`__).

6.0.0 (2024-09-21)
------------------

New
~~~

- Add wheels for Python 3.13
  (`#193 <https://github.com/bluescarni/heyoka.py/pull/193>`__).
- Non-number exponents for the ``pow()`` function
  are now supported in Taylor integrators
  (`#189 <https://github.com/bluescarni/heyoka.py/pull/189>`__).
- It is now possible to initialise a scalar Taylor integrator
  with an empty initial state vector, or a batch integrator
  with a 2D state vector whose first dimension is zero. This will result
  in zero-initialization of the state vector
  (`#189 <https://github.com/bluescarni/heyoka.py/pull/189>`__).
- Implement parallel compilation for Taylor integrators
  and compiled functions
  (`#188 <https://github.com/bluescarni/heyoka.py/pull/188>`__).
- Add the possibility of specifying the LLVM code model
  used for JIT compilation
  (`#188 <https://github.com/bluescarni/heyoka.py/pull/188>`__).

Changes
~~~~~~~

- **BREAKING**: the array of parameter values passed to the
  constructor of a Taylor integrator must now either be empty
  (in which case the parameter values will be zero-inited),
  or have the correct size
  (`#189 <https://github.com/bluescarni/heyoka.py/pull/189>`__).
  This is a :ref:`breaking change <bchanges_6_0_0>`.
- heyoka.py now requires version 6.0.0 of the
  heyoka C++ library
  (`#188 <https://github.com/bluescarni/heyoka.py/pull/188>`__).

Fix
~~~

- Fix build system warnings when using recent versions of
  CMake and Boost
  (`#188 <https://github.com/bluescarni/heyoka.py/pull/188>`__).

5.1.0 (2024-07-23)
------------------

New
~~~

- Add a fully differentiable implementation of the SGP4 analytical propagator
  (`#183 <https://github.com/bluescarni/heyoka.py/pull/183>`__).
- Add the ``select()`` primitive to the expression system
  (`#183 <https://github.com/bluescarni/heyoka.py/pull/183>`__).
- Add relational and logical operators to the expression system
  (`#183 <https://github.com/bluescarni/heyoka.py/pull/183>`__).
- Add tutorial on Taylor map inversion
  (`#182 <https://github.com/bluescarni/heyoka.py/pull/182>`__).
- Add tutorial on solving inversion problems with the variational equations
  (`#181 <https://github.com/bluescarni/heyoka.py/pull/181>`__).

Changes
~~~~~~~

- The minimum supported SymPy version is now 1.13.0
  (`#183 <https://github.com/bluescarni/heyoka.py/pull/183>`__).
- The binary wheels are now built on top of ``manylinux_2_28``
  (`#183 <https://github.com/bluescarni/heyoka.py/pull/183>`__).
- heyoka.py now requires version 5.1.0 of the
  heyoka C++ library
  (`#183 <https://github.com/bluescarni/heyoka.py/pull/183>`__).

Fix
~~~

- Fix test failures when using recent SymPy versions
  (`#183 <https://github.com/bluescarni/heyoka.py/pull/183>`__).

5.0.1 (2024-06-14)
------------------

Fix
~~~

- Fix an input size check that would wrongly throw on valid code
  (`#179 <https://github.com/bluescarni/heyoka.py/pull/179>`__).

5.0.0 (2024-06-13)
------------------

New
~~~

- Add support for variational ODE systems and Taylor map computation
  (`#177 <https://github.com/bluescarni/heyoka.py/pull/177>`__).
- Add thermonets: neural, differentiable, high-performance
  models for the Earth's thermosphere density
  (`#176 <https://github.com/bluescarni/heyoka.py/pull/176>`__).
- Add a vectorised implementation of ``diff()``
  (`#173 <https://github.com/bluescarni/heyoka.py/pull/173>`__).

Changes
~~~~~~~

- Several automatic simplifications and normalisations in the expression system
  have been removed as they caused drastic slowdowns in symbolic operations when
  working with large and highly recursive computational graphs
  (`#174 <https://github.com/bluescarni/heyoka.py/pull/174>`__).
- **BREAKING**: as a consequence of the removal of most automatic simplifications,
  several now-obsolete functions have also been removed
  (`#174 <https://github.com/bluescarni/heyoka.py/pull/174>`__).
  These are :ref:`breaking changes <bchanges_5_0_0>`.
- heyoka.py now requires version 5.0.0 of the
  heyoka C++ library
  (`#173 <https://github.com/bluescarni/heyoka.py/pull/173>`__).

4.0.0 (2024-03-03)
------------------

New
~~~

- New convenience :func:`~heyoka.dtens.hessian()` method to fetch the Hessian
  from a :class:`~heyoka.dtens` object
  (`#171 <https://github.com/bluescarni/heyoka.py/pull/171>`__).
- Compiled functions now support multithreaded parallelisation
  for batched evaluations
  (`#168 <https://github.com/bluescarni/heyoka.py/pull/168>`__).
- Add new example on gravity-gradient stabilisation
  (`#159 <https://github.com/bluescarni/heyoka.py/pull/159>`__).
- Add support for Lagrangian and Hamiltonian mechanics
  (`#156 <https://github.com/bluescarni/heyoka.py/pull/156>`__).
- It is now possible to pass a list of step callbacks to the
  ``propagate_*()`` functions
  (`#155 <https://github.com/bluescarni/heyoka.py/pull/155>`__).
- New ``angle_reducer`` step callback to automatically reduce
  angular state variables to the :math:`\left[0, 2\pi\right)` range
  (`#155 <https://github.com/bluescarni/heyoka.py/pull/155>`__).
- New ``callback`` module containing ready-made step and event callbacks
  (`#155 <https://github.com/bluescarni/heyoka.py/pull/155>`__).

Changes
~~~~~~~

- **BREAKING**: the function to construct compiled functions
  has been renamed from ``make_cfunc()`` to ``cfunc()``
  (`#168 <https://github.com/bluescarni/heyoka.py/pull/168>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- **BREAKING**: compiled functions now require contiguous arrays
  as input/output arguments. The compiled functions API is also now
  more restrictive with respect to on-the-fly type conversions
  (`#168 <https://github.com/bluescarni/heyoka.py/pull/168>`__).
  These are :ref:`breaking changes <bchanges_4_0_0>`.
- **BREAKING**: it is now mandatory to supply a list of differentiation
  arguments to :func:`~heyoka.diff_tensors()`
  (`#164 <https://github.com/bluescarni/heyoka.py/pull/164>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- Improve performance when creating compiled functions
  (`#162 <https://github.com/bluescarni/heyoka.py/pull/162>`__).
- **BREAKING**: :ref:`compiled functions <cfunc_tut>` now require
  the list of input variables to be always supplied by the user
  (`#162 <https://github.com/bluescarni/heyoka.py/pull/162>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- **BREAKING**: the :py:func:`~heyoka.make_vars()` function
  now returns a single expression (rather than a list of expressions)
  if a single argument is passed in input
  (`#161 <https://github.com/bluescarni/heyoka.py/pull/161>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- **BREAKING**: the signature of callbacks for terminal events
  has been simplified
  (`#158 <https://github.com/bluescarni/heyoka.py/pull/158>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- **BREAKING**: the ``propagate_*()`` functions
  now return the (optional) step callback that can be
  passed in input
  (`#155 <https://github.com/bluescarni/heyoka.py/pull/155>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- **BREAKING**: the ``propagate_grid()`` methods of the
  adaptive integrators now require the first element of the
  time grid to be equal to the current integrator time
  (`#154 <https://github.com/bluescarni/heyoka.py/pull/154>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- The binary wheels are now built on top of ``manylinux2014``
  (`#153 <https://github.com/bluescarni/heyoka.py/pull/153>`__).
- heyoka.py now requires C++20 when building from source
  (`#153 <https://github.com/bluescarni/heyoka.py/pull/153>`__).
- heyoka.py now requires version 4.0.0 of the
  heyoka C++ library
  (`#153 <https://github.com/bluescarni/heyoka.py/pull/153>`__).

3.2.0 (2023-11-29)
------------------

New
~~~

- New example on a differentiable atmosphere model via
  neural networks
  (`#151 <https://github.com/bluescarni/heyoka.py/pull/151>`__).
- New example on interfacing pytorch and heyoka.py
  (`#151 <https://github.com/bluescarni/heyoka.py/pull/151>`__).
- Add wheels for Python 3.12
  (`#150 <https://github.com/bluescarni/heyoka.py/pull/150>`__).
- Add support for single-precision computations
  (`#150 <https://github.com/bluescarni/heyoka.py/pull/150>`__).
- Add model implementing the ELP2000 analytical lunar theory
  (`#149 <https://github.com/bluescarni/heyoka.py/pull/149>`__).

Changes
~~~~~~~

- heyoka.py now requires version 3.2.0 of the
  heyoka C++ library
  (`#149 <https://github.com/bluescarni/heyoka.py/pull/149>`__).

Fix
~~~

- Fix wrong truncation to double precision in the dtime setter for the
  scalar integrator
  (`#150 <https://github.com/bluescarni/heyoka.py/pull/150>`__).

3.1.0 (2023-11-13)
------------------

New
~~~

- New example notebooks on neural ODEs
  (`#143 <https://github.com/bluescarni/heyoka.py/pull/143>`__,
  `#142 <https://github.com/bluescarni/heyoka.py/pull/142>`__).
- Add a model for feed-forward neural networks
  (`#142 <https://github.com/bluescarni/heyoka.py/pull/142>`__).
- Implement (leaky) ``ReLU`` and its derivative in the expression
  system (`#141 <https://github.com/bluescarni/heyoka.py/pull/141>`__).
- Implement the eccentric longitude :math:`F` in the expression
  system (`#140 <https://github.com/bluescarni/heyoka.py/pull/140>`__).
- Implement the delta eccentric anomaly :math:`\Delta E` in the expression
  system (`#140 <https://github.com/bluescarni/heyoka.py/pull/140>`__).
  Taylor derivatives are not implemented yet.
- Implement convenience properties to fetch the gradient/Jacobian
  from a ``dtens`` object
  (`#140 <https://github.com/bluescarni/heyoka.py/pull/140>`__).
- New example notebook implementing Lagrange propagation
  (`#140 <https://github.com/bluescarni/heyoka.py/pull/140>`__).
- New example notebook on the continuation of periodic orbits
  in the CR3BP (`#97 <https://github.com/bluescarni/heyoka.py/pull/97>`__).

Changes
~~~~~~~

- heyoka.py now requires version 3.1.0 of the
  heyoka C++ library
  (`#140 <https://github.com/bluescarni/heyoka.py/pull/140>`__).

Fix
~~~

- Fix slow performance when creating very large compiled functions
  (`#144 <https://github.com/bluescarni/heyoka.py/pull/144>`__).
- Fix building against Python 3.12
  (`#139 <https://github.com/bluescarni/heyoka.py/pull/139>`__).

3.0.0 (2023-10-07)
------------------

Changes
~~~~~~~

- heyoka.py now requires version 3.0.0 of the
  heyoka C++ library
  (`#137 <https://github.com/bluescarni/heyoka.py/pull/137>`__).

2.0.0 (2023-09-22)
------------------

New
~~~

- Add model for the circular restricted three-body problem
  (`#135 <https://github.com/bluescarni/heyoka.py/pull/135>`__).
- The LLVM SLP vectorizer can now be enabled
  (`#134 <https://github.com/bluescarni/heyoka.py/pull/134>`__).
  This feature is opt-in due to the fact that enabling it
  can considerably increase JIT compilation times.
- Implement an in-memory cache for ``llvm_state``. The cache is used
  to avoid re-optimising and re-compiling LLVM code which has
  already been optimised and compiled during the program execution
  (`#134 <https://github.com/bluescarni/heyoka.py/pull/134>`__).
- It is now possible to get the LLVM bitcode of
  an ``llvm_state``
  (`#134 <https://github.com/bluescarni/heyoka.py/pull/134>`__).

1.0.0 (2023-08-11)
------------------

New
~~~

- The step callbacks can now optionally implement a ``pre_hook()``
  method that will be called before the first step
  is taken by a ``propagate_*()`` function
  (`#128 <https://github.com/bluescarni/heyoka.py/pull/128>`__).
- Introduce several vectorised overloads in the expression
  API. These vectorised overloads allow to perform the same
  operation on a list of expressions more efficiently
  than performing the same operation repeatedly on individual
  expressions
  (`#127 <https://github.com/bluescarni/heyoka.py/pull/127>`__).
- New API to compute high-order derivatives
  (`#127 <https://github.com/bluescarni/heyoka.py/pull/127>`__).
- Implement substitution of generic subexpressions
  (`#127 <https://github.com/bluescarni/heyoka.py/pull/127>`__).
- The state variables and right-hand side of a system of ODEs
  are now available as read-only properties in the integrator
  classes
  (`#122 <https://github.com/bluescarni/heyoka.py/pull/122>`__).
- Several additions to the :ref:`compiled functions <cfunc_tut>` API:
  compiled functions can now
  be pickled/unpickled, and they expose several information as
  read-only properties (e.g., list of variables, outputs, etc.)
  (`#120 <https://github.com/bluescarni/heyoka.py/pull/120>`__).
- Expressions now support hashing
  (`#120 <https://github.com/bluescarni/heyoka.py/pull/120>`__).
- New ``model`` submodule containing ready-made dynamical models
  (`#119 <https://github.com/bluescarni/heyoka.py/pull/119>`__).

Changes
~~~~~~~

- **BREAKING**: the VSOP2013 functions have been moved from the
  main module to the new ``model`` submodule
  (`#130 <https://github.com/bluescarni/heyoka.py/pull/130>`__).
  This is a :ref:`breaking change <bchanges_1_0_0>`.
- The custom NumPy memory manager that prevents memory leaks
  with ``real`` arrays is now disabled by default
  (`#129 <https://github.com/bluescarni/heyoka.py/pull/129>`__).
- The step callbacks are now deep-copied in multithreaded
  :ref:`ensemble propagations <ensemble_prop>`
  rather then being shared among threads. The aim of this change
  is to reduce the likelihood of data races
  (`#128 <https://github.com/bluescarni/heyoka.py/pull/128>`__).
- Comprehensive overhaul of the expression system, including:
  enhanced automatic simplification capabilities for sums,
  products and powers, removal of several specialised primitives
  (such as ``square()``, ``neg()``, ``sum_sq()``, etc.),
  re-implementation of division and subtraction as special
  cases of product and sum, and more
  (`#127 <https://github.com/bluescarni/heyoka.py/pull/127>`__).
- heyoka.py now requires at least version 1.0.0 of the
  heyoka C++ library
  (`#127 <https://github.com/bluescarni/heyoka.py/pull/127>`__).
- **BREAKING**: the ``make_nbody_sys()`` helper has been replaced by an equivalent
  function in the new ``model`` submodule
  (`#119 <https://github.com/bluescarni/heyoka.py/pull/119>`__).
  This is a :ref:`breaking change <bchanges_1_0_0>`.

0.21.8 (2023-07-03)
-------------------

Fix
~~~

- Fix building against NumPy 1.25
  (`#125 <https://github.com/bluescarni/heyoka.py/pull/125>`__).

0.21.7 (2023-02-16)
-------------------

New
~~~

- Add support for installation via ``pip`` on Linux
  (`#115 <https://github.com/bluescarni/heyoka.py/pull/115>`__).
- Time-dependent functions can now be compiled
  (`#113 <https://github.com/bluescarni/heyoka.py/pull/113>`__).

Changes
~~~~~~~

- heyoka.py now requires at least version 0.21.0 of the
  heyoka C++ library
  (`#113 <https://github.com/bluescarni/heyoka.py/pull/113>`__).

0.20.0 (2022-12-18)
-------------------

New
~~~

- Implement arbitrary-precision computations
  (`#108 <https://github.com/bluescarni/heyoka.py/pull/108>`__).
- Implement the ``isnan()`` and ``isinf()`` NumPy ufuncs for
  ``real128``
  (`#108 <https://github.com/bluescarni/heyoka.py/pull/108>`__).
- Several JIT-related settings can now be tweaked via keyword arguments
  (`#107 <https://github.com/bluescarni/heyoka.py/pull/107>`__).

Changes
~~~~~~~

- heyoka.py now requires CMake >= 3.18 when building from source
  (`#109 <https://github.com/bluescarni/heyoka.py/pull/109>`__).
- heyoka.py now requires at least version 0.20.0 of the
  heyoka C++ library
  (`#107 <https://github.com/bluescarni/heyoka.py/pull/107>`__).

Fix
~~~

- Fix the ``real128`` NumPy comparison operator to be consistent
  with ``float`` with respect to NaN values
  (`#108 <https://github.com/bluescarni/heyoka.py/pull/108>`__).
- Prevent the ``real128`` constructor from being called with keyword arguments
  (`#108 <https://github.com/bluescarni/heyoka.py/pull/108>`__).
- Fix a build issue with Python 3.11
  (`#107 <https://github.com/bluescarni/heyoka.py/pull/107>`__).

0.19.0 (2022-09-19)
-------------------

New
~~~

- Add a tutorial on extended-precision computations
  (`#99 <https://github.com/bluescarni/heyoka.py/pull/99>`__).
- The way quadruple-precision computations are supported via ``real128``
  has been completely overhauled: ``real128`` is now exposed as a
  NumPy-enabled Python type, meaning that ``real128``
  can now be used in exactly the same way as ``float`` and
  ``np.longdouble`` in the heyoka.py API
  (`#99 <https://github.com/bluescarni/heyoka.py/pull/99>`__,
  `#98 <https://github.com/bluescarni/heyoka.py/pull/98>`__).
  This is a :ref:`breaking change <bchanges_0_19_0>`.
- Add the capability to compile multivariate vector functions at runtime
  (`#96 <https://github.com/bluescarni/heyoka.py/pull/96>`__).

Changes
~~~~~~~

- **BREAKING**: heyoka.py is now more strict with respect
  to type conversions. See the :ref:`breaking changes <bchanges_0_19_0>`
  section for more details.
- heyoka.py now compiles without deprecation warnings against
  the latest fmt versions
  (`#98 <https://github.com/bluescarni/heyoka.py/pull/98>`__).
- New version requirements: heyoka>=0.19, CMake>=3.16, pybind11>=2.10
  (`#98 <https://github.com/bluescarni/heyoka.py/pull/98>`__,
  `#96 <https://github.com/bluescarni/heyoka.py/pull/96>`__).

0.18.0 (2022-05-11)
-------------------

New
~~~

- Add a function to build (N+1)-body problems
  (`#92 <https://github.com/bluescarni/heyoka.py/pull/92>`__).
- Expose numerical solvers for Kepler's elliptic equation
  (`#91 <https://github.com/bluescarni/heyoka.py/pull/91>`__).
- Implement parallel mode
  for the automatic parallelisation of an individual integration step
  (`#88 <https://github.com/bluescarni/heyoka.py/pull/88>`__).

Changes
~~~~~~~

- heyoka.py does not depend on the spdlog library any more
  (`#89 <https://github.com/bluescarni/heyoka.py/pull/89>`__).
- heyoka.py now depends on the `TBB <https://github.com/oneapi-src/oneTBB>`__ library
  (`#88 <https://github.com/bluescarni/heyoka.py/pull/88>`__).
- heyoka.py now requires at least version 0.18.0 of the
  heyoka C++ library
  (`#88 <https://github.com/bluescarni/heyoka.py/pull/88>`__).
- In case of an early interruption, the ``propagate_grid()`` function will now
  process all available grid points before the interruption time before exiting
  (`#88 <https://github.com/bluescarni/heyoka.py/pull/88>`__).
- The ``propagate_grid()`` callbacks are now invoked also if the integration
  is interrupted by a stopping terminal event
  (`#88 <https://github.com/bluescarni/heyoka.py/pull/88>`__).

Fix
~~~

- Fix an issue in the ``propagate_grid()`` functions
  that could lead to invalid results in certain corner cases
  (`#88 <https://github.com/bluescarni/heyoka.py/pull/88>`__).

0.17.0 (2022-01-25)
-------------------

New
~~~

- It is now possible to access the adaptive integrators'
  time values as double-length floats
  (`#86 <https://github.com/bluescarni/heyoka.py/pull/86>`__).
- Add support for ensemble propagations
  (`#85 <https://github.com/bluescarni/heyoka.py/pull/85>`__).
- Several functions in the batch integration API
  now also accept scalar time values in input,
  instead of just vectors. The scalar values
  are automatically splatted into vectors
  of the appropriate size
  (`#85 <https://github.com/bluescarni/heyoka.py/pull/85>`__).
- Copy operations on the main heyoka.py classes now preserve
  dynamic attributes
  (`#85 <https://github.com/bluescarni/heyoka.py/pull/85>`__).
- Add a function to compute the suggested SIMD size for
  the CPU in use
  (`#84 <https://github.com/bluescarni/heyoka.py/pull/84>`__).

Changes
~~~~~~~

- heyoka.py now requires at least version 0.17.0 of the
  heyoka C++ library
  (`#84 <https://github.com/bluescarni/heyoka.py/pull/84>`__).

Fix
~~~

- Fix build failures when using recent versions of ``fmt``
  (`#86 <https://github.com/bluescarni/heyoka.py/pull/86>`__).

0.16.0 (2021-11-20)
-------------------

New
~~~

- **BREAKING**: add support for continuous output
  to the ``propagate_for/until()`` methods
  (`#81 <https://github.com/bluescarni/heyoka.py/pull/81>`__).
  This is a :ref:`breaking change <bchanges_0_16_0>`.
- Event detection is now available also in batch mode
  (`#80 <https://github.com/bluescarni/heyoka.py/pull/80>`__).
- Attributes can now be dynamically added to the main heyoka.py
  classes (`#78 <https://github.com/bluescarni/heyoka.py/pull/78>`__).
- Add a tutorial on the computation of event sensitivity
  (`#77 <https://github.com/bluescarni/heyoka.py/pull/77>`__).

Changes
~~~~~~~

- heyoka.py now requires at least version 0.16.0 of the
  heyoka C++ library
  (`#75 <https://github.com/bluescarni/heyoka.py/pull/75>`__).

0.15.1 (2021-10-10)
-------------------

Fix
~~~

- Fix conversion from SymPy rationals.

0.15.0 (2021-09-28)
-------------------

New
~~~

- Add a tutorial on the simulation of tides
  (`#70 <https://github.com/bluescarni/heyoka.py/pull/70>`__).
- The conversion of expressions from/to SymPy now takes advantage
  of reference semantics, which leads to large
  performance improvements when dealing with expressions
  with a high degree of internal repetition
  (`#70 <https://github.com/bluescarni/heyoka.py/pull/70>`__).
- Add the possibility to customise the behaviour of the
  ``from_sympy()`` function
  (`#70 <https://github.com/bluescarni/heyoka.py/pull/70>`__).
- Add :math:`\pi` as a symbolic constant to the expression system
  (`#70 <https://github.com/bluescarni/heyoka.py/pull/70>`__).
- Add a function to compute the size of an expression
  (`#69 <https://github.com/bluescarni/heyoka.py/pull/69>`__).
- Add an example on the computation of definite integrals
  (`#68 <https://github.com/bluescarni/heyoka.py/pull/68>`__).
- Add an implementation of the VSOP2013 analytical solution
  for the motion of the planets of the Solar System, usable
  in the definition of differential equations
  (`#67 <https://github.com/bluescarni/heyoka.py/pull/67>`__).
  An example describing this new feature is available in
  the documentation.
- Add support for the two-argument inverse tangent function
  ``atan2()`` in the expression system
  (`#64 <https://github.com/bluescarni/heyoka.py/pull/64>`__).

Changes
~~~~~~~

- heyoka.py now requires at least version 0.15.0 of the
  heyoka C++ library
  (`#64 <https://github.com/bluescarni/heyoka.py/pull/64>`__).

Fix
~~~

- Test fixes on PPC64
  (`#69 <https://github.com/bluescarni/heyoka.py/pull/69>`__).

0.14.0 (2021-08-03)
-------------------

New
~~~

- Add a new example on the numerical detection of integrals
  of motion
  (`#59 <https://github.com/bluescarni/heyoka.py/pull/59>`__).
- The tolerance value is now stored in the integrator objects
  (`#58 <https://github.com/bluescarni/heyoka.py/pull/58>`__).

Changes
~~~~~~~

- heyoka.py now requires at least version 0.14.0 of the
  heyoka C++ library
  (`#58 <https://github.com/bluescarni/heyoka.py/pull/58>`__).

0.12.0 (2021-07-23)
-------------------

New
~~~

- Add support for 64-bit ARM processors
  (`#55 <https://github.com/bluescarni/heyoka.py/pull/55>`__).
- Pickling support has been added to all classes
  (`#53 <https://github.com/bluescarni/heyoka.py/pull/53>`__).
- Event properties can now be accessed after construction
  (`#53 <https://github.com/bluescarni/heyoka.py/pull/53>`__).

Changes
~~~~~~~

- heyoka.py now depends on the
  `Boost <https://www.boost.org/>`__ C++ libraries
  (`#53 <https://github.com/bluescarni/heyoka.py/pull/53>`__).
- heyoka.py now requires at least version 0.12.0 of the
  heyoka C++ library
  (`#53 <https://github.com/bluescarni/heyoka.py/pull/53>`__).

0.11.0 (2021-07-06)
-------------------

New
~~~

- New tutorial on transit timing variations
  (`#50 <https://github.com/bluescarni/heyoka.py/pull/50>`__).

Changes
~~~~~~~

- heyoka.py now requires at least version 0.11.0 of the
  heyoka C++ library
  (`#50 <https://github.com/bluescarni/heyoka.py/pull/50>`__).

0.10.0 (2021-06-09)
-------------------

New
~~~

- The callback that can be passed to the ``propagate_*()`` methods
  can now be used to stop the integration
  (`#48 <https://github.com/bluescarni/heyoka.py/pull/48>`__).
- New tutorial on SymPy interoperability
  (`#47 <https://github.com/bluescarni/heyoka.py/pull/47>`__).
- Add a pairwise product primitive
  (`#46 <https://github.com/bluescarni/heyoka.py/pull/46>`__).
- heyoka.py expressions can now be converted to/from SymPy expressions
  (`#46 <https://github.com/bluescarni/heyoka.py/pull/46>`__).

Changes
~~~~~~~

- **BREAKING**: a :ref:`breaking change <bchanges_0_10_0>`
  in the ``propagate_*()`` callback API
  (`#48 <https://github.com/bluescarni/heyoka.py/pull/48>`__).
- Division by zero in the expression system now raises an error
  (`#48 <https://github.com/bluescarni/heyoka.py/pull/48>`__).
- heyoka.py now requires at least version 0.10.0 of the
  heyoka C++ library
  (`#46 <https://github.com/bluescarni/heyoka.py/pull/46>`__).

0.9.0 (2021-05-25)
------------------

New
~~~

- Add time polynomials to the expression system
  (`#44 <https://github.com/bluescarni/heyoka.py/pull/44>`__).
- New tutorial on Mercury's relativistic precession
  (`#42 <https://github.com/bluescarni/heyoka.py/pull/42>`__).
- Add the inverse of Kepler's elliptic equation to the expression system
  (`#41 <https://github.com/bluescarni/heyoka.py/pull/41>`__).
- New tutorial on planetary embryos
  (`#39 <https://github.com/bluescarni/heyoka.py/pull/39>`__).
- Initial exposition of the ``llvm_state`` class
  (`#39 <https://github.com/bluescarni/heyoka.py/pull/39>`__).

Changes
~~~~~~~

- heyoka.py now requires at least version 0.9.0 of the
  heyoka C++ library
  (`#41 <https://github.com/bluescarni/heyoka.py/pull/41>`__).

0.8.0 (2021-04-28)
------------------

New
~~~

- The ``propagate_for/until()`` functions now support writing
  the Taylor coefficients at the end of each timestep
  (`#37 <https://github.com/bluescarni/heyoka.py/pull/37>`__).

Changes
~~~~~~~

- **BREAKING**: :ref:`breaking changes <bchanges_0_8_0>`
  in the event detection API
  (`#37 <https://github.com/bluescarni/heyoka.py/pull/37>`__).
- heyoka.py now requires at least version 0.8.0 of the
  heyoka C++ library
  (`#37 <https://github.com/bluescarni/heyoka.py/pull/37>`__).

0.7.0 (2021-04-22)
------------------

New
~~~

- The ``propagate_*()`` functions now accept an optional
  ``max_delta_t`` argument to limit the size of a timestep,
  and an optional ``callback`` argument that will be invoked
  at the end of each timestep
  (`#34 <https://github.com/bluescarni/heyoka.py/pull/34>`__).
- ``update_d_output()`` can now be called with a relative
  (rather than absolute) time argument
  (`#34 <https://github.com/bluescarni/heyoka.py/pull/34>`__).

Changes
~~~~~~~

- **BREAKING**: the time coordinates in batch integrators
  cannot be directly modified any more, and the new
  ``set_time()`` function must be used instead
  (`#34 <https://github.com/bluescarni/heyoka.py/pull/34>`__).
- heyoka.py now requires at least version 0.7.0 of the
  heyoka C++ library
  (`#34 <https://github.com/bluescarni/heyoka.py/pull/34>`__).

0.6.1 (2021-04-08)
------------------

New
~~~

- Add the wavy ramp tutorial
  (`#32 <https://github.com/bluescarni/heyoka.py/pull/32>`__).

Changes
~~~~~~~

- heyoka.py now requires at least version 0.6.1 of the
  heyoka C++ library
  (`#32 <https://github.com/bluescarni/heyoka.py/pull/32>`__).

0.6.0 (2021-04-06)
------------------

New
~~~

- Add a tutorial about Brouwer's law
  (`#31 <https://github.com/bluescarni/heyoka.py/pull/31>`__).
- Add a tutorial about batch mode
  (`#30 <https://github.com/bluescarni/heyoka.py/pull/30>`__).
- Add tutorials about gravitational billiards
  (`#29 <https://github.com/bluescarni/heyoka.py/pull/29>`__,
  `#28 <https://github.com/bluescarni/heyoka.py/pull/28>`__).
- Expose propagation over a time grid for the batch integrator
  (`#29 <https://github.com/bluescarni/heyoka.py/pull/29>`__).
- Add a tutorial about the computation of Poincaré sections
  (`#27 <https://github.com/bluescarni/heyoka.py/pull/27>`__).
- Add a tutorial on optimal control
  (`#24 <https://github.com/bluescarni/heyoka.py/pull/24>`__).
- Initial version of the event detection system
  (`#23 <https://github.com/bluescarni/heyoka.py/pull/23>`__).
- Expose low-level functions to compute the jet of derivatives
  for an ODE system
  (`#21 <https://github.com/bluescarni/heyoka.py/pull/21>`__).

Changes
~~~~~~~

- **BREAKING**: the ``propagate_grid()`` method now requires
  monotonically-ordered grid points
  (`#25 <https://github.com/bluescarni/heyoka.py/pull/25>`__).
- heyoka.py now depends on the `spdlog <https://github.com/gabime/spdlog>`__ library
  (`#23 <https://github.com/bluescarni/heyoka.py/pull/23>`__).
- heyoka.py now requires at least version 0.6.0 of the
  heyoka C++ library
  (`#21 <https://github.com/bluescarni/heyoka.py/pull/21>`__).

Fix
~~~

- Properly restore the original ``mpmath`` precision after
  importing heyoka.py
  (`#21 <https://github.com/bluescarni/heyoka.py/pull/21>`__).

0.5.0 (2021-02-25)
------------------

New
~~~

- Expose symbolic differentiation.
- Add a new tutorial (restricted three-body problem).

Changes
~~~~~~~

- The interface of the integrator in batch mode has changed
  to work with arrays in which the batch size has its own dimension,
  instead of being flattened out
  (`#20 <https://github.com/bluescarni/heyoka.py/pull/20>`__).
- heyoka.py now depends on the `{fmt} <https://fmt.dev/latest/index.html>`__ library
  (`#20 <https://github.com/bluescarni/heyoka.py/pull/20>`__).
- heyoka.py now requires at least version 0.5.0 of the
  heyoka C++ library
  (`#20 <https://github.com/bluescarni/heyoka.py/pull/20>`__).

0.4.0 (2021-02-20)
------------------

New
~~~

- Expose the new ``powi()`` function from heyoka 0.4.0
  (`#18 <https://github.com/bluescarni/heyoka.py/pull/18>`__).
- Add support for ``propagate_grid()``
  (`#17 <https://github.com/bluescarni/heyoka.py/pull/17>`__).
- Add support for dense output and for storing
  the Taylor coefficients at the end of a timestep
  (`#11 <https://github.com/bluescarni/heyoka.py/pull/11>`__).
- Various doc additions
  (`#15 <https://github.com/bluescarni/heyoka.py/pull/15>`__,
  `#14 <https://github.com/bluescarni/heyoka.py/pull/14>`__,
  `#13 <https://github.com/bluescarni/heyoka.py/pull/13>`__,
  `#12 <https://github.com/bluescarni/heyoka.py/pull/12>`__,
  `#11 <https://github.com/bluescarni/heyoka.py/pull/11>`__).

Changes
~~~~~~~

- heyoka.py now requires at least version 0.4.0 of the
  heyoka C++ library.

0.3.0 (2021-02-13)
------------------

- This is the initial public release of heyoka.py
