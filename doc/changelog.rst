.. _changelog:

Changelog
=========

0.16.0 (unreleased)
-------------------

New
~~~

- Add a tutorial on the computation of event sensitivity
  (`#77 <https://github.com/bluescarni/heyoka.py/pull/77`__).

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
