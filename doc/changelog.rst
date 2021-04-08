.. _changelog:

Changelog
=========

0.6.1 (unreleased)
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
