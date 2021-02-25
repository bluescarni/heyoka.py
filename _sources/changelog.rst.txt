.. _changelog:

Changelog
=========

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
