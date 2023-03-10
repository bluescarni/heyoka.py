.. _breaking_changes:

Breaking changes
================

.. _bchanges_0_22_0:

0.22.0
------

- The ``make_nbody_sys()`` function has been replaced by
  the ``model.nbody()`` function, with identical semantics.

.. _bchanges_0_19_0:

0.19.0
------

- Quadruple-precision computations via the ``real128`` type now
  use the same API as the other supported floating-point types
  (:py:class:`float` and :py:class:`numpy.longdouble`). Additionally,
  the way extended-precision mode is enabled has changed - please
  refer to the `extended-precision tutorial <https://bluescarni.github.io/heyoka.py/notebooks/ext_precision.html>`__
  for detailed information.
- The heyoka.py API is now more strict with respect to type conversions.
  For instance, attempting to initialise an integrator object with a state
  vector consisting of an array of integers will now raise an error:

  .. code-block:: ipython

     >>> ta = hy.taylor_adaptive(sys, [-1, 0])
     [...]
     TypeError: __init__(): incompatible constructor arguments.

  The solution here is to pass the initial state as an array of floats
  instead, i.e., ``[-1., 0.]``. Similarly:

  .. code-block:: ipython

     >>> ta.propagate_until(10)
     [...]
     TypeError: propagate_until(): incompatible function arguments.

  Again, the problem here is that a floating-point value is expected by
  the ``propagate_until()`` method, but an integral value was passed instead.
  The solution is to use ``propagate_until(10.)`` instead.
  In a similar fashion, if your code
  is raising :py:exc:`TypeError` exceptions with heyoka.py >=0.19,
  the solution is to ensure that values of the correct
  type are passed to the heyoka.py API (especially whenever floating-point arguments
  are expected).

.. _bchanges_0_16_0:

0.16.0
------

- The tuple returned by the ``propagate_for/until()`` methods
  in a scalar integrator has now 5 elements, rather than 4.
  The new return value at index 4 is the continuous output
  function object. This change can break code which assumes
  that the tuple returned by the ``propagate_for/until()`` functions
  has a size of 4, such as:

  .. code-block:: python

     r0, r1, r2, r3 = ta.propagate_until(...)

  The fix should be straightforward in most cases, e.g.:

  .. code-block:: python

     r0, r1, r2, r3, r4 = ta.propagate_until(...)

  Similarly, the ``propagate_for/until()`` functions in a batch integrator,
  which previously returned nothing, now return the continuous output
  function object (if requested).

.. _bchanges_0_10_0:

0.10.0
------

- The callback that can (optionally) be passed to
  the ``propagate_*()`` methods must now return
  a ``bool`` indicating whether the integration should
  continue or not. The callback used to return ``None``.

.. _bchanges_0_8_0:

0.8.0
-----

- An ``int`` argument has been appended to the signature of
  the events' callbacks. This new argument represents the sign
  of the derivative of the event equation at the event trigger
  time, and its value will be -1 for negative derivative,
  1 for positive derivative and 0 for zero derivative.
