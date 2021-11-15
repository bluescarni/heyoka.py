.. _breaking_changes:

Breaking changes
================

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
