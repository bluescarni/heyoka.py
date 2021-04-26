.. _breaking_changes:

Breaking changes
================

.. _bchanges_0_8_0:

0.8.0
-----

- An ``int`` argument has been appended to the signature of
  the events' callbacks. This new argument represents the sign
  of the derivative of the event equation at the event trigger
  time, and its value will be -1 for negative derivative,
  1 for positive derivative and 0 for zero derivative.
