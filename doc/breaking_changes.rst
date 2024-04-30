.. _breaking_changes:

Breaking changes
================

.. currentmodule:: heyoka

.. _bchanges_5_0_0:

5.0.0
-----

In heyoka.py 5, the expression system has undergone several changes.

Removal of automatic simplifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most automatic simplifications and normalisations in the expression
system have been removed due to their performance impact on large and highly recursive
computational graphs.

While the removal of these automatic simplifications has not resulted in API breaks, the best
practices for creating and manipulating expressions efficiently have changed. Please see the updated
:ref:`tutorials <ex_sys_tutorials>` for more information.

Removed/changed functions
~~~~~~~~~~~~~~~~~~~~~~~~~

As a consequence of the removal of automatic simplifications, several functions have been removed
as they are now obsolete. These include:

- the ``fix()``, ``fix_nn()`` and ``unfix()`` functions. These functions were used
  to temporarily disable automatic simplifications, and thus they serve no purpose
  any more;
- the ``normalise()`` function. This function was used to manually trigger the application
  of automatic simplifications.

Additionally, the ``normalise = True`` flag in the :func:`subs()` function has also been removed.

Note that the removed functions were used mostly internally and the impact on user code is expected
to be minimal.

.. _bchanges_4_0_0:

4.0.0
-----

heyoka.py 4 includes several backwards-incompatible changes.

Changes to compiled functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function to create :ref:`compiled functions <cfunc_tut>` has been renamed from
``make_cfunc()`` to simply ``cfunc()``.

Compiled functions have also gained the ability to use multiple
threads of execution during batched evaluations. As a consequence, compiled functions
now require contiguous NumPy arrays to be passed as input/output arguments (whereas
in previous versions compiled functions would work also with non-contiguous
arrays). The NumPy function :py:func:`numpy.ascontiguousarray()` can be used to turn
non-contiguous arrays into contiguous arrays.

Finally, compiled functions are now stricter with respect to type conversions: if a NumPy
array with the wrong datatype is passed as an input/output argument, an error will be raised
(whereas previously heyoka.py would convert the array to the correct datatype on-the-fly).
The NumPy method :py:meth:`numpy.ndarray.astype()` can be used for datatype conversions.

A more explicit API
~~~~~~~~~~~~~~~~~~~

Several functions and classes have been changed to explicitly require
the user to pass a list of variables in input. The previous behaviour, where
heyoka.py would try to automatically infer a list of variables from other
input arguments, turned out to be in practice confusing and a source of bugs.

The affected APIs include:

- :ref:`compiled functions <cfunc_tut>`, which now require the list of input
  variables to be always supplied by the user;
- :func:`~heyoka.diff_tensors()`, which now requires the differentiation
  arguments to be always provided by the user.

The tutorials and the documentation have been updated accordingly.

Changes to :py:func:`~heyoka.make_vars()`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:func:`~heyoka.make_vars()` function now returns a single expression (rather than a list of expressions)
if a single argument is passed in input. This means that code such as

.. code-block:: python

    x, = make_vars("x")
    y = make_vars("y")[0]

needs to be rewritten like this:

.. code-block:: python

    x = make_vars("x")
    y = make_vars("y")

Terminal events callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~

The second argument in the signature of callbacks for terminal events, a ``bool`` conventionally
called ``mr``, has been removed. This flag was meant to signal the possibility of multiple roots
in the event function within the cooldown period, but it never worked as intended and
it has thus been dropped.

Adapting existing code for this API change is straightforward: you just have to remove the second argument
from the signature of a terminal event callback.

Step callbacks and ``propagate_*()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The step callbacks that can (optionally) be passed to the ``propagate_*()`` methods of the
adaptive integrators are now part of the return value. Specifically:

- for the scalar ``propagate_for()`` and ``propagate_until()`` methods, the step callback is
  the sixth element of the return tuple, while for the batch variants the step callback
  is the second element of the return tuple;
- for the scalar ``propagate_grid()`` method, the step callback is the fifth element of the return
  tuple, while for the batch variant the step callback is the first element of the return
  tuple.

:ref:`The ensemble propagation <ensemble_prop>` functions have been modified in an analogous way.

Adapting existing code for the new API should be straightforward. In most cases it should be just
a matter of:

- adapting unpacking declarations to account for the new element in the return tuple
  of scalar propagations,
- adjusting the indexing into the return tuple when fetching a specific element,
- accounting for the fact that batch propagations now return a tuple of two elements
  rather than a single value.

Changes to ``propagate_grid()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``propagate_grid()`` methods of the adaptive integrators now require the first element of the
time grid to be equal to the current integrator time. Previously, in case of a difference between the
integrator time and the first grid point, heyoka.py would propagate the state of the system up to the
first grid point with ``propagate_until()``.

If you want to recover the previous behaviour, you will have to invoke manually ``propagate_until(grid[0])``
before invoking ``propagate_grid()``.

.. _bchanges_1_0_0:

1.0.0
-----

- The VSOP2013 functions have been moved into the
  ``model`` submodule. The semantics of the functions
  have not changed.
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
