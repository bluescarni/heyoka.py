.. _api_common_kwargs:

Common keyword arguments
========================

.. currentmodule:: heyoka

Several sets of keyword arguments appear multiple times in the API docs for heyoka.py's
classes and functions. This page documents these common sets.

.. _api_common_kwargs_fp_type:

The ``fp_type`` kwarg
---------------------

Several classes and functions in heyoka.py allow to select the floating-point type to be used in
the computations (as seen, for instance, in the :ref:`extended precision tutorial<ext_precision_tutorial>`).

The floating-point type is selected via the ``fp_type`` keyword argument, which must
be one of the floating-point types supported by heyoka.py. The types :py:class:`float`
(double precision) and :py:class:`numpy.single` (single precision) are always available,
while support for other floating-point types varies depending on the platform and on
the installed optional dependencies.

By default, ``fp_type`` is always
:py:class:`float` - that is, heyoka.py operates by default in double precision.

.. _api_common_kwargs_llvm:

LLVM kwargs
-----------

These are keyword arguments influencing just-in-time (JIT) compilation via LLVM.

- ``opt_level``: an integer in the [0, 3] range, representing the optimisation
  level to be employed during JIT compilation. Lower optimisation levels
  result in slower code and faster compilation times, and viceversa, higher optimisation
  levels result in faster code and slower compilation time. The default value is 3.
- ``slp_vectorize``: a boolean flag indicating whether or not to enable the LLVM
  `SLP vectorizer <https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer>`__.
  The SLP vectorizer can improve performance in some situations, but it results
  in longer compilation times. The default value is ``False``.
- ``fast_math``: a boolean flag indicating whether or not to enable optimisations
  which may improve floating-point performance at the expense of accuracy and/or strict conformance
  to the IEEE 754 standard. The default value is ``False``.
- ``code_model``: an enumerator of type :py:class:`code_model` representing the code model
  to be used for JIT compilation. The default code model is ``small``.

  .. versionadded:: 6.0.0
- ``parjit``: a boolean flag indicating that the JIT compilation process should be
  parallelised. This flag has an effect only when compact mode is enabled. The default value is ``False``.

  .. versionadded:: 6.0.0

  .. warning::

     Due to several LLVM issues around parallel JIT compilation, the ``parjit`` flag is currently **off**
     by default on Unix platforms and completely **disabled** on Windows. Turning on ``parjit`` on Unix
     platforms is considered safe but can very rarely result in a runtime exception being thrown.

     We expect the ``parjit`` feature to become more stable in later LLVM releases, at which point it will
     be turned on by default.

.. _api_common_kwargs_cfunc:

``cfunc`` kwargs
----------------
