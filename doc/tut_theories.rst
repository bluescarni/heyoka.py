Analytical theories and models
==============================

heyoka.py provides builtin implementations of several established analytical and
semi-analytical theories describing astronomical and geophysical phenomena.

These theories and models are implemented and exposed directly in the expression
system as differentiable functions. Differentiability is a key property: it is
both required by Taylor's method for ODE integration, and
it also enables the formulation of variational equations (which, in turn, underpin
gradient-based inverse problems and optimisation tasks).

.. toctree::
   :maxdepth: 1

   notebooks/vsop2013
   notebooks/elp2000
   notebooks/iau2006
   notebooks/egm2008
