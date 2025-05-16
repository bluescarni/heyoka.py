.. _api_model:

Models
======

.. currentmodule:: heyoka.model

Dynamics
--------

Functions
^^^^^^^^^

.. autosummary::
   :toctree: autosummary_generated

   fixed_centres
   pendulum

Coordinate transformations
--------------------------

Functions
^^^^^^^^^

.. autosummary::
   :toctree: autosummary_generated

   cart2geo
   geo2cart
   rot_fk5j2000_icrs
   rot_icrs_fk5j2000
   rot_itrs_icrs
   rot_icrs_itrs

Time transformations
--------------------

Functions
^^^^^^^^^

.. autosummary::
   :toctree: autosummary_generated

   delta_tdb_tt

Attributes
^^^^^^^^^^

.. autosummary::
   :toctree: autosummary_generated

   delta_tt_tai

Atmospheric models
------------------

Functions
^^^^^^^^^

.. autosummary::
   :toctree: autosummary_generated

   nrlmsise00_tn
   jb08_tn

Earth orientation and space weather
-----------------------------------

Functions
^^^^^^^^^

.. autosummary::
   :toctree: autosummary_generated

   era
   erap
   pm_x
   pm_xp
   pm_y
   pm_yp
   dX
   dXp
   dY
   dYp
   Ap_avg
   f107
   f107a_center81

Analytical theories and models
------------------------------

Functions
^^^^^^^^^

.. autosummary::
   :toctree: autosummary_generated

   iau2006
   egm2008_pot
   egm2008_acc
   vsop2013_cartesian_icrf
   vsop2013_cartesian
   vsop2013_elliptic
   get_vsop2013_mus

SGP4 propagation
----------------

Classes
^^^^^^^

.. autosummary::
   :toctree: autosummary_generated
   :template: custom-class-template.rst

   sgp4_propagator_dbl
   sgp4_propagator_flt

Functions
^^^^^^^^^

.. autosummary::
   :toctree: autosummary_generated

   sgp4
   gpe_is_deep_space
   sgp4_propagator
