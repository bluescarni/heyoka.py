EOP and space weather data
==========================

heyoka.py provides builtin support for Earth orientation parameters (EOP) and
space weather (SW) data. These are empirical datasets that are essential for the
high-fidelity dynamical modelling of Earth-orbiting satellites. EOP data underpins
accurate transformations between terrestrial and celestial reference frames,
while SW data drives the atmospheric density variations that govern orbital
drag.

.. toctree::
  :maxdepth: 1

  notebooks/eop_data
  notebooks/sw_data
