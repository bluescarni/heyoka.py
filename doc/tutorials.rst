.. _tutorials:

Tutorials
=========

.. only:: html

  .. important::

     .. raw:: html

        <p>
        Most of these tutorials can be launched as online interactive notebooks
        thanks to the infrastructure provided by <a href="https://mybinder.org/">binder</a>.
        Look for the rocket icon <i class="fas fa-rocket"></i> on top of each page!
        </p>

  .. important::

     Some tutorials may use features not available yet in the latest stable release
     of heyoka.py, and thus might fail to execute correctly in the online interactive
     notebooks. Please refer to the :ref:`changelog <changelog>` for an overview of
     the features currently
     available only in the development version of heyoka.py.

Basics
-------

.. toctree::
  :maxdepth: 1

  tut_taylor_method
  notebooks/The expression system.ipynb
  notebooks/The adaptive integrator.ipynb
  notebooks/Customising the adaptive integrator.ipynb
  notebooks/ODEs with parameters.ipynb
  notebooks/Non-autonomous systems.ipynb
  notebooks/Dense output.ipynb
  notebooks/sympy_interop.ipynb
  notebooks/pickling.ipynb

Advanced features
-----------------

.. toctree::
  :maxdepth: 1

  notebooks/Batch mode overview
  notebooks/ensemble_mode.ipynb
  notebooks/ensemble_batch_perf.ipynb
  notebooks/parallel_mode.ipynb
  notebooks/ext_precision.ipynb

Event detection
---------------

.. toctree::
  :maxdepth: 1

  notebooks/Event detection
  notebooks/Sampling events
  notebooks/Poincaré sections
  notebooks/The Keplerian billiard
  notebooks/The two-fixed elliptic billiard
  notebooks/The wavy ramp
  notebooks/The Maxwell-Boltzmann distribution
  notebooks/ev_sensitivity
