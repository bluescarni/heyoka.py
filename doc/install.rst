Installation
============

.. _installation_deps:

Dependencies
------------

heyoka.py has several Python and C++ dependencies. On the C++ side, heyoka.py depends on:

* the `heyoka C++ library <https://github.com/bluescarni/heyoka>`__,
  version 7.2.x (**mandatory**),
* the `Boost <https://www.boost.org/>`__ C++ libraries (**mandatory**),
* the `{fmt} <https://fmt.dev/latest/index.html>`__ library (**mandatory**),
* the `TBB <https://github.com/oneapi-src/oneTBB>`__ library (**mandatory**),
* the `mp++ <https://github.com/bluescarni/mppp>`__ library (**mandatory** if the
  heyoka C++ library was compiled with the ``HEYOKA_WITH_MPPP`` option enabled
  and the mp++ installation supports quadruple-precision computations via
  the :cpp:class:`mppp::real128` type and/or arbitrary-precision computations
  via the :cpp:class:`mppp::real` type - see the
  :ref:`heyoka <hy:installation>` and :ref:`mp++ <mppp:installation>` installation
  instructions).

On the Python side, heyoka.py requires at least Python 3.9 and depends on:

* `NumPy <https://numpy.org/>`__ (version >=2, **mandatory**),
* `cloudpickle <https://github.com/cloudpipe/cloudpickle>`__ (**mandatory**),
* `SymPy <https://www.sympy.org/en/index.html>`__ (version >=1.13.0) and `mpmath <https://mpmath.org/>`__
  (*optional*, for converting heyoka.py expressions to/from SymPy expressions),
* `skyfield <https://rhodesmill.org/skyfield/>`__ (*optional*, needed by the
  :ref:`SGP4 propagator <tut_sgp4_propagator>`).

The tested and supported CPU architectures at this time are x86-64, 64-bit ARM and 64-bit PowerPC.

Packages
--------

conda
^^^^^

heyoka.py is available via the `conda <https://docs.conda.io/en/latest/>`__
package manager for Linux, OSX and Windows
thanks to the infrastructure provided by `conda-forge <https://conda-forge.org/>`__.
In order to install heyoka.py via conda, you just need to add ``conda-forge``
to the channels, and then we can immediately install heyoka.py:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict
   $ conda install heyoka.py

The conda packages for heyoka.py are maintained by the core development team,
and they are regularly updated when new heyoka.py versions are released.

Please refer to the `conda documentation <https://docs.conda.io/en/latest/>`__
for instructions on how to setup and manage
your conda installation.

pip
^^^

heyoka.py packages are available on `PyPI <https://pypi.org/project/heyoka/>`__.
You can install them via ``pip``:

.. code-block:: console

   $ pip install heyoka

We currently offer prebuilt binary wheels for x86-64 Linux, along with a source distribution (sdist).
When using the sdist for installation, you are responsible for managing the installation of the C++ dependencies.

We plan to add in the near future prebuilt wheels for additional operating systems and CPU architectures.

.. note::

   heyoka.py relies on a stack of C++ dependencies which are bundled in the prebuilt binary wheels.
   There is a non-negligible chance of conflicts with other packages which might also depend on and bundle
   the same C++ libraries, which can lead to unpredictable runtime errors and hard-to-diagnose
   issues.

   We encourage users to install heyoka.py via conda rather than ``pip`` whenever possible.

FreeBSD
^^^^^^^

A community-supported FreeBSD port via `pkg <https://www.freebsd.org/doc/handbook/pkgng-intro.html>`__ is available for
heyoka.py. In order to install heyoka.py using pkg, execute the following command:

.. code-block:: console

   $ pkg install py39-heyoka

Installation from source
------------------------

.. warning::

   When installing from source, it is important to ensure that the compiler that was used
   to build heyoka.py's dependencies is ABI-compatible with the compiler that is being used
   to build heyoka.py. Mixing incompatible compilers will likely lead to hard-to-diagnose
   build and/or runtime issues.

heyoka.py is written in modern C++, and it requires a compiler able to understand
at least C++20. The library is regularly tested on
a continuous integration pipeline which currently includes several
compilers (GCC, Clang, MSVC) on several operating systems (Linux, OSX, Windows)
and several CPU architectures (x86-64, 64-bit ARM and 64-bit PowerPC).

In addition to the C++ dependencies enumerated :ref:`earlier <installation_deps>`,
installation from source requires also:

* `pybind11 <https://github.com/pybind/pybind11>`__ (version >= 2.10),
* `CMake <https://cmake.org/>`__, version 3.18 or later.

Note that heyoka.py makes use of the :ref:`NumPy C API <numpy:c-api>`
and thus NumPy must be installed **before** compiling heyoka.py from source.
The other Python dependencies need not to be installed at compilation time.

After making sure the dependencies are installed on your system, you can
download the heyoka.py source code from the
`GitHub release page <https://github.com/bluescarni/heyoka.py/releases>`__. Alternatively,
and if you like living on the bleeding edge, you can get the very latest
version of heyoka.py via ``git``:

.. code-block:: console

   $ git clone https://github.com/bluescarni/heyoka.py.git

We follow the usual PR-based development workflow, thus heyoka.py's ``main``
branch is normally kept in a working state.

After downloading and/or unpacking heyoka.py's
source code, go to heyoka.py's
source tree, create a ``build`` directory and ``cd`` into it. E.g.,
on a Unix-like system:

.. code-block:: console

   $ cd /path/to/heyoka.py
   $ mkdir build
   $ cd build

Once you are in the ``build`` directory, you must configure your build
using ``cmake``. There are various useful CMake variables you can set,
such as:

* ``CMAKE_BUILD_TYPE``: the build type (``Release``, ``Debug``, etc.),
  defaults to ``Release``.
* ``CMAKE_PREFIX_PATH``: additional paths that will be searched by CMake
  when looking for dependencies.
* ``HEYOKA_PY_INSTALL_PATH``: the path into which the heyoka.py module
  will be installed. If left empty (the default), heyoka.py will be installed
  in the global modules directory of your Python installation.
* ``HEYOKA_PY_ENABLE_IPO``: set this flag to ``ON`` to compile heyoka.py
  with link-time optimisations. Requires compiler support,
  defaults to ``OFF``.

Please consult `CMake's documentation <https://cmake.org/cmake/help/latest/>`_
for more details about CMake's variables and options.

The ``HEYOKA_PY_INSTALL_PATH`` option is particularly important. If you
want to install heyoka.py locally instead of globally (which is in general
a good idea), you can set this variable to the output of
``python -m site --user-site``.

After configuring the build with CMake, we can then proceed to actually
building heyoka.py:

.. code-block:: console

   $ cmake --build .

Finally, we can install heyoka.py with the command:

.. code-block:: console

   $ cmake  --build . --target install

Verifying the installation
--------------------------

You can verify that heyoka.py was successfully compiled and
installed by running the test suite with the following command:

.. code-block:: bash

   $ python -c "import heyoka; heyoka.test.run_test_suite();"

If this command executes without any error, then
your heyoka.py installation is ready for use.

Note that the test suite makes use of the optional dependencies, thus if they
are not installed some tests will fail.

Getting help
------------

If you run into troubles installing heyoka.py, please do not hesitate
to contact us by opening an issue report on `github <https://github.com/bluescarni/heyoka.py/issues>`__.
