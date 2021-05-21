heyoka.py
=========

[![Build Status](https://img.shields.io/circleci/project/github/bluescarni/heyoka.py/main.svg?style=for-the-badge)](https://circleci.com/gh/bluescarni/heyoka.py)
[![Build Status](https://img.shields.io/appveyor/ci/bluescarni/heyoka-py/main.svg?logo=appveyor&style=for-the-badge)](https://ci.appveyor.com/project/bluescarni/heyoka-py)
[![Build Status](https://img.shields.io/github/workflow/status/bluescarni/heyoka.py/GitHub%20CI?style=for-the-badge)](https://github.com/bluescarni/heyoka.py/actions?query=workflow%3A%22GitHub+CI%22)

[![Anaconda-Server Badge](https://img.shields.io/conda/vn/conda-forge/heyoka.py.svg?style=for-the-badge)](https://anaconda.org/conda-forge/heyoka.py)

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/bluescarni/heyoka.py">
    <img src="doc/images/white_logo.png" alt="Logo" width="280">
  </a>
  <p align="center">
    Modern Taylor's method via just-in-time compilation
    <br />
    <a href="https://bluescarni.github.io/heyoka.py/index.html"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/bluescarni/heyoka.py/issues/new/choose">Report bug</a>
    ·
    <a href="https://github.com/bluescarni/heyoka.py/issues/new/choose">Request feature</a>
    ·
    <a href="https://github.com/bluescarni/heyoka.py/discussions">Discuss</a>
  </p>
</p>

> The [heyókȟa](https://en.wikipedia.org/wiki/Heyoka>) [...] is a kind of
> sacred clown in the culture of the Sioux (Lakota and Dakota people)
> of the Great Plains of North America. The heyoka is a contrarian, jester,
> and satirist, who speaks, moves and reacts in an opposite fashion to the
> people around them.

heyoka.py is a Python library for the integration of ordinary differential equations
(ODEs) via Taylor's method. Notable features include:

* support for both double-precision and extended-precision floating-point types
  (80-bit and 128-bit),
* the ability to maintain machine precision accuracy over
  tens of billions of timesteps,
* high-precision zero-cost dense output,
* accurate and reliable event detection,
* batch mode integration to harness the power of modern
  [SIMD](https://en.wikipedia.org/wiki/SIMD) instruction sets,
* a high-performance implementation of Taylor's method based
  on automatic differentiation techniques and aggressive just-in-time
  compilation via [LLVM](https://llvm.org/).

heyoka.py is based on the [heyoka C++ library](https://github.com/bluescarni/heyoka).

If you are using heyoka.py as part of your research, teaching, or other activities, we would be grateful if you could star
the repository and/or cite our work. For citation purposes, you can use the following BibTex entry, which refers
to the heyoka.py paper ([arXiv preprint](https://arxiv.org/abs/2105.00800)):

```bibtex
@article{10.1093/mnras/stab1032,
    author = {Biscani, Francesco and Izzo, Dario},
    title = "{Revisiting high-order Taylor methods for astrodynamics and celestial mechanics}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {504},
    number = {2},
    pages = {2614-2628},
    year = {2021},
    month = {04},
    issn = {0035-8711},
    doi = {10.1093/mnras/stab1032},
    url = {https://doi.org/10.1093/mnras/stab1032},
    eprint = {https://academic.oup.com/mnras/article-pdf/504/2/2614/37750349/stab1032.pdf}
}
```

Documentation
-------------

The full documentation can be found [here](https://bluescarni.github.io/heyoka.py/).

Authors
-------

* Francesco Biscani (Max Planck Institute for Astronomy)
* Dario Izzo (European Space Agency)

License
-------

heyoka.py is released under the [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)
license.
