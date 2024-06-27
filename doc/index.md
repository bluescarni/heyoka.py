# heyoka.py

> The [heyókȟa](https://en.wikipedia.org/wiki/Heyoka) \[\...\] is a kind
> of sacred clown in the culture of the Sioux (Lakota and Dakota people)
> of the Great Plains of North America. The heyoka is a contrarian,
> jester, and satirist, who speaks, moves and reacts in an opposite
> fashion to the people around them.

heyoka.py is a Python library for the integration of ordinary differential equations
(ODEs) via Taylor's method, based on automatic differentiation techniques and aggressive just-in-time
compilation via [LLVM](https://llvm.org/). Notable features include:

- support for single-precision, double-precision, extended-precision (80-bit and 128-bit),
  and arbitrary-precision floating-point types,
- high-precision zero-cost dense output,
- accurate and reliable event detection,
- builtin support for analytical mechanics - bring your own Lagrangians/Hamiltonians
  and let heyoka.py formulate and solve the equations of motion,
- builtin support for high-order variational equations - compute not only the solution,
  but also its partial derivatives,
- builtin support for machine learning applications via neural network models,
- the ability to maintain machine precision accuracy over
  tens of billions of timesteps,
- batch mode integration to harness the power of modern
  [SIMD](https://en.wikipedia.org/wiki/SIMD) instruction sets
  (including AVX/AVX2/AVX-512/Neon/VSX),
- ensemble simulations and automatic parallelisation,
- interoperability with [SymPy](https://www.sympy.org/en/index.html).

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

heyoka.py's novel event detection system is described in the following paper ([arXiv preprint](https://arxiv.org/abs/2204.09948)):

```bibtex
@article{10.1093/mnras/stac1092,
    author = {Biscani, Francesco and Izzo, Dario},
    title = "{Reliable event detection for Taylor methods in astrodynamics}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {513},
    number = {4},
    pages = {4833-4844},
    year = {2022},
    month = {04},
    issn = {0035-8711},
    doi = {10.1093/mnras/stac1092},
    url = {https://doi.org/10.1093/mnras/stac1092},
    eprint = {https://academic.oup.com/mnras/article-pdf/513/4/4833/43796551/stac1092.pdf}
}
```

heyoka.py is released under the [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/FAQ/) license.
The authors are Francesco Biscani and Dario Izzo (European Space Agency).

```{toctree}
:maxdepth: 1
:caption: Main
 
install
changelog
breaking_changes
benchmarks
acknowledgement

```

```{toctree}
:maxdepth: 2
:caption: Tutorials
 
basic_tutorials
ex_sys_tutorials
advanced_tutorials
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples_astro
examples_event
examples_ml
examples_var_ode_sys
examples_others
```

```{toctree}
:maxdepth: 2
:caption: API reference

api_exsys
api_integrators
api_var_ode_sys
api_lagham
api_model
```
