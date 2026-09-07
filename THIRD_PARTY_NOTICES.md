# Attribution and external materials

Project-authored code is licensed under GPL-3.0-or-later; see LICENSE.
It was extracted from the authors' Helsinki Asteroid Challenge research
implementation; `provenance/source_extraction.json` records the source correspondence.

Challenge observations, reference meshes, videos, and the organizers'
evaluation source are obtained separately from the official open data release.
They are not covered by this project's license or included in this repository.
The input manifest identifies the exact real-intensity tables used here.

Official challenge page:
https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/

The module named `damit_model.py` contains this project's local Python
implementation of the spherical grid and harmonic representation described
in the cited lightcurve-inversion literature. No external DAMIT C source,
Numerical Recipes source, or compiled DAMIT executable is distributed or
required. Mathematical references are given in `docs/method.tex`.

NumPy, SciPy, JAX, and their dependencies retain their own licenses. The
optional MATLAB evaluation uses the organizers' source and requires a
separately licensed MATLAB installation and toolboxes.
