# Packing Optimization Demo

This repository contains small demonstrations of solving weighted packing
problems with Google's OR-Tools CP-SAT solver. The rectangle workflow lives in
`rectangle_packing.py`, while the analogous circle example is implemented in
`circle_packing.py`. Each script builds and solves an instance, then produces a
visualization (`rectangular_solution.png` or `circle_solution.png`).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python rectangle_packing.py
python circle_packing.py
```

The same dependencies are also declared in `pyproject.toml`, so you can install
the project as a package instead:

```bash
pip install .
rectangle-packing
circle-packing
```

The rectangle solver prints the selected pieces, their top-left coordinates,
and stores a plot in `rectangular_solution.png`. The circle solver reports the
retained discs, their centers, and writes `circle_solution.png`. Coordinates are
enumerated on an integer grid; when you allow a non-zero overlap percentage the
models round the permitted area up to the nearest grid cell so that fractional
allowances such as 10% remain achievable in the discrete setting.
