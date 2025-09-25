# Packing Optimization Demo

This repository contains a small demonstration of solving a weighted rectangle
packing problem with Google's OR-Tools CP-SAT solver. A single script,
`rectangle_packing_demo.py`, builds and solves an instance, then produces a
visualization saved to `solution.png`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python rectangle_packing_demo.py
```

The same dependencies are also declared in `pyproject.toml`, so you can install
the project as a package instead:

```bash
pip install .
rectangle-packing-demo
```

The solver will print the selected rectangles, their positions, and store a
plot of the layout in `solution.png`. Coordinates are enumerated on an integer
grid; when you allow a non-zero overlap percentage the model rounds the
permitted area up to the nearest grid cell so that fractional allowances such
as 10% remain achievable in the discrete setting.
