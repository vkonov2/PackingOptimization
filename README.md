# Packing Optimization Demo

This repository contains a small demonstration of solving weighted rectangle
packing problems with Google's OR-Tools CP-SAT solver. The workflow lives in
`rectangle_packing.py`, which builds an inventory automatically, solves an
instance, and produces a visualization (`rectangular_solution.png`).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python rectangle_packing.py
```

The same dependencies are also declared in `pyproject.toml`, so you can install
the project as a package instead:

```bash
pip install .
rectangle-packing
```

The solver prints the selected pieces, their top-left coordinates, and stores a
plot in `rectangular_solution.png`.

## Inventory generation

The demo inventory is derived from a base rectangle (2×3 with weight 1) and two
growth percentages that control how much the dimensions expand along the axes.
Every combination of horizontal and vertical expansions that can fit inside the
10×5.8 container in either orientation is enumerated. For each feasible size the
script adds enough copies so that the total area of that size exceeds the
container area, ensuring ample supply for the optimizer. Rotated versions are
included whenever they fit, which allows the CP-SAT model to choose between
upright and 90° layouts for each rectangle type.
