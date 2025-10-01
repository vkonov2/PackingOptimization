# Packing Optimization Demo

This repository contains a small demonstration of solving weighted rectangle
packing problems with Google's OR-Tools CP-SAT solver. The workflow lives in
`rectangle_packing.py`, which builds and solves an instance before producing a
visualization (`rectangular_solution.png`).

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
