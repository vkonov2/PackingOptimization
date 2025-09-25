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
plot of the layout in `solution.png`.
