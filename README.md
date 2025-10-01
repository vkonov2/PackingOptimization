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

The demo inventory now follows the fixed list of rectangles requested in the
task specification.  The vertical dimension is listed first below; each entry is
added with enough copies so that its total area exceeds the area of the
10×6 container.

```
3.0×2.0  (w=1)   3.0×3.8  (w=2)   3.0×5.6  (w=3)   3.0×7.4  (w=4)   3.0×9.2  (w=5)
5.85×2.0 (w=2)   5.85×3.8 (w=4)   5.85×5.6 (w=6)   5.85×7.4 (w=8)   5.85×9.2 (w=10)
2.0×3.0  (w=1)   2.0×5.85 (w=2)   2.0×8.7  (w=3)
3.8×3.0  (w=2)   3.8×5.85 (w=4)   3.8×8.7  (w=6)
5.6×3.0  (w=3)   5.6×5.85 (w=6)   5.6×8.7  (w=9)
```

The visualization panel groups rectangles by size, highlights how many copies
were used in the optimal solution, and leaves the unused inventory unshaded so
the layout is easy to inspect.

## Gradio application

For an interactive setup that lets you tweak the rectangle inventory, launch the
Gradio interface:

```bash
python rectangle_packing_gradio.py
```

A local web server will open in your browser.  Use the editable table to add or
remove rectangle types, then press **«упаковать»** to solve the packing problem
and display the resulting layout.
