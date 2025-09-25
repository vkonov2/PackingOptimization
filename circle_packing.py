"""Circle packing demo using OR-Tools CP-SAT with limited overlaps.

This module mirrors the rectangle packing example but targets circular
pieces.  Each circle has a weight and can either be omitted or placed inside
an axis-aligned rectangular container.  Circles are allowed to overlap up to a
fraction of their diameters, letting the solver trade controlled overlaps for
higher retained weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Rectangle
from ortools.sat.python import cp_model


@dataclass(frozen=True)
class WeightedCircle:
    """Description of an input circle.

    Attributes:
        name: Identifier used in reports and the visualization.
        radius: Radius of the circle measured in the same units as the
            container.
        weight: Contribution to the objective when this circle is selected.
    """

    name: str
    radius: float
    weight: float

    @property
    def diameter(self) -> float:
        return self.radius * 2


class CirclePackingModel:
    """Solver wrapper that picks weighted circles and their locations."""

    def __init__(
        self,
        container_size: Tuple[float, float],
        circles: Sequence[WeightedCircle],
        grid_step: float = 1.5,
        max_overlap_fraction: float = 0.1,
        coordinate_scale: int = 10,
    ) -> None:
        if grid_step <= 0:
            raise ValueError("grid_step must be positive")
        if not 0 <= max_overlap_fraction <= 1:
            raise ValueError("max_overlap_fraction must be between 0 and 1")
        if coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be positive")

        self.scale = coordinate_scale
        self.container_width = int(round(container_size[0] * self.scale))
        self.container_height = int(round(container_size[1] * self.scale))
        self.circles: List[WeightedCircle] = list(circles)
        self.grid_step = grid_step
        self.max_overlap_fraction = max_overlap_fraction

        self.scaled_radii: List[int] = [
            int(round(circle.radius * self.scale)) for circle in self.circles
        ]

        self.model = cp_model.CpModel()
        self.placement_vars: List[List[cp_model.BoolVar]] = []
        self.use_circle: List[cp_model.BoolVar] = []
        self.candidate_positions: List[List[Tuple[int, int]]] = []

        self._build_variables()
        self._add_overlap_constraints()
        self._set_objective()

    # ------------------------------------------------------------------
    # Model construction helpers
    # ------------------------------------------------------------------
    def _build_variables(self) -> None:
        step = int(round(self.grid_step * self.scale))
        if step <= 0:
            raise ValueError("grid_step is too small after scaling")

        for idx, (circle, radius) in enumerate(zip(self.circles, self.scaled_radii)):
            x_positions = self._enumerate_axis_positions(
                radius, self.container_width - radius, step
            )
            y_positions = self._enumerate_axis_positions(
                radius, self.container_height - radius, step
            )

            candidates: List[Tuple[int, int]] = [
                (x, y) for x in x_positions for y in y_positions
            ]

            if not candidates:
                # Circle cannot fit anywhere inside the container.
                candidates = []

            position_vars: List[cp_model.BoolVar] = []
            for pos_id, (x, y) in enumerate(candidates):
                var = self.model.NewBoolVar(f"place_{circle.name}_{pos_id}")
                position_vars.append(var)

            self.placement_vars.append(position_vars)
            self.candidate_positions.append(candidates)

            use_var = self.model.NewBoolVar(f"use_{circle.name}")
            if not candidates:
                self.model.Add(use_var == 0)

            self.model.Add(sum(position_vars) == use_var)
            self.use_circle.append(use_var)

    def _enumerate_axis_positions(
        self, start: int, end: int, step: int
    ) -> List[int]:
        if start > end:
            return []

        positions = list(range(start, end + 1, step))
        if not positions or positions[-1] != end:
            positions.append(end)
        return positions

    def _allowable_penetration(self, idx_a: int, idx_b: int) -> int:
        radius_a = self.scaled_radii[idx_a]
        radius_b = self.scaled_radii[idx_b]
        allowed = self.max_overlap_fraction * min(radius_a, radius_b) * 2
        return int(math.floor(allowed + 1e-9))

    def _required_distance(self, idx_a: int, idx_b: int) -> int:
        return max(
            0,
            self.scaled_radii[idx_a]
            + self.scaled_radii[idx_b]
            - self._allowable_penetration(idx_a, idx_b),
        )

    def _add_overlap_constraints(self) -> None:
        for i, j in combinations(range(len(self.circles)), 2):
            required = self._required_distance(i, j)
            if required <= 0:
                continue

            placements_i = self.placement_vars[i]
            placements_j = self.placement_vars[j]
            positions_i = self.candidate_positions[i]
            positions_j = self.candidate_positions[j]

            required_sq = required * required

            for var_i, (x_i, y_i) in zip(placements_i, positions_i):
                for var_j, (x_j, y_j) in zip(placements_j, positions_j):
                    dx = x_i - x_j
                    dy = y_i - y_j
                    if abs(dx) >= required or abs(dy) >= required:
                        continue
                    if dx * dx + dy * dy >= required_sq:
                        continue
                    self.model.Add(var_i + var_j <= 1)

    def _set_objective(self) -> None:
        weighted_usage = [
            int(round(circle.weight * 1000)) * use
            for circle, use in zip(self.circles, self.use_circle)
        ]
        self.model.Maximize(sum(weighted_usage))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self, time_limit: float = 30.0) -> Tuple[float, Dict[str, Tuple[float, float]]]:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 8

        status = solver.Solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("No feasible solution found")

        placements: Dict[str, Tuple[float, float]] = {}
        total_weight = 0.0

        for circle, use_var, candidates, position_vars in zip(
            self.circles, self.use_circle, self.candidate_positions, self.placement_vars
        ):
            if math.isclose(solver.Value(use_var), 1.0):
                total_weight += circle.weight
                for coords, placement_var in zip(candidates, position_vars):
                    if math.isclose(solver.Value(placement_var), 1.0):
                        x, y = coords
                        placements[circle.name] = (x / self.scale, y / self.scale)
                        break

        return total_weight, placements


def visualize_solution(
    container_size: Tuple[float, float],
    circles: Sequence[WeightedCircle],
    centers: Dict[str, Tuple[float, float]],
    output_path: str = "circle_solution.png",
) -> None:
    used = set(centers)
    fig, (ax_inventory, ax_layout) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    _draw_circle_inventory(ax_inventory, circles, used)
    _draw_circle_layout(ax_layout, container_size, circles, centers)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def _circle_color(circle: WeightedCircle) -> str:
    return "#1f77b4" if circle.weight >= 2 else "#ff7f0e"


def _draw_circle_inventory(
    ax: plt.Axes, circles: Sequence[WeightedCircle], used: set[str]
) -> None:
    ax.set_title("Инвентарь окружностей")
    ax.axis("off")

    if not circles:
        return

    sorted_circles = sorted(circles, key=lambda c: (c.radius, c.weight), reverse=True)
    cols = max(5, int(math.ceil(math.sqrt(len(sorted_circles)))))
    rows = int(math.ceil(len(sorted_circles) / cols))

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")

    max_radius = max(circle.radius for circle in sorted_circles)
    radius_scale = 0.45 / max_radius if max_radius > 0 else 0.45

    for idx, circle in enumerate(sorted_circles):
        row = idx // cols
        col = idx % cols
        cx = col + 0.5
        cy = rows - row - 0.5
        patch = CirclePatch((cx, cy), circle.radius * radius_scale)
        patch.set_facecolor(_circle_color(circle))
        patch.set_alpha(0.85 if circle.name in used else 0.25)
        patch.set_edgecolor("#333333")
        ax.add_patch(patch)
        ax.text(
            cx,
            cy - 0.55,
            f"{circle.name}\n r={circle.radius}, w={circle.weight}",
            ha="center",
            va="top",
            fontsize=8,
        )


def _draw_circle_layout(
    ax: plt.Axes,
    container_size: Tuple[float, float],
    circles: Sequence[WeightedCircle],
    centers: Dict[str, Tuple[float, float]],
) -> None:
    width, height = container_size
    ax.set_title("Размещение в контейнере")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")

    container_rect = Rectangle((0, 0), width, height, fill=False, lw=2)
    ax.add_patch(container_rect)

    circle_lookup = {circle.name: circle for circle in circles}
    for name, (cx, cy) in centers.items():
        circle = circle_lookup[name]
        patch = CirclePatch((cx, cy), circle.radius)
        patch.set_facecolor(_circle_color(circle))
        patch.set_alpha(0.6)
        patch.set_edgecolor("#444444")
        ax.add_patch(patch)
        ax.text(
            cx,
            cy,
            name,
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )


def report_overlap_statistics(
    circles: Sequence[WeightedCircle],
    centers: Dict[str, Tuple[float, float]],
    max_fraction: float,
) -> None:
    lookup = {circle.name: circle for circle in circles}
    used = sorted(centers)

    print("Контроль перекрытия окружностей (порог = {:.0%}):".format(max_fraction))
    if len(used) < 2:
        print("  Перекрытия отсутствуют")
        return

    violation = False
    for name_a, name_b in combinations(used, 2):
        circle_a = lookup[name_a]
        circle_b = lookup[name_b]
        center_a = centers[name_a]
        center_b = centers[name_b]

        dx = center_a[0] - center_b[0]
        dy = center_a[1] - center_b[1]
        distance = math.hypot(dx, dy)

        allowed_intrusion = max_fraction * min(circle_a.diameter, circle_b.diameter)
        required_distance = circle_a.radius + circle_b.radius - allowed_intrusion
        required_distance = max(required_distance, 0.0)
        overlap_depth = circle_a.radius + circle_b.radius - distance

        status = "OK"
        if overlap_depth > allowed_intrusion + 1e-9:
            status = "VIOLATION"
            violation = True

        print(
            f"  {name_a} ↔ {name_b}: дистанция={distance:.2f}, "
            f"допустимо≥{required_distance:.2f}, проникновение={max(overlap_depth, 0):.2f} → {status}"
        )

    if not violation:
        print("  Все перекрытия укладываются в пределы")


def main() -> None:
    container = (10.0, 6.0)
    large_circles = [
        WeightedCircle(f"C{i:02d}", radius=1.0, weight=2.0) for i in range(10)
    ]
    small_circles = [
        WeightedCircle(f"C{i:02d}", radius=0.6, weight=1.0)
        for i in range(10, 30)
    ]
    circles = large_circles + small_circles

    model = CirclePackingModel(
        container_size=container,
        circles=circles,
        grid_step=1.5,
        max_overlap_fraction=0.1,
        coordinate_scale=20,
    )

    weight, placements = model.solve()
    print(f"Итоговый вес: {weight:.1f}")
    print("Использованные окружности:")
    for name in sorted(placements):
        cx, cy = placements[name]
        print(f"  {name}: центр=({cx:.2f}, {cy:.2f})")

    report_overlap_statistics(circles, placements, model.max_overlap_fraction)
    visualize_solution(container, circles, placements)


if __name__ == "__main__":
    main()

