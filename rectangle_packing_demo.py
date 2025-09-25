import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ortools.sat.python import cp_model


@dataclass(frozen=True)
class SmallRectangle:
    name: str
    width: int
    height: int
    weight: float

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class Placement:
    rect_index: int
    x: int
    y: int


class WeightedPackingModel:
    def __init__(
        self,
        container_size: Tuple[int, int],
        rectangles: List[SmallRectangle],
        grid_step: int = 1,
        max_overlap_fraction: float = 0.1,
    ) -> None:
        if grid_step <= 0:
            raise ValueError("grid_step must be positive")
        if not 0 <= max_overlap_fraction <= 1:
            raise ValueError("max_overlap_fraction must be between 0 and 1")

        self.container_width, self.container_height = container_size
        self.rectangles = rectangles
        self.grid_step = grid_step
        self.max_overlap_fraction = max_overlap_fraction

        self.model = cp_model.CpModel()
        self._build_variables()
        self._add_selection_constraints()
        self._add_overlap_constraints()
        self._set_objective()

    def _build_variables(self) -> None:
        self.placements_by_rect: List[List[Placement]] = []
        self.x: Dict[Tuple[int, int], cp_model.IntVar] = {}
        self.use_rect: List[cp_model.IntVar] = []

        for idx, rect in enumerate(self.rectangles):
            placements: List[Placement] = []
            max_x = self.container_width - rect.width
            max_y = self.container_height - rect.height
            if max_x >= 0 and max_y >= 0:
                x_positions = list(range(0, max_x + 1, self.grid_step))
                if x_positions[-1] != max_x:
                    x_positions.append(max_x)
                y_positions = list(range(0, max_y + 1, self.grid_step))
                if y_positions[-1] != max_y:
                    y_positions.append(max_y)

                for x in x_positions:
                    for y in y_positions:
                        placement = Placement(idx, x, y)
                        placements.append(placement)
                        var = self.model.NewBoolVar(f"rect_{rect.name}_at_{x}_{y}")
                        self.x[(idx, len(placements) - 1)] = var
            self.placements_by_rect.append(placements)
            use_var = self.model.NewBoolVar(f"use_{rect.name}")
            self.use_rect.append(use_var)

    def _add_selection_constraints(self) -> None:
        for idx, placements in enumerate(self.placements_by_rect):
            selection_vars = [self.x[(idx, p_idx)] for p_idx in range(len(placements))]
            if selection_vars:
                self.model.Add(sum(selection_vars) == self.use_rect[idx])
            else:
                # No feasible placement, force rectangle to be unused.
                self.model.Add(self.use_rect[idx] == 0)

    def _round_up_to_grid(self, raw_depth: float) -> int:
        """Return the smallest grid-aligned depth not below the requested amount."""

        if raw_depth <= 0:
            return 0
        cells = math.ceil(raw_depth / self.grid_step - 1e-9)
        return cells * self.grid_step

    def _allowed_axis_overlap(self, i: int, j: int, axis: str) -> int:
        """Maximum overlap depth along the chosen axis for the pair."""

        if axis == "x":
            dims = (self.rectangles[i].width, self.rectangles[j].width)
        elif axis == "y":
            dims = (self.rectangles[i].height, self.rectangles[j].height)
        else:
            raise ValueError("axis must be 'x' or 'y'")

        allowances = [
            min(dim, self._round_up_to_grid(self.max_overlap_fraction * dim))
            for dim in dims
        ]
        return min(allowances)

    def _add_overlap_constraints(self) -> None:
        for i in range(len(self.rectangles)):
            for j in range(i + 1, len(self.rectangles)):
                allowed_x = self._allowed_axis_overlap(i, j, "x")
                allowed_y = self._allowed_axis_overlap(i, j, "y")
                for p_idx, placement_i in enumerate(self.placements_by_rect[i]):
                    for q_idx, placement_j in enumerate(self.placements_by_rect[j]):
                        rect_i = self.rectangles[i]
                        rect_j = self.rectangles[j]

                        if placement_i.x + rect_i.width <= placement_j.x:
                            continue
                        if placement_j.x + rect_j.width <= placement_i.x:
                            continue
                        if placement_i.y + rect_i.height <= placement_j.y:
                            continue
                        if placement_j.y + rect_j.height <= placement_i.y:
                            continue
                        x_overlap = max(
                            0,
                            min(placement_i.x + rect_i.width, placement_j.x + rect_j.width)
                            - max(placement_i.x, placement_j.x),
                        )
                        y_overlap = max(
                            0,
                            min(placement_i.y + rect_i.height, placement_j.y + rect_j.height)
                            - max(placement_i.y, placement_j.y),
                        )

                        disallowed = False
                        if allowed_x <= 0 and x_overlap > 0:
                            disallowed = True
                        elif allowed_y <= 0 and y_overlap > 0:
                            disallowed = True
                        elif x_overlap > allowed_x or y_overlap > allowed_y:
                            disallowed = True

                        if disallowed:
                            self.model.Add(
                                self.x[(i, p_idx)] + self.x[(j, q_idx)] <= 1
                            )

    def _set_objective(self) -> None:
        objective_terms = [
            int(round(rect.weight * 1000)) * self.use_rect[idx]
            for idx, rect in enumerate(self.rectangles)
        ]
        self.model.Maximize(sum(objective_terms))

    def solve(self) -> Tuple[float, Dict[str, Tuple[int, int]]]:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        solver.parameters.num_search_workers = 8

        status = solver.Solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("No feasible solution found")

        selected_positions: Dict[str, Tuple[int, int]] = {}
        total_weight = 0.0
        for idx, placements in enumerate(self.placements_by_rect):
            rect = self.rectangles[idx]
            if math.isclose(solver.Value(self.use_rect[idx]), 1.0):
                total_weight += rect.weight
                for p_idx, placement in enumerate(placements):
                    if math.isclose(solver.Value(self.x[(idx, p_idx)]), 1.0):
                        selected_positions[rect.name] = (placement.x, placement.y)
                        break
        return total_weight, selected_positions


def visualize_solution(
    container_size: Tuple[int, int],
    rectangles: List[SmallRectangle],
    positions: Dict[str, Tuple[int, int]],
    output_path: str = "solution.png",
) -> None:
    used_rectangles = {name for name in positions}
    fig, (ax_inventory, ax_layout) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    _draw_inventory_panel(ax_inventory, rectangles, used_rectangles)
    _draw_layout_panel(ax_layout, container_size, rectangles, positions)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def _draw_inventory_panel(
    ax: plt.Axes, rectangles: List[SmallRectangle], used_rectangles: Set[str]
) -> None:
    columns = min(5, max(len(rectangles), 1))
    spacing = 0.3
    cell_w = 3.0
    cell_h = 3.6
    rows = math.ceil(len(rectangles) / columns)
    panel_width = spacing + columns * (cell_w + spacing)
    panel_height = spacing + rows * (cell_h + spacing)

    ax.set_xlim(0, panel_width)
    ax.set_ylim(0, panel_height)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Инвентарь прямоугольников")

    for idx, rect in enumerate(rectangles):
        col = idx % columns
        row = idx // columns
        origin_x = spacing + col * (cell_w + spacing)
        origin_y = panel_height - (row + 1) * (cell_h + spacing)

        scale = min((cell_w - 0.6) / rect.width, (cell_h - 1.0) / rect.height)
        scaled_w = rect.width * scale
        scaled_h = rect.height * scale
        rect_x = origin_x + (cell_w - scaled_w) / 2
        rect_y = origin_y + (cell_h - scaled_h) / 2

        is_used = rect.name in used_rectangles
        facecolor = "#1f77b4"
        alpha = 0.85 if is_used else 0.25

        ax.add_patch(
            Rectangle(
                (rect_x, rect_y),
                scaled_w,
                scaled_h,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=1.0,
                alpha=alpha,
            )
        )

        label_y = rect_y + scaled_h / 2
        ax.text(
            rect_x + scaled_w / 2,
            label_y,
            f"{rect.name}\n{rect.width}×{rect.height}\n{rect.weight}",
            ha="center",
            va="center",
            fontsize=8,
        )


def _draw_layout_panel(
    ax: plt.Axes,
    container_size: Tuple[int, int],
    rectangles: List[SmallRectangle],
    positions: Dict[str, Tuple[int, int]],
) -> None:
    container_width, container_height = container_size
    ax.add_patch(
        Rectangle((0, 0), container_width, container_height, fill=False, edgecolor="black")
    )

    colors = plt.get_cmap("tab20", max(len(rectangles), 1))

    for idx, rect in enumerate(rectangles):
        if rect.name not in positions:
            continue
        x, y = positions[rect.name]
        patch = Rectangle(
            (x, y),
            rect.width,
            rect.height,
            facecolor=colors(idx % colors.N),
            alpha=0.6,
            edgecolor="black",
        )
        ax.add_patch(patch)
        ax.text(
            x + rect.width / 2,
            y + rect.height / 2,
            f"{rect.name}\n{rect.weight}",
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(0, container_width)
    ax.set_ylim(0, container_height)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Размещение внутри контейнера")
    ax.set_aspect("equal")


def main() -> None:
    # The container is intentionally smaller in area than the total supply of
    # small rectangles so that the optimizer must choose a profitable subset.
    container_size = (10, 7)

    rectangles: List[SmallRectangle] = []
    for idx in range(10):
        rectangles.append(
            SmallRectangle(
                name=f"L{idx + 1:02d}",
                width=2,
                height=3,
                weight=6.0,
            )
        )
    for idx in range(20):
        rectangles.append(
            SmallRectangle(
                name=f"S{idx + 1:02d}",
                width=1,
                height=2,
                weight=3.0,
            )
        )

    model = WeightedPackingModel(
        container_size=container_size,
        rectangles=rectangles,
        grid_step=1,
        max_overlap_fraction=0.10,
    )
    total_weight, positions = model.solve()

    used_rectangles = [rect for rect in rectangles if rect.name in positions]
    print(f"Всего прямоугольников: {len(rectangles)}")
    print(f"Использовано: {len(used_rectangles)}")
    print("Выбранные прямоугольники и их позиции:")
    for rect in used_rectangles:
        x, y = positions[rect.name]
        print(f"  {rect.name}: левый нижний угол в ({x}, {y})")
    print(f"Суммарный вес: {total_weight:.1f}")

    visualize_solution(container_size, rectangles, positions)
    print("Визуализация сохранена в solution.png")


if __name__ == "__main__":
    main()
