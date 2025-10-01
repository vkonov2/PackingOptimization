import math
from dataclasses import dataclass
from itertools import combinations
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


class WeightedPackingModel:
    def __init__(
        self,
        container_size: Tuple[int, int],
        rectangles: List[SmallRectangle],
        grid_step: int = 1,
        coordinate_scale: int = 10,
    ) -> None:
        if grid_step <= 0:
            raise ValueError("grid_step must be positive")
        if coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be positive")

        self.scale = coordinate_scale
        self.container_width = int(container_size[0] * self.scale)
        self.container_height = int(container_size[1] * self.scale)
        self.rectangles = rectangles
        self.grid_step = grid_step

        self.scaled_dimensions: List[Tuple[int, int]] = [
            (int(rect.width * self.scale), int(rect.height * self.scale))
            for rect in self.rectangles
        ]
        self.max_scaled_width = max((w for w, _ in self.scaled_dimensions), default=0)
        self.max_scaled_height = max((h for _, h in self.scaled_dimensions), default=0)

        stride_padding = (
            self.container_width
            + self.container_height
            + max(self.max_scaled_width, self.max_scaled_height)
            + self.scale
        )
        self.sentinel_stride = max(stride_padding, 1)
        self.min_coord = -self.sentinel_stride * (len(self.rectangles) + 2)

        self.model = cp_model.CpModel()
        self._build_variables()
        self._add_non_overlap_constraints()
        self._set_objective()

    def _build_variables(self) -> None:
        self.use_rect: List[cp_model.BoolVar] = []
        self.x_coord: List[cp_model.IntVar] = []
        self.y_coord: List[cp_model.IntVar] = []
        self.can_place: List[bool] = []
        self.sentinel_locations: List[int] = []

        for idx, rect in enumerate(self.rectangles):
            width, height = self.scaled_dimensions[idx]
            sentinel = -self.sentinel_stride * (idx + 1)
            self.sentinel_locations.append(sentinel)

            max_x = self.container_width - width
            max_y = self.container_height - height
            feasible = max_x >= 0 and max_y >= 0
            self.can_place.append(feasible)

            use_var = self.model.NewBoolVar(f"use_{rect.name}")
            if not feasible:
                self.model.Add(use_var == 0)

            x_var = self.model.NewIntVar(
                self.min_coord, self.container_width, f"x_{rect.name}"
            )
            y_var = self.model.NewIntVar(
                self.min_coord, self.container_height, f"y_{rect.name}"
            )

            self.model.Add(x_var == sentinel).OnlyEnforceIf(use_var.Not())
            self.model.Add(y_var == sentinel).OnlyEnforceIf(use_var.Not())

            if feasible:
                self.model.Add(x_var >= 0).OnlyEnforceIf(use_var)
                self.model.Add(y_var >= 0).OnlyEnforceIf(use_var)
                self.model.Add(x_var + width <= self.container_width).OnlyEnforceIf(
                    use_var
                )
                self.model.Add(y_var + height <= self.container_height).OnlyEnforceIf(
                    use_var
                )

            self.use_rect.append(use_var)
            self.x_coord.append(x_var)
            self.y_coord.append(y_var)

    def _add_non_overlap_constraints(self) -> None:
        if not self.rectangles:
            return

        for i in range(len(self.rectangles)):
            for j in range(i + 1, len(self.rectangles)):
                x_i = self.x_coord[i]
                y_i = self.y_coord[i]
                x_j = self.x_coord[j]
                y_j = self.y_coord[j]
                width_i, height_i = self.scaled_dimensions[i]
                width_j, height_j = self.scaled_dimensions[j]

                i_left_j = self.model.NewBoolVar(f"i_left_j_{i}_{j}")
                j_left_i = self.model.NewBoolVar(f"j_left_i_{i}_{j}")
                i_below_j = self.model.NewBoolVar(f"i_below_j_{i}_{j}")
                j_below_i = self.model.NewBoolVar(f"j_below_i_{i}_{j}")

                self.model.Add(x_i + width_i <= x_j).OnlyEnforceIf(i_left_j)
                self.model.Add(x_j + width_j <= x_i).OnlyEnforceIf(j_left_i)
                self.model.Add(y_i + height_i <= y_j).OnlyEnforceIf(i_below_j)
                self.model.Add(y_j + height_j <= y_i).OnlyEnforceIf(j_below_i)

                for indicator in (i_left_j, j_left_i, i_below_j, j_below_i):
                    self.model.AddImplication(indicator, self.use_rect[i])
                    self.model.AddImplication(indicator, self.use_rect[j])

                self.model.AddBoolOr(
                    [
                        self.use_rect[i].Not(),
                        self.use_rect[j].Not(),
                        i_left_j,
                        j_left_i,
                        i_below_j,
                        j_below_i,
                    ]
                )

    def _set_objective(self) -> None:
        objective_terms = [
            int(round(rect.weight * 1000)) * self.use_rect[idx]
            for idx, rect in enumerate(self.rectangles)
        ]
        self.model.Maximize(sum(objective_terms))

    def solve(self) -> Tuple[float, Dict[str, Tuple[float, float]]]:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        solver.parameters.num_search_workers = 8

        status = solver.Solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("No feasible solution found")

        selected_positions: Dict[str, Tuple[float, float]] = {}
        total_weight = 0.0
        for idx, rect in enumerate(self.rectangles):
            if math.isclose(solver.Value(self.use_rect[idx]), 1.0):
                total_weight += rect.weight
                x_value = solver.Value(self.x_coord[idx]) / self.scale
                y_value = solver.Value(self.y_coord[idx]) / self.scale
                selected_positions[rect.name] = (x_value, y_value)
        return total_weight, selected_positions


def visualize_solution(
    container_size: Tuple[int, int],
    rectangles: List[SmallRectangle],
    positions: Dict[str, Tuple[float, float]],
    output_path: str = "rectangular_solution.png",
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


def report_overlap_statistics(
    rectangles: List[SmallRectangle],
    positions: Dict[str, Tuple[float, float]],
) -> None:
    name_to_rect = {rect.name: rect for rect in rectangles}
    used = [name for name in positions]
    print("Проверка отсутствия перекрытий:")
    any_overlap = False
    violation_found = False

    for name_i, name_j in combinations(sorted(used), 2):
        rect_i = name_to_rect[name_i]
        rect_j = name_to_rect[name_j]
        x_i, y_i = positions[name_i]
        x_j, y_j = positions[name_j]

        x_overlap = max(
            0.0,
            min(x_i + rect_i.width, x_j + rect_j.width) - max(x_i, x_j),
        )
        y_overlap = max(
            0.0,
            min(y_i + rect_i.height, y_j + rect_j.height) - max(y_i, y_j),
        )

        if x_overlap <= 1e-9 or y_overlap <= 1e-9:
            continue

        any_overlap = True
        violation_found = True
        print(
            "  {} ↔ {}: перекрытие {:.2f}×{:.2f} ← нарушение".format(
                name_i, name_j, x_overlap, y_overlap
            )
        )

    if not any_overlap:
        print("  Перекрытия отсутствуют.")
    elif violation_found:
        print("  Найдены перекрытия, проверьте модель размещения.")


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
    positions: Dict[str, Tuple[float, float]],
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


def _format_coord(value: float) -> str:
    rounded = round(value + 1e-9, 1)
    if math.isclose(rounded, round(rounded)):
        return str(int(round(rounded)))
    return f"{rounded:.1f}"


def main() -> None:
    # The container is intentionally smaller in area than the total supply of
    # small rectangles so that the optimizer must choose a profitable subset.
    container_size = (10, 5.8)

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
    )
    total_weight, positions = model.solve()

    used_rectangles = [rect for rect in rectangles if rect.name in positions]
    print(f"Всего прямоугольников: {len(rectangles)}")
    print(f"Использовано: {len(used_rectangles)}")
    print("Выбранные прямоугольники и их позиции:")
    for rect in used_rectangles:
        x, y = positions[rect.name]
        print(
            f"  {rect.name}: левый нижний угол в "
            f"({_format_coord(x)}, {_format_coord(y)})"
        )
    print(f"Суммарный вес: {total_weight:.1f}")

    report_overlap_statistics(rectangles, positions)

    visualize_solution(container_size, rectangles, positions)
    print("Визуализация сохранена в rectangular_solution.png")


if __name__ == "__main__":
    main()
