import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
        max_overlap_fraction: float = 0.2,
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
            for x in range(0, self.container_width - rect.width + 1, self.grid_step):
                for y in range(0, self.container_height - rect.height + 1, self.grid_step):
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

    def _allowed_overlap_area(self, i: int, j: int) -> float:
        min_area = min(self.rectangles[i].area, self.rectangles[j].area)
        return self.max_overlap_fraction * min_area

    @staticmethod
    def _overlap_area(
        p1: Placement,
        rect1: SmallRectangle,
        p2: Placement,
        rect2: SmallRectangle,
    ) -> int:
        x_overlap = max(
            0,
            min(p1.x + rect1.width, p2.x + rect2.width)
            - max(p1.x, p2.x),
        )
        y_overlap = max(
            0,
            min(p1.y + rect1.height, p2.y + rect2.height)
            - max(p1.y, p2.y),
        )
        return x_overlap * y_overlap

    def _add_overlap_constraints(self) -> None:
        for i in range(len(self.rectangles)):
            for j in range(i + 1, len(self.rectangles)):
                allowed_area = self._allowed_overlap_area(i, j)
                for p_idx, placement_i in enumerate(self.placements_by_rect[i]):
                    for q_idx, placement_j in enumerate(self.placements_by_rect[j]):
                        overlap = self._overlap_area(
                            placement_i,
                            self.rectangles[i],
                            placement_j,
                            self.rectangles[j],
                        )
                        if overlap > allowed_area + 1e-9:
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
    fig, ax = plt.subplots(figsize=(8, 6))
    container_width, container_height = container_size
    ax.add_patch(
        Rectangle((0, 0), container_width, container_height, fill=False, edgecolor="black")
    )

    colors = plt.get_cmap("tab10", max(len(positions), 1))

    for idx, rect in enumerate(rectangles):
        if rect.name not in positions:
            continue
        x, y = positions[rect.name]
        patch = Rectangle(
            (x, y),
            rect.width,
            rect.height,
            facecolor=colors(idx % colors.N),
            alpha=0.5,
            edgecolor="black",
        )
        ax.add_patch(patch)
        ax.text(
            x + rect.width / 2,
            y + rect.height / 2,
            f"{rect.name}\n{rect.weight}",
            ha="center",
            va="center",
        )

    ax.set_xlim(0, container_width)
    ax.set_ylim(0, container_height)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Weighted packing with controlled overlap")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    container_size = (12, 9)
    rectangles = [
        SmallRectangle("A", width=5, height=4, weight=9.0),
        SmallRectangle("B", width=4, height=3, weight=7.5),
        SmallRectangle("C", width=6, height=2, weight=6.0),
        SmallRectangle("D", width=3, height=3, weight=4.0),
        SmallRectangle("E", width=2, height=5, weight=3.5),
        SmallRectangle("F", width=4, height=4, weight=8.0),
    ]

    model = WeightedPackingModel(
        container_size=container_size,
        rectangles=rectangles,
        grid_step=1,
        max_overlap_fraction=0.25,
    )
    total_weight, positions = model.solve()

    print("Выбранные прямоугольники и их позиции:")
    for rect in rectangles:
        if rect.name in positions:
            x, y = positions[rect.name]
            print(f"  {rect.name}: левый нижний угол в ({x}, {y})")
    print(f"Суммарный вес: {total_weight:.1f}")

    visualize_solution(container_size, rectangles, positions)
    print("Визуализация сохранена в solution.png")


if __name__ == "__main__":
    main()
