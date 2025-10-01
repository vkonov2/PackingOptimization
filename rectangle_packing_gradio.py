"""Gradio interface for configuring and solving the rectangle packing demo."""
from __future__ import annotations

import math
import tempfile
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import gradio as gr
import pandas as pd

from rectangle_packing import (
    SmallRectangle,
    WeightedPackingModel,
    generate_rectangle_inventory,
    render_inventory_preview,
    visualize_solution,
)

DEFAULT_CONTAINER_SIZE: Tuple[float, float] = (10.0, 6.0)
DEFAULT_BASE_WIDTH = 2.0
DEFAULT_BASE_HEIGHT = 3.0
DEFAULT_BASE_WEIGHT = 1.0
DEFAULT_HORIZONTAL_GROWTH = 0.10
DEFAULT_VERTICAL_GROWTH = 0.05
INVENTORY_COLUMNS = ["width", "height", "weight", "count"]


def _inventory_dataframe(rectangles: Sequence[SmallRectangle]) -> pd.DataFrame:
    grouped: Dict[Tuple[float, float, float], List[SmallRectangle]] = defaultdict(list)
    for rect in rectangles:
        grouped[(rect.width, rect.height, rect.weight)].append(rect)

    rows = []
    for (width, height, weight), items in grouped.items():
        rows.append(
            {
                "width": width,
                "height": height,
                "weight": weight,
                "count": len(items),
            }
        )

    rows.sort(key=lambda row: (row["height"], row["width"], row["weight"]))
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def _coerce_table(table: Iterable[Iterable[float]] | None) -> pd.DataFrame:
    if table is None:
        dataframe = pd.DataFrame(columns=INVENTORY_COLUMNS)
    else:
        dataframe = pd.DataFrame(table, columns=INVENTORY_COLUMNS)
    return dataframe[INVENTORY_COLUMNS]


def _expand_inventory(table: pd.DataFrame) -> List[SmallRectangle]:
    rectangles: List[SmallRectangle] = []
    working_table = _coerce_table(table)
    for idx, row in working_table[INVENTORY_COLUMNS].iterrows():
        width = float(row["width"])
        height = float(row["height"])
        weight = float(row["weight"])
        count_value = row["count"]
        if any(math.isnan(value) for value in (width, height, weight)):
            continue
        try:
            count = int(round(float(count_value)))
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0 or weight <= 0 or count <= 0:
            continue

        for copy_idx in range(count):
            name = f"U{idx + 1:02d}_{copy_idx + 1:02d}"
            rectangles.append(SmallRectangle(name=name, width=width, height=height, weight=weight))

    return rectangles


def _render_inventory_snapshot(rectangles: Sequence[SmallRectangle]) -> str | None:
    if not rectangles:
        return None

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_file.close()
    render_inventory_preview(rectangles, temp_file.name)
    return temp_file.name


def _solve_and_render(table: pd.DataFrame) -> Tuple[str, str]:
    rectangles = _expand_inventory(table)
    if not rectangles:
        return "Не удалось построить инвентарь. Проверьте введенные размеры.", gr.update(value=None)

    total_inventory_weight = sum(rect.weight for rect in rectangles)

    try:
        model = WeightedPackingModel(container_size=DEFAULT_CONTAINER_SIZE, rectangles=rectangles)
        total_weight, positions = model.solve()
    except Exception as exc:  # noqa: BLE001 - surface error message in UI
        return f"Ошибка при расчете: {exc}", gr.update(value=None)

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_file.close()

    visualize_solution(
        DEFAULT_CONTAINER_SIZE,
        rectangles,
        positions,
        total_weight,
        total_inventory_weight,
        output_path=temp_file.name,
    )

    summary_lines = [
        f"Всего прямоугольников: {len(rectangles)}",
        f"Использовано: {len(positions)}",
        f"Суммарный вес: {total_weight:.2f} из {total_inventory_weight:.2f}",
    ]

    return "\n".join(summary_lines), temp_file.name


def _on_pack(table: Iterable[Iterable[float]]) -> Tuple[str, str]:
    dataframe = _coerce_table(table)
    return _solve_and_render(dataframe)


def _on_inventory_change(table: Iterable[Iterable[float]]) -> gr.Update:
    dataframe = _coerce_table(table)
    rectangles = _expand_inventory(dataframe)
    snapshot = _render_inventory_snapshot(rectangles)
    return gr.update(value=snapshot)


def build_demo() -> gr.Blocks:
    default_inventory = generate_rectangle_inventory(
        container_size=DEFAULT_CONTAINER_SIZE,
        base_width=DEFAULT_BASE_WIDTH,
        base_height=DEFAULT_BASE_HEIGHT,
        base_weight=DEFAULT_BASE_WEIGHT,
        horizontal_growth_percent=DEFAULT_HORIZONTAL_GROWTH,
        vertical_growth_percent=DEFAULT_VERTICAL_GROWTH,
    )
    default_table = _inventory_dataframe(default_inventory)
    default_inventory_snapshot = _render_inventory_snapshot(default_inventory)

    with gr.Blocks(title="Упаковка прямоугольников") as demo:
        gr.Markdown(
            """
            # Упаковка прямоугольников
            Настройте инвентарь прямоугольников в таблице ниже. Вы можете менять размеры, вес и количество типов
            прямоугольников, добавлять новые строки или деактивировать тип, установив его количество в ноль.
            После настройки нажмите кнопку **«упаковать»**, чтобы запустить расчет и увидеть полученную раскладку.
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=400):
                inventory_editor = gr.Dataframe(
                    headers=INVENTORY_COLUMNS,
                    value=default_table,
                    datatype=["number", "number", "number", "number"],
                    row_count=(len(default_table), "dynamic"),
                    col_count=(len(INVENTORY_COLUMNS), "fixed"),
                    label="Инвентарь прямоугольников",
                )
            with gr.Column(scale=1, min_width=400):
                inventory_preview = gr.Image(
                    value=default_inventory_snapshot,
                    label="Визуализация инвентаря",
                    type="filepath",
                )

        pack_button = gr.Button("упаковать")
        status = gr.Textbox(label="Результат", interactive=False)
        result_image = gr.Image(label="Визуализация", type="filepath")

        pack_button.click(_on_pack, inputs=inventory_editor, outputs=[status, result_image])
        inventory_editor.change(_on_inventory_change, inputs=inventory_editor, outputs=inventory_preview)
    return demo


def main() -> None:
    """Launch the Gradio demo."""
    app = build_demo()
    app.launch()


if __name__ == "__main__":
    main()
