from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "outputs/final_sentiment_sweden/final_directional_spread_correlation_matrix.csv"
OUTFILE = ROOT / "outputs/final_sentiment_sweden/final_directional_spread_correlation_heatmap.png"

DISPLAY = {
    "ME": "ME",
    "age": "Age",
    "risk": "Risk",
    "IVOL_FF3": "IVOL FF3",
    "E_plus_BE": "E+BE",
    "UNPROFITABLE": "Unprof.",
    "NON_D_PAYER": "Nonpayer",
    "PPE_A": "PPE/A",
    "GS": "GS",
    "ILLIQ": "ILLIQ",
    "XTURN": "XTURN",
}


def greedy_correlation_order(matrix: np.ndarray, columns: list[str]) -> list[int]:
    """Order characteristics by a simple nearest-neighbor path on correlation distance."""
    start = columns.index("ME") if "ME" in columns else 0
    order = [start]
    remaining = set(range(len(columns))) - {start}
    while remaining:
        last = order[-1]
        next_index = max(remaining, key=lambda index: matrix[last, index])
        order.append(next_index)
        remaining.remove(next_index)
    return order


def main() -> None:
    with INFILE.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        columns = header[1:]
        rows = []
        values = []
        for row in reader:
            rows.append(row[0])
            values.append([float(value) for value in row[1:]])

    row_lookup = {name: i for i, name in enumerate(rows)}
    matrix = np.array([values[row_lookup[column]] for column in columns], dtype=float)

    order = greedy_correlation_order(matrix, columns)
    ordered = matrix[np.ix_(order, order)]

    labels = [DISPLAY.get(columns[index], columns[index]) for index in order]

    fig, ax = plt.subplots(figsize=(9.2, 7.2), dpi=220)
    image = ax.imshow(ordered, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = ordered[i, j]
            color = "white" if abs(value) >= 0.55 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
