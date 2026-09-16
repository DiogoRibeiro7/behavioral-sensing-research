#!/usr/bin/env python3
"""Render the pre-specified Paper 1 household correctness-AUC figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

DIAGNOSTICS = (
    ("Confidence", "confidence_correctness_auc"),
    ("Posterior\nmargin", "margin_correctness_auc"),
    ("Normalised\nentropy", "normalised_entropy_correctness_auc"),
    ("Evidence\nstrength", "evidence_strength_correctness_auc"),
    ("Information\ngain", "information_gain_correctness_auc"),
)


def load_panel(path: Path) -> list[dict[str, object]]:
    """Load and minimally validate the frozen Paper 1 panel artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    homes = payload.get("homes")
    if not isinstance(homes, list) or len(homes) != 22:
        raise ValueError("Paper 1 figure requires the frozen 22-home panel")
    return homes


def render(homes: list[dict[str, object]], output: Path) -> None:
    """Write one household point per diagnostic with the fixed AUC=0.5 reference."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for x, (_, key) in enumerate(DIAGNOSTICS):
        values = []
        for home in homes:
            uncertainty = home.get("uncertainty")
            if not isinstance(uncertainty, dict):
                raise ValueError("home is missing uncertainty diagnostics")
            value = uncertainty.get(key)
            if not isinstance(value, (int, float)):
                raise ValueError(f"home is missing numeric {key}")
            values.append(float(value))
        ax.scatter([x] * len(values), values, alpha=0.7, s=28)

    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(DIAGNOSTICS)), [label for label, _ in DIAGNOSTICS])
    ax.set_ylabel("Within-home correctness AUC")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Correctness separation across the 22-home development panel")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    # Omitting the creation timestamp keeps re-renders byte-identical.
    fig.savefig(output, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/paper1/uncertainty_panel.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "papers/when-confidence-is-not-information/uncertainty_panel_auc.pdf"
        ),
    )
    args = parser.parse_args()
    render(load_panel(args.input), args.output)


if __name__ == "__main__":
    main()
