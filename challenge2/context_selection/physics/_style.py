"""Shared look and output helpers for the physics figures.

Restrained on purpose: white background, one accent colour, grey for context,
labelled axes with units, no decoration that carries no information.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402

ACCENT = "#1b6ca8"      # the quantity the figure is about
ACCENT_2 = "#b23a48"    # second meaningful series (for example the infrared photon)
INK = "#1f2933"         # text and axes
MUTED = "#8a8f98"       # context series, reference lines
FAINT = "#d5d9de"       # background structure

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("PHYSICS_OUT", REPO / "figures" / "physics"))


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.top": False,
        "ytick.right": False,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "font.family": "DejaVu Sans",
    })


def save(fig, name: str, caption: str, data: dict | None = None) -> Path:
    """Write ``name.png`` (200 dpi), ``name.txt`` (caption) and ``name.json`` (numbers)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    caption = " ".join(caption.split())
    if chr(0x2014) in caption or chr(0x2013) in caption:   # no em or en dashes
        raise ValueError("captions use plain punctuation only")
    (OUT_DIR / f"{name}.txt").write_text(caption + "\n", encoding="utf-8")
    if data is not None:
        (OUT_DIR / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return png
