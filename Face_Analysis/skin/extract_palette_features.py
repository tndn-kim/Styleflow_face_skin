"""Load palette prototype files for inference."""
from __future__ import annotations
import json
from pathlib import Path


def load_prototypes(path: str | Path) -> dict:
    """Load palette_prototypes.json. Returns {"season": {"Spring": {...}, ...}, "subtype": {...}}."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_axis_prototypes(path: str | Path) -> dict:
    """Load palette_axis_prototypes.json. Returns {"Spring": {...}, "Summer": {...}, ...}."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
