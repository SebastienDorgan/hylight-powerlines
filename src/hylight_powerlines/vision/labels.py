"""Label utilities (prompt composition, normalization, matching)."""

from __future__ import annotations


def combined_prompt(targets: list[str]) -> str:
    """Return a single combined prompt for open-vocabulary detection."""
    return " . ".join(targets)


def _norm(s: str) -> str:
    cleaned = (
        s.strip().lower().replace("-", " ").replace("_", " ").replace("/", " ").replace(",", " ")
    )
    return " ".join(cleaned.split())


def normalize_label(raw_label: str, targets: list[str]) -> str | None:
    """Map a raw predicted label/phrase to one of `targets`.

    Policy (deterministic):
      0) Apply a small alias map for common synonyms (e.g. "utility pole" -> "tower").
      1) Exact match after normalization.
      2) Longest target that appears as a substring in the normalized label.
      3) Otherwise drop (return None).
    """
    raw = _norm(raw_label)
    if not raw:
        return None

    alias_map: dict[str, str] = {
        # Tower-like structures (common for overhead distribution lines).
        "pylon": "tower",
        "electric pylon": "tower",
        "utility pole": "tower",
        "power pole": "tower",
        "electric pole": "tower",
        "distribution pole": "tower",
        "power line pole": "tower",
        "wood pole": "tower",
        "concrete pole": "tower",
        # Hardware.
        "stockbridge damper": "damper",
        "spacer damper": "spacer",
        "insulator string": "insulator",
        "power line insulator": "insulator",
        "tower plate": "tower_plate",
        "crossarm plate": "tower_plate",
        "cross arm plate": "tower_plate",
    }
    # Alias exact match.
    if raw in alias_map and alias_map[raw] in targets:
        return alias_map[raw]

    targets_by_norm = {t: _norm(t) for t in targets}
    for t, tn in targets_by_norm.items():
        if raw == tn:
            return t

    # Alias substring match (e.g. "wood utility pole" should still map to tower).
    for alias, canonical in alias_map.items():
        if canonical in targets and alias in raw:
            return canonical

    best: tuple[int, str] | None = None
    for t, tn in targets_by_norm.items():
        if tn and tn in raw:
            cand = (len(tn), t)
            if best is None or cand[0] > best[0]:
                best = cand
    return best[1] if best is not None else None
