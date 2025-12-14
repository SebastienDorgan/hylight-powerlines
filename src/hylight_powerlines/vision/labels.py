"""Label utilities (prompt composition, normalization, matching)."""

def combined_prompt(targets: list[str]) -> str:
    """Return a single combined prompt for open-vocabulary detection."""
    return " . ".join(targets)


def _norm(s: str) -> str:
    cleaned = (
        s.strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
        .replace(",", " ")
    )
    return " ".join(cleaned.split())


def normalize_label(raw_label: str, targets: list[str]) -> str | None:
    """Map a raw predicted label/phrase to one of `targets`.

    Policy (deterministic):
      1) Exact match after normalization.
      2) Longest target that appears as a substring in the normalized label.
      3) Otherwise drop (return None).
    """
    raw = _norm(raw_label)
    if not raw:
        return None

    targets_by_norm = {t: _norm(t) for t in targets}
    for t, tn in targets_by_norm.items():
        if raw == tn:
            return t

    best: tuple[int, str] | None = None
    for t, tn in targets_by_norm.items():
        if tn and tn in raw:
            cand = (len(tn), t)
            if best is None or cand[0] > best[0]:
                best = cand
    return best[1] if best is not None else None
