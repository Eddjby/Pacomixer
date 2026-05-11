"""
metrics.py — Transition cost function f(s, a) for the Auto-DJ agent.

Combines three components, each normalised to roughly [0, 1]:
    1. Harmonic distance — based on the Circle of Fifths
    2. Tempo distance   — relative BPM difference with a tolerance band
    3. Semantic distance — cosine distance between embedding vectors

The total cost is a weighted sum: f(s, s') = w_h * d_harm + w_t * d_tempo + w_e * d_emb

A feasibility check is also provided: transitions where any single component
exceeds its threshold are considered impossible (no DJ would attempt them),
which removes edges from the action space and forces the agent to find
intermediate stepping stones.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from .models import Song


# Circle of Fifths positions (0-11) — major and minor keys mapped consistently.
# Two keys at the same position are enharmonic equivalents (e.g., F# = Gb).
_MAJOR_POS: Dict[str, int] = {
    'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5,
    'F#': 6, 'Gb': 6, 'C#': 7, 'Db': 7, 'G#': 8, 'Ab': 8,
    'D#': 9, 'Eb': 9, 'A#': 10, 'Bb': 10, 'F': 11,
}

_MINOR_POS: Dict[str, int] = {
    'Am': 0, 'Em': 1, 'Bm': 2, 'F#m': 3, 'Gbm': 3,
    'C#m': 4, 'Dbm': 4, 'G#m': 5, 'Abm': 5,
    'D#m': 6, 'Ebm': 6, 'A#m': 7, 'Bbm': 7,
    'Fm': 8, 'Cm': 9, 'Gm': 10, 'Dm': 11,
}

# Default weights — sum to 1.0 for interpretability
DEFAULT_WEIGHTS: Dict[str, float] = {
    "harmonic": 0.40,
    "tempo": 0.30,
    "semantic": 0.30,
}

# Feasibility thresholds — beyond these, the transition is judged unmixable
DEFAULT_FEASIBILITY = {
    "harmonic_max": 0.55,   # at most ~3 steps on circle of fifths + mode change
    "tempo_max": 0.18,      # at most ~18% raw BPM diff after tolerance
    "semantic_max": 0.75,   # cosine distance < 0.75 (very loose)
}


def _key_position(key: str) -> Optional[int]:
    """Return (position, is_minor) tuple or None if key is unknown."""
    key = key.strip()
    if key.endswith('m'):
        return _MINOR_POS.get(key)
    return _MAJOR_POS.get(key)


def harmonic_distance(key1: str, key2: str) -> float:
    """
    Distance between two musical keys on the Circle of Fifths, normalised to [0, 1.15].
    Adds a small penalty (+0.15) when the two keys differ in mode (major vs minor).
    """
    is_minor1 = key1.strip().endswith('m')
    is_minor2 = key2.strip().endswith('m')

    pos1 = _key_position(key1)
    pos2 = _key_position(key2)

    if pos1 is None or pos2 is None:
        return 1.0  # unknown key — treat as harshest

    diff = abs(pos1 - pos2)
    circular_dist = min(diff, 12 - diff)  # range [0, 6]
    base = circular_dist / 6.0
    mode_penalty = 0.0 if is_minor1 == is_minor2 else 0.15
    return base + mode_penalty


def tempo_distance(bpm1: float, bpm2: float, tolerance: float = 0.08) -> float:
    """
    Relative BPM difference with a tolerance band (default ±8%).
    Accounts for the DJ practice of half-time and double-time beatmatching:
    a 87 BPM track can be perceptually aligned to 174 BPM (2x) or vice-versa.
    We therefore compare bpm1 against the closest of {bpm2, 2*bpm2, bpm2/2}.
    Zero if within tolerance, scales linearly beyond, capped at 1.0.
    """
    if bpm1 <= 0:
        return 1.0
    candidates = [bpm2, 2.0 * bpm2, 0.5 * bpm2]
    rel_diff = min(abs(bpm1 - c) / bpm1 for c in candidates)
    if rel_diff <= tolerance:
        return 0.0
    return min(1.0, (rel_diff - tolerance) / 0.42)  # caps at ~50% BPM diff


def semantic_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine distance between embeddings, normalised to [0, 1]."""
    n1 = float(np.linalg.norm(emb1))
    n2 = float(np.linalg.norm(emb2))
    if n1 == 0.0 or n2 == 0.0:
        return 1.0
    cos_sim = float(np.dot(emb1, emb2) / (n1 * n2))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return (1.0 - cos_sim) / 2.0


def transition_cost(
    s1: Song,
    s2: Song,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Total transition cost f(s, a) where a means 'play s2 next after s1'.
    Returns the weighted sum of the three component distances.
    """
    w = weights or DEFAULT_WEIGHTS
    h_d = harmonic_distance(s1.key, s2.key)
    t_d = tempo_distance(s1.bpm, s2.bpm)
    s_d = semantic_distance(s1.embedding, s2.embedding)
    return w["harmonic"] * h_d + w["tempo"] * t_d + w["semantic"] * s_d


def transition_components(s1: Song, s2: Song) -> Dict[str, float]:
    """Return the three raw components (for logging and analysis)."""
    return {
        "harmonic": harmonic_distance(s1.key, s2.key),
        "tempo": tempo_distance(s1.bpm, s2.bpm),
        "semantic": semantic_distance(s1.embedding, s2.embedding),
    }


def is_feasible(
    s1: Song,
    s2: Song,
    thresholds: Optional[Dict[str, float]] = None,
) -> bool:
    """
    True if every individual component is below its threshold.
    A transition is judged 'mixable' only if no single dimension is too harsh.
    """
    th = thresholds or DEFAULT_FEASIBILITY
    return (
        harmonic_distance(s1.key, s2.key) <= th["harmonic_max"]
        and tempo_distance(s1.bpm, s2.bpm) <= th["tempo_max"]
        and semantic_distance(s1.embedding, s2.embedding) <= th["semantic_max"]
    )
