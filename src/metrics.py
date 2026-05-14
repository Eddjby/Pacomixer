"""
metrics.py — Transition cost function f(s, a) for the Auto-DJ agent.

Combines four components, each normalised to roughly [0, 1]:
    1. Harmonic distance — based on the Circle of Fifths
    2. Tempo distance   — relative BPM difference with a tolerance band
    3. Semantic distance — cosine distance between embedding vectors
    4. Genre distance   — penalty for mixing across genres (soft preference)

The total cost is a weighted sum:
    f(s, s') = w_h·d_harm + w_t·d_tempo + w_e·d_emb + w_g·d_genre

A feasibility check is also provided: transitions where any single audio
component (harmonic, tempo, semantic) exceeds its threshold are considered
impossible. Genre is *not* used for feasibility — it is a soft preference
that the A* may override when the ruta lo amerita.
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
    "harmonic": 0.30,
    "tempo": 0.25,
    "semantic": 0.25,
    "genre": 0.20,
}

# Feasibility thresholds — beyond these, the transition is judged unmixable.
# Genre is intentionally NOT in feasibility: it only affects cost.
DEFAULT_FEASIBILITY = {
    "harmonic_max": 0.55,   # at most ~3 steps on circle of fifths + mode change
    "tempo_max": 0.18,      # at most ~18% raw BPM diff after tolerance
    "semantic_max": 0.75,   # cosine distance < 0.75 (very loose)
}


def _key_position(key: str) -> Optional[int]:
    """Return position on the Circle of Fifths or None if unknown."""
    key = key.strip()
    if key.endswith('m'):
        return _MINOR_POS.get(key)
    return _MAJOR_POS.get(key)


def _normalize_genre(g: str) -> str:
    """Lowercase, collapse separators and whitespace for fair comparison.
    Examples:
        'Hip-Hop'   → 'hip hop'
        'HIP_HOP '  → 'hip hop'
        'Rock/Pop'  → 'rock pop'
    """
    if not g:
        return ""
    g = g.lower().strip()
    g = g.replace("-", " ").replace("/", " ").replace("_", " ")
    return " ".join(g.split())


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


def genre_distance(g1: str, g2: str) -> float:
    """
    Soft penalty for mixing across genres.

    Rules:
        0.0 — same genre (after normalisation)
        0.3 — either genre is empty or 'unknown' (neutral, no penalty for ignorance)
        1.0 — different genres

    Notes:
        - Comparison is normalised (case-insensitive, separators collapsed):
          'Hip-Hop' and 'hip hop' and 'HIP_HOP' are all considered equal.
        - This is intentionally a coarse rule. It does NOT understand
          that 'Indie Rock' is closer to 'Rock' than to 'Reggaeton';
          for that the semantic (CLAP) component already captures fine
          stylistic similarity.
    """
    n1 = _normalize_genre(g1)
    n2 = _normalize_genre(g2)

    if not n1 or not n2 or n1 == "unknown" or n2 == "unknown":
        return 0.3
    if n1 == n2:
        return 0.0
    return 1.0


def transition_cost(
    s1: Song,
    s2: Song,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Total transition cost f(s, a) where a means 'play s2 next after s1'.
    Weighted sum of the four component distances.
    """
    w = weights or DEFAULT_WEIGHTS
    h_d = harmonic_distance(s1.key, s2.key)
    t_d = tempo_distance(s1.bpm, s2.bpm)
    s_d = semantic_distance(s1.embedding, s2.embedding)
    g_d = genre_distance(s1.genre, s2.genre)
    return (
        w.get("harmonic", 0.0) * h_d
        + w.get("tempo", 0.0) * t_d
        + w.get("semantic", 0.0) * s_d
        + w.get("genre", 0.0) * g_d
    )


def transition_components(s1: Song, s2: Song) -> Dict[str, float]:
    """Return the four raw components (for logging and analysis)."""
    return {
        "harmonic": harmonic_distance(s1.key, s2.key),
        "tempo": tempo_distance(s1.bpm, s2.bpm),
        "semantic": semantic_distance(s1.embedding, s2.embedding),
        "genre": genre_distance(s1.genre, s2.genre),
    }


def is_feasible(
    s1: Song,
    s2: Song,
    thresholds: Optional[Dict[str, float]] = None,
) -> bool:
    """
    True if every individual audio component is below its threshold.
    A transition is judged 'mixable' only if no single dimension is too harsh.

    Genre is NOT checked here — it only affects cost, allowing the A* to
    cross genres when the ruta lo amerita (e.g., to reach the goal).
    """
    th = thresholds or DEFAULT_FEASIBILITY
    return (
        harmonic_distance(s1.key, s2.key) <= th["harmonic_max"]
        and tempo_distance(s1.bpm, s2.bpm) <= th["tempo_max"]
        and semantic_distance(s1.embedding, s2.embedding) <= th["semantic_max"]
    )