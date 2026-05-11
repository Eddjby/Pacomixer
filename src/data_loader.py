"""
data_loader.py — Library construction utilities.

Generates a synthetic, fully-reproducible song library structured in
genre clusters. Each genre has a characteristic BPM range, a small pool
of compatible keys, and a centroid in embedding space. Songs are sampled
from each cluster with Gaussian noise added to the embedding.

This structure produces a problem where:
    - Transitions WITHIN a genre tend to be cheap.
    - Transitions BETWEEN distant genres are often infeasible directly.
    - The agent must discover intermediate stepping-stones (e.g.,
      a downtempo track that bridges hip-hop and ambient).

Replace this with a real audio analysis pipeline (librosa + madmom +
essentia + CLAP, as in the original repo) when you have an actual library.
"""
from __future__ import annotations
from typing import List
import numpy as np

from .models import Song


# Each genre: (bpm_centre, bpm_std, compatible_keys, embedding_centroid_index)
_GENRE_PROFILES = {
    "ambient":    {"bpm": (75, 6),   "keys": ["Am", "Em", "Dm", "Gm", "C", "F"],   "cluster": 0},
    "downtempo":  {"bpm": (90, 7),   "keys": ["Am", "Em", "C", "F", "G", "Dm"],    "cluster": 1},
    "hiphop":     {"bpm": (95, 8),   "keys": ["Am", "Dm", "Gm", "Cm", "F", "Bb"],  "cluster": 2},
    "indie":      {"bpm": (110, 8),  "keys": ["C", "G", "D", "Am", "Em", "F"],     "cluster": 3},
    "house":      {"bpm": (124, 4),  "keys": ["Am", "Em", "Bm", "F#m", "G", "D"],  "cluster": 4},
    "techno":     {"bpm": (132, 4),  "keys": ["Am", "Em", "F#m", "C#m", "G#m"],    "cluster": 5},
    "dnb":        {"bpm": (174, 3),  "keys": ["Am", "Em", "F#m", "Dm", "C"],       "cluster": 6},
}

# Pre-defined centroids in 8-dim embedding space.
# We place them so adjacent genres on the BPM spectrum are also near in
# embedding space, with techno → dnb being the largest jump.
_EMBEDDING_CENTROIDS = {
    0: np.array([ 1.0,  0.8, -0.5, -0.3,  0.0,  0.2, -0.1,  0.0]),   # ambient
    1: np.array([ 0.7,  0.5,  0.0, -0.2,  0.3,  0.4, -0.1,  0.1]),   # downtempo
    2: np.array([ 0.3,  0.2,  0.8,  0.4,  0.1,  0.0,  0.5,  0.2]),   # hiphop
    3: np.array([ 0.0,  0.1,  0.2,  0.7,  0.6,  0.2,  0.3,  0.4]),   # indie
    4: np.array([-0.3, -0.1,  0.0,  0.3,  0.8,  0.7,  0.5,  0.3]),   # house
    5: np.array([-0.6, -0.3, -0.2,  0.0,  0.5,  0.9,  0.7,  0.4]),   # techno
    6: np.array([-0.8, -0.6,  0.1,  0.0,  0.2,  0.6,  0.9,  0.7]),   # dnb
}

_ARTIST_POOL = [
    "Lunaris", "K. Halberd", "Mira Voss", "Cyrus Lan", "Echo & Field",
    "Petra Solis", "Atlas Drift", "Nyx Carter", "Ovid Park", "S. Kobayashi",
    "Velour", "M. Hadid", "Theron West", "Iris Bloom", "F. Antonelli",
    "Quartz", "B. Achebe", "Indra Pavón", "Cael Storm", "R. Vega",
]

_TITLE_FRAGMENTS_A = [
    "Glass", "Neon", "Hollow", "Ember", "Drift", "Quiet", "Static",
    "Velvet", "Crystal", "Slow", "Open", "Magnet", "Wired", "Pale",
    "Liquid", "Iron", "Mirror", "Folded", "Dust", "Outer",
]
_TITLE_FRAGMENTS_B = [
    "Bloom", "Signal", "Hour", "Mile", "River", "Engine", "Room",
    "Architecture", "Threshold", "Stairs", "Magnet", "Tide", "Field",
    "Phase", "Daughter", "Window", "Echo", "Lantern", "Promise", "Wave",
]


def generate_synthetic_library(
    n_songs: int = 28,
    seed: int = 42,
    embedding_noise_std: float = 0.25,
) -> List[Song]:
    """Build a reproducible synthetic library with genre clusters."""
    rng = np.random.default_rng(seed)
    genres = list(_GENRE_PROFILES.keys())
    songs: List[Song] = []

    # Distribute songs across genres roughly evenly
    per_genre_base = n_songs // len(genres)
    leftover = n_songs - per_genre_base * len(genres)
    counts = [per_genre_base + (1 if i < leftover else 0) for i in range(len(genres))]

    song_idx = 0
    for genre, count in zip(genres, counts):
        profile = _GENRE_PROFILES[genre]
        bpm_centre, bpm_std = profile["bpm"]
        centroid = _EMBEDDING_CENTROIDS[profile["cluster"]]
        for _ in range(count):
            song_idx += 1
            bpm = float(np.clip(rng.normal(bpm_centre, bpm_std), 60, 200))
            key = str(rng.choice(profile["keys"]))
            noise = rng.normal(0.0, embedding_noise_std, size=centroid.shape).astype(np.float32)
            embedding = (centroid + noise).astype(np.float32)
            artist = str(rng.choice(_ARTIST_POOL))
            title = f"{rng.choice(_TITLE_FRAGMENTS_A)} {rng.choice(_TITLE_FRAGMENTS_B)}"
            songs.append(Song(
                id=f"s{song_idx:02d}",
                title=title,
                artist=artist,
                bpm=round(bpm, 1),
                key=key,
                embedding=embedding,
                genre=genre,
            ))

    return songs


def print_library_summary(library: List[Song]) -> None:
    """Print a short summary grouped by genre."""
    by_genre: dict = {}
    for s in library:
        by_genre.setdefault(s.genre, []).append(s)
    print(f"Library: {len(library)} songs across {len(by_genre)} genres")
    for g, songs in sorted(by_genre.items()):
        bpms = [s.bpm for s in songs]
        print(f"  {g:10s}  n={len(songs):2d}  BPM range {min(bpms):5.1f}–{max(bpms):5.1f}  "
              f"e.g. {songs[0].id} '{songs[0].title}' ({songs[0].key})")
