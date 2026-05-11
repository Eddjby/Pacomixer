"""
models.py — Data model for songs in the library.

Each Song carries the audio features used by the agent's cost function:
    - bpm: tempo in beats per minute
    - key: musical key (e.g., 'C', 'Am', 'F#') — major if no 'm' suffix
    - embedding: vector representing sonic character (analogous to CLAP embeddings
      in the original repo; here lower-dimensional for synthetic experiments)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import json
import numpy as np


@dataclass
class Song:
    id: str
    title: str
    artist: str
    bpm: float
    key: str
    embedding: np.ndarray
    genre: str = "unknown"  # used only for synthetic data inspection

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return isinstance(other, Song) and self.id == other.id

    def __repr__(self) -> str:
        return (f"Song({self.id} | '{self.title}' — {self.artist} | "
                f"BPM={self.bpm:.1f} | Key={self.key} | Genre={self.genre})")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "bpm": self.bpm,
            "key": self.key,
            "embedding": self.embedding.tolist(),
            "genre": self.genre,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Song":
        return cls(
            id=d["id"],
            title=d["title"],
            artist=d["artist"],
            bpm=float(d["bpm"]),
            key=d["key"],
            embedding=np.array(d["embedding"], dtype=np.float32),
            genre=d.get("genre", "unknown"),
        )


def save_library(songs: List[Song], path: str) -> None:
    """Persist a list of Song objects to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in songs], f, indent=2)


def load_library(path: str) -> List[Song]:
    """Load a list of Song objects from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Song.from_dict(d) for d in data]
