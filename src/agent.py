"""
agent.py — The Auto-DJ transition agent.

Formalisation (matches the project rubric):

    State space S:
        s = (current_song, target_song, history, library)
        where 'history' = songs already played, 'library' = available songs.

    Action space A(s):
        A(s) = { s' ∈ library : s' ≠ current,  s' ∉ history,
                                is_feasible(current, s') }

    Cost function f(s, a):
        f(s, s') = w_h·d_harm + w_t·d_tempo + w_e·d_embed + w_g·d_genre
        (see metrics.transition_cost)

    Decision rule (rubric form):
        a* = argmin_{a ∈ A(s)} [ g(s) + f(s, a) + h(a) ]
        Equivalently: a* = argmax R(s, a) with R = -[g + f + h].

The agent supports:
    1. Full planning   — produce an entire path from start to goal up-front.
    2. Step-by-step    — perceive a state, return the single next action.
    3. Replanning      — react to environment changes (songs removed,
                         target changed) and recompute from the current state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import copy

from .models import Song
from .metrics import transition_cost, transition_components, is_feasible, DEFAULT_WEIGHTS
from .astar import astar_search, SearchResult


@dataclass
class AgentState:
    """The state s perceived by the agent at any moment."""
    current_id: str
    target_id: str
    history: List[str] = field(default_factory=list)
    library_ids: Set[str] = field(default_factory=set)

    def copy(self) -> "AgentState":
        return AgentState(
            current_id=self.current_id,
            target_id=self.target_id,
            history=list(self.history),
            library_ids=set(self.library_ids),
        )


class TransitionAgent:
    """
    The Auto-DJ intelligent agent.

    Usage:
        agent = TransitionAgent(library, weights={'harmonic':0.4, ...})
        result = agent.plan(start_id='s01', goal_id='s17')
        for song_id in result.path: print(song_id)

    Or step-by-step:
        state = AgentState(current_id='s01', target_id='s17',
                           library_ids={s.id for s in library})
        action = agent.perceive(state)
    """

    def __init__(
        self,
        library: List[Song],
        weights: Optional[Dict[str, float]] = None,
        use_heuristic: bool = True,
        use_feasibility: bool = True,
        feasibility_thresholds: Optional[Dict[str, float]] = None,
        verbose: bool = False,
    ):
        self.library = list(library)
        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self.use_heuristic = use_heuristic
        self.use_feasibility = use_feasibility
        self.feasibility_thresholds = (
            dict(feasibility_thresholds) if feasibility_thresholds else None
        )
        self.verbose = verbose
        self._index: Dict[str, Song] = {s.id: s for s in self.library}

    # ----- formal interface required by the rubric -----

    def perceive(self, state: AgentState) -> Optional[str]:
        """
        Receive a state s, return the next action a* (= ID of the song to play next).
        Returns None if no feasible action exists or goal already reached.

        This is the explicit a* = argmin [g + f + h] decision, applied to the
        full remaining search problem from the current state.
        """
        if state.current_id == state.target_id:
            return None  # goal reached

        # Active library: everything available minus what's been played
        active = [self._index[i] for i in state.library_ids
                  if i in self._index and i not in state.history]
        # The current song must be in the search library
        if state.current_id not in {s.id for s in active}:
            active.append(self._index[state.current_id])

        result = astar_search(
            library=active,
            start_id=state.current_id,
            goal_id=state.target_id,
            weights=self.weights,
            use_heuristic=self.use_heuristic,
            use_feasibility=self.use_feasibility,
            feasibility_thresholds=self.feasibility_thresholds,
            log_decisions=self.verbose,
        )
        if not result.found or len(result.path) < 2:
            return None
        return result.path[1]  # the immediate next action

    # ----- high-level planning -----

    def plan(
        self,
        start_id: str,
        goal_id: str,
        excluded_ids: Optional[Set[str]] = None,
        log_decisions: bool = False,
    ) -> SearchResult:
        """Full path planning from start to goal."""
        excluded = excluded_ids or set()
        active = [s for s in self.library if s.id not in excluded]
        result = astar_search(
            library=active,
            start_id=start_id,
            goal_id=goal_id,
            weights=self.weights,
            use_heuristic=self.use_heuristic,
            use_feasibility=self.use_feasibility,
            feasibility_thresholds=self.feasibility_thresholds,
            log_decisions=log_decisions,
        )
        if self.verbose:
            print(f"[agent.plan] {start_id} → {goal_id}: {result.summary()}")
        return result

    def replan_from(
        self,
        current_id: str,
        new_goal_id: str,
        played_so_far: List[str],
        excluded_ids: Optional[Set[str]] = None,
    ) -> SearchResult:
        """Replan after an environment change (removed song, new target, etc.)."""
        excluded = set(excluded_ids or set())
        excluded.update(p for p in played_so_far if p != current_id)
        return self.plan(current_id, new_goal_id, excluded_ids=excluded)

    # ----- path inspection utilities (for reports and videos) -----

    def describe_path(self, result: SearchResult) -> str:
        """Human-readable rendering of a found path with per-step components."""
        if not result.found:
            return "[no feasible path]"
        lines = []
        path_songs = [self._index[i] for i in result.path]
        lines.append(f"Path of {len(path_songs)} songs, total cost = {result.total_cost:.4f}")
        lines.append("")
        for i, song in enumerate(path_songs):
            lines.append(f"  [{i}] {song.title} — {song.artist}  "
                         f"(BPM={song.bpm:.1f}, Key={song.key}, Genre={song.genre})")
            if i + 1 < len(path_songs):
                comps = transition_components(song, path_songs[i + 1])
                step_cost = transition_cost(song, path_songs[i + 1], self.weights)
                lines.append(f"        ↓  cost={step_cost:.4f}  "
                             f"[harm={comps['harmonic']:.3f}, "
                             f"tempo={comps['tempo']:.3f}, "
                             f"sem={comps['semantic']:.3f}, "
                             f"genre={comps['genre']:.3f}]")
        return "\n".join(lines)