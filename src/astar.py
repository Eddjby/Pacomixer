"""
astar.py — A* search algorithm for the Auto-DJ agent.

The agent's decision rule at every expansion is:
    a* = argmin_{a in A(s)}  [ g(s) + f(s, a) + h(a) ]
where:
    g(s) = cumulative cost so far,
    f(s, a) = edge cost (the transition_cost function in metrics.py),
    h(a) = admissible heuristic estimating remaining cost to the goal.

Setting h ≡ 0 collapses A* to Dijkstra, which we use as a baseline.
"""
from __future__ import annotations
import heapq
import time
from typing import Callable, Dict, List, Optional, Iterable, Tuple
from dataclasses import dataclass, field

from .models import Song
from .metrics import transition_cost, is_feasible


@dataclass
class SearchResult:
    path: Optional[List[str]]            # ordered list of song IDs, or None if no path
    total_cost: float                    # cumulative cost along the path
    nodes_expanded: int                  # number of node pops from the frontier
    edges_evaluated: int                 # number of transition_cost calls during search
    time_seconds: float                  # wall-clock time
    decision_log: List[Dict] = field(default_factory=list)  # per-step decision trace

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def num_transitions(self) -> int:
        return 0 if self.path is None else max(0, len(self.path) - 1)

    def summary(self) -> str:
        if not self.found:
            return (f"NO PATH FOUND | expanded={self.nodes_expanded} | "
                    f"edges={self.edges_evaluated} | time={self.time_seconds*1000:.2f}ms")
        return (f"path_len={len(self.path)} transitions={self.num_transitions} "
                f"cost={self.total_cost:.4f} expanded={self.nodes_expanded} "
                f"edges={self.edges_evaluated} time={self.time_seconds*1000:.2f}ms")


def astar_search(
    library: List[Song],
    start_id: str,
    goal_id: str,
    weights: Optional[Dict[str, float]] = None,
    use_heuristic: bool = True,
    use_feasibility: bool = True,
    feasibility_thresholds: Optional[Dict[str, float]] = None,
    max_path_length: int = 12,
    log_decisions: bool = False,
) -> SearchResult:
    """
    A* search for the lowest-cost sequence of transitions from start_id to goal_id.

    Args:
        library: list of all candidate Song objects
        start_id, goal_id: IDs of the source and target songs
        weights: cost weights for transition_cost; None uses defaults
        use_heuristic: if False, runs as Dijkstra (h ≡ 0)
        use_feasibility: if True, infeasible transitions are excluded from A(s)
        feasibility_thresholds: thresholds for is_feasible(); None uses defaults
        max_path_length: hard cap on number of songs in path (safety bound)
        log_decisions: if True, record details of each expansion

    Returns:
        SearchResult with path (as list of IDs), cost, and search statistics.
    """
    t0 = time.perf_counter()
    index = {s.id: s for s in library}

    if start_id not in index or goal_id not in index:
        return SearchResult(None, float("inf"), 0, 0, time.perf_counter() - t0)

    start = index[start_id]
    goal = index[goal_id]

    # Heuristic: direct linear transition cost (admissible because actual paths
    # through intermediates can only add non-negative edge costs on top of this
    # lower-bound estimate when feasibility filtering excludes the direct edge).
    def h(song: Song) -> float:
        if not use_heuristic:
            return 0.0
        return transition_cost(song, goal, weights)

    edges_evaluated = 0
    decision_log: List[Dict] = []

    # Priority queue entries: (f_score, tiebreak, current_id, g_score, path_so_far)
    counter = 0
    frontier: List[Tuple[float, int, str, float, List[str]]] = []
    heapq.heappush(frontier, (h(start), counter, start_id, 0.0, [start_id]))

    # best_g[node_id] = best g-score found for that node so far
    best_g: Dict[str, float] = {start_id: 0.0}
    nodes_expanded = 0

    while frontier:
        f_score, _, current_id, g_score, path = heapq.heappop(frontier)

        if g_score > best_g.get(current_id, float("inf")):
            continue  # stale entry — already found a better path here

        nodes_expanded += 1
        current = index[current_id]

        if current_id == goal_id:
            elapsed = time.perf_counter() - t0
            return SearchResult(
                path=path,
                total_cost=g_score,
                nodes_expanded=nodes_expanded,
                edges_evaluated=edges_evaluated,
                time_seconds=elapsed,
                decision_log=decision_log,
            )

        if len(path) >= max_path_length:
            continue  # don't expand beyond the safety bound

        # Generate the action space A(s): neighbours we may transition to next
        candidates: List[Tuple[str, float, float]] = []  # (neighbor_id, edge_cost, f_value)
        for neighbor in library:
            if neighbor.id == current_id:
                continue
            if neighbor.id in path:  # no revisits within a single path
                continue
            if use_feasibility and not is_feasible(current, neighbor, feasibility_thresholds):
                continue

            edge_cost = transition_cost(current, neighbor, weights)
            edges_evaluated += 1
            new_g = g_score + edge_cost
            new_f = new_g + h(neighbor)
            candidates.append((neighbor.id, edge_cost, new_f))

            if new_g < best_g.get(neighbor.id, float("inf")):
                best_g[neighbor.id] = new_g
                counter += 1
                heapq.heappush(frontier, (new_f, counter, neighbor.id, new_g, path + [neighbor.id]))

        if log_decisions and candidates:
            # Record the argmin decision at this state — the agent's "thought"
            best = min(candidates, key=lambda x: x[2])
            decision_log.append({
                "step": nodes_expanded,
                "state": current_id,
                "g_so_far": round(g_score, 4),
                "candidates_evaluated": len(candidates),
                "chosen_next_lowest_f": best[0],
                "edge_cost": round(best[1], 4),
                "f_value": round(best[2], 4),
            })

    elapsed = time.perf_counter() - t0
    return SearchResult(
        path=None,
        total_cost=float("inf"),
        nodes_expanded=nodes_expanded,
        edges_evaluated=edges_evaluated,
        time_seconds=elapsed,
        decision_log=decision_log,
    )
