"""
Experiment 1 — A* vs Dijkstra.

Both algorithms are guaranteed to find optimal paths in this problem
(the heuristic in A* is admissible). The expected finding is that A*
expands fewer nodes and evaluates fewer edges than Dijkstra while
returning paths of identical total cost.

This experiment runs both algorithms on a battery of (start, goal) pairs
and writes a CSV of the comparison plus a printable summary table.
"""
from __future__ import annotations
import sys
import os
import csv
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import generate_synthetic_library
from src.agent import TransitionAgent


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run():
    library = generate_synthetic_library(n_songs=28, seed=42)
    pairs = [
        ("s01", "s28"),   # ambient → dnb (very far)
        ("s05", "s24"),   # downtempo → techno
        ("s09", "s20"),   # hiphop → house
        ("s13", "s25"),   # indie → dnb
        ("s01", "s17"),   # ambient → house
        ("s02", "s22"),   # ambient → techno
        ("s10", "s26"),   # hiphop → dnb
        ("s14", "s23"),   # indie → techno
        ("s06", "s19"),   # downtempo → house
        ("s03", "s27"),   # ambient → dnb (different start)
    ]

    rows = []
    print(f"{'pair':>14} | {'algorithm':>10} | {'cost':>7} | "
          f"{'len':>3} | {'expanded':>8} | {'edges':>6} | {'time_ms':>8}")
    print("-" * 80)
    for start, goal in pairs:
        for algo_name, use_h in [("Dijkstra", False), ("A*", True)]:
            agent = TransitionAgent(library, use_heuristic=use_h)
            r = agent.plan(start, goal)
            row = {
                "start": start,
                "goal": goal,
                "algorithm": algo_name,
                "found": r.found,
                "total_cost": round(r.total_cost, 4) if r.found else None,
                "path_length": len(r.path) if r.found else 0,
                "nodes_expanded": r.nodes_expanded,
                "edges_evaluated": r.edges_evaluated,
                "time_ms": round(r.time_seconds * 1000, 3),
            }
            rows.append(row)
            cost_str = f"{r.total_cost:.4f}" if r.found else "  NA  "
            print(f"{start+' → '+goal:>14} | {algo_name:>10} | {cost_str:>7} | "
                  f"{len(r.path) if r.found else 0:>3} | {r.nodes_expanded:>8} | "
                  f"{r.edges_evaluated:>6} | {r.time_seconds*1000:>8.3f}")
        print()

    csv_path = os.path.join(RESULTS_DIR, "exp1_astar_vs_dijkstra.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {csv_path}")

    # Aggregate analysis
    print("\nAggregate over all pairs where both found a path:")
    paired = list(zip(rows[0::2], rows[1::2]))
    same_cost_count = sum(1 for d, a in paired
                          if d["found"] and a["found"]
                          and abs((d["total_cost"] or 0) - (a["total_cost"] or 0)) < 1e-6)
    avg_dij_exp = sum(d["nodes_expanded"] for d, _ in paired) / len(paired)
    avg_astar_exp = sum(a["nodes_expanded"] for _, a in paired) / len(paired)
    avg_dij_edg = sum(d["edges_evaluated"] for d, _ in paired) / len(paired)
    avg_astar_edg = sum(a["edges_evaluated"] for _, a in paired) / len(paired)

    print(f"  same-cost rate     : {same_cost_count}/{len(paired)} (optimality confirmation)")
    print(f"  avg nodes expanded : Dijkstra={avg_dij_exp:.1f}, A*={avg_astar_exp:.1f} "
          f"({100*(1-avg_astar_exp/avg_dij_exp):.1f}% reduction)")
    print(f"  avg edges evaluated: Dijkstra={avg_dij_edg:.1f}, A*={avg_astar_edg:.1f} "
          f"({100*(1-avg_astar_edg/avg_dij_edg):.1f}% reduction)")


if __name__ == "__main__":
    run()
