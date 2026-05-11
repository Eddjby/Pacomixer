"""
Experiment 2 — Weight sensitivity.

Sweep the three weights (w_harmonic, w_tempo, w_semantic) on the unit
simplex (each weight in {0.0, 0.1, ..., 1.0}, summing to 1.0) and observe
how the chosen path changes for a fixed (start, goal) pair.

We log: total cost, path length, the average value of each cost component
along the chosen path. This shows the agent's preferences responding to
how we prioritise the three audio dimensions.
"""
from __future__ import annotations
import sys
import os
import csv
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import generate_synthetic_library
from src.agent import TransitionAgent
from src.metrics import transition_components

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run(start: str = "s01", goal: str = "s23", step: float = 0.1):
    library = generate_synthetic_library(n_songs=28, seed=42)
    index = {s.id: s for s in library}

    # Generate all (w_h, w_t, w_e) on the simplex with given step
    n = int(round(1.0 / step))
    triples = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            triples.append((i * step, j * step, k * step))

    rows = []
    unique_paths = {}
    for w_h, w_t, w_e in triples:
        weights = {"harmonic": w_h, "tempo": w_t, "semantic": w_e}
        agent = TransitionAgent(library, weights=weights)
        r = agent.plan(start, goal)
        if not r.found:
            rows.append({
                "w_harmonic": round(w_h, 2),
                "w_tempo": round(w_t, 2),
                "w_semantic": round(w_e, 2),
                "found": False,
                "total_cost": None,
                "path_length": 0,
                "path": "",
                "avg_harm": None, "avg_tempo": None, "avg_sem": None,
            })
            continue

        # Per-step components along the chosen path
        h_vals, t_vals, s_vals = [], [], []
        for a_id, b_id in zip(r.path[:-1], r.path[1:]):
            c = transition_components(index[a_id], index[b_id])
            h_vals.append(c["harmonic"])
            t_vals.append(c["tempo"])
            s_vals.append(c["semantic"])

        path_str = " → ".join(r.path)
        unique_paths.setdefault(path_str, []).append((w_h, w_t, w_e))

        rows.append({
            "w_harmonic": round(w_h, 2),
            "w_tempo": round(w_t, 2),
            "w_semantic": round(w_e, 2),
            "found": True,
            "total_cost": round(r.total_cost, 4),
            "path_length": len(r.path),
            "path": path_str,
            "avg_harm": round(sum(h_vals) / len(h_vals), 4),
            "avg_tempo": round(sum(t_vals) / len(t_vals), 4),
            "avg_sem": round(sum(s_vals) / len(s_vals), 4),
        })

    csv_path = os.path.join(RESULTS_DIR, "exp2_weight_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"Pair: {start} → {goal}")
    print(f"Total weight combinations evaluated: {len(rows)}")
    n_found = sum(1 for r in rows if r["found"])
    print(f"Combinations yielding a feasible path: {n_found}/{len(rows)}")
    print(f"Number of distinct paths discovered    : {len(unique_paths)}")
    print()
    print("Top 5 most-common paths and the weight regions that produced them:")
    for path_str, weight_list in sorted(unique_paths.items(),
                                        key=lambda x: -len(x[1]))[:5]:
        print(f"  path ({len(weight_list):2d} configs): {path_str}")
        # average weights that selected this path
        avg_w = [sum(w[i] for w in weight_list) / len(weight_list) for i in range(3)]
        print(f"    average weights in regime  → w_h={avg_w[0]:.2f}  "
              f"w_t={avg_w[1]:.2f}  w_e={avg_w[2]:.2f}")
    print()
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    run()
