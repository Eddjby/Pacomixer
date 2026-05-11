"""
Experiment 3 — Robustness to environment changes.

Demonstrates that the agent responds to changes in the environment
(the requirement "responde a cambios en el entorno" in the rubric).

Three scenarios per (start, goal) pair:

    A) Baseline plan with the full library.
    B) Removed-song replan: at runtime, one of the intermediate stepping
       stones becomes unavailable. The agent replans from its current
       position toward the same goal.
    C) Goal-change replan: mid-set the target changes. The agent replans
       from its current position toward the new target.

For each scenario we record path, cost, and number of replan operations
needed before reaching the (new or original) goal.
"""
from __future__ import annotations
import sys
import os
import csv
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import generate_synthetic_library
from src.agent import TransitionAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def simulate_removal_scenario(agent: TransitionAgent, start: str, goal: str, rng: random.Random):
    """Plan, then remove a random non-endpoint song from the path, then replan."""
    initial = agent.plan(start, goal)
    if not initial.found or len(initial.path) <= 2:
        return None

    # Pick a song in the middle of the path to remove
    middle_idx = rng.randint(1, len(initial.path) - 2)
    removed = initial.path[middle_idx]

    # The agent is currently at position middle_idx - 1 in the path.
    current = initial.path[middle_idx - 1]
    played_so_far = initial.path[:middle_idx - 1]

    replanned = agent.replan_from(
        current_id=current,
        new_goal_id=goal,
        played_so_far=played_so_far,
        excluded_ids={removed},
    )

    return {
        "scenario": "song_removed",
        "start": start,
        "goal": goal,
        "removed_song": removed,
        "removed_at_step": middle_idx,
        "original_path": " → ".join(initial.path),
        "original_cost": round(initial.total_cost, 4),
        "replanned_found": replanned.found,
        "replanned_path": " → ".join(replanned.path) if replanned.found else "",
        "replanned_cost": round(replanned.total_cost, 4) if replanned.found else None,
        "extra_cost": (round(replanned.total_cost - initial.total_cost, 4)
                       if replanned.found else None),
    }


def simulate_goal_change(agent: TransitionAgent, start: str, original_goal: str,
                         new_goal: str):
    """Plan toward original goal, then halfway through change to a new target."""
    initial = agent.plan(start, original_goal)
    if not initial.found or len(initial.path) < 2:
        return None
    midway_idx = max(1, len(initial.path) // 2)
    current = initial.path[midway_idx - 1]
    played_so_far = initial.path[:midway_idx - 1]

    replanned = agent.replan_from(
        current_id=current,
        new_goal_id=new_goal,
        played_so_far=played_so_far,
    )
    return {
        "scenario": "goal_changed",
        "start": start,
        "goal": original_goal,
        "new_goal": new_goal,
        "switched_at_step": midway_idx,
        "switched_at_song": current,
        "original_path": " → ".join(initial.path),
        "original_cost": round(initial.total_cost, 4),
        "replanned_found": replanned.found,
        "replanned_path": " → ".join(replanned.path) if replanned.found else "",
        "replanned_cost": round(replanned.total_cost, 4) if replanned.found else None,
    }


def run():
    library = generate_synthetic_library(n_songs=28, seed=42)
    agent = TransitionAgent(library)
    rng = random.Random(123)

    scenarios = [
        ("s01", "s28"),  # ambient → dnb
        ("s05", "s24"),  # downtempo → techno
        ("s09", "s20"),  # hiphop → house
        ("s13", "s25"),  # indie → dnb
        ("s02", "s22"),  # ambient → techno
    ]

    rows = []
    print("=== Scenario B: a song becomes unavailable mid-set ===\n")
    for start, goal in scenarios:
        r = simulate_removal_scenario(agent, start, goal, rng)
        if r is None:
            continue
        rows.append(r)
        print(f"  {start} → {goal}")
        print(f"    original   : {r['original_path']}  (cost {r['original_cost']})")
        print(f"    removed    : '{r['removed_song']}' at step {r['removed_at_step']}")
        if r["replanned_found"]:
            print(f"    replanned  : {r['replanned_path']}  (cost {r['replanned_cost']}, "
                  f"Δ={r['extra_cost']:+.4f})")
        else:
            print(f"    replanned  : NO PATH FOUND")
        print()

    print("\n=== Scenario C: goal changes mid-set ===\n")
    goal_changes = [
        ("s01", "s17", "s28"),  # was going to house, now want dnb
        ("s09", "s22", "s05"),  # was going to techno, now want downtempo
        ("s13", "s25", "s17"),  # was going to dnb, now want house
        ("s02", "s28", "s20"),  # was going to dnb, now want house
    ]
    for start, original_goal, new_goal in goal_changes:
        r = simulate_goal_change(agent, start, original_goal, new_goal)
        if r is None:
            continue
        rows.append(r)
        print(f"  {start} → {original_goal}, then switched to {new_goal} at step {r['switched_at_step']}")
        print(f"    original   : {r['original_path']}  (cost {r['original_cost']})")
        if r["replanned_found"]:
            print(f"    replanned  : {r['replanned_path']}  (cost {r['replanned_cost']})")
        else:
            print(f"    replanned  : NO PATH FOUND")
        print()

    csv_path = os.path.join(RESULTS_DIR, "exp3_environment_changes.csv")
    # Flatten field set (each row has different keys depending on scenario)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    run()
