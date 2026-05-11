"""
main.py — CLI demonstration of the Auto-DJ agent.

Usage:
    python main.py                          # default demo (ambient → DnB)
    python main.py --start s01 --goal s28
    python main.py --start s01 --goal s28 --no-heuristic   # run Dijkstra
    python main.py --list                   # print library summary

The demo shows:
    (1) The library being loaded.
    (2) The agent perceiving an initial state and planning a full path.
    (3) Per-step transition costs broken into harmonic / tempo / semantic.
    (4) A second run that demonstrates replanning after a song is removed.
"""
from __future__ import annotations
import argparse
import sys
import os

# Make the src package importable when running this file directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import generate_synthetic_library, print_library_summary
from src.agent import TransitionAgent, AgentState
from src.models import save_library, load_library


def main():
    parser = argparse.ArgumentParser(description="Pacomixer — transition path planner")
    parser.add_argument("--start", default="s01", help="Starting song ID (default s01)")
    parser.add_argument("--goal", default="s28", help="Target song ID (default s28)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic library")
    parser.add_argument("--n-songs", type=int, default=28, help="Library size")
    parser.add_argument("--no-heuristic", action="store_true",
                        help="Disable A* heuristic (run Dijkstra)")
    parser.add_argument("--no-feasibility", action="store_true",
                        help="Allow any transition (no feasibility filter)")
    parser.add_argument("--list", action="store_true",
                        help="Print library summary and exit")
    parser.add_argument("--save-library", metavar="PATH",
                        help="Save the generated library to this JSON path")
    parser.add_argument("--load-library", metavar="PATH",
                        help="Load library from JSON instead of generating")
    args = parser.parse_args()

    # ---- 1. Build / load the library ----
    if args.load_library:
        library = load_library(args.load_library)
        print(f"Loaded library from {args.load_library}")
    else:
        library = generate_synthetic_library(n_songs=args.n_songs, seed=args.seed)

    if args.save_library:
        save_library(library, args.save_library)
        print(f"Saved library to {args.save_library}")

    print_library_summary(library)
    print()

    if args.list:
        for s in library:
            print(f"  {s}")
        return

    # ---- 2. Build the agent ----
    agent = TransitionAgent(
        library=library,
        use_heuristic=not args.no_heuristic,
        use_feasibility=not args.no_feasibility,
        verbose=False,
    )
    algo_name = "Dijkstra" if args.no_heuristic else "A*"
    print(f"=== Agent ready: algorithm = {algo_name}, "
          f"feasibility = {not args.no_feasibility} ===\n")

    # ---- 3. Plan a path ----
    print(f">>> Planning transition: {args.start}  →  {args.goal}")
    result = agent.plan(args.start, args.goal, log_decisions=True)
    print(result.summary())
    print()
    print(agent.describe_path(result))
    print()

    if result.found and result.decision_log:
        print("First few argmin decisions (perceive → action):")
        for entry in result.decision_log[:5]:
            print(f"  step {entry['step']:2d}: state={entry['state']} "
                  f"g={entry['g_so_far']:.3f}  candidates={entry['candidates_evaluated']}  "
                  f"→ chose {entry['chosen_next_lowest_f']} (f={entry['f_value']:.3f})")
        print()

    # ---- 4. Demonstrate replanning (environment change) ----
    if result.found and len(result.path) >= 3:
        removed = result.path[1]  # remove the agent's first chosen stepping stone
        print(f">>> Environment change: song '{removed}' becomes unavailable. Replanning…")
        result2 = agent.replan_from(
            current_id=args.start,
            new_goal_id=args.goal,
            played_so_far=[],
            excluded_ids={removed},
        )
        print(result2.summary())
        if result2.found:
            old_path = " → ".join(result.path)
            new_path = " → ".join(result2.path)
            print(f"  original path : {old_path}")
            print(f"  new path      : {new_path}")
            if new_path != old_path:
                print("  ✓ agent re-routed around the removed song")


if __name__ == "__main__":
    main()
