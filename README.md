# Auto-DJ Agent — Intelligent Song Transition Planner

Final project for *Inteligencia Artificial*, Universidad de las Américas Puebla.

This system implements an **intelligent agent** that, given a starting song
and a target song from a library, finds the lowest-cost sequence of
transitions that takes one to the other. The agent uses **A* search** with
an admissible heuristic over a cost function combining three audio dimensions:
**harmonic compatibility, tempo distance, and sonic similarity** (embedding
cosine distance).

---

## 1. Problem and objective

DJs face the problem of moving from a current track to a desired track
without harsh transitions. Going *directly* between two distant tracks
(e.g. a 75 BPM ambient piece in G minor and a 174 BPM drum-and-bass piece
in F# minor) is rarely possible — but the right *sequence* of intermediate
tracks makes it smooth. Our agent automates this decision.

## 2. Formal agent definition

| Component | Definition |
|---|---|
| **State space S** | `s = (current_song, target_song, history, library)` |
| **Action space A(s)** | `{ s' ∈ library : s' ≠ current, s' ∉ history, is_feasible(current, s') }` |
| **Cost function f(s, a)** | `f(s, s') = w_h·d_harm + w_t·d_tempo + w_e·d_embed` |
| **Decision rule** | `a* = argmin_{a ∈ A(s)} [ g(s) + f(s, a) + h(a) ]` |

The three component distances are:
- **`d_harm`**: distance on the Circle of Fifths (normalised to `[0, 1]`,
  with a `+0.15` penalty when mode changes between major and minor).
- **`d_tempo`**: relative BPM difference with an `8%` tolerance band. We
  account for the standard DJ practice of half-time and double-time
  beatmatching: 87 BPM perceptually aligns with 174 BPM (2×) and vice-versa.
- **`d_embed`**: cosine distance between sonic-character embedding vectors
  (analogue of CLAP embeddings in the original repo).

The agent uses the **direct transition cost** as the A* heuristic, which is
admissible because the cost function is a weighted sum of metric distances
and therefore satisfies the triangle inequality on the unrestricted graph.

## 3. Project structure

```
Pacomixer/
├── src/
│   ├── models.py          Song dataclass + JSON load/save
│   ├── metrics.py         f(s, a): harmonic + tempo + semantic distances
│   ├── astar.py           A* search (and Dijkstra as a special case)
│   ├── agent.py           TransitionAgent: perceive(s) → a*, plan, replan
│   └── data_loader.py     Synthetic library generator (genre clusters)
├── experiments/
│   ├── exp1_astar_vs_dijkstra.py     A* vs Dijkstra comparison
│   ├── exp2_weight_sensitivity.py    Weight-simplex sweep
│   └── exp3_environment_changes.py   Replanning under perturbations
├── results/                Output CSVs from experiments
├── data/                   (saved libraries, optional)
├── main.py                 CLI demonstration
├── requirements.txt
└── README.md
```

## 4. Installation

```bash
git clone <repo-url>
cd Pacomixer
pip install -r requirements.txt
```

Only `numpy` is required for the synthetic-data version. If you wish to
use real audio (the original repo's pipeline), additionally install
`librosa`, `madmom`, `essentia`, and the CLAP model.

## 5. Usage

### Quick demo

```bash
python main.py
```

This generates a 28-song synthetic library across 7 genres (ambient,
downtempo, hip-hop, indie, house, techno, drum-and-bass) and plans a
transition from `s01` (ambient) to `s28` (DnB). It then demonstrates
replanning when one of the intermediate songs becomes unavailable.

### Custom start and goal

```bash
python main.py --start s05 --goal s24
```

### Run as Dijkstra (no heuristic)

```bash
python main.py --start s01 --goal s28 --no-heuristic
```

### List the library

```bash
python main.py --list
```

### Run the experiments

```bash
python experiments/exp1_astar_vs_dijkstra.py
python experiments/exp2_weight_sensitivity.py
python experiments/exp3_environment_changes.py
```

CSV results are written to `results/`.

### Python API

```python
from src.data_loader import generate_synthetic_library
from src.agent import TransitionAgent, AgentState

library = generate_synthetic_library(n_songs=28, seed=42)
agent = TransitionAgent(library)

# Full planning
result = agent.plan(start_id="s01", goal_id="s28")
print(agent.describe_path(result))

# Step-by-step perceive → action
state = AgentState(current_id="s01", target_id="s28",
                   library_ids={s.id for s in library})
next_song_id = agent.perceive(state)

# Replanning after an environment change
new_result = agent.replan_from(
    current_id="s06", new_goal_id="s28",
    played_so_far=["s01", "s03"],
    excluded_ids={"s15"},   # song became unavailable
)
```

## 6. Example output

```
Library: 28 songs across 7 genres
  ambient     n= 4  BPM range  70.1– 82.3
  downtempo   n= 4  BPM range  85.5– 91.1
  ...

>>> Planning transition: s01  →  s28
path_len=4 transitions=3 cost=0.5580 expanded=4 edges=26 time=0.87ms

Path of 4 songs, total cost = 0.5580
  [0] Open Architecture — Ovid Park   (BPM=76.8, Key=Gm, Genre=ambient)
        ↓  cost=0.0859  [harm=0.167, tempo=0.000, sem=0.064]
  [1] Crystal Room — Indra Pavón      (BPM=82.3, Key=Dm, Genre=ambient)
        ↓  cost=0.3228  [harm=0.333, tempo=0.000, sem=0.631]
  [2] Static Echo — Lunaris           (BPM=175.2, Key=Em, Genre=dnb)
        ↓  cost=0.1493  [harm=0.333, tempo=0.000, sem=0.053]
  [3] Iron Promise — S. Kobayashi     (BPM=172.9, Key=F#m, Genre=dnb)

First few argmin decisions (perceive → action):
  step  1: state=s01 g=0.000  candidates=6  → chose s03 (f=0.527)
  step  2: state=s03 g=0.086  candidates=9  → chose s26 (f=0.558)
  step  3: state=s26 g=0.409  candidates=11  → chose s28 (f=0.558)
```

## 7. Experimental results (summary)

**Experiment 1 — A* vs Dijkstra.** Over 10 (start, goal) pairs, A*
expands on average **82.8%** fewer nodes and evaluates **86.2%** fewer
edges than Dijkstra, with identical path cost in 8/10 cases.

**Experiment 2 — Weight sensitivity.** Sweeping `(w_h, w_t, w_e)` over
66 points of the unit simplex for a fixed pair `s01 → s23` produces
**4 distinct paths**. Tempo-dominated weight regimes select longer
paths with smoother tempo transitions; embedding-dominated regimes
prefer paths through sonically similar bridges.

**Experiment 3 — Environment changes.** When an intermediate song
becomes unavailable mid-set, the agent successfully replans from its
current state in every test case, with an average cost delta of
`+0.08`. When the goal changes mid-set, the agent also adapts and in
some cases finds *better* paths than the original plan.

## 8. Known limitation

The path-storage A* implementation enforces a "no song twice within a
path" constraint. Combined with the standard `best_g`-based pruning,
this can occasionally produce paths slightly worse than the unrestricted
optimum (about 1–7% in our experiments). A fully optimal solution would
require indexing `best_g` by `(node, history)`, which is exponential in
the library size. The current trade-off is acceptable for libraries of
∼30 tracks.

## 9. Differences from the base repository

This project takes the analytical idea from
[TheMexicanTarzan/Auto-DJ](https://github.com/TheMexicanTarzan/Auto-DJ)
(BPM, key, and CLAP embedding for transition cost) and rebuilds it
around an **explicit agent abstraction** that the rubric requires:

| Original repo | This project |
|---|---|
| Implicit cost function inside `metrics.py` | Explicit `f(s, a)` documented as the agent's objective |
| Dijkstra over a static graph | A* with admissible heuristic + Dijkstra baseline for comparison |
| No agent class | `TransitionAgent` with `perceive(state) → action` interface |
| No experiment suite | Three experiments with CSV outputs |
| Static planning only | `replan_from` supports environment changes |
| Heavy audio dependencies | Audio pipeline decoupled from the agent; synthetic data for reproducible demos |
