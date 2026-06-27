# Pacomixer — Intelligent Song Transition Planner

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
| --- | --- |
| **State space S** | `s = (current_song, target_song, history, library)` |
| **Action space A(s)** | `{ s' ∈ library : s' ≠ current, s' ∉ history, is_feasible(current, s') }` |
| **Cost function f(s, a)** | `f(s, s') = w_h·d_harm + w_t·d_tempo + w_e·d_embed` |
| **Decision rule** | `a* = argmin_{a ∈ A(s)} [ g(s) + f(s, a) + h(a) ]` |

The three component distances are:

- **`d_harm`**: distance on the Circle of Fifths (normalised to `[0, 1]`,
  with a `+0.15` penalty when mode changes between major and minor).
- **`d_tempo`**: relative BPM difference with an `8%` tolerance band.
  We account for half-time and double-time beatmatching.
- **`d_embed`**: cosine distance between sonic-character embedding vectors.

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
│   ├── exp1_astar_vs_dijkstra.py
│   ├── exp2_weight_sensitivity.py
│   └── exp3_environment_changes.py
├── results/               Output CSVs from experiments
├── music/                 Your audio files (.mp3, .flac, .wav, etc.)
├── data/                  Generated library.json
├── app.py                 Streamlit graphical interface
├── main.py                CLI demonstration
├── extract_library.py     Real audio analyzer (CLAP/essentia/mutagen, incremental cache)
├── run.bat                One-click Windows launcher (conda env + Streamlit)
├── requirements.txt
└── README.md
```

## 4. Installation

```bash
git clone https://github.com/Eddjby/Pacomixer.git
cd Pacomixer
pip install -r requirements.txt
```

> **Optional analyzer enhancements.** `extract_library.py` works out of the
> box with `librosa` (BPM via `beat_track`, key via Krumhansl chroma, MFCC
> embeddings). For higher quality it auto-detects and uses, if installed:
>
> - **CLAP** 512-dim embeddings on GPU — `pip install msclap torch`
> - **essentia** key detection — `pip install essentia`
> - **mutagen** genre tags — `pip install mutagen`
>
> All are optional; the analyzer falls back gracefully when any is missing.

## 5. Usage

### Graphical interface (recommended)

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`. Select start and goal songs
from the dropdowns and click **Planear transición**.

> **Windows shortcut:** double-click `run.bat` to auto-activate the `Pacomixer`
> conda environment and launch the interface in one step.

### Analyze your own music library

Place your audio files (`.mp3`, `.flac`, `.wav`, `.ogg`, `.m4a`, `.aiff`) in a
`music/` folder, then run:

```bash
python extract_library.py ./music
```

This generates `data/library.json` with BPM, key, embedding, and genre for
every song. Name your files as `Artist - Title.mp3` for automatic metadata
parsing.

**Incremental by default.** Re-running only analyzes new or modified files
(each song is fingerprinted by name + size + mtime), so adding a few tracks is
fast. Useful flags:

```bash
python extract_library.py ./music --force      # re-analyze everything
python extract_library.py ./music --prune      # drop songs no longer present
python extract_library.py ./music --output data/custom.json
```

### CLI — quick demo (synthetic library)

```bash
python main.py
```

### CLI — plan a transition with your library

```bash
python main.py --load-library data/library.json --start s001 --goal s098
```

### CLI — list all songs

```bash
python main.py --load-library data/library.json --list
```

### CLI — run as Dijkstra (no heuristic)

```bash
python main.py --load-library data/library.json --start s001 --goal s098 --no-heuristic
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

result = agent.plan(start_id="s01", goal_id="s28")
print(agent.describe_path(result))
```

## 6. Example output

```
Library: 28 songs across 7 genres

>>> Planning transition: s01 → s28
path_len=4 transitions=3 cost=0.5580 expanded=4 edges=26 time=0.87ms

Path of 4 songs, total cost = 0.5580
  [0] Open Architecture — Ovid Park   (BPM=76.8, Key=Gm, Genre=ambient)
        ↓  cost=0.0859  [harm=0.167, tempo=0.000, sem=0.064]
  [1] Crystal Room — Indra Pavón      (BPM=82.3, Key=Dm, Genre=ambient)
        ↓  cost=0.3228  [harm=0.333, tempo=0.000, sem=0.631]
  [2] Static Echo — Lunaris           (BPM=175.2, Key=Em, Genre=dnb)
        ↓  cost=0.1493  [harm=0.333, tempo=0.000, sem=0.053]
  [3] Iron Promise — S. Kobayashi     (BPM=172.9, Key=F#m, Genre=dnb)
```

## 7. Experimental results (summary)

**Experiment 1 — A* vs Dijkstra.** Over 10 (start, goal) pairs, A*
expands on average **82.8%** fewer nodes and evaluates **86.2%** fewer
edges than Dijkstra, with identical path cost in 8/10 cases.

**Experiment 2 — Weight sensitivity.** Sweeping `(w_h, w_t, w_e)` over
66 points of the unit simplex produces **4 distinct paths**.

**Experiment 3 — Environment changes.** When an intermediate song becomes
unavailable mid-set, the agent successfully replans in every test case,
with an average cost delta of `+0.08`.

## 8. Known limitations

- `genre` is read from each file's ID3 tags (via `mutagen`); files without a
  genre tag are labelled `"Unknown"`. Genre is **not** inferred from the audio
  signal itself — content-based detection (e.g. CLAP text similarity) is still
  planned.
- The path-storage A* implementation enforces a "no song twice within a
  path" constraint, which can occasionally produce paths ~1–7% worse than
  the unrestricted optimum.

## 9. Roadmap

- [x] CLAP audio embeddings (512 dims) with MFCC (16 dims) fallback
- [x] GPU acceleration via PyTorch + CUDA for CLAP embeddings
- [x] Genre detection from file metadata (ID3 tags via `mutagen`)
- [ ] Content-based genre detection from the audio signal (CLAP text similarity)
- [ ] essentia-based key detection bundled and enabled by default

## 10. Differences from the base repository

| Original repo | This project |
| --- | --- |
| Implicit cost function | Explicit `f(s, a)` documented as the agent's objective |
| Dijkstra over a static graph | A* with admissible heuristic + Dijkstra baseline |
| No agent class | `TransitionAgent` with `perceive(state) → action` interface |
| No experiment suite | Three experiments with CSV outputs |
| Static planning only | `replan_from` supports environment changes |
| Heavy audio dependencies | Audio pipeline decoupled; synthetic data for reproducible demos |
| CLI only | Streamlit graphical interface (`app.py`) |
