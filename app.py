"""
app.py — Pacomixer · Interfaz gráfica con Streamlit

Uso:
    streamlit run app.py          (o doble clic en run.bat)

Requiere que data/library.json exista, o genéralo desde la pestaña Extracción.
"""
from __future__ import annotations
import sys
import os
import subprocess
import time
from datetime import datetime

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import load_library
from src.agent import TransitionAgent
from src.metrics import transition_cost, transition_components, DEFAULT_WEIGHTS

# ─── Configuración ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pacomixer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LIBRARY_PATH = "data/library.json"
MUSIC_DIR    = "./music"

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Ocultar chrome default */
    #MainMenu, header, footer { visibility: hidden; }
    [data-testid="stToolbar"]    { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    .stDeployButton              { display: none !important; }
    [data-testid="stSidebar"]    { display: none !important; }

    /* Base */
    .stApp { background: #0e0e13; color: #e4e4e7; }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* Tabs */
    [data-testid="stTabs"] { margin-top: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid #22222b;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #71717a;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        padding: 0.6rem 1.2rem;
        border-bottom: 2px solid transparent;
        transition: color 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #e4e4e7 !important;
        border-bottom: 2px solid #a78bfa !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

    /* Tipografía */
    .pcm-title {
        font-size: 2rem; font-weight: 700; color: #e4e4e7;
        letter-spacing: -0.025em; margin: 0; line-height: 1.1;
    }
    .pcm-title .dot { color: #a78bfa; }
    .pcm-subtitle {
        color: #71717a; font-size: 0.88rem;
        margin: 0.3rem 0 0 0; font-weight: 400;
    }
    .pcm-header-meta {
        text-align: right; padding-top: 0.6rem;
        color: #71717a; font-size: 0.82rem;
        font-family: ui-monospace, 'SF Mono', Monaco, monospace;
        font-feature-settings: "tnum";
    }
    .pcm-header-meta strong { color: #e4e4e7; font-weight: 600; }
    .pcm-section {
        font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: #71717a;
        font-weight: 600; margin: 1.8rem 0 0.9rem 0;
    }
    .pcm-section:first-child { margin-top: 0.3rem; }

    /* Stat cards */
    .pcm-stat {
        background: #16161e; border: 1px solid #22222b;
        border-radius: 10px; padding: 0.95rem 1.15rem;
    }
    .pcm-stat-value {
        font-size: 1.55rem; font-weight: 700; color: #e4e4e7;
        font-family: ui-monospace, 'SF Mono', Monaco, monospace;
        font-feature-settings: "tnum"; line-height: 1.15;
        letter-spacing: -0.01em;
    }
    .pcm-stat-value.accent { color: #a78bfa; }
    .pcm-stat-label {
        font-size: 0.68rem; color: #71717a;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin-top: 0.3rem; font-weight: 600;
    }

    /* Song row */
    .pcm-song {
        background: #16161e; border: 1px solid #22222b;
        border-left: 3px solid #3f3f46; border-radius: 10px;
        padding: 0.85rem 1.15rem; margin: 0;
    }
    .pcm-song.start { border-left-color: #10b981; }
    .pcm-song.end   { border-left-color: #f43f5e; }
    .pcm-song.mid   { border-left-color: #a78bfa; }
    .pcm-song-title {
        font-size: 0.95rem; font-weight: 600; color: #e4e4e7;
    }
    .pcm-song-meta {
        font-size: 0.78rem; color: #a1a1aa; margin-top: 0.3rem;
        font-family: ui-monospace, 'SF Mono', Monaco, monospace;
        font-feature-settings: "tnum";
    }
    .pcm-song-meta .sep   { color: #3f3f46; margin: 0 0.35rem; }
    .pcm-song-meta .genre { color: #c4b5fd; }
    .pcm-id {
        display: inline-block;
        font-family: ui-monospace, 'SF Mono', Monaco, monospace;
        font-size: 0.7rem; color: #71717a; background: #22222b;
        padding: 1px 7px; border-radius: 4px;
        margin-right: 0.6rem; font-weight: 500;
    }

    /* Connector */
    .pcm-connector {
        margin: 0.2rem 0 0.2rem 1.55rem;
        padding: 0.45rem 0 0.45rem 1.1rem;
        border-left: 2px dashed #2d2d38;
        font-family: ui-monospace, 'SF Mono', Monaco, monospace;
        font-size: 0.73rem; color: #a1a1aa;
        display: flex; gap: 1.2rem; flex-wrap: wrap; align-items: center;
    }
    .pcm-connector .total { color: #a78bfa; font-weight: 700; }
    .pcm-connector .lbl {
        color: #52525b; margin-right: 0.25rem;
        text-transform: uppercase; font-size: 0.65rem; letter-spacing: 0.05em;
    }

    /* Controles (dentro de columna izquierda) */
    .pcm-ctrl-label {
        font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: #71717a;
        font-weight: 600; margin: 1.3rem 0 0.4rem 0;
    }
    .pcm-ctrl-label:first-child { margin-top: 0; }
    .stSelectbox label,
    .stMultiSelect label,
    .stSlider label,
    [data-testid="stToggle"] label {
        font-size: 0.8rem !important;
        color: #d4d4d8 !important;
        font-weight: 500 !important;
    }
    .pcm-caption {
        color: #71717a; font-size: 0.73rem;
        margin-top: -0.3rem; margin-bottom: 0.5rem;
        font-family: ui-monospace, 'SF Mono', Monaco, monospace;
    }
    .pcm-caption.warn { color: #fbbf24; }

    /* Botón primario */
    .stButton button[kind="primary"] {
        background: #7c3aed; border: none;
        font-weight: 600; letter-spacing: 0.01em;
    }
    .stButton button[kind="primary"]:hover { background: #6d28d9; }

    /* Separador sutil dentro de controles */
    .pcm-divider {
        border: none; border-top: 1px solid #22222b;
        margin: 1.2rem 0;
    }

    /* Diff replanificación */
    .pcm-diff {
        background: #16161e; border: 1px solid #22222b;
        border-radius: 10px; padding: 1rem 1.15rem;
    }
    .pcm-diff h4 {
        color: #71717a; font-size: 0.72rem;
        text-transform: uppercase; letter-spacing: 0.1em;
        font-weight: 600; margin: 0 0 0.7rem 0;
    }
    .pcm-diff ol {
        padding-left: 1.2rem; margin: 0;
        color: #d4d4d8; font-size: 0.86rem; line-height: 1.7;
    }
    .pcm-diff .removed { color: #f43f5e; text-decoration: line-through; }
    .pcm-diff .new     { color: #10b981; font-weight: 600; }

    /* Extracción: log */
    .pcm-log-header {
        font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: #71717a;
        font-weight: 600; margin-bottom: 0.5rem;
    }
    .pcm-status-card {
        background: #16161e; border: 1px solid #22222b;
        border-radius: 10px; padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
    }
    .pcm-status-card .lbl {
        font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: #71717a; font-weight: 600;
    }
    .pcm-status-card .val {
        font-size: 0.92rem; color: #e4e4e7;
        font-family: ui-monospace, 'SF Mono', Monaco, monospace;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Carga de librería ────────────────────────────────────────────────────────

@st.cache_resource
def load_lib(path: str):
    return load_library(path)

library_exists = os.path.exists(LIBRARY_PATH)
library = load_lib(LIBRARY_PATH) if library_exists else []
song_map   = {s.id: s for s in library}
all_genres = sorted({s.genre for s in library})

# ─── Cabecera ─────────────────────────────────────────────────────────────────

h1, h2 = st.columns([3, 2])
with h1:
    st.markdown('<p class="pcm-title">Pacomixer<span class="dot">.</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="pcm-subtitle">Planificador de transiciones · armonía · tempo · timbre · género</p>', unsafe_allow_html=True)
with h2:
    if library:
        st.markdown(
            f"<div class='pcm-header-meta'>"
            f"<strong>{len(library)}</strong> canciones · "
            f"<strong>{len(all_genres)}</strong> géneros</div>",
            unsafe_allow_html=True,
        )

# Banner de éxito tras extracción
if st.session_state.get("extraction_done"):
    st.success(st.session_state.pop("extraction_done"))

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_plan, tab_lib, tab_extract = st.tabs(["Planificador", "Librería", "Extracción"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Planificador
# ══════════════════════════════════════════════════════════════════════════════

with tab_plan:
    if not library:
        st.info("No hay librería cargada. Ve a la pestaña **Extracción** para analizar tu música.")
    else:
        ctrl, _, results = st.columns([1, 0.08, 2.3])

        # ─ Columna de controles ─
        with ctrl:
            st.markdown('<div class="pcm-ctrl-label">Género</div>', unsafe_allow_html=True)
            selected_genres = st.multiselect(
                "Filtrar por género",
                options=all_genres,
                default=all_genres,
                label_visibility="collapsed",
                help="Filtra los selectores de canción. El A* usa siempre toda la librería.",
            )

            filtered_lib = [s for s in library if s.genre in selected_genres] or library

            def song_label(s) -> str:
                return f"{s.artist} — {s.title}  ·  {s.bpm:.0f} BPM · {s.key} · {s.genre}"

            song_options = {song_label(s): s.id for s in filtered_lib}
            song_labels  = list(song_options.keys())

            st.markdown('<div class="pcm-ctrl-label">Canción inicial</div>', unsafe_allow_html=True)
            start_label = st.selectbox("Inicial", song_labels, index=0, label_visibility="collapsed")

            st.markdown('<div class="pcm-ctrl-label">Canción objetivo</div>', unsafe_allow_html=True)
            goal_label = st.selectbox(
                "Objetivo", song_labels,
                index=min(len(song_labels) - 1, 10),
                label_visibility="collapsed",
            )

            st.markdown('<hr class="pcm-divider">', unsafe_allow_html=True)
            st.markdown('<div class="pcm-ctrl-label">Pesos</div>', unsafe_allow_html=True)

            w_harm  = st.slider("Armonía", 0.0, 1.0, float(DEFAULT_WEIGHTS["harmonic"]), 0.05, format="%.2f")
            w_tempo = st.slider("Tempo",   0.0, 1.0, float(DEFAULT_WEIGHTS["tempo"]),    0.05, format="%.2f")
            w_emb   = st.slider("Timbre",  0.0, 1.0, float(DEFAULT_WEIGHTS["semantic"]), 0.05, format="%.2f")
            w_genre = st.slider("Género",  0.0, 1.0, float(DEFAULT_WEIGHTS["genre"]),    0.05, format="%.2f")

            total_w = w_harm + w_tempo + w_emb + w_genre
            if total_w <= 0:
                st.markdown('<div class="pcm-caption warn">Todos los pesos en 0 · usando defaults</div>', unsafe_allow_html=True)
                weights = dict(DEFAULT_WEIGHTS)
            elif abs(total_w - 1.0) > 0.001:
                st.markdown(f'<div class="pcm-caption">Suma: {total_w:.2f} · se normalizará a 1.00</div>', unsafe_allow_html=True)
                weights = {
                    "harmonic": w_harm  / total_w,
                    "tempo":    w_tempo / total_w,
                    "semantic": w_emb   / total_w,
                    "genre":    w_genre / total_w,
                }
            else:
                weights = {"harmonic": w_harm, "tempo": w_tempo, "semantic": w_emb, "genre": w_genre}

            st.markdown('<hr class="pcm-divider">', unsafe_allow_html=True)
            st.markdown('<div class="pcm-ctrl-label">Algoritmo</div>', unsafe_allow_html=True)

            use_heuristic   = st.toggle("Heurística A*", value=True,
                                        help="Desactiva para correr Dijkstra puro")
            use_feasibility = st.toggle("Filtro de factibilidad", value=True,
                                        help="Excluye transiciones que excedan umbrales")

            st.markdown("")
            plan_btn = st.button("Planear transición", use_container_width=True, type="primary")

        # ─ Columna de resultados ─
        with results:
            if plan_btn:
                start_id = song_options[start_label]
                goal_id  = song_options[goal_label]

                if start_id == goal_id:
                    st.warning("La canción inicial y objetivo son la misma.")
                    st.stop()

                agent = TransitionAgent(
                    library=library,
                    weights=weights,
                    use_heuristic=use_heuristic,
                    use_feasibility=use_feasibility,
                    verbose=False,
                )

                with st.spinner("Calculando ruta óptima..."):
                    result = agent.plan(start_id, goal_id, log_decisions=True)

                if not result.found:
                    st.error("No se encontró ruta. Prueba ampliar el filtro de género o desactivar el filtro de factibilidad.")
                    st.stop()

                path_songs = [song_map[sid] for sid in result.path]
                algo_name  = "A*" if use_heuristic else "Dijkstra"

                # Stats
                c1, c2, c3, c4 = st.columns(4)
                for col, (val, label, accent) in zip(
                    [c1, c2, c3, c4],
                    [
                        (str(len(result.path)),      "canciones",    False),
                        (str(len(result.path) - 1),  "transiciones", False),
                        (f"{result.total_cost:.3f}", "coste total",  True),
                        (algo_name,                   "algoritmo",    False),
                    ],
                ):
                    col.markdown(
                        f'<div class="pcm-stat">'
                        f'<div class="pcm-stat-value {"accent" if accent else ""}">{val}</div>'
                        f'<div class="pcm-stat-label">{label}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Edge costs
                edge_data = []
                for i in range(len(path_songs) - 1):
                    s1, s2 = path_songs[i], path_songs[i + 1]
                    comps = transition_components(s1, s2)
                    total = transition_cost(s1, s2, weights)
                    edge_data.append({**comps, "total": total})

                # Timeline
                st.markdown('<div class="pcm-section">Ruta</div>', unsafe_allow_html=True)
                for i, song in enumerate(path_songs):
                    role = "start" if i == 0 else ("end" if i == len(path_songs) - 1 else "mid")
                    st.markdown(
                        f'<div class="pcm-song {role}">'
                        f'<div class="pcm-song-title">'
                        f'<span class="pcm-id">{song.id}</span>{song.artist} — {song.title}'
                        f'</div>'
                        f'<div class="pcm-song-meta">'
                        f'{song.bpm:.1f} BPM'
                        f'<span class="sep">·</span>{song.key}'
                        f'<span class="sep">·</span><span class="genre">{song.genre}</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if i < len(edge_data):
                        ec = edge_data[i]
                        st.markdown(
                            f'<div class="pcm-connector">'
                            f'<span class="total">↓ {ec["total"]:.3f}</span>'
                            f'<span><span class="lbl">arm</span>{ec["harmonic"]:.2f}</span>'
                            f'<span><span class="lbl">tmp</span>{ec["tempo"]:.2f}</span>'
                            f'<span><span class="lbl">tim</span>{ec["semantic"]:.2f}</span>'
                            f'<span><span class="lbl">gen</span>{ec["genre"]:.2f}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Desglose
                st.markdown('<div class="pcm-section">Desglose</div>', unsafe_allow_html=True)
                rows = []
                for i, ec in enumerate(edge_data):
                    s1, s2 = path_songs[i], path_songs[i + 1]
                    rows.append({
                        "De":      f"{s1.artist} — {s1.title}",
                        "A":       f"{s2.artist} — {s2.title}",
                        "Armonía": round(ec["harmonic"], 3),
                        "Tempo":   round(ec["tempo"], 3),
                        "Timbre":  round(ec["semantic"], 3),
                        "Género":  round(ec["genre"], 3),
                        "Total":   round(ec["total"], 3),
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

                # Replanificación
                if len(result.path) >= 3:
                    st.markdown('<div class="pcm-section">Replanificación</div>', unsafe_allow_html=True)
                    removed_id   = result.path[1]
                    removed_song = song_map[removed_id]
                    st.markdown(
                        f"<p style='color:#a1a1aa; font-size:0.88rem; margin-bottom:0.9rem;'>"
                        f"Qué pasa si <strong style='color:#e4e4e7;'>"
                        f"{removed_song.artist} — {removed_song.title}</strong> deja de estar disponible:</p>",
                        unsafe_allow_html=True,
                    )

                    with st.spinner("Recalculando..."):
                        result2 = agent.replan_from(
                            current_id=start_id,
                            new_goal_id=goal_id,
                            played_so_far=[],
                            excluded_ids={removed_id},
                        )

                    if result2.found:
                        new_path = [song_map[sid] for sid in result2.path]
                        old_ids  = {s.id for s in path_songs}
                        ca, cb = st.columns(2)
                        with ca:
                            items = [
                                f'<li class="{"removed" if s.id == removed_id else ""}">'
                                f'{s.artist} — {s.title}</li>'
                                for s in path_songs
                            ]
                            st.markdown(
                                f'<div class="pcm-diff"><h4>Original</h4><ol>{"".join(items)}</ol></div>',
                                unsafe_allow_html=True,
                            )
                        with cb:
                            items = [
                                f'<li class="{"new" if s.id not in old_ids else ""}">'
                                f'{s.artist} — {s.title}</li>'
                                for s in new_path
                            ]
                            st.markdown(
                                f'<div class="pcm-diff"><h4>Alternativa</h4><ol>{"".join(items)}</ol></div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.warning("No hay ruta alternativa sin esa canción.")

            else:
                st.markdown(
                    "<p style='color:#52525b; font-size:0.9rem; padding-top: 4rem; text-align:center;'>"
                    "Configura los parámetros y pulsa<br>"
                    "<strong style='color:#a78bfa;'>Planear transición</strong></p>",
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Librería
# ══════════════════════════════════════════════════════════════════════════════

with tab_lib:
    if not library:
        st.info("No hay librería. Genera una desde la pestaña **Extracción**.")
    else:
        bpms = [s.bpm for s in library]
        c1, c2, c3, c4 = st.columns(4)
        for col, (val, label, accent) in zip(
            [c1, c2, c3, c4],
            [
                (str(len(library)),                    "canciones",   False),
                (f"{min(bpms):.0f} – {max(bpms):.0f}", "rango BPM",   True),
                (str(len({s.key for s in library})),   "tonalidades", False),
                (str(len(all_genres)),                 "géneros",     False),
            ],
        ):
            col.markdown(
                f'<div class="pcm-stat">'
                f'<div class="pcm-stat-value {"accent" if accent else ""}">{val}</div>'
                f'<div class="pcm-stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="pcm-section">Filtros</div>', unsafe_allow_html=True)
        fa, fb = st.columns([2, 1])
        with fa:
            search = st.text_input("Buscar artista o título", placeholder="Escribe para filtrar...", label_visibility="collapsed")
        with fb:
            genre_filter = st.multiselect("Género", all_genres, default=all_genres, label_visibility="collapsed")

        filtered = [
            s for s in library
            if s.genre in genre_filter
            and (not search or search.lower() in s.artist.lower() or search.lower() in s.title.lower())
        ]

        st.markdown(
            f'<div style="color:#71717a; font-size:0.78rem; margin-bottom:0.6rem;">'
            f'Mostrando {len(filtered)} de {len(library)} canciones</div>',
            unsafe_allow_html=True,
        )

        df = pd.DataFrame([{
            "ID":        s.id,
            "Artista":   s.artist,
            "Título":    s.title,
            "BPM":       round(s.bpm, 1),
            "Tonalidad": s.key,
            "Género":    s.genre,
        } for s in filtered])

        st.dataframe(df, width='stretch', hide_index=True, height=520)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Extracción
# ══════════════════════════════════════════════════════════════════════════════

with tab_extract:
    ec1, ec2 = st.columns([1.6, 1])

    with ec1:
        # Estado actual de la librería
        if library_exists:
            mtime = os.path.getmtime(LIBRARY_PATH)
            last_run = datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M")
            st.markdown(
                f'<div class="pcm-status-card">'
                f'<div class="lbl">Librería actual</div>'
                f'<div class="val">{len(library)} canciones · último análisis {last_run}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pcm-status-card">'
                '<div class="lbl">Librería actual</div>'
                '<div class="val" style="color:#71717a;">Sin librería — ejecuta el análisis</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Carpeta de música
        music_ok = os.path.isdir(MUSIC_DIR)
        n_files  = len([
            f for f in os.listdir(MUSIC_DIR)
            if os.path.splitext(f)[1].lower() in {'.mp3','.flac','.wav','.ogg','.m4a','.aiff','.aif'}
        ]) if music_ok else 0

        if music_ok:
            files_status = f'<span style="color:#10b981;">{n_files} archivos de audio</span>'
        else:
            files_status = '<span style="color:#f43f5e;">No encontrada</span>'

        st.markdown(
            f'<div class="pcm-status-card">'
            f'<div class="lbl">Carpeta de música</div>'
            f'<div class="val">{MUSIC_DIR} · {files_status}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Opciones
        st.markdown('<div class="pcm-section">Opciones</div>', unsafe_allow_html=True)
        use_force = st.checkbox(
            "--force   Re-analizar todas las canciones, ignorando el cache",
            value=False,
        )
        use_prune = st.checkbox(
            "--prune   Eliminar del JSON las canciones que ya no están en la carpeta",
            value=False,
        )

        st.markdown("")
        run_btn = st.button(
            "Analizar librería",
            use_container_width=True,
            type="primary",
            disabled=not music_ok,
        )

    with ec2:
        st.markdown(
            "<p style='color:#71717a; font-size:0.82rem; padding-top:0.3rem;'>"
            "El análisis corre <code>extract_library.py</code> en segundo plano "
            "y actualiza <code>data/library.json</code> automáticamente.<br><br>"
            "Con <strong>--force</strong> se re-procesan todas las canciones "
            "(útil si cambias el extractor).<br><br>"
            "Con <strong>--prune</strong> se eliminan del JSON canciones "
            "que ya no están en <code>./music</code>.<br><br>"
            "Sin ninguna opción, solo se analizan las canciones <strong>nuevas o modificadas</strong>."
            "</p>",
            unsafe_allow_html=True,
        )

    if run_btn:
        cmd = [sys.executable, "extract_library.py", MUSIC_DIR]
        if use_force:
            cmd.append("--force")
        if use_prune:
            cmd.append("--prune")

        st.markdown('<div class="pcm-section">Output</div>', unsafe_allow_html=True)
        log_placeholder = st.empty()
        lines: list[str] = []

        try:
            # Forzar UTF-8 en el subprocess (evita el cp1252 default de Windows)
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
                env=env,
            )

            for line in iter(process.stdout.readline, ""):
                stripped = line.rstrip()
                if stripped:
                    lines.append(stripped)
                    log_placeholder.code("\n".join(lines), language=None)

            process.stdout.close()
            process.wait()

            if process.returncode == 0:
                load_lib.clear()
                st.session_state["extraction_done"] = (
                    f"Librería actualizada correctamente desde {MUSIC_DIR}"
                )
                st.rerun()
            else:
                st.error(f"El proceso terminó con error (código {process.returncode}). Revisa el output de arriba.")

        except Exception as e:
            st.error(f"No se pudo lanzar el extractor: {e}")