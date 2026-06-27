"""
extract_library.py — Analiza tu carpeta de música y genera data/library.json

Usa (en orden de preferencia):
  - BPM:   librosa beat_track
  - Key:   essentia KeyExtractor  →  Krumhansl (librosa chroma)
  - Embed: CLAP (512 dims, GPU si disponible)  →  MFCCs (16 dims)
  - Genre: tag ID3 (mutagen)  →  "Unknown"

Soporta: .mp3 .flac .wav .ogg .m4a .aiff

Extracción incremental: si library.json ya existe, solo analiza canciones nuevas
o modificadas. Cada canción se identifica por (filename, size, mtime).

Uso:
    python extract_library.py ./mi_musica              # incremental por default
    python extract_library.py ./mi_musica --force      # re-analizar todo
    python extract_library.py ./mi_musica --prune      # borrar las que ya no están
"""
from __future__ import annotations
import sys, os, json, argparse, warnings
import numpy as np

# Forzar UTF-8 en stdout (Windows usa cp1252 por default cuando stdout
# está capturado por otro proceso, lo que rompe caracteres como ✓)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Silence verbose warnings from audio libs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import Song, save_library

# ─── Detección de librerías disponibles ───────────────────────────────────────

try:
    import essentia.standard as es
    HAS_ESSENTIA = True
except ImportError:
    HAS_ESSENTIA = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import torch
    from msclap import CLAP
    _clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    HAS_CLAP = True
    CLAP_DEVICE = "GPU ✓" if torch.cuda.is_available() else "CPU"
except Exception:
    HAS_CLAP = False
    CLAP_DEVICE = None

try:
    from mutagen import File as MutaFile
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

if not HAS_LIBROSA:
    print("Error: librosa es obligatorio. Instálalo con: pip install librosa soundfile")
    sys.exit(1)

# ─── Constantes ───────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {'.mp3', '.flac', '.wav', '.ogg', '.m4a', '.aiff', '.aif'}

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

# ─── Módulo BPM ───────────────────────────────────────────────────────────────

def detect_bpm(filepath: str, y: np.ndarray, sr: int) -> float:
    """Detección de tempo con librosa beat_track."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(float(np.atleast_1d(tempo)[0]), 2)

# ─── Módulo Key ───────────────────────────────────────────────────────────────

def detect_key_essentia(filepath: str) -> str:
    loader = es.MonoLoader(filename=filepath, sampleRate=44100)
    audio = loader()
    key, scale, _ = es.KeyExtractor()(audio)
    return key if scale == 'major' else key + 'm'

def detect_key_krumhansl(y: np.ndarray, sr: int) -> str:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = chroma.mean(axis=1)
    best_score, best_key = -np.inf, 'C'
    for root in range(12):
        rotated = np.roll(mean_chroma, -root)
        maj = float(np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1])
        min_ = float(np.corrcoef(rotated, _MINOR_PROFILE)[0, 1])
        if maj > best_score:
            best_score, best_key = maj, _NOTE_NAMES[root]
        if min_ > best_score:
            best_score, best_key = min_, _NOTE_NAMES[root] + 'm'
    return best_key

def detect_key(filepath: str, y: np.ndarray, sr: int) -> str:
    if HAS_ESSENTIA:
        try:
            return detect_key_essentia(filepath)
        except Exception:
            pass
    return detect_key_krumhansl(y, sr)

# ─── Módulo Genre ─────────────────────────────────────────────────────────────

def read_genre(filepath: str) -> str:
    """Detecta el género desde los tags ID3/metadatos del archivo."""
    if HAS_MUTAGEN:
        try:
            audio = MutaFile(filepath, easy=True)
            if audio and "genre" in audio:
                g = audio["genre"][0].strip()
                if g:
                    return g
        except Exception:
            pass
    return "Unknown"

# ─── Módulo Embedding ─────────────────────────────────────────────────────────

def extract_embedding_clap(filepath: str) -> np.ndarray:
    embeddings = _clap_model.get_audio_embeddings([filepath], resample=True)
    raw = embeddings[0]
    if hasattr(raw, 'cpu'):
        raw = raw.cpu()
    emb = np.array(raw, dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 0 else emb

def extract_embedding_mfcc(y: np.ndarray, sr: int, dim: int = 16) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim - 2).mean(axis=1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean() / (sr / 2)
    rms = librosa.feature.rms(y=y).mean()
    emb = np.append(mfcc, [centroid, rms]).astype(np.float32)
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 0 else emb

def extract_embedding(filepath: str, y: np.ndarray, sr: int) -> np.ndarray:
    if HAS_CLAP:
        try:
            return extract_embedding_clap(filepath)
        except Exception:
            pass
    return extract_embedding_mfcc(y, sr)

# ─── Análisis completo de un archivo ─────────────────────────────────────────

def analyze_file(filepath: str) -> tuple[float, str, np.ndarray, str]:
    """Carga el audio y extrae BPM, key, embedding y género."""
    y, sr = librosa.load(filepath, sr=22050, mono=True, duration=120)
    bpm   = detect_bpm(filepath, y, sr)
    key   = detect_key(filepath, y, sr)
    emb   = extract_embedding(filepath, y, sr)
    genre = read_genre(filepath)
    return bpm, key, emb, genre

# ─── Cache / Extracción incremental ──────────────────────────────────────────

def compute_fingerprint(filepath: str) -> str:
    """Identificador único de archivo basado en nombre, tamaño y mtime."""
    st = os.stat(filepath)
    name = os.path.basename(filepath)
    return f"{name}|{st.st_size}|{int(st.st_mtime)}"

def load_existing_library(path: str) -> tuple[dict[str, Song], int]:
    """
    Lee library.json existente. Devuelve:
      - dict: fingerprint → Song
      - int:  máximo número de ID encontrado (para asignar IDs nuevos)
    """
    if not os.path.exists(path):
        return {}, 0
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Aviso: no pude leer '{path}' ({e}). Re-analizando todo.")
        return {}, 0

    items = data.get("songs", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return {}, 0

    cache: dict[str, Song] = {}
    max_id = 0

    for d in items:
        try:
            n = int(str(d.get("id", "s0"))[1:])
            max_id = max(max_id, n)
        except (ValueError, IndexError):
            pass

        fp = d.get("fingerprint", "")
        if not fp:
            continue

        try:
            cache[fp] = Song(
                id=d["id"],
                title=d["title"],
                artist=d["artist"],
                bpm=float(d["bpm"]),
                key=d["key"],
                embedding=np.array(d["embedding"], dtype=np.float32),
                genre=d.get("genre", "Unknown"),
                fingerprint=fp,
            )
        except Exception:
            continue

    return cache, max_id

# ─── Scanner de directorio ────────────────────────────────────────────────────

def scan_directory(music_dir: str, output_path: str = "data/library.json",
                   force: bool = False, prune: bool = False) -> list[Song]:
    files = sorted(
        f for f in os.listdir(music_dir)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    )
    if not files:
        print(f"No se encontraron archivos de audio en '{music_dir}'.")
        return []

    cache: dict[str, Song] = {}
    max_id = 0
    if not force:
        cache, max_id = load_existing_library(output_path)

    emb_info   = f"CLAP 512 dims ({CLAP_DEVICE})" if HAS_CLAP else "MFCC 16 dims (fallback)"
    genre_info = "ID3 tags (mutagen)" if HAS_MUTAGEN else "No disponible"

    print(f"\nEncontré {len(files)} archivos en '{music_dir}'")
    if force:
        print(f"  --force: ignorando cache, re-analizando todo")
    elif cache:
        print(f"  Cache:  {len(cache)} canciones previamente analizadas")
    print(f"  BPM:    librosa beat_track")
    print(f"  Key:    {'essentia KeyExtractor' if HAS_ESSENTIA else 'Krumhansl/chroma (fallback)'}")
    print(f"  Emb:    {emb_info}")
    print(f"  Genre:  {genre_info}\n")

    songs:  list[Song] = []
    errors: list[str]  = []
    seen_fingerprints:  set[str] = set()
    reused   = 0
    analyzed = 0

    for i, fname in enumerate(files, 1):
        fpath = os.path.join(music_dir, fname)
        fp    = compute_fingerprint(fpath)
        seen_fingerprints.add(fp)

        if fp in cache:
            cached = cache[fp]
            songs.append(cached)
            reused += 1
            print(f"  [{i:3d}/{len(files)}] {fname[:55]:<55} ⟳  CACHE  ID={cached.id}  Genre={cached.genre}")
            continue

        max_id  += 1
        song_id  = f"s{max_id:03d}"

        base   = os.path.splitext(fname)[0]
        parts  = base.split(' - ', 1)
        artist = parts[0].strip() if len(parts) == 2 else "Unknown"
        title  = parts[1].strip() if len(parts) == 2 else base

        print(f"  [{i:3d}/{len(files)}] {fname[:55]:<55}", end=' ', flush=True)
        try:
            bpm, key, emb, genre = analyze_file(fpath)
            songs.append(Song(
                id=song_id, title=title, artist=artist,
                bpm=bpm, key=key, embedding=emb, genre=genre,
                fingerprint=fp,
            ))
            analyzed += 1
            print(f"✓  ID={song_id}  BPM={bpm:6.1f}  Key={key}  Emb={len(emb)}d  Genre={genre}")
        except Exception as e:
            print(f"✗  ERROR: {e}")
            errors.append(fname)

    disappeared = [s for fp, s in cache.items() if fp not in seen_fingerprints]

    if disappeared:
        print(f"\n{len(disappeared)} canciones en el cache ya no están en la carpeta:")
        for s in disappeared[:10]:
            print(f"  - {s.artist} - {s.title}  (ID={s.id})")
        if len(disappeared) > 10:
            print(f"  ... y {len(disappeared) - 10} más")
        if prune:
            print(f"  --prune: descartadas del library.json")
        else:
            print(f"  Conservadas en library.json. Usa --prune para eliminarlas.")
            songs.extend(disappeared)

    if errors:
        print(f"\nArchivos con error ({len(errors)}): {', '.join(errors)}")

    pruned_note = " (eliminadas)" if prune and disappeared else ""
    print(f"\nResumen:  {reused} reusadas  |  {analyzed} nuevas  |  {len(disappeared)} desaparecidas{pruned_note}")
    return songs

# ─── Entrada principal ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extrae características de audio y genera library.json")
    parser.add_argument("music_dir", help="Carpeta con archivos de audio")
    parser.add_argument("--output",  default="data/library.json",
                        help="Destino del JSON (default: data/library.json)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-analizar todas las canciones, ignorando el cache existente")
    parser.add_argument("--prune",   action="store_true",
                        help="Eliminar del library.json las canciones que ya no están en la carpeta")
    args = parser.parse_args()

    if not os.path.isdir(args.music_dir):
        print(f"Error: '{args.music_dir}' no es una carpeta válida.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    songs = scan_directory(args.music_dir, output_path=args.output,
                           force=args.force, prune=args.prune)

    if not songs:
        sys.exit(1)

    save_library(songs, args.output)

    print(f"\n{'─'*60}")
    print(f"  Guardado: {args.output}  ({len(songs)} canciones)")
    print(f"{'─'*60}")
    print(f"\nPróximos pasos:")
    print(f"  python main.py --load-library {args.output} --list")
    print(f"  python main.py --load-library {args.output} --start s001 --goal s045")

if __name__ == "__main__":
    main()
