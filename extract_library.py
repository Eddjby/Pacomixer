"""
extract_library.py — Analiza tu carpeta de música y genera data/library.json

Usa (en orden de preferencia):
  - BPM:   madmom RNN + DBN  →  fallback librosa beat_track
  - Key:   essentia KeyExtractor  →  fallback Krumhansl (librosa chroma)
  - Embed: MFCCs via librosa (dim=16, más rico que la versión anterior)

Soporta: .mp3 .flac .wav .ogg .m4a .aiff

Uso:
    python extract_library.py ./mi_musica
    python extract_library.py ./mi_musica --workers 4   # paralelo
"""
from __future__ import annotations
import sys, os, json, argparse, warnings
import numpy as np

# Silence verbose warnings from audio libs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import Song, save_library

# ─── Detección de librerías disponibles ───────────────────────────────────────

try:
    import madmom
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    HAS_MADMOM = True
except ImportError:
    HAS_MADMOM = False

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

if not HAS_LIBROSA:
    print("Error: librosa es obligatorio. Instálalo con: pip install librosa soundfile")
    sys.exit(1)

# ─── Constantes ───────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {'.mp3', '.flac', '.wav', '.ogg', '.m4a', '.aiff', '.aif'}

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

# ─── Módulo BPM ───────────────────────────────────────────────────────────────

def detect_bpm_madmom(filepath: str) -> float:
    """
    Detección de tempo con madmom.
    Usa RNNBeatProcessor (red neuronal recurrente) + DBN (red bayesiana dinámica).
    """
    act = RNNBeatProcessor()(filepath)
    beats = DBNBeatTrackingProcessor(fps=100)(act)
    if len(beats) > 1:
        intervals = np.diff(beats)
        # Mediana de intervalos → más robusto que la media ante outliers
        return round(60.0 / float(np.median(intervals)), 2)
    return 120.0  # fallback si no detecta suficientes beats

def detect_bpm_librosa(y: np.ndarray, sr: int) -> float:
    """Fallback: beat tracking con librosa."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(float(tempo), 2)

def detect_bpm(filepath: str, y: np.ndarray, sr: int) -> float:
    if HAS_MADMOM:
        try:
            return detect_bpm_madmom(filepath)
        except Exception:
            pass
    return detect_bpm_librosa(y, sr)

# ─── Módulo Key ───────────────────────────────────────────────────────────────

def detect_key_essentia(filepath: str) -> str:
    """
    Detección de tonalidad con essentia KeyExtractor.
    Retorna strings como 'C', 'F#m', 'Bbm', etc.
    """
    loader = es.MonoLoader(filename=filepath, sampleRate=44100)
    audio = loader()
    key, scale, _ = es.KeyExtractor()(audio)
    # essentia retorna scale='major' o 'minor'
    return key if scale == 'major' else key + 'm'

def detect_key_krumhansl(y: np.ndarray, sr: int) -> str:
    """Fallback: correlación de chroma con perfiles de Krumhansl."""
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

# ─── Módulo Embedding ─────────────────────────────────────────────────────────

def extract_embedding(y: np.ndarray, sr: int, dim: int = 16) -> np.ndarray:
    """
    Embedding de 16 dimensiones basado en características espectrales:
      - MFCCs (promedio temporal) → captura timbre / color sonoro
      - Spectral centroid normalizado → brillo
      - RMS normalizado → energía

    Normalizado a norma unitaria para que la distancia coseno sea coherente.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim - 2).mean(axis=1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean() / (sr / 2)
    rms = librosa.feature.rms(y=y).mean()
    emb = np.append(mfcc, [centroid, rms]).astype(np.float32)
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 0 else emb

# ─── Análisis completo de un archivo ─────────────────────────────────────────

def analyze_file(filepath: str) -> tuple[float, str, np.ndarray]:
    """Carga el audio y extrae las tres características."""
    y, sr = librosa.load(filepath, sr=22050, mono=True, duration=120)
    bpm = detect_bpm(filepath, y, sr)
    key = detect_key(filepath, y, sr)
    embedding = extract_embedding(y, sr)
    return bpm, key, embedding

# ─── Scanner de directorio ────────────────────────────────────────────────────

def scan_directory(music_dir: str) -> list[Song]:
    files = sorted(
        f for f in os.listdir(music_dir)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    )
    if not files:
        print(f"No se encontraron archivos de audio en '{music_dir}'.")
        return []

    print(f"\nEncontré {len(files)} archivos en '{music_dir}'")
    print(f"  BPM:  {'madmom (RNN + DBN)' if HAS_MADMOM else 'librosa (fallback)'}")
    print(f"  Key:  {'essentia KeyExtractor' if HAS_ESSENTIA else 'Krumhansl/chroma (fallback)'}")
    print(f"  Emb:  MFCC + spectral centroid (16 dims)\n")

    songs: list[Song] = []
    errors: list[str] = []

    for i, fname in enumerate(files, 1):
        fpath = os.path.join(music_dir, fname)
        song_id = f"s{i:03d}"

        base = os.path.splitext(fname)[0]
        parts = base.split(' - ', 1)
        artist = parts[0].strip() if len(parts) == 2 else "Unknown"
        title  = parts[1].strip() if len(parts) == 2 else base

        print(f"  [{i:3d}/{len(files)}] {fname[:55]:<55}", end=' ', flush=True)
        try:
            bpm, key, emb = analyze_file(fpath)
            songs.append(Song(
                id=song_id, title=title, artist=artist,
                bpm=bpm, key=key, embedding=emb, genre="real"
            ))
            print(f"✓  BPM={bpm:6.1f}  Key={key}")
        except Exception as e:
            print(f"✗  ERROR: {e}")
            errors.append(fname)

    if errors:
        print(f"\nArchivos con error ({len(errors)}): {', '.join(errors)}")
    return songs

# ─── Entrada principal ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extrae características de audio y genera library.json")
    parser.add_argument("music_dir", help="Carpeta con archivos de audio")
    parser.add_argument("--output", default="data/library.json",
                        help="Destino del JSON (default: data/library.json)")
    args = parser.parse_args()

    if not os.path.isdir(args.music_dir):
        print(f"Error: '{args.music_dir}' no es una carpeta válida.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    songs = scan_directory(args.music_dir)

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
  
