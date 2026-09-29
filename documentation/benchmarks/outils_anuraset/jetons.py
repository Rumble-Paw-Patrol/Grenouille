"""Jetons de toutes les fenêtres d'un stock AnuraSet (n° 151 : un transformer se juge sur ses
jetons). Relit l'audio et repasse chaque fenêtre dans l'encodeur (second passage, après
`encoder.py`). Grille (temps, fréquence) moyennée sur la fréquence (« Jetons et mémoire »,
`encodeurs-bacpipe.md`), float16, par paquets de 50 enregistrements (reprise après un
redémarrage). Les jetons restent sur la machine : ni git, ni branche de données.

Usage : jetons.py <encodeur> [--sortie data/tokens_anuraset]
Lecture : `charger(<stock>, window_ids)` (utilisé par global_bench.py --tokens).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("data/tokens_anuraset")
PACK = 50  # enregistrements par paquet


def stock_of(name: str, embeddings: Path) -> str:
    """Nom du stock d'un encodeur (`<nom>-<version>@o0`), trouvé sur le disque."""
    found = sorted(p.name for p in Path(embeddings).glob(f"{name}-*@o0"))
    if len(found) != 1:
        raise ValueError(f"{name} : stock introuvable ou ambigu dans {embeddings} ({found})")
    return found[0]


def charger(stock: str, window_ids, root: Path = ROOT) -> np.ndarray:
    """Jetons (fenêtres, temps, dim) alignés sur `window_ids` ; erreur s'il en manque."""
    parts = sorted((Path(root) / stock).glob("paquet_*.npz"))
    if not parts:
        raise FileNotFoundError(f"aucun jeton pour {stock} dans {root} (lancer jetons.py)")
    ids, values = [], []
    for p in parts:
        z = np.load(p)
        ids.extend(z["window_ids"].tolist())
        values.append(z["tokens"])
    index = {w: i for i, w in enumerate(ids)}
    missing = [w for w in window_ids if w not in index]
    if missing:
        raise ValueError(f"{stock} : {len(missing)} fenêtres sans jetons (jetons.py à relancer)")
    tokens = np.concatenate(values)
    return tokens[[index[w] for w in window_ids]]


def main() -> None:
    from blanci.audio import cut_windows, load_audio
    from blanci.config import config_path, load_config
    from blanci.dataset import recordings_table
    from blanci.db import connect
    from blanci.encoders import get_encoder
    from blanci.store import EmbeddingStore

    name = sys.argv[1]
    root = Path(sys.argv[sys.argv.index("--sortie") + 1]) if "--sortie" in sys.argv else ROOT
    cfg = load_config(Path("config/anuraset.yaml"))
    con = connect(config_path(cfg, "db"))
    stock = stock_of(name, config_path(cfg, "embeddings"))
    meta, _ = EmbeddingStore(config_path(cfg, "embeddings"), stock).load()
    encoder = get_encoder(name, cfg)
    if not encoder.has_tokens:
        raise SystemExit(f"{name} n'expose pas de jetons")
    out = root / stock
    out.mkdir(parents=True, exist_ok=True)
    paths = recordings_table(con).set_index("recording_id")["path"]
    recordings = sorted(meta["recording_id"].unique())
    t0 = time.time()
    for k in range(0, len(recordings), PACK):
        target = out / f"paquet_{k // PACK:04d}.npz"
        if target.exists():
            continue
        ids, values = [], []
        for rid in recordings[k : k + PACK]:
            group = meta[meta["recording_id"] == rid]
            wav, sr = load_audio(config_path(cfg, "raw") / paths[rid], cfg["audio"]["channel"])
            windows = [(o, encoder.window_s) for o in group["offset_s"]]
            tokens = encoder.embed_tokens(cut_windows(wav, sr, windows), sr)
            if tokens.ndim == 4:  # (fenêtres, temps, fréquence, dim) → moyenne sur la fréquence
                tokens = tokens.mean(axis=2)
            ids.extend(group["window_id"].tolist())
            values.append(tokens.astype(np.float16))
        tmp = target.with_suffix(".tmp.npz")
        np.savez(tmp, window_ids=np.array(ids), tokens=np.concatenate(values))
        tmp.replace(target)
        done = min(k + PACK, len(recordings))
        print(f"{name} : {done}/{len(recordings)} enr., {time.time() - t0:.0f} s", flush=True)
    print(f"FINI jetons {name} ({stock}) : {out}", flush=True)


if __name__ == "__main__":
    main()
