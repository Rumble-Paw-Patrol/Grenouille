"""Stockage des embeddings : Parquet partitionné <encodeur>/<jeu>/<site>/<aaaamm>.parquet (§13.3).

Colonnes : window_id, recording_id, offset_s, emb (liste de taille fixe float16[dim]) ; `gated`
(booléen) dans un stock encodé avec des portes : une fenêtre arrêtée a un embedding nul, jamais
appris, et le score le plus bas (module séquentiel en amont, `blanci/sequential.py`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

META_COLUMNS = ["window_id", "recording_id", "offset_s"]
GATED = "gated"  # colonne optionnelle : fenêtre arrêtée par une porte (seuillage en amont)
PARTITION_KEYS = ("dataset", "site", "month")


def gated_mask(meta: pd.DataFrame) -> np.ndarray:
    """Fenêtres arrêtées par une porte (toutes à False dans un stock sans portes)."""
    if GATED not in meta:
        return np.zeros(len(meta), dtype=bool)
    return meta[GATED].astype("boolean").fillna(False).to_numpy(dtype=bool)


def _as_set(value: str | list[str] | None) -> set[str] | None:
    if value is None:
        return None
    return {value} if isinstance(value, str) else set(value)


@dataclass
class EmbeddingStore:
    root: Path  # data/embeddings
    encoder_id: str

    @property
    def directory(self) -> Path:
        return Path(self.root) / self.encoder_id

    def partition_path(self, dataset: str, site: str, month: str) -> Path:
        return self.directory / dataset / site / f"{month}.parquet"

    def write(
        self, meta: pd.DataFrame, emb: np.ndarray, dataset: str, site: str, month: str
    ) -> Path:
        """Ajoute des fenêtres à une partition ; une fenêtre déjà présente est remplacée."""
        if len(meta) != len(emb):
            raise ValueError("meta et emb n'ont pas le même nombre de lignes")
        path = self.partition_path(dataset, site, month)
        columns = META_COLUMNS + ([GATED] if GATED in meta else [])
        meta = meta[columns].reset_index(drop=True)
        emb = np.asarray(emb, dtype=np.float16)
        if path.exists():
            old_meta, old_emb = self.read(path)
            keep = ~old_meta["window_id"].isin(meta["window_id"]).to_numpy()
            meta = pd.concat([old_meta[keep], meta], ignore_index=True)
            if GATED in meta:
                meta[GATED] = meta[GATED].astype("boolean").fillna(False).astype(bool)
            emb = np.concatenate([old_emb[keep].astype(np.float16), emb])
        dim = emb.shape[1]
        values = pa.array(emb.reshape(-1), type=pa.float16())
        table = pa.Table.from_pandas(meta, preserve_index=False).append_column(
            "emb", pa.FixedSizeListArray.from_arrays(values, dim)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)  # écriture atomique : pas de partition à moitié écrite
        return path

    def read_meta(self, path: Path) -> pd.DataFrame:
        """Métadonnées seules : évite de charger les embeddings pour savoir ce qui est déjà fait."""
        return pq.read_table(path, columns=META_COLUMNS).to_pandas()

    def read(self, path: Path) -> tuple[pd.DataFrame, np.ndarray]:
        table = pq.read_table(path)
        column = table.column("emb").combine_chunks()
        dim = column.type.list_size
        emb = column.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim)
        return table.drop(["emb"]).to_pandas(), emb

    def fragments(self, filters: dict | None = None) -> Iterator[Path]:
        """Partitions retenues par les filtres {dataset, site, month} (valeur ou liste)."""
        filters = filters or {}
        unknown = set(filters) - set(PARTITION_KEYS)
        if unknown:
            raise ValueError(f"filtres inconnus : {sorted(unknown)} (attendus : {PARTITION_KEYS})")
        wanted = {key: _as_set(filters.get(key)) for key in PARTITION_KEYS}
        for path in sorted(self.directory.glob("*/*/*.parquet")):
            parts = (path.parent.parent.name, path.parent.name, path.stem)
            values = dict(zip(PARTITION_KEYS, parts, strict=True))
            if all(wanted[k] is None or values[k] in wanted[k] for k in PARTITION_KEYS):
                yield path

    def load(self, filters: dict | None = None) -> tuple[pd.DataFrame, np.ndarray]:
        parts = [self.read(path) for path in self.fragments(filters)]
        if not parts:
            return pd.DataFrame(columns=META_COLUMNS), np.zeros((0, 0), dtype=np.float16)
        return (
            pd.concat([m for m, _ in parts], ignore_index=True),
            np.concatenate([e for _, e in parts]),
        )

    def sample(
        self, n: int, filters: dict | None = None, seed: int = 0
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """`n` fenêtres tirées au hasard, réparties entre partitions au prorata de leur taille.

        Une partition à la fois en mémoire : l'échantillon d'un stock de plusieurs millions de
        fenêtres (clustering C0, §5 bis) tient dans la mémoire du portable.
        """
        rng = np.random.default_rng(seed)
        paths = list(self.fragments(filters))
        sizes = np.array([pq.ParquetFile(p).metadata.num_rows for p in paths], dtype=float)
        if not len(paths) or sizes.sum() == 0:
            return pd.DataFrame(columns=META_COLUMNS), np.zeros((0, 0), dtype=np.float16)
        n = min(n, int(sizes.sum()))
        quota = np.floor(n * sizes / sizes.sum()).astype(int)
        for i in rng.choice(len(paths), n - quota.sum(), replace=False, p=sizes / sizes.sum()):
            quota[i] += 1
        metas, embs = [], []
        for path, k in zip(paths, quota, strict=True):
            if k == 0:
                continue
            meta, emb = self.read(path)
            rows = np.sort(rng.choice(len(meta), size=min(k, len(meta)), replace=False))
            metas.append(meta.iloc[rows])
            embs.append(emb[rows])
        return pd.concat(metas, ignore_index=True), np.concatenate(embs)
