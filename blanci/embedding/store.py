"""Stockage des embeddings : Parquet partitionné <encodeur>/<jeu>/<site>/<aaaamm>.parquet (§13.3).

Colonnes : window_id, recording_id, offset_s, emb (liste de taille fixe float16[dim]) ; `gated`
(booléen) dans un stock encodé avec des portes : une fenêtre arrêtée a un embedding nul, jamais
appris, et le score le plus bas (module séquentiel en amont, `blanci/heads/sequential.py`).

Pendant un encodage, les fenêtres arrivent par morceaux `<aaaamm>.part-<n>.parquet` (`append`),
lus comme leur partition et fondus dans son fichier en fin de partition (`consolidate`).
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
PART_MARK = ".part-"  # <aaaamm>.part-<n>.parquet : morceau en attente de `consolidate`
# Dictionnaire pour les identifiants seulement : sur les embeddings (presque tous distincts),
# il ralentit l'écriture d'un tiers et la lecture d'un facteur 4 sans rien gagner en taille.
WRITE_OPTIONS = {"use_dictionary": ["window_id", "recording_id"]}


def gated_mask(meta: pd.DataFrame) -> np.ndarray:
    """Fenêtres arrêtées par une porte (toutes à False dans un stock sans portes)."""
    if GATED not in meta:
        return np.zeros(len(meta), dtype=bool)
    return meta[GATED].astype("boolean").fillna(False).to_numpy(dtype=bool)


def _as_set(value: str | list[str] | None) -> set[str] | None:
    if value is None:
        return None
    return {value} if isinstance(value, str) else set(value)


def _table(meta: pd.DataFrame, emb: np.ndarray) -> pa.Table:
    values = pa.array(emb.reshape(-1), type=pa.float16())
    return pa.Table.from_pandas(meta, preserve_index=False).append_column(
        "emb", pa.FixedSizeListArray.from_arrays(values, emb.shape[1])
    )


def _write_atomic(table: pa.Table, path: Path) -> None:
    """Écriture atomique : pas de fichier à moitié écrit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, **WRITE_OPTIONS)
    tmp.replace(path)


def _merged(
    old: tuple[pd.DataFrame, np.ndarray], new: tuple[pd.DataFrame, np.ndarray]
) -> tuple[pd.DataFrame, np.ndarray]:
    """`old` suivi de `new` ; une fenêtre de `old` reprise dans `new` est remplacée."""
    (old_meta, old_emb), (meta, emb) = old, new
    keep = ~old_meta["window_id"].isin(meta["window_id"]).to_numpy()
    meta = pd.concat([old_meta[keep], meta], ignore_index=True)
    if GATED in meta:
        meta[GATED] = meta[GATED].astype("boolean").fillna(False).astype(bool)
    return meta, np.concatenate([old_emb[keep], emb]).astype(np.float16, copy=False)


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
            meta, emb = _merged(self.read(path), (meta, emb))
        _write_atomic(_table(meta, emb), path)
        return path

    def part_paths(self, dataset: str, site: str, month: str) -> list[Path]:
        """Morceaux d'une partition pas encore fondus dans son fichier, dans l'ordre d'écriture."""
        directory = self.partition_path(dataset, site, month).parent
        return sorted(directory.glob(f"{month}{PART_MARK}*.parquet"))

    def append(
        self, meta: pd.DataFrame, emb: np.ndarray, dataset: str, site: str, month: str
    ) -> Path:
        """Ajoute des fenêtres à une partition sans la relire : elles vont dans un morceau
        `<aaaamm>.part-<n>.parquet`, à côté de son fichier.

        `write` relit et réécrit toute la partition : 42 s et 8 Go de mémoire pour ajouter 50
        enregistrements à une partition de 0,9 Go (Mataroni, janvier 2026, à mi-encodage), contre
        0,2 s ici. Un morceau se lit comme la partition (`fragments`) ; `consolidate` les fond
        dans son fichier.
        """
        if len(meta) != len(emb):
            raise ValueError("meta et emb n'ont pas le même nombre de lignes")
        columns = META_COLUMNS + ([GATED] if GATED in meta else [])
        parts = self.part_paths(dataset, site, month)
        number = int(parts[-1].stem.split(PART_MARK)[1]) + 1 if parts else 1
        path = self.partition_path(dataset, site, month).with_name(
            f"{month}{PART_MARK}{number:06d}.parquet"
        )
        table = _table(meta[columns].reset_index(drop=True), np.asarray(emb, dtype=np.float16))
        _write_atomic(table, path)
        return path

    def consolidate(self, dataset: str, site: str, month: str) -> Path:
        """Fond les morceaux d'une partition dans son fichier, puis les supprime.

        Sans doublon de `window_id` (cas normal : la reprise saute ce qui est fait), les fichiers
        sont recopiés groupe de lignes par groupe de lignes, sans charger la partition. Sinon,
        comme `write` : la dernière écriture d'une fenêtre l'emporte. Interrompue, la fonte se
        refait à l'identique : le fichier est remplacé d'un coup, les morceaux supprimés après.
        """
        path = self.partition_path(dataset, site, month)
        parts = self.part_paths(dataset, site, month)
        if not parts:
            return path
        files = ([path] if path.exists() else []) + parts
        ids = pd.concat(
            [pq.read_table(f, columns=["window_id"]).to_pandas() for f in files], ignore_index=True
        )["window_id"]
        schemas = [pq.read_schema(f) for f in files]
        tmp = path.with_suffix(".parquet.tmp")
        if ids.is_unique and all(s.equals(schemas[0]) for s in schemas[1:]):
            with pq.ParquetWriter(tmp, schemas[0], **WRITE_OPTIONS) as writer:
                for f in files:
                    with pq.ParquetFile(f) as source:
                        for group in range(source.num_row_groups):
                            writer.write_table(source.read_row_group(group))
            tmp.replace(path)
        else:
            meta, emb = self.read(files[0])
            for f in files[1:]:
                meta, emb = _merged((meta, emb), self.read(f))
            _write_atomic(_table(meta, emb), path)
        for part in parts:
            part.unlink()
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
            month = path.stem.split(PART_MARK)[0]  # un morceau est lu comme sa partition
            parts = (path.parent.parent.name, path.parent.name, month)
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
