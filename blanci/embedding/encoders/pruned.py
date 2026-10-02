"""Graphe ONNX réduit aux sorties dont on se sert (DECISIONS n° 180).

Le graphe de perch_v2 calcule à chaque fenêtre, en plus de l'embedding, les logits de ses
14 795 classes (364 Mo de prototypes sur 413) et rend son spectrogramme. ONNX Runtime exécute
tout le graphe quelles que soient les sorties demandées : il faut retirer ces branches pour ne
plus les payer. L'embedding et les jetons sont inchangés, au bit près.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def pruned_model(source: Path, outputs: list[str], cache_dir: Path) -> Path:
    """Fichier ONNX ne calculant que `outputs`, rangé dans `cache_dir` (fait une fois ; refait
    si le modèle d'origine change de taille). Sans le paquet `onnx`, rend `source`."""
    try:
        import onnx
        from onnx.utils import Extractor
    except ImportError:
        return source
    source = Path(source)
    target = Path(cache_dir) / f"{source.stem}-{'+'.join(outputs)}-{source.stat().st_size}.onnx"
    if not target.exists():
        model = onnx.load(str(source))
        inputs = [value.name for value in model.graph.input]
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".onnx.tmp")
        onnx.save(Extractor(model).extract_model(inputs, outputs), str(tmp))
        tmp.replace(target)
    return target


class PrunedSession:
    """Session d'un graphe élagué, vue comme celle du graphe entier : `run(None, …)` rend les
    sorties dans l'ordre d'origine (`names`), vides (lot, 0) pour celles qui ont été retirées."""

    def __init__(self, session: Any, kept: list[str], names: list[str]):
        self.session, self.kept, self.names = session, list(kept), list(names)

    def get_providers(self) -> list[str]:
        return self.session.get_providers()

    def run(self, output_names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        found = dict(zip(self.kept, self.session.run(None, feeds), strict=True))
        n = len(next(iter(feeds.values())))
        return [found.get(name, np.zeros((n, 0), dtype=np.float32)) for name in self.names]
