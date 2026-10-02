"""Préparation en avance, dans des fils à part.

Lire un fichier audio et calculer sur le signal (numpy, scipy) relâche le verrou de Python :
ces étapes peuvent tourner pendant que le fil principal fait autre chose (l'encodeur, la base).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def ahead(
    items: Iterable[Any], prepare: Callable[[Any], Any], ahead: int = 2, workers: int = 1
) -> Iterator[tuple[Any, Any]]:
    """(élément, prepare(élément)), dans l'ordre ; `prepare` tourne dans `workers` fils à part,
    avec `ahead` éléments d'avance sur celui qui est rendu (0 : chacun à son tour). Une
    exception de `prepare` remonte ici, à son tour."""
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="blanci-lecture")
    pending: deque = deque()
    try:
        for item in items:
            pending.append((item, pool.submit(prepare, item)))
            if len(pending) > ahead:
                first, future = pending.popleft()
                yield first, future.result()
        while pending:
            first, future = pending.popleft()
            yield first, future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
