"""R67 : classes des sons de chaque fenêtre, pour la tête multiclasse."""

from __future__ import annotations

import numpy as np

# --- R67 : classes des sons ----------------------------------------------------------------------

# Labels d'annotation → classe de la tête multi-classes. Les négatifs présumés (sans label)
# rejoignent le fond : par définition, aucun événement n'y a été noté (n° 124).
CLASS_OF_LABEL = {
    "bird": "oiseau",
    "amphibian": "amphibien",
    "amphibian_contact_call": "amphibien",
    "orthoptera": "orthoptère",
    "rain": "pluie",
    "background": "fond",
    "artefact_in_bag": "artefact",
    "other": "autre",
}
POSITIVE_CLASS = "blanci"


def window_classes(
    labels: np.ndarray, y: np.ndarray, presumed: np.ndarray | None = None
) -> np.ndarray:
    """R67 : classe de chaque fenêtre — `blanci` pour les positifs, la classe de son label
    d'annotation sinon (`CLASS_OF_LABEL`), `fond` pour les négatifs présumés et les labels
    absents, `autre` pour un label inconnu."""
    y = np.asarray(y).astype(int)
    out = []
    for i, label in enumerate(np.asarray(labels, dtype=object)):
        if y[i] == 1:
            out.append(POSITIVE_CLASS)
        elif (presumed is not None and bool(presumed[i])) or label is None or label != label:
            out.append("fond")
        else:
            out.append(CLASS_OF_LABEL.get(str(label), "autre"))
    return np.array(out, dtype=object)


def merge_rare_classes(classes: np.ndarray, min_count: int = 10) -> np.ndarray:
    """R67 : une classe de moins de `min_count` fenêtres (hors `blanci`) rejoint `autre` ;
    si `autre` reste sous le seuil, elle rejoint `fond`. Une classe de 3 exemples ne s'apprend
    pas, elle ajoute du bruit."""
    classes = np.asarray(classes, dtype=object).copy()
    names, counts = np.unique(classes, return_counts=True)
    for name, count in zip(names, counts, strict=True):
        if name not in (POSITIVE_CLASS, "fond", "autre") and count < min_count:
            classes[classes == name] = "autre"
    if 0 < (classes == "autre").sum() < min_count:
        classes[classes == "autre"] = "fond"
    return classes
