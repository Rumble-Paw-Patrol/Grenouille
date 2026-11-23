# Présentation n° 7 : PNA A. blanci (restitution de stage)

Sortie : `documentation/prez/presentation_7_pna.pptx`, animée par la transition Morphose (PowerPoint 2019 / Microsoft 365). Données intermédiaires dans `rapports/presentation_7/` (jamais committées). Tout se lit en lecture seule : la base (`mode=ro`), le stock Perch v2 (`F:/data/embeddings`), l'audio (disque `BLANCI_2023-24`).

Ordre (depuis la racine du dépôt) :

```sh
uv run python documentation/prez/scripts/presentation_7/extraire.py          # ≈ 10 min : fenêtres au hasard
uv run python documentation/prez/scripts/presentation_7/extraire_labels.py   # ≈ 25 min : fenêtres annotées, signatures d'enregistrements
uv run python documentation/prez/scripts/presentation_7/figures_audio.py     # enveloppe, spectrogramme, extrait audio
uv run python documentation/prez/scripts/presentation_7/donnees.py           # clustering, rejeu d'apprentissage actif, nuage
uv run --with python-pptx python documentation/prez/scripts/presentation_7/deck.py
uv run --with python-pptx --with pillow python documentation/prez/scripts/presentation_7/verifier_textes.py documentation/prez/presentation_7_pna.pptx
```

Ne pas relancer `deck.py` par-dessus une version retouchée à la main : passer un autre chemin en argument.
