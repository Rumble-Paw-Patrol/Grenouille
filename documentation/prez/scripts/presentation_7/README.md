# Présentation n° 7 : PNA A. blanci (restitution de stage)

Sorties : `documentation/prez/presentation_7_pna.pptx` (v1, Perch, retouchée à la main par Léonard : ne plus la régénérer) et `documentation/prez/presentation_7_pna_v2.pptx` (v2, BirdNET 2.4, fenêtres de 3 s). Animation par la transition Morphose (PowerPoint 2019 / Microsoft 365). Données intermédiaires dans `rapports/presentation_7/` (jamais committées). Tout se lit en lecture seule : la base (`mode=ro`), les stocks d'embeddings, l'audio (disque `BLANCI_2023-24`).

Ordre pour la v2 (depuis la racine du dépôt) :

```sh
uv run python documentation/prez/scripts/presentation_7/extraire_birdnet.py  # ≈ 30 min : fenêtres au hasard et annotées, stock BirdNET
uv run python documentation/prez/scripts/presentation_7/figures_audio.py     # spectrogramme et extrait audio (diapo 3)
uv run python documentation/prez/scripts/presentation_7/figures_pna.py       # inventaire, forme d'onde, spectrogramme de la fenêtre, part du chant
uv run python documentation/prez/scripts/presentation_7/donnees.py           # variance par facteur, rejeu d'apprentissage actif, nuage
uv run --with python-pptx python documentation/prez/scripts/presentation_7/deck.py
uv run --with python-pptx --with pillow python documentation/prez/scripts/presentation_7/verifier_textes.py documentation/prez/presentation_7_pna_v2.pptx
```

`extraire.py` et `extraire_labels.py` sont ceux de la v1 (stock Perch v2).

Ne pas relancer `deck.py` par-dessus une version retouchée à la main : passer un autre chemin en argument.
