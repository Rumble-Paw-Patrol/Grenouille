# Présentation n° 7 : PNA A. blanci (restitution de stage)

Sorties : `documentation/prez/presentation_7_pna.pptx` (v1, Perch, retouchée à la main par Léonard : ne plus la régénérer), `documentation/prez/presentation_7_pna_v2.pptx` (v2, BirdNET 2.4, fenêtres de 3 s) et `documentation/prez/presentation_7_pna_v3.pptx` (v3 : zoom dans BirdNET avec un réseau de neurones animé en GIF, scanner vertical, repère XYZ, file d'écoute en trois clics ; aucune diapo n'avance seule, sinon le retour en arrière est bloqué), `documentation/prez/presentation_7_pna_v4.pptx` (v4 : poussée latérale entre diapos sans animation, tête de classification en un neurone animé, file d'écoute à droite, diapo « regrouper ou apprendre », dernière diapo carte SD → chaîne → carte de présence) et `documentation/prez/presentation_7_pna_v5.pptx` (v5 : la tête apprend ses poids passe par passe, frontière de décision qui bouge à chaque version de la tête, légende des scores dans le panneau puis en bas, carte de la Guyane avec Kaw, Molokoï et Trésor, textes repris). Animation par la transition Morphose (PowerPoint 2019 / Microsoft 365). Données intermédiaires dans `rapports/presentation_7/` (jamais committées). Tout se lit en lecture seule : la base (`mode=ro`), les stocks d'embeddings, l'audio (disque `BLANCI_2023-24`).

Ordre pour la v5 (depuis la racine du dépôt) ; `figures_v5.py` lit aussi `rapports/presentation_7/carte/region-guyane.geojson`, contour IGN de la Guyane (36 Ko), à télécharger une fois depuis le dépôt public `gregoiredavid/france-geojson` (`regions/guyane/region-guyane.geojson`) :

```sh
uv run python documentation/prez/scripts/presentation_7/extraire_birdnet.py  # ≈ 30 min : fenêtres au hasard et annotées, stock BirdNET
uv run python documentation/prez/scripts/presentation_7/figures_audio.py     # spectrogramme et extrait audio (diapo 3)
uv run python documentation/prez/scripts/presentation_7/figures_pna.py       # inventaire, forme d'onde, spectrogramme de la fenêtre, part du chant
uv run python documentation/prez/scripts/presentation_7/animation_reseau.py  # GIF du réseau de neurones, spectrogramme sombre
uv run python documentation/prez/scripts/presentation_7/donnees.py           # variance par facteur, rejeu d'apprentissage actif, nuage
uv run python documentation/prez/scripts/presentation_7/figures_v4.py        # spectrogramme agrandi, courbe, carte de présence, GIF de la tête
uv run python documentation/prez/scripts/presentation_7/figures_v5.py        # carte de la Guyane, GIF de l'entraînement de la tête
uv run --with python-pptx python documentation/prez/scripts/presentation_7/deck.py
uv run --with python-pptx --with pillow python documentation/prez/scripts/presentation_7/verifier_textes.py documentation/prez/presentation_7_pna_v5.pptx
```

`extraire.py` et `extraire_labels.py` sont ceux de la v1 (stock Perch v2).

Ne pas relancer `deck.py` par-dessus une version retouchée à la main : passer un autre chemin en argument.
