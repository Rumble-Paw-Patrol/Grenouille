# Tableau de bord

Page claude.ai privée, pour Léonard et ses encadrants :
<https://claude.ai/artifact/DLfBZopGjpLyJPBi9ViLi8>

Sections : vue d'ensemble, en développement (travail en cours), données, chaîne de traitement,
annotation, temps d'encodage, AnuraSet (introduction et jeu réduit, puis trois sous-sections :
benchmark des encodeurs, explorateur des têtes, amorcer un site), rapports de benchmark,
tableaux, bibliographie. Chaque carte a un bouton « Commenter » ; les commentaires sont
partagés entre tous ceux qui ont accès à la page. La page n'est visible qu'une fois partagée
depuis son menu « Partager » sur claude.ai.

## Fichiers

| Fichier | Rôle | Qui le tient |
|---|---|---|
| `en_cours.yaml` | les chantiers en cours | **Léonard, à la main** (le seul) |
| `structure.yaml` | titre, objectif, critères, étapes de la chaîne et leur état, plan d'annotation | Claude, skill `tableau-de-bord` |
| `spectrogramme.jpg` | bandeau : un vrai chant d'A. blanci (Mataroni), 3,5–6,3 kHz | `spectrogramme.py` |
| `poste-annotation.png` | capture du poste d'annotation Streamlit, affichée en fin de section Annotation si elle existe | Léonard (capture d'écran) |
| `annotations.json` | avancement des annotations, compté dans `data/db/blanci.sqlite` | `construire.py` |
| `encodage.json` | temps d'encodage du corpus par encodeur, relu dans `models.params_json.totals` (cumul des passages d'`embed`) | `construire.py` |
| `publication.json` | commit de la dernière publication, point de départ du repérage des changements | `construire.py --publie` |
| `modele.html` | la page, sans données | — |
| `construire.py` | lit le dépôt et écrit `index.html` et `fichiers.json` (ignorés par git) | — |

## Mettre à jour

`/tableau-de-bord` dans Claude Code, éventuellement suivi de ce qui a changé dans les
chantiers (« /tableau-de-bord le plan de tirage est validé, je commence l'annotation »).
Le skill repère ce qui a été ajouté, modifié ou supprimé depuis la dernière publication
(`construire.py --changements`), met la page en accord, pose les questions qu'il ne peut pas
trancher seul, reconstruit la page et la republie à la même URL.

Le compteur d'annotations ne se rafraîchit que sur une machine qui a la base locale ; ailleurs,
la page garde le dernier comptage (`annotations.json`, versionné).
