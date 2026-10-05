# Tableau de bord

Page claude.ai privée, pour Léonard et ses encadrants :
<https://claude.ai/artifact/DLfBZopGjpLyJPBi9ViLi8>

Sections : vue d'ensemble, travail en cours, chaîne de traitement, données,
encodeurs, explorateur des têtes (comparaison à la carte), rapports de benchmark, tableaux,
bibliographie, glossaire. Chaque carte a un bouton « Commenter » ; les commentaires sont
partagés entre tous ceux qui ont accès à la page. La page n'est visible qu'une fois partagée
depuis son menu « Partager » sur claude.ai.

## Fichiers

| Fichier | Rôle | Qui le tient |
|---|---|---|
| `en_cours.yaml` | les chantiers en cours | **Léonard, à la main** (le seul) |
| `structure.yaml` | objectif, critères, étapes de la chaîne et leur état, débit, cibles d'annotation | Claude, skill `tableau-de-bord` |
| `annotations.json` | avancement des annotations, compté dans `data/db/blanci.sqlite` | `construire.py` |
| `modele.html` | la page, sans données | — |
| `construire.py` | lit le dépôt et écrit `index.html` et `fichiers.json` (ignorés par git) | — |

## Mettre à jour

`/tableau-de-bord` dans Claude Code, éventuellement suivi de ce qui a changé dans les
chantiers (« /tableau-de-bord le plan de tirage est validé, je commence l'annotation »).
Le skill reporte les chantiers, reconstruit la page et la republie à la même URL.

Le compteur d'annotations ne se rafraîchit que sur une machine qui a la base locale ; ailleurs,
la page garde le dernier comptage (`annotations.json`, versionné).
