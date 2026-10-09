# Tableau de bord, version dashboard claude.ai

Le tableau de bord du dossier parent refait avec le type « Dashboard » de claude.ai :
<https://claude.ai/artifact/VAfDVGTLfR5stBZv7Y9dyJ>. Privé tant qu'il n'est pas partagé depuis
son menu « Partager ».

Chaque chiffre, graphique et tableau de la page vient d'une **source** (un CSV ou un JSON
attaché au dashboard), visible dans le panneau Sources avec sa description, sa fraîcheur et ses
lignes (téléchargeables en CSV). Les totaux et les parts sont des **calculs** (`dash.calc`),
eux aussi lisibles dans Sources.

## Fichiers

| Fichier | Rôle |
|---|---|
| `donnees.py` | écrit les 22 sources dans `donnees/` (ignoré par git), avec les fonctions de `../construire.py` |
| `images.py` | écrit `page/bandeau.css` et `page/poste.css` (ignorés par git) : spectrogramme, planche des 24 images de la grenouille 8-bit, capture du poste d'annotation, en `data:` URL |
| `page/index.html` | balisage et styles ; les textes de `../textes.yaml` y sont repris tels quels |
| `page/commun.js` | outils communs (formats français, état des sources, infobulle, pages) |
| `page/vue.js` | Vue d'ensemble, En développement, Données, Chaîne, Annotation, Temps d'encodage |
| `page/anuraset.js` | AnuraSet : jeu réduit, encodeurs, explorateur des têtes, amorçage |
| `page/lectures.js` | Rapports, tableaux de référence, bibliographie |
| `page/bandeau.js` | animation de la grenouille, chiffres marqués, navigation entre les pages |

Sources (identifiant dans le dashboard = nom du fichier) : `inventaire`, `fenetres`,
`controle_qualite`, `chaine`, `tests`, `annotation_sites`, `annotation_micros`, `encodage`,
`chantiers`, `commits`, `reperes`, `anuraset_sites`, `anuraset_positifs`, `anuraset_especes`,
`encodeurs`, `transfert`, `comparaisons`, `tetes_01_06`, `amorcage`, `rapports`, `tableaux`,
`biblio`.

## Mettre à jour

Les sources sont des fichiers figés : elles ne se rafraîchissent pas seules. Demander à Claude
« mets à jour le dashboard » : il relance `donnees.py`, téléverse les fichiers qui ont changé
dans le dashboard et met à jour leur source (adresse du fichier et date). Un changement de la
page (`page/`) se republie de la même façon, fichier par fichier.

Ce qui n'a pas été repris de l'ancienne page : le bouton « Écouter le chant » (le dashboard
n'accepte pas de son), le texte complet des rapports et les images des tableaux (liens vers
GitHub à la place), les boutons « Commenter » (le dashboard a ses propres commentaires).
