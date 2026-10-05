---
name: prez
description: Crée une présentation PowerPoint (.pptx) de suivi de stage ou de réunion pour le projet Grenouille, générée par un script pptxgenjs versionné, avec la charte des présentations précédentes, des figures tirées des données du dépôt et un script de notes orateur. À utiliser quand Léonard demande une prez, des diapos, un support de réunion, un point d'étape ou une présentation de suivi.
---

# Présentation de suivi

Modèle : `documentation/prez/presentation-suivi-2/`. Lire `deck.js` en entier avant de
commencer : on reprend ses aides (`content`, `card`, `dot`, `table`, `bullets`,
`richBullets`, `notes`…) et sa charte telles quelles, on ne les réinvente pas.

## 1. Cadrer avec Léonard

Avant d'écrire, demander (une seule question groupée, sauf si la demande y répond déjà) :
- le public et la date (réunion de suivi, encadrants, ONF, soutenance) ;
- la durée, donc le nombre de diapos (≈ 1 par minute et demie) ;
- les sujets à couvrir, dans l'ordre.

Sans réponse sur les sujets, proposer un plan tiré de ce qui a changé depuis la présentation
précédente (`git log` depuis sa date, rapports de `documentation/benchmarks/` plus récents,
`README.md` « Où en est le projet »), et le faire valider.

## 2. Dossier

`documentation/prez/presentation-suivi-<N>/` (N = le suivant), avec :

| Fichier | Rôle |
|---|---|
| `deck.js` | construit le .pptx et écrit `script.md` ; en-tête : comment le lancer |
| `generer_figures.py` | toutes les figures (PNG) et extraits audio, depuis les données du dépôt |
| `figures/`, `audio/` | sorties de `generer_figures.py`, committées |
| `script.md` | notes orateur, une section par diapo, généré par `deck.js` |

Sortie : `documentation/prez/presentation_suivi_<N>.pptx`.

## 3. Charte (celle des présentations n° 1 et 2)

- Format 16:9 large (`LAYOUT_WIDE`, 13,333 × 7,5 in), marges 0,6 in.
- Famille de verts, accent or : `dark 0E2A24`, `dark2 173D34`, `ink 1B2A24`, `muted 5B6B63`,
  `sage A7BFB2`, `pale E4EEE8`, `paler F3F7F4`, `gold E9B44C`, `teal 2A7F72`, `terra C8553D`.
- Graphiques : `2E8B57, D9822B, 3E7CB1, C0492F, 8A6BBE`, dans cet ordre. Pour un graphique
  matplotlib, charger le skill `dataviz` avant d'écrire le code.
- Titres en Cambria, texte en Calibri.
- Diapo de titre sur fond `dark` avec image à droite ; diapos de contenu sur fond blanc,
  pastille de rubrique en haut à gauche, titre Cambria 28 pt, numéro de page en bas à droite.
- Auteur : Léonard Laplace-Palette. Date en toutes lettres (« 30 septembre 2026 »).

## 4. Contenu

- Une idée par diapo ; le titre de la diapo est cette idée, en phrase (« Le rythme du chant,
  mesuré directement sur le signal »), pas un mot-clé.
- Chiffres : seulement ceux qui portent une conclusion, tirés des CSV de
  `documentation/benchmarks/*/donnees/` ou recalculés par `generer_figures.py`, jamais
  recopiés de mémoire. Virgule décimale, espaces fines dans les milliers.
- Tableaux : 6 lignes au plus ; au-delà, une figure.
- Spectrogrammes et extraits audio : partir des enregistrements locaux (non versionnés) ;
  le script de figures dit d'où viennent les fichiers et comment le relancer.
- Notes orateur rédigées comme on parle, assez complètes pour présenter sans la diapo (voir
  `presentation-suivi-2/script.md`). Expliquer les termes techniques pour des naturalistes.
- Pas de renvoi à `DECISIONS.md` ni aux numéros de décision : la raison est dite dans la diapo
  ou les notes.
- Dernière diapo : ce qui est fait d'ici la prochaine réunion, et les décisions attendues des
  encadrants.

## 5. Construire et vérifier

```sh
npm install --prefix /tmp/pptx pptxgenjs     # une fois ; ou npm i -g pptxgenjs
uv run --group notebook python documentation/prez/presentation-suivi-<N>/generer_figures.py
NODE_PATH=/tmp/pptx/node_modules node documentation/prez/presentation-suivi-<N>/deck.js
```

Puis contrôle visuel avec le skill `anthropic-skills:pptx` : rendre chaque diapo en image et
regarder débordements de texte, chevauchements, figures déformées, texte trop petit (< 12 pt
hors légendes). Corriger, reconstruire, une seule passe.

## 6. Rendre

Envoyer le .pptx à Léonard (SendUserFile), avec le plan en une ligne par diapo. Ne pas
committer sans qu'il le demande ; s'il le demande, committer le dossier source et le .pptx.
