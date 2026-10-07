---
name: tableau-de-bord
description: Met à jour et republie le tableau de bord claude.ai du projet Grenouille en un seul appel — reprend les textes et les chantiers que Léonard a modifiés à la main (textes.yaml, en_cours.yaml), repère ce qui a été ajouté, modifié ou supprimé dans le dépôt depuis la dernière publication, met la page en accord, recompte les annotations, republie à la même URL. Argument facultatif — texte libre sur les chantiers ou les textes. À utiliser pour « /tableau-de-bord », « mets à jour le tableau de bord », ou après un benchmark, un audit ou un changement notable.
---

# Mettre à jour le tableau de bord

Dossier : `documentation/tableau-de-bord/` (voir `LISEZMOI.md`). URL fixe :
<https://claude.ai/artifact/DLfBZopGjpLyJPBi9ViLi8>. Ne jamais publier ailleurs.

But : **un seul appel suffit**. `/tableau-de-bord` sans argument veut dire « reprends mes
fichiers et republie » : ne rien demander de plus. Ne poser de question que pour ce qu'on ne
peut pas déduire du dépôt, toutes en un seul message, avec une proposition par question.

## Qui tient quoi

| Fichier | Contenu | Tenu par |
|---|---|---|
| `textes.yaml` | tous les textes affichés : projet, critères, menu, titres, sous-titres, cartes, phrases, textes des étapes de la chaîne (`chaine:`), annotation, introduction AnuraSet | **Léonard** |
| `en_cours.yaml` | les chantiers de la section « En développement » | **Léonard** |
| `structure.yaml` | la chaîne côté code (module, tests, commandes, `etat`, `suite`), lieux, encodeurs suivis, cibles d'annotation | Claude |
| `modele.html` | la mise en page et le code, **sans texte rédigé** | Claude |
| `construire.py` | lit tout le dépôt et écrit `index.html`, `fichiers.json` | Claude |

Les fichiers de Léonard font foi : **ne jamais réécrire un texte qu'il a modifié** sans qu'il le
demande. Un texte nouveau (nouvelle carte, nouvelle section) va dans `textes.yaml`, jamais en
dur dans `modele.html`. Si une demande de Léonard porte sur un texte, l'appliquer dans
`textes.yaml` (ou `en_cours.yaml`).

## 1. Récupérer les modifications de Léonard

Il édite sur sa machine ou sur GitHub : `git fetch origin main` puis fusionner `origin/main`
dans la branche de travail avant tout le reste, pour partir de ses derniers textes.

## 2. Ce qui a changé depuis la dernière publication

```sh
uv run python documentation/tableau-de-bord/construire.py --changements
```

| Rubrique du rapport | Quoi faire |
|---|---|
| Commits | les lire : ils disent ce que Léonard a fait (sert aux étapes 3 et 4) |
| Pris en compte tout seul | rien : benchmarks, tableaux, biblio, inventaire, tests, fichiers du tableau de bord sont relus par `construire.py`. Vérifier seulement qu'un fichier supprimé ou renommé ne casse pas la construction |
| À reporter dans structure.yaml | modules de `blanci/` ajoutés, supprimés, renommés ; commandes nouvelles ou disparues. Mettre à jour l'étape de `chaine:` concernée (module, commandes, état), ou `commandes_hors_chaine:` pour un outil de recherche. Une étape nouvelle : l'ajouter dans `structure.yaml`, ses textes dans `textes.yaml` (`chaine:`), sa position dans `POS` de `modele.html` |
| Sans place dans le tableau de bord | un ajout que la page ne montre pas (nouveau dossier de documentation, nouveau type de résultat, notebook…) : proposer une carte ou une section ; une suppression : retirer ce qui s'y rapportait |

Un chiffre de la page (débit, repère d'étape) ne change que s'il vient d'un commit, d'un
rapport ou d'un CSV. Ne rien inventer.

Jamais de nom de branche de travail (`claude/…`) sur la page : un commit de fusion garde son
numéro mais affiche le commit qu'il apporte (`construire.py` s'en charge).

## 3. Chantiers (`en_cours.yaml`)

- d'après l'argument du skill s'il y en a un (nouveau chantier, fini, bloqué, prochaine
  action) : l'appliquer directement ;
- d'après les commits : si un commit termine visiblement un chantier ou en ouvre un, le
  **proposer** (« Le chantier X semble fini (commit n° 72) : je le retire ? ») sans
  l'appliquer avant sa réponse ; le reste de la mise à jour continue en attendant.

Pas de date future ni d'échéance.

## 4. Modifier la page si besoin

Une nouvelle section ou carte : la donnée dans `construire.py`, l'affichage dans
`modele.html` (composants existants : `section(id)`, `carteT(section, cle, …)`,
`boutonCommenter`, tokens de couleur), **ses textes dans `textes.yaml`** (`sections.<id>` :
`menu`, `titre`, `sous_titre`, `surtitre`, `cartes`, `phrases`). Une section nouvelle s'ajoute
aussi dans `SECTIONS` de `modele.html` (ordre, et section parente pour un sous-menu).

Mise en page pensée pour un écran d'ordinateur : utiliser toute la largeur, pas de largeur
maximale sur les blocs. Vérifier une fois le rendu (Chromium : `/opt/pw-browsers`,
Playwright), clair et sombre ; à 400 px de large, seulement qu'il n'y a ni erreur de script
ni défilement horizontal.

## 5. Construire et publier

```sh
uv run python documentation/tableau-de-bord/construire.py
```

- Le script compte les annotations dans `data/db/blanci.sqlite`. Sans base (session cloud),
  il garde le dernier comptage : le dire en une ligne, avec sa date.
- Une erreur de lecture de `textes.yaml` (indentation, virgule dans un titre de liste) :
  la corriger a minima et dire à Léonard quelle ligne posait problème.
- En cas d'erreur, corriger avant de publier ; ne jamais publier une page cassée.

Publication avec l'outil Artifact : `file_path` = chemin absolu de `index.html`, `url` =
l'URL ci-dessus (la lire d'abord avec `action: "read"` si cette session ne l'a ni lue ni
publiée), `files` = contenu de `fichiers.json` avec chaque source en chemin absolu. Ne pas
passer `icon` ni `capabilities`.

Puis noter la publication :

```sh
uv run python documentation/tableau-de-bord/construire.py --publie
```

## 6. Rendre compte et committer

En trois à cinq lignes : ce qui a changé sur la page, ce qui attend une réponse, le lien.
Committer les fichiers du dossier modifiés (« Tableau de bord : … »), pousser sur la branche
de travail, puis sur main (`git push origin HEAD:main`, en avance rapide après la fusion de
l'étape 1) : Léonard édite ses fichiers sur main.
