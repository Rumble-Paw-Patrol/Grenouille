---
name: tableau-de-bord
description: Met à jour et republie le tableau de bord claude.ai du projet Grenouille en un seul appel — repère ce qui a été ajouté, modifié ou supprimé dans le dépôt depuis la dernière publication, met la page en accord (structure, chantiers, nouvelles sections si besoin), recompte les annotations, republie à la même URL. Argument facultatif — texte libre sur les chantiers. À utiliser pour « /tableau-de-bord », « mets à jour le tableau de bord », ou après un benchmark, un audit ou un changement notable.
---

# Mettre à jour le tableau de bord

Dossier : `documentation/tableau-de-bord/` (voir `LISEZMOI.md`). URL fixe :
<https://claude.ai/artifact/DLfBZopGjpLyJPBi9ViLi8>. Ne jamais publier ailleurs.

But : **un seul appel suffit**. Ne poser de question que pour ce qu'on ne peut pas déduire
du dépôt, toutes les questions en un seul message, avec une proposition par question.

## 1. Ce qui a changé depuis la dernière publication

```sh
uv run python documentation/tableau-de-bord/construire.py --changements
```

Le rapport range tout ce qui a changé depuis le commit noté dans `publication.json` :

| Rubrique du rapport | Quoi faire |
|---|---|
| Commits | les lire : ils disent ce que Léonard a fait (sert aux étapes 2 et 3) |
| Pris en compte tout seul | rien : benchmarks, tableaux, biblio, glossaire, inventaire, tests sont relus par `construire.py`. Vérifier seulement qu'un fichier supprimé ou renommé ne casse pas la construction |
| À reporter dans structure.yaml | modules de `blanci/` ajoutés, supprimés, renommés ; commandes nouvelles ou disparues. Mettre à jour l'étape de `chaine:` concernée (module, commandes, état, résumé, repère chiffré), ou `commandes_hors_chaine:` pour un outil de recherche. Une étape nouvelle : l'ajouter avec sa position dans `POS` de `modele.html` |
| Sans place dans le tableau de bord | un ajout que la page ne montre pas (nouveau dossier de documentation, nouveau type de résultat, notebook…) : proposer une carte ou une section ; une suppression : retirer ce qui s'y rapportait |

Un chiffre de la page (débit, repère d'étape) ne change que s'il vient d'un commit, d'un
rapport ou d'un CSV. Ne rien inventer.

## 2. Chantiers (`en_cours.yaml`)

C'est le seul fichier de Léonard. Le mettre à jour :
- d'après l'argument du skill s'il y en a un (nouveau chantier, fini, bloqué, prochaine
  action) : l'appliquer directement ;
- d'après les commits : si un commit termine visiblement un chantier ou en ouvre un, le
  **proposer** (« Le chantier X semble fini (commit n° 72) : je le retire ? ») sans
  l'appliquer avant sa réponse ; le reste de la mise à jour continue en attendant.

Pas de date future ni d'échéance.

## 3. Ce qu'on ne peut pas déduire : demander

Si une rubrique « Sans place… » ou un changement de structure laisse un doute (où ranger,
faut-il le montrer, une section disparue doit-elle partir), poser les questions en un seul
message court, chacune avec la proposition par défaut. Appliquer les réponses, puis
continuer. Si Léonard ne répond pas sur un point, ne pas le bloquer : publier sans ce point
et le rappeler en une ligne.

## 4. Modifier la page si besoin

Une nouvelle section ou carte : `construire.py` (lire la donnée) et `modele.html` (l'afficher,
en reprenant les composants existants : `section`, `carte`, `boutonCommenter`, tokens de
couleur). Puis vérifier une fois le rendu (Chromium : `/opt/pw-browsers`, Playwright),
clair, sombre et 400 px de large : aucune erreur de script, pas de défilement horizontal.

## 5. Construire et publier

```sh
uv run python documentation/tableau-de-bord/construire.py
```

- Le script compte les annotations dans `data/db/blanci.sqlite`. Sans base (session cloud),
  il garde le dernier comptage : le dire en une ligne, avec sa date.
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
Committer `structure.yaml`, `en_cours.yaml`, `annotations.json`, `publication.json` et les
fichiers de la page modifiés (« Tableau de bord : … »), puis pousser sur la branche de travail.
