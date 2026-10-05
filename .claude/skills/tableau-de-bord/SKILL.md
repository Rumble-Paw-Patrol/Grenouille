---
name: tableau-de-bord
description: Met à jour et republie le tableau de bord claude.ai du projet Grenouille (reconstruction depuis le dépôt, compteur d'annotations, structure de la chaîne, publication à la même URL avec les images). Argument facultatif — texte libre sur les chantiers en cours à reporter dans en_cours.yaml. À utiliser quand Léonard dit « mets à jour le tableau de bord », « republie le dashboard », « /tableau-de-bord », ou après un benchmark, un audit ou un changement notable du code.
---

# Mettre à jour le tableau de bord

Dossier : `documentation/tableau-de-bord/` (voir son `LISEZMOI.md`). URL fixe :
<https://claude.ai/artifact/DLfBZopGjpLyJPBi9ViLi8>. Ne jamais publier ailleurs.

## 1. Travail en cours (`en_cours.yaml`)

C'est le seul fichier que Léonard tient à la main.
- Si l'argument décrit des chantiers (nouveau, fini, bloqué, prochaine action), les reporter
  dans `en_cours.yaml` au format du fichier (titre, etat, detail, suite) ; retirer ceux qu'il dit
  finis. Montrer le résultat en trois lignes au plus.
- Sinon, ne pas y toucher.
- Pas de date future ni d'échéance dans ce fichier.

## 2. Structure (`structure.yaml`)

Tenue par Claude. Lire `git log` depuis le dernier commit qui a touché
`documentation/tableau-de-bord/`. Si un commit change l'état réel d'une étape de la chaîne
(module écrit, commande ajoutée ou renommée, étape finie, nouveau repère chiffré comme le débit
de l'encodage), mettre à jour l'entrée de `chaine:` ou `debit:` correspondante. Ne rien inventer :
un chiffre vient d'un commit, d'un rapport ou d'un CSV.

## 3. Construire

```sh
uv run python documentation/tableau-de-bord/construire.py
```

(`--config <fichier>` si la configuration locale n'est pas `config/local.yaml`.)

Le script compte les annotations dans la base locale (`data/db/blanci.sqlite`) et écrit
`annotations.json`. Sans base (session cloud), il relit le dernier `annotations.json` : le
dire en une ligne à Léonard, avec la date du comptage, et lui rappeler qu'il suffit de lancer
le skill une fois sur sa machine pour rafraîchir le compteur.

En cas d'erreur du script, la corriger avant de publier ; ne jamais publier une page cassée.

## 4. Publier

Outil Artifact, action publish :
- `file_path` : chemin absolu de `documentation/tableau-de-bord/index.html` ;
- `url` : l'URL ci-dessus (si cette session n'a encore ni lu ni publié l'artifact, le lire
  d'abord avec `action: "read"`, comme l'exige l'outil) ;
- `files` : le contenu de `fichiers.json`, chaque source préfixée par la racine du dépôt
  (chemin absolu) ;
- ne pas passer `icon` ni `capabilities` (ils sont gardés).

## 5. Rendre compte et committer

En deux à quatre lignes : ce qui a changé sur la page (chantiers, étapes, compteur, nouveaux
rapports) et le lien. Committer `en_cours.yaml`, `structure.yaml` et `annotations.json` s'ils
ont changé (message : « Tableau de bord : … »), puis pousser sur la branche de travail.
