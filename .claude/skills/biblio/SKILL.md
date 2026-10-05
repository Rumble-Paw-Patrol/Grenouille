---
name: biblio
description: Ajoute à documentation/biblio/biblio.md, dans la bonne section, les références bibliographiques (articles, preprints, dépôts, jeux de données, pages de concours) citées ou trouvées dans la conversation en cours, au format du fichier. À utiliser quand Léonard dit « biblio », « ajoute ça à la biblio », « mets les références dans la bibliographie », ou en fin de recherche bibliographique.
---

# Ajouter les références de la conversation à la bibliographie

Fichier : `documentation/biblio/biblio.md`. Le lire en entier avant d'écrire.

## 1. Rassembler

Relever dans la conversation chaque lien vers une **source** : article, preprint (arXiv,
bioRxiv), DOI, page d'éditeur, dépôt de code ou de modèle (GitHub, Hugging Face, Zenodo), jeu
de données, page de concours ou compte rendu. Ignorer les liens de navigation, de
documentation d'outils (pytest, uv…), et les pages de recherche.

Écarter ce qui est déjà dans le fichier : chercher l'URL **et** le titre (une même référence
peut y être sous une autre URL, arXiv contre DOI). Si l'entrée existe et que la conversation
apporte du neuf (version publiée, chiffre, autre lien), compléter l'entrée existante au lieu
d'en créer une.

Pour chaque référence nouvelle, vérifier titre, auteurs, année et lieu de publication à la
source (WebFetch du lien) plutôt que de mémoire. Ce qui n'a pas pu être vérifié est écrit
« non vérifié ».

## 2. Classer

Une référence va dans la section dont le titre décrit le mieux son apport **pour le projet**
(lire les titres `##` du fichier au moment de l'ajout, ils peuvent avoir changé). Repères
actuels :

- modèle ou encodeur bioacoustique → « Modèles de fondation en bioacoustique » ;
- concours BirdCLEF et leurs comptes rendus → « BirdCLEF+ 2026 » ;
- transfert vers les anoures → « Anoures : transfert depuis les modèles de fondation » ;
- biologie, chant, phénologie d'A. blanci ou des congénères → « Phénologie et écologie » ;
- jeu de données, banc d'essai → « Jeux de données et bancs d'essai » ;
- méthode d'apprentissage (attention, régularisation, domaine, perte, calibration) →
  « Apprentissage : attention, régularisation, domaine » ;
- annotation, détection, logiciel d'écoute → « Annotation, détection et outils ».

Si aucune section ne convient et qu'au moins deux références la rempliraient, proposer une
nouvelle section à Léonard avant de la créer ; sinon, la section la plus proche.

## 3. Écrire

Format exact d'une entrée (une puce, deux espaces d'indentation pour la suite) :

```
- **Titre exact** — Premier auteur et al., année, lieu (revue, conférence).
  <https://lien-principal> · <https://second-lien-éventuel>
  Une à trois lignes : ce que la source montre, avec le chiffre qui le porte.
  → ce qu'on en retient pour A. blanci (ou « à lire », s'il n'y a pas encore d'avis).
```

- Liens entre chevrons, séparés par « · ».
- Français, phrases courtes, virgule décimale.
- La ligne « → » dit ce que la référence change pour le projet (encodeur à tester, choix
  confirmé, piste écartée). Pas de renvoi à `DECISIONS.md`.
- Ordre dans la section : à la suite des entrées existantes, sauf si elle complète un groupe
  (même famille de modèles, même concours) : alors juste après lui.

Mettre à jour la date de la phrase d'en-tête (« Mise à jour du JJ/MM/AAAA… »).

## 4. Rendre compte

Lister à Léonard ce qui a été ajouté (titre → section), ce qui a été complété, ce qui a été
écarté comme doublon, et ce qui reste « non vérifié ». Ne pas committer sans qu'il le demande.
