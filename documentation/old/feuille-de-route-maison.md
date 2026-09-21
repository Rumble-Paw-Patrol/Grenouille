


# Feuille de route fait main

## 1. Lecture bibliographique

Lire la doc de l'ONF
Check l'édition 2026 de BirdCLEF : m'inspirer des meilleurs candidats. C'est une excellente source, l'édition 2026 portant sur la faune du Pantanal au Brésil et limitant le budget d'inférence des candidats au CPU. Regarder en particulier les working notes des participants très instructives, disponibles fin septembre.

Check PAMGuard : s'en inspirer pour l'appli de bureau pour naturalistes de l'ONF. Trois limites. Pas de ré-entraînement dans l'application. Le post-traitement séquentiel (intervalles entre notes) n'entre pas dans un graphe ONNX simple. L'ergonomie est celle d'un logiciel d'acousticien, c'est lourd.

Check AnuraSet : faire des tests de mes modèles avec ce database

Rééditer ma feuille de route avec les informations que j'ai collecté depuis le début de mon stage


## 2. Construire la database

Décider si on part du jeu de données annotées ou si on part de zéro et on fait du clustering pour ensuite annoter les clusters (garder quand même les données annotées).

## 3. Coder en python ma pipeline

Faire de l'apprentissage actif, le jeu de données grandissant avec le temps.

audio brut (format et f_e inconnus, H1)
  └─ ingest      inventaire, métadonnées (site, horodatage, enregistreur) → SQLite
  └─ decode      float32 mono, f_e native conservée (soundfile ; ffmpeg si exotique)
  └─ grid        fenêtres (recording_id, offset_s, dur_s), INDÉPENDANTES de l'encodeur
  └─ encoder[k]  rééchantillonnage vers f_e(k) ; E_k ∈ R^{N×d_k}
  └─ store       Parquet partitionné encoder_id / site / mois, float16
  └─ index       RAG, similarité cosinus, exhaustive par fragments (FAISS optionnel), Un produit scalaire dans un bon espace de représentation bat un algorithme sophistiqué dans un mauvais espace. L'intelligence est dans l'encodeur. Apparier les positifs et les négatifs par point d'éoute et heure si possible
  └─ head[v]     régression logistique → score par fenêtre
  └─ sequential  descripteurs fins des profils de chants → tête de fusion avec head
  └─ aggregate   fenêtre → enregistrement → point × période
  └─ queue       file de vérification (GUI) → labels append-only → ré-entraînement de head

## 4. Tester les modèles, les comparer
Ordre d'approches :
indices acoustique via scikit-maad pour nettoyer la database s'il y a des micros disfonctionnels par exemple
template matching (baseline)
few-shot linear probing
linear probing sur foundation model via bacpipe
attentive probing sur foundation model via bacpipe
PEFT LoRA sur modèle préexistant
fine-tuning sur modèle préexistant
modèle maison : la distillation (idée de Claude, approfondir)

décider d'un protocole d'évaluation des modèles

faire le benchmark des approches

## 5. Optimiser
faire de l'active learning

optimiser les approches avec :
seuillage spectral
phénologie
durée d'une fenêtre
data augmentation
negative mining
module séquentiel (à approfondir, très important)
autres outils à déterminer

faire le benchmark des approches après les optimisations

## 6. Combiner les modèles
combiner les approches :
soft voting
soft voting pondéré
stacking
concaténation
sélection gloutonne avec remise

faire le benchmark des combinaisons

pour maximiser l'intérêt de combiner les approches, il faut faire varier au plus les approches : l'architecture, le modèle pré-entraîné, la résolution du spectrogramme, les augmentations, le découpage temporel

