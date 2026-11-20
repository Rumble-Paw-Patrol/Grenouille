# File de candidats « etouffe_milieu »

Écrite le 20/11/2026 à 15:01. 112 candidats, 112 enregistrements.

`candidats.csv` ne contient que des références (enregistrement, début, durée) : le poste d'annotation (`blanci annotate`) retrouve l'audio sur les disques branchés.

## Réglages

- origine : scripts/son_etouffe/files_milieu_bordure.py (carte CUSUM, référence 0.05, un créneau manqué toléré)
- contenu : enregistrements retenus au-dessus du seuil, en plein milieu d'un épisode, avec leurs deux voisins au programme ; extraits de 30 s

## Motifs

- etouffe_milieu/CDR/candidat retenu : 53
- etouffe_milieu/CDR/voisin sain : 33
- etouffe_milieu/CDR/voisin retenu : 23
- etouffe_milieu/PatawaOuest/voisin sain : 2
- etouffe_milieu/PatawaOuest/candidat retenu : 1

## Sites

- CDR : 109
- PatawaOuest : 3
