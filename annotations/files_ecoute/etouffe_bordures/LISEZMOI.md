# File de candidats « etouffe_bordures »

Écrite le 20/11/2026 à 15:01. 167 candidats, 167 enregistrements.

`candidats.csv` ne contient que des références (enregistrement, début, durée) : le poste d'annotation (`blanci annotate`) retrouve l'audio sur les disques branchés.

## Réglages

- origine : scripts/son_etouffe/files_milieu_bordure.py (carte CUSUM, référence 0.05, un créneau manqué toléré)
- contenu : premier et dernier enregistrement retenus de chaque épisode, avec leurs deux voisins au programme (dont un hors épisode) ; extraits de 30 s

## Motifs

- etouffe_bordures/CDR/voisin sain : 87
- etouffe_bordures/CDR/candidat retenu : 60
- etouffe_bordures/CDR/voisin retenu : 14
- etouffe_bordures/PatawaOuest/voisin sain : 4
- etouffe_bordures/PatawaOuest/candidat retenu : 2

## Sites

- CDR : 161
- PatawaOuest : 6
