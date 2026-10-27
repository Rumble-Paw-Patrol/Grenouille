# Encodage

Les scripts PowerShell (Windows) qui lancent l'encodage de tout le corpus. Reprenables : les enregistrements déjà encodés sont sautés. Les journaux vont dans `rapports/` (ignoré par git).

- `ENCODAGE.txt` : les commandes PowerShell pour lancer l'encodage, voir où il en est et l'arrêter.
- `encode_v3.ps1` : la boucle d'encodage BirdNET 3.0, qui empêche la mise en veille.
- `weekend.ps1` : la chaîne du week-end, copie de D: vers F:, puis encodage perch_v2, puis BirdNET.
- `05_embeddings_vers_disque2.ps1` : déplace `data/embeddings` vers le disque 2, y pointe `paths.embeddings` et relance l'encodage.
