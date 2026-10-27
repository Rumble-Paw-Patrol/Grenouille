# Encodage

Le script PowerShell (Windows) qui lance l'encodage de tout le corpus. Reprenable : les enregistrements déjà encodés sont sautés. Les fichiers de log (`.log`) vont dans `rapports/` (ignoré par git).

- `encode_modele.ps1` : le modèle de départ pour encoder avec un de nos encodeurs (`-Encoder <nom>`). Il relance `blanci embed` jusqu'à la fin, journalise dans `rapports/encodage.log` et `rapports/embed_<encodeur>.log`, et empêche la mise en veille. Pour un besoin particulier, copier ce fichier et l'adapter.
- `ENCODAGE.txt` : les commandes PowerShell pour lancer l'encodage, voir où il en est et l'arrêter.
