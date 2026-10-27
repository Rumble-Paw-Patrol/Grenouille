# Encodage

Le script PowerShell (Windows) qui lance l'encodage de tout le corpus. Le script est conçu pour être reprenable : les enregistrements déjà encodés sont sautés.

- `encode_modele.ps1` : le modèle de départ pour encoder avec un de nos encodeurs (`-Encoder <nom>`). Il relance `blanci embed` jusqu'à la fin, journalise dans `rapports/encodage.log` et `rapports/embed_<encodeur>.log`, et empêche la mise en veille.

- `ENCODAGE.txt` : les commandes PowerShell pour lancer l'encodage, voir où il en est et l'arrêter.
