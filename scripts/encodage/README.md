# Encodage

Le script PowerShell (Windows) qui lance l'encodage de tout le corpus. Le script est conçu pour être reprenable : les enregistrements déjà encodés sont sautés.

- `encode_modele.ps1` : le modèle de départ pour encoder avec un de nos encodeurs (`-Encoder <nom>`). Il relance `blanci embed` jusqu'à la fin, journalise dans `rapports/encodage.log` et `rapports/embed_<encodeur>.log`, et empêche la mise en veille.

- `ENCODAGE.txt` : les commandes PowerShell pour lancer l'encodage, voir où il en est et l'arrêter.

- `encode_modele.ps1 -Recordings <liste.csv> -Embeddings <dossier>` : n'encode que les enregistrements de la liste (colonne `recording_id`), dans un autre dossier de stocks que `paths.embeddings` (ex. `data\embeddings` pour `perch_v2`, sur F:). Sort en code 0 si l'encodage est allé au bout, 1 après trois échecs rapides.

- `file_pistes.py` : la file « bonnes pistes ». Pour chaque stock (`perch_v2`, `birdnet`, `birdnet_v3`, `audioprotopnet`), ce qui est encodé sur une autre piste que celle de la règle (piste écoutée pour un enregistrement annoté, sinon `audio.channel_rules` et le repli du son étouffé) (à retirer avec `blanci pistes --apply`), et ce qui reste à encoder, avec la durée estimée au débit mesuré. Lecture seule ; écrit sous `rapports/encodage_pistes/` : `<modèle>_pistes.csv` (les trois groupes : piste 0 étouffée, Kaw D au micro 0 sourd, gain inversé), `<modèle>.csv` (tout ce qui manque au stock), `resume.csv`, `extraits_annotes.csv` (piste écoutée contre piste encodée), `LISEZMOI.md`.

- `file_piste_cassee.py` : la file d'écoute `piste_cassee` : les extraits annotés en écoutant une piste cassée (piste 0 sourde de Kaw D, piste étouffée avec l'autre saine), aux bornes de leur dernier envoi, avec la colonne `channel` qui ouvre chacun sur la piste valable dans le poste. Un enregistrement annoté s'encode sur la piste écoutée (`channels.annotation_channels`) : le réenvoi le fait passer sur la piste valable. Le gain inversé n'est pas une casse : ces extraits gardent leur piste. Lecture seule de la base ; écrit la file dans `paths.queues`.

- `encode_pistes.ps1` : enchaîne `perch_v2`, `birdnet`, `birdnet_v3` sur leur `<modèle>_pistes.csv`, puis `audioprotopnet` sur tout le corpus, par `encode_modele.ps1`. `-Depuis <modèle>` reprend à ce modèle. Marche à suivre complète (listes, retrait, lancement) : `ENCODAGE.txt`.
