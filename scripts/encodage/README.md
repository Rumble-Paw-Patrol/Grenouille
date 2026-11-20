# Encodage

Le script PowerShell (Windows) qui lance l'encodage de tout le corpus. Le script est conçu pour être reprenable : les enregistrements déjà encodés sont sautés.

- `encode_modele.ps1` : le modèle de départ pour encoder avec un de nos encodeurs (`-Encoder <nom>`). Il relance `blanci embed` jusqu'à la fin, journalise dans `rapports/encodage.log` et `rapports/embed_<encodeur>.log`, et empêche la mise en veille.

- `ENCODAGE.txt` : les commandes PowerShell pour lancer l'encodage, voir où il en est et l'arrêter.

- `encode_modele.ps1 -Recordings <liste.csv> -Embeddings <dossier>` : n'encode que les enregistrements de la liste (colonne `recording_id`), dans un autre dossier de stocks que `paths.embeddings` (ex. `data\embeddings` pour `perch_v2`, sur F:). Sort en code 0 si l'encodage est allé au bout, 1 après trois échecs rapides.

- `file_pistes.py` : la file « bonnes pistes ». Pour chaque stock (`perch_v2`, `birdnet`, `birdnet_v3`, `audioprotopnet`), ce qui est encodé sur une autre piste que celle de la règle (piste écoutée pour un enregistrement annoté, sinon `audio.channel_rules` et le repli du son étouffé) (à retirer avec `blanci pistes --apply`), et ce qui reste à encoder, avec la durée estimée au débit mesuré. Lecture seule ; écrit sous `rapports/encodage_pistes/` : `<modèle>_pistes.csv` (les trois groupes : piste 0 étouffée, Kaw D au micro 0 sourd, gain inversé), `<modèle>.csv` (tout ce qui manque au stock), `resume.csv`, `extraits_annotes.csv` (piste écoutée contre piste encodée), `LISEZMOI.md`.

- `encode_pistes.ps1` : enchaîne `perch_v2`, `birdnet`, `birdnet_v3` sur leur `<modèle>.csv` (tout ce qui manque au stock, dont ce qui vient d'en être retiré), puis `audioprotopnet` sur tout le corpus, sur F: (`data\embeddings` : C: n'a pas la place), par `encode_modele.ps1`. `-Depuis <modèle>` reprend à ce modèle. Marche à suivre complète (sauvegarde, listes, retrait, lancement, vérification) : `ENCODAGE.txt`.

- `manifeste_corpus.py` : le corpus entériné (contrôle qualité clos le 20/11/2026), dans `annotations/corpus_<aaaammjj>/` : `corpus.csv` (chaque enregistrement retenu par `select_recordings`, sa piste d'encodage et son origine : règle `broken`, commentaire, piste cassée, intervalles, envoi, règle, repli, défaut ; positif) et `LISEZMOI.md` (comptes, empreinte de la base `database_fingerprint`, commit). Lecture seule de la base.
