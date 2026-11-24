# Corpus entériné du 23/11/2026

Contrôle qualité clos. `corpus.csv` : les enregistrements que garde le corpus et la piste sur laquelle chacun s'encode (`scripts/encodage/manifeste_corpus.py`).

- 107 293 enregistrements retenus sur 108 012 en base, 3 576,6 h ; 186 avec A. blanci entendu.
- jeu 2018 : 20 enregistrements, 1,0 h
- jeu 2023 : 66 754 enregistrements, 2 225,1 h
- jeu 2026 : 40 513 enregistrements, 1 350,4 h
- jeu reference : 6 enregistrements, 0,0 h

## Piste d'encodage

- piste 0, défaut : 86 998
- piste 0, envoi : 575
- piste 0, intervalles : 107
- piste 1, commentaire : 19
- piste 1, envoi : 11
- piste 1, intervalles : 19
- piste 1, piste cassée : 4
- piste 1, repli son étouffé : 462
- piste 1, règle : gain inversé : 16 164
- piste 1, règle broken : 2 934

## Empreinte

- base : `C:\Users\ll34272\base_blanci\blanci.sqlite` (migration 7)
- inventaire : `2fe95fef5b7ddf661c540c399da84aa78d30e7512f413d53633ce852f6b0e0a8`
- drapeaux : `5ac39dd0b2a72a1b99024a4e4987768db6b303dc7348b77a7d404508768aaa5b`
- dernière annotation : extrait 2026-11-20T19:31:04Z, fenêtre 2026-10-05T15:00:02Z
- code : commit `cb38a23` ; configuration : `config/default.yaml` de ce commit

Une base dont l'empreinte des drapeaux diffère n'a plus ce corpus : ne relancer ni `blanci flag`, ni `blanci qc`, ni `blanci ingest`. Une annotation sur l'autre piste d'un enregistrement déjà encodé change sa piste (`blanci pistes`).
