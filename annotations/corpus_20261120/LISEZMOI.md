# Corpus entériné du 20/11/2026

Contrôle qualité clos. `corpus.csv` : les enregistrements que garde le corpus et la piste sur laquelle chacun s'encode (`scripts/encodage/manifeste_corpus.py`).

- 95 633 enregistrements retenus sur 96 349 en base, 3 187,9 h ; 186 avec A. blanci entendu.
- jeu 2018 : 20 enregistrements, 1,0 h
- jeu 2023 : 66 754 enregistrements, 2 225,1 h
- jeu 2026 : 28 853 enregistrements, 961,7 h
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
- piste 1, règle : gain inversé : 4 504
- piste 1, règle broken : 2 934

## Empreinte

- base : `C:\Users\ll34272\base_blanci\blanci.sqlite` (migration 7)
- inventaire : `92b0ecc4b7569d7bb21f7e6f4c12561df10f36a0839dd828acff8039657da23b`
- drapeaux : `1d0a3eb7b6ea30f1d690f2b123acfaeac0c0ecb33e92bf73302cccdd0d60d757`
- dernière annotation : extrait 2026-11-20T19:31:04Z, fenêtre 2026-10-05T15:00:02Z
- code : commit `6fd087d` ; configuration : `config/default.yaml` de ce commit

Une base dont l'empreinte des drapeaux diffère n'a plus ce corpus : ne relancer ni `blanci flag`, ni `blanci qc`, ni `blanci ingest`. Une annotation sur l'autre piste d'un enregistrement déjà encodé change sa piste (`blanci pistes`).
