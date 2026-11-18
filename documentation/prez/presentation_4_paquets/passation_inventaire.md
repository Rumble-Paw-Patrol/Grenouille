# Passation : les trouvailles du module d'inventaire, à reporter dans la présentation n° 4

Écrit le 18/11/2026 par la session « module d'exploration du corpus ». Elle a écrit `blanci/exploration/inventaire.py` (61 fonctions en lecture seule, 38 tests) et ses fiches `blanci/exploration/inventaire.md`, puis rejoué les chiffres de la présentation n° 4 et corrigé deux erreurs de la base. Ce fichier dit, diapo par diapo, ce qui change dans la présentation et quel code citer sur chaque diapo.

## À lire avant tout

- **Ne recopier aucun chiffre de ce fichier.** Léonard réannote des enregistrements écartés à tort : les chiffres bougent d'un jour à l'autre. Avant d'écrire un chiffre sur une diapo, rejouer :

```
uv run python scripts/verification/chiffres_presentation_4.py
```

- **Ce que fait le script** : il réécrit `documentation/prez/chiffres_presentation_4.md`. On y trouve, pour chaque chiffre, la valeur annoncée dans les notes orateur, la valeur recalculée, la fonction qui la calcule, le statut (« ok » ou « différent »), l'empreinte de la base et le commit du code. Un chiffre qui n'y est pas se calcule avec une fonction de `inventaire.py` (voir `inventaire.md`, une fiche par fonction), jamais à la main en conversation.
- **Les chiffres cités ici** datent du 18/11/2026 vers 18 h 40 :
  - empreinte de l'inventaire `92b0ecc4b7569d7b`, des drapeaux `c78479120a36fe7b` ;
  - ils servent à repérer ce qui doit changer, pas à être recopiés ;
  - si l'empreinte des drapeaux a changé, les comptes d'écartés ont changé aussi.
- **Règles des paquets** : elles sont dans `LISEZMOI.md` (même dossier). Léonard édite les .pptx à la main : ne jamais piloter PowerPoint (COM), ne jamais écraser un fichier ouvert, écrire un nouveau fichier sous un autre nom si besoin.
- **Notes orateur** : elles sont dans `../scripts/presentation_4_inventaire_qc_v2_script.md`, à corriger en même temps que les diapos.

## Citer le code sur les diapos (traçabilité)

Léonard veut pouvoir dire « ce graphe vient de qc.py ». Les trois fichiers sont `blanci/inputs/ingest.py` (l'inventaire : un fichier → une ligne de `recordings`), `blanci/inputs/qc.py` (les drapeaux et l'exclusion) et `blanci/exploration/inventaire.py` (les chiffres et les vérifications, en lecture seule). Proposition : une ligne discrète en pied de diapo, « Source : `qc.py` (`metadata_flags`) · `inventaire.py` (`campaign_blocks`) », même police que le numéro de diapo.

| Diapo | Code à citer |
|---|---|
| 3 | `ingest.py` (la table `recordings`) ; `inventaire.py` : `corpus_totals`, `recording_programme`, `programme_counts` |
| 4 | `inventaire.py` : `counts_by`, `near_campaign_orphans` ; `qc.py` : `metadata_flags` (règle du hors relevé) |
| 5 | `ingest.py` : `read_guano`, `parse_songmeter_name`, `start_utc`, `describe_recording` ; `scripts/verification/entete_wav.py` ; `inventaire.py` : `filename_conformity`, `mic_folders` |
| 5 bis | `inventaire.py` : `database_fingerprint` (taille, empreinte) ; figure : `presentation_4_figures.py` (`fig_base`) |
| 6 | `scripts/controle_qualite/echantillon_pistes.py` (la mesure) ; `qc.py` : `qc_indices` (RMS) ; `inventaire.py` : `gain_settings`, `track_gap_by_point` |
| 7 | `inventaire.py` : `recording_programme`, `mic_days`, `completeness_summary`, `empty_days`, `station_successions` |
| 8 | `ingest.py` : `ingest`, `same_recording` ; `scripts/controle_qualite/verifier_doublons.py` ; `inventaire.py` : `duplicate_report_summary`, `homonyms` |
| 9 | `ingest.py` : `apply_clock_corrections` ; `inventaire.py` : `clock_lag_correlation`, `clock_offsets`, `clock_summary`, `clock_corrected_counts` |
| 9 ter, 11 | `qc.py` : `metadata_flags`, `apply_audio_flags`, `is_excluded`, `keep_flags` ; `inventaire.py` : `flag_counts`, `protected_positives` |
| 10 | `qc.py` : `metadata_flags` ; `inventaire.py` : `campaign_blocks`, `duration_classes`, `recording_detections` |
| 12 à 16 | `qc.py` : `qc_indices`, `_welch`, `muffled_candidate`, `muffled_runs`, `apply_audio_flags` ; `notebooks/06_controle_qualite.ipynb` |
| 17 | `inventaire.py` : `track_flags_summary` |
| 19 | `inventaire.py` : `counts_by` (écartés par site et micro) |
| 20 | `echantillon_pistes.py` ; `inventaire.py` : `track_gap_by_point` ; la série de Kaw D : notebook 06 |
| 21 | `inventaire.py` : `ingest_errors_summary` (fichiers illisibles), `position_outliers` (GPS) |
| 24 | `inventaire.py` : `flag_counts` |

Les chiffres des diapos 13, 15, 16, 18, 19 (en partie), 20 (Kaw D), 22 et 23 viennent du notebook 06 ou des files d'écoute, pas d'une fonction testée. Citer « notebook 06 » pour ceux-là, sans les attribuer à `inventaire.py`.

## Corrections faites dans la base le 18/11 (à dire, pas à cacher)

- **Un positif était écarté** : Mataroni SMA13417, 06/01/2026 11 h 30, 104 s (`9e453c33941c80ce`), celui que la diapo 10 montre, écouté le 12/11 avec A. blanci de 0 à 67 s. Il portait `confirmed_duration_off` (confirmation en bloc) sans `keep`, ce qui contredisait la règle de la diapo 11 (« jamais écarté s'il a été entendu »). `keep` a été posé par `qc.keep_flags`, après sauvegarde des drapeaux dans `base_blanci/sauvegarde_start_utc_qc_flags_20261118.csv`. C'est une matière pour la partie « Difficultés rencontrées » : la confirmation en bloc a écrasé une écoute.
- **Un drapeau `off_campaign` était périmé** (Kaw 2023, `f670702208cbfa39`, 18 s), corrigé par `blanci flag`.
- **Effet** : 681 → 680 écartés, 173 → 172 pour la durée, 108 → 107 écartés détectés par BlanciNet, dont 19 → 18 au-dessus de 0,5. Le rejeu vérifie désormais qu'aucun positif n'est écarté (« 0 », statut « ok »).

## Diapo par diapo

### Diapo 3

- Tiennent : 96 310 enregistrements, 3 207 h, 164 points, 95 enregistreurs, 4 lieux, 74 007 (5 h → 19 h 30), 22 289 (6 h → 18 h 30).
- À dire en note : la base a 39 lignes de plus (31 du jeu `2018`, l'enregistreur `TEST-BLANCI` ; 8 du jeu `reference`, chants dirigés), que la présentation ne compte pas (`dataset_counts`).
- 14 enregistrements n'ont pas de programme lisible (PatawaOuest SMA14826, posé le 30/12/2025, arrêté le 01/01/2026, trois jours avant la relève) : 74 007 + 22 289 + 14 = 96 310.

### Diapo 4

- Notes orateur : « 25 enregistrements de 12 micros, les 30 et 31 décembre » est **faux sur les micros**. Il y a 26 enregistrements de 20 micros ces deux jours-là, dont 25 de 19 micros séparés de la pose par moins de 7 jours (donc pas hors relevé) ; le 26ᵉ (2LA03555, 0,8 s) est déjà hors relevé. Tous sont écartés pour leur durée.
- `LISEZMOI.md` (paquet 1) reprend le même « 12 micros » : à corriger aussi.

### Diapo 5

- Chiffre utile pour justifier « le micro vient du GUANO, pas du nom » : 893 fichiers de CDR ont un préfixe qui n'est pas le numéro de série (`LAGOTHRIX` 350, `SAIMIRI` 350, `ALOUATTA` 193), réglé dans l'application. Lire le nom aurait donné 893 micros faux (`filename_conformity`).
- De même pour les dossiers : en 2023, le même enregistreur est rangé sous 2 à 5 noms de dossier (Kaw SMA14079 : « SM D_SMA14079 », « SM D_SMA14079 disfonction », « SM4 D », « SMA14079 », « SMA14079 D ») (`mic_folders`).
- « Rien ne vérifie le site, alors que la position GPS du GUANO le pourrait » : c'est **faux en l'état**, voir la diapo 21.

### Diapo 5 bis

- 6,7 Go tient (6,68 Go). On peut ajouter l'empreinte de la base en note : c'est ce qui permet de dire que deux personnes ont la même base.

### Diapo 6

- Tiennent : 143 points en 6 / 18 dB, 21 en 18 / 6 dB (18 enregistreurs 2MA, 3 SMA), 164 points.
- Notes orateur à corriger :
  - « un enregistrement par micro et par jour, soit 3 607 enregistrements » est inexact : 3 607 = 3 466 « un par micro et par jour » + 141 lus aux bascules d'état (le dernier sain avant une suite étouffée, etc.) ;
  - la phrase « ils n'ont pas pu être recomptés ce jour (disque F: absent) » est périmée : `echantillon_pistes.csv` était lisible le 18/11 et les chiffres sont recomptés par `gain_settings`.
- Confirmé sur le disque : un fichier par point lu le 18/11 redonne 143 / 21 ; tout 2023 est en 6 / 18 dB ; aucun point ne change de réglage.

### Diapo 7

- Tiennent : 3 137 jours-micros, 3 128 complets, 99,7 %, 9 partiels ; 2023 : 2 217 jours, 99,8 % ; 2026 : 920 jours, 99,5 % ; Molokoï E 54 jours ; Kaw D 5 jours ; RNRT, 5 micros muets le 10/02/2026. Ces trois trous sont **tous** ceux du corpus (`empty_days`).
- Trouvaille pour le trou de Molokoï E : SMA14157 s'arrête le 21/11/2023 à midi. Son fichier de 12 h 30 existe sur le disque BLANCI_2023-24 mais pas en base : 262 144 octets (exactement 256 Kio), illisible, sans en-tête. C'est un fichier commencé et jamais refermé, la signature d'une coupure d'alimentation en pleine écriture (`series_ends`, `base_vs_disk`). Les Song Meter Mini n'écrivent aucun champ de batterie dans l'en-tête : on ne peut pas dire que c'est la batterie, seulement que l'alimentation a lâché.
- Kaw D : SMA14079 s'arrête le 28/08/2024 sur un dernier jour complet (un retrait plutôt qu'une panne), SMA15429 reprend le 03/09/2024 (`station_successions`).

### Diapo 8

- 3 673 tient, recompté depuis le rapport `ingest_duplicates_2026_Mataroni.csv` (`duplicate_report_summary`). Aucun homonyme en base, tous jeux confondus (`homonyms`).
- Toujours impossible : vérifier les 3 673 par le contenu. Les copies écartées ne sont que sur le disque de Biophonia ; ni E: ni F: ne les ont.

### Diapo 9

- Tiennent : 1 521 enregistrements, corrélation 0,57 à −1 h contre 0,34 sans décalage, maximum à −1 h, 1 521 créneaux communs.
- **Nouveau et fort** : les 9 459 en-têtes de SMA14636 ont été relus sur E: le 18/11 (`clock_offsets`, `clock_summary`). Exactement 1 521 sont écrits en « -4:00 », du 20/02/2024 16 h 30 au 11/04/2024 11 h 30. Avant et après, ils sont en « -3:00 », sauf un test de juillet 2023 écrit en UTC. Les bornes de la correction sont donc exactes à l'enregistrement près, et l'heure d'avance est mesurée sur tous les fichiers, pas seulement déduite du son. Le passage en UTC−4 coïncide avec les relèves du 20/02 et du 11/04 (réglage à la relève, probablement ; non vérifié).
- Détail possible : SMA14636 a le firmware 3.7, plus ancien que ses voisins de 2023 (3.9) ; mais SMA15429, même firmware, a une horloge juste. Pas une cause, au plus une piste (`header_outliers`).
- Spectrogrammes : les fichiers du 03/04/2024 sont sur E:. `fig_molokoi_spectros(racine=Path("E:/audio/R3_2024-04_avril/Molokoi 11042024"))` (`presentation_4_figures.py`) les refait depuis le son, car le chemin par défaut est D:. Les spectrogrammes actuels sont découpés dans l'image du 06/11.

### Diapo 9 ter

- Depuis la correction : 680 écartés (et non 681), dont 506 pour le son étouffé (399 par le calcul, 107 confirmés), 172 pour la durée (86 seule, 86 avec hors relevé), 2 pour le hors relevé seul.

### Diapo 10

- Tiennent : 238 blocs, 74 petits, 106 enregistrements hors relevé dont 20 de 2 min, 88 écartés ; 173 fichiers de durée anormale = 125 tests + 34 coupés + 14 longs ; 8 349 enregistrements de 2026 scorés par BlanciNet ; 3 coupés à 0,5 ou plus.
- Le premier des 3 coupés (Mataroni SMA13417, 104 s, 0,97) est désormais **gardé** (`keep`) : la diapo doit le dire, car c'est la correction du 18/11.

### Diapo 11

- « Un enregistrement où A. blanci a été entendu n'est jamais écarté » : c'était faux pour un enregistrement jusqu'au 18/11, c'est vrai depuis. Le rejeu le vérifie à chaque fois (`protected_positives`).
- 1 521 `clock_corrected` et 78 `clock_off` tiennent. Les 78 `clock_off` sont tous des déclenchements de test dont l'en-tête est écrit en UTC (−180 min) ou décalé (deux tests de CDR à +60 min), déjà écartés pour leur durée (`name_start_offsets`).
- La ligne affichée (`fig_ligne_drapeau`, `9e453c33941c80ce`) est justement l'enregistrement corrigé : son `qc_flags` contient maintenant `"keep": true`. Si la figure est refaite, elle montrera `keep`. C'est à choisir : refaire la figure et le dire, ou garder l'image d'avant et le dire.

### Diapo 17

- Tiennent : 399 (deux pistes, écartés), 407 (piste 0 étouffée, piste 1 bonne, gardés), 85 (piste 0 étouffée, piste 1 douteuse, écartés), 268 candidats isolés de la piste 0 (`track_flags_summary`).
- À corriger : « une quarantaine » de pistes 1 étouffées seules → 46 (38 gardés, 8 écartés).

### Diapo 19

- 452 tient, mais se compte par (site, micro) : les six points les plus touchés, tous à CDR (2MA01489 84, 2LA04443 77, 2MA10139 75, 2LA05244 73, 2LA04185 72, 2LA05017 71). Par micro seul, on trouve 455. Depuis la correction, l'écarté de Mataroni ne touche pas ces six-là.

### Diapo 20

- « Cinq points s'éloignent de plus de 6 dB » tient (`track_gap_by_point` sur l'échantillon des pistes). La perte de 30 dB de Kaw D et les 1 871 enregistrements encodés sur la piste sourde viennent du notebook 06 : non rejoués par une fonction.
- Précision méthodologique : la base n'a les indices de la piste 1 que pour 1 404 enregistrements, ceux que le contrôle du son étouffé a eu besoin de lire. L'écart entre pistes ne se lit donc que sur l'échantillon des pistes (3 607 fichiers), pas sur toute la base (`track_levels`).

### Diapo 21 (« tous les critères possibles »)

- « Le fichier illisible » : 26 fichiers, tous `LibsndfileError` (2 au premier relevé de 2023, 7 à CDR, 8 à Patawa, 9 à Mataroni), plus le fichier tronqué de SMA14157, absent des rapports d'erreurs gardés (`ingest_errors_summary`, `base_vs_disk`).
- **« Vérifier le site avec la position GPS de l'en-tête » ne marche pas en l'état** (`mic_positions`, `position_outliers`). À changer sur la diapo : de « le critère qui manque » à « un critère essayé, inutilisable en l'état ». Trois raisons :
  - les enregistreurs 2MA écrivent la longitude sans signe moins (52,29 au lieu de −52,29), ce qui les place à 11 568 km ;
  - en 2026, la position médiane des micros de CDR, Mataroni, PatawaOuest et RNRT est à 2,6 à 4,2 km du centre de Cayenne ;
  - en 2023, au premier relevé, plusieurs stations de Kaw et de Molokoï portent la même position (4,90323 ; −52,27165), et des micros « bougent » de 43 à 48 km d'un relevé à l'autre.
- Hypothèse plausible, non vérifiée : l'en-tête garde la position du lieu où l'enregistreur a été réglé. À 2 km de tolérance, 58 points sur 164 sont loin du centre des autres micros de leur site.
- Contrôles ajoutés et passés le 18/11 :
  - base contre disque : toutes les lignes de 2023 (66 779) sont sur E:, toutes celles de 2026 sur F:, et tout fichier audio de terrain lisible est en base ;
  - tailles : aucun fichier vide ni tronqué sur 2 390 tirés (un par point en 2026, un par micro et par jour en 2023) ;
  - en-têtes : tous complets sur les fichiers lus.

### Diapo 24

- « 681 enregistrements sur 96 292 » et « 95 611 encodables » sont **périmés deux fois**. Le relevé 7 de Mataroni (18 enregistrements du 19/05/2026, inventoriés depuis) fait passer à 96 310 ; la correction du 18/11 fait passer à 680 écartés et 95 630 encodables. Ces chiffres doivent aussi bouger avec la réannotation en cours : rejouer avant d'écrire.
- Le titre de la diapo dit 96 292 alors que la diapo 3 dit 96 310 : un seul total dans toute la présentation.

## Trouvailles sans diapo aujourd'hui (pour les notes, la partie « Difficultés rencontrées » ou plus tard)

- **Formats** : 116 fichiers ne sont pas en 48 kHz stéréo (24 ou 32 kHz, souvent mono) ; ce sont tous des déclenchements de test de quelques secondes, écartés pour leur durée (`audio_formats`).
- **Relevés** : le relevé de CDR commence le 19/07/2024 et celui de Mataroni le 21/07/2024 dans la base, à cause de fichiers de test restés sur les cartes, tous écartés (`survey_folders`).
- **Un chant dirigé jamais inventorié** : `audio/R2_2024-02_fevrier/Trésor 20022024/Anomaloglossus blanci_Trésor_déc2023.wav` (2 min, 05/12/2023 7 h 30), à verser dans le jeu `reference` si c'est voulu.
- **Doublons de contenu** : seules 57 lignes ont une empreinte SHA-256 (0,06 %) ; on ne peut donc rien dire des doublons de contenu sous un autre nom (`content_duplicates`).
- **Écoute** : 941 enregistrements écoutés, 186 positifs. Parts écoutées : 0,35 à 0,50 % par site en 2023, 1,1 à 2,9 % en 2026. RNRT (tenu à l'écart pour le test) : 83 écoutés, aucun positif (`annotation_coverage_by`).
- **Strates du tirage** : aucune strate horaire n'est vide à aucun site, mais elles sont très inégales (Mataroni : 994 enregistrements d'aube pour 6 025 de journée) (`stratum_effort`).
- **Notebooks 05 et 06** : ils chargent tous les jeux, y compris les 39 lignes de `2018` et `reference`. Relancés, ils affichent 96 349 enregistrements et 212 durées anormales, pas 96 310 et 173. C'est voulu dans les notebooks, mais une valeur de notebook ne doit pas être recopiée sur une diapo qui compte les jeux 2023 et 2026.

## Ce qui reste hors de portée d'une fonction

- **Diapo 8** : la vérification des 3 673 doublons par le contenu (disque de Biophonia).
- **Diapos 13, 15, 16, 18, 19 (9 pistes à la pose, 14 dans la nuit du 24 au 25/12), 20 (Kaw D), 22, 23** : notebook 06, files d'écoute, paquets d'annotation. À rejouer en relançant le notebook 06 et à dater ; s'ils doivent devenir traçables comme les autres, il faut les passer dans `inventaire.py`, avec fiche et test.
