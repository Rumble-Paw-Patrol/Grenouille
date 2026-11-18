# `blanci/exploration/inventaire.py` : l'inventaire, fonction par fonction

Fiche écrite le 18/11/2026 avec le module (commit qui ajoute ces deux fichiers). Pour savoir si elle est périmée : `git log --oneline -- blanci/exploration/inventaire.py` ; un commit plus récent que celui de cette fiche veut dire qu'il faut la relire.

Les résultats cités sont ceux de la base du 18/11/2026 : `blanci.sqlite`, schéma v5, 96 349 lignes dans `recordings`, empreinte de l'inventaire `92b0ecc4b7569d7b`, des drapeaux `f19991208922aecf` (`database_fingerprint`). Une base d'une autre empreinte peut donner d'autres chiffres : c'est attendu, et l'empreinte dit laquelle des deux bases on a.

## Pourquoi ce module

Plusieurs chiffres de la présentation n° 4 venaient de calculs faits une fois, en conversation ou dans une cellule de notebook, sans fonction ni test. Ce module rassemble tous les calculs d'inventaire : chaque chiffre sur le corpus doit sortir d'une fonction, qu'on peut relire, tester et relancer. Ce n'est pas du code de la chaîne : `blanci/inputs/ingest.py` et `qc.py` restent seuls à écrire dans la base. Ici tout se lit.

- **Lecture seule** : la base s'ouvre en `mode=ro` (`open_database`) ; aucune fonction n'écrit dans une table ni dans `qc_flags` ; l'audio et ses en-têtes sont lus, jamais écrits.
- **Entrées et sorties explicites** : une connexion, un DataFrame ou la configuration déjà chargée en entrée ; un DataFrame ou un dict en sortie ; rien n'est affiché ni écrit par les fonctions.
- **Rien n'est réécrit** : les drapeaux, le motif des noms, la lecture du GUANO, la recherche des fichiers et la règle d'exclusion viennent de `blanci.inputs.ingest`, `blanci.inputs.qc` et `blanci.core.locate`.
- **Le disque** : `read_headers`, `file_size_check` et `disk_inventory` lisent le disque ; elles prennent un échantillon explicite et lèvent `FileNotFoundError` si aucun fichier n'est trouvé.
- **Le point de départ** est `load_recordings` : une ligne par enregistrement, avec l'heure locale, le jour, le créneau et les drapeaux lus (colonne `fl`).

Rejouer tous les chiffres de la présentation n° 4 : `uv run python scripts/verification/chiffres_presentation_4.py`, qui écrit [`documentation/prez/chiffres_presentation_4.md`](../../documentation/prez/chiffres_presentation_4.md) (chiffre annoncé, chiffre recalculé, fonction, statut, empreinte de la base, commit). Tests : [`tests/exploration/test_inventaire.py`](../../tests/exploration/test_inventaire.py), un jeu fictif de trois micros sur quatre jours, avec une horloge décalée et un doublon, où chaque réponse se compte à la main (38 tests).

## Sommaire

0. [Socle et provenance](#0-socle-et-provenance) : `open_database`, `database_fingerprint`, `load_recordings`, `dataset_counts`
1. [Structure du corpus](#1-structure-du-corpus) : `corpus_totals`, `counts_by`, `audio_formats`, `survey_folders`, `station_successions`
2. [Programme et complétude](#2-programme-et-complétude) : `recording_programme`, `programme_of`, `programme_counts`, `off_programme_recordings`, `mic_days`, `completeness_summary`, `empty_days`
3. [Noms et arborescence](#3-noms-et-arborescence) : `filename_conformity`, `homonyms`, `mic_folders`
4. [En-têtes GUANO](#4-en-têtes-guano) : `parse_guano_fields`, `sample_recordings`, `read_headers`, `header_field_summary`, `header_outliers`, `gain_settings`
5. [Horloge](#5-horloge) : `name_start_offsets`, `clock_offsets`, `clock_summary`, `clock_corrected_counts`, `clock_lag_correlation`
6. [Position](#6-position) : `mic_positions`, `position_outliers`
7. [Température et fin de série](#7-température-et-fin-de-série) : `temperature_series`, `series_ends`
8. [Durées et tailles](#8-durées-et-tailles) : `duration_classes`, `expected_size`, `file_size_check`
9. [Doublons et rapports d'inventaire](#9-doublons-et-rapports-dinventaire) : `duplicate_report_summary`, `duplicate_verification_summary`, `content_duplicates`, `ingest_errors_summary`
10. [Pistes](#10-pistes) : `track_levels`, `track_gap_by_point`, `track_flags_summary`
11. [Drapeaux](#11-drapeaux) : `exclusion_cause`, `flag_counts`, `campaign_blocks`, `near_campaign_orphans`, `stale_flags`
12. [Base et disque](#12-base-et-disque) : `disk_inventory`, `base_vs_disk`
13. [Annotations et scores](#13-annotations-et-scores) : `listened_recordings`, `annotation_coverage_by`, `score_coverage`, `recording_detections`, `excluded_with_detections`, `protected_positives`
14. [Comparaisons](#14-comparaisons) : `compare_groups`, `stratum_effort`
15. [Rejeu de la présentation n° 4](#15-rejeu-de-la-présentation-n-4) : `presentation_4_numbers`
16. [Chiffres de la présentation n° 4 : ce qui se rejoue, ce qui ne se rejoue pas](#16-chiffres-de-la-présentation-n-4--ce-qui-se-rejoue-ce-qui-ne-se-rejoue-pas)
17. [Écarts relevés](#17-écarts-relevés)

## 0. Socle et provenance

#### `open_database(cfg)` — [inventaire.py:67](inventaire.py#L67)

- Question : sur quelle base calcule-t-on, sans risque d'y écrire ?
- Entrée : la configuration chargée (`load_config`). Sortie : une connexion SQLite en `mode=ro`.
- Règle : `paths.db` de la configuration, ouvert par `explore.open_readonly` (rien de réécrit).
- Limite : `FileNotFoundError` si la base n'existe pas.
- Prez n° 4 : toutes les diapos chiffrées. Résultat : toute écriture lève `sqlite3.OperationalError` (testé). Ce qu'on peut dire : aucun calcul de ce module n'a modifié la base. Ce qu'on ne peut pas dire : que la base est la même que celle d'hier (voir `database_fingerprint`).

#### `database_fingerprint(con)` — [inventaire.py:80](inventaire.py#L80)

- Question : la base sur laquelle je rejoue un chiffre est-elle celle d'où il vient ?
- Entrée : une connexion. Sortie : dict `user_version`, `size_bytes`, `rows` (lignes par table), `inventory_sha256` (empreinte de `recordings` sans `qc_flags`), `flags_sha256` (empreinte des seuls `qc_flags`), `last_span`, `last_label`.
- Règle : SHA-256 des lignes triées par `recording_id`. Deux empreintes séparées parce que les drapeaux bougent plus souvent que l'inventaire.
- Limite : quelques secondes (elle compte la table `scores`, 27 millions de lignes) ; elle dit qu'une base a changé, pas ce qui a changé.
- Prez n° 4 : diapo 5 bis (taille de la base). Résultat : 6,68 Go, annoncé 6,7 Go. Ce qu'on peut dire : deux personnes qui trouvent `92b0ecc4b7569d7b` ont le même inventaire au bit près. Ce qu'on ne peut pas dire : quel chiffre a changé entre deux empreintes différentes (il faut rejouer).

#### `load_recordings(con, cfg, datasets=("2023", "2026"))` — [inventaire.py:121](inventaire.py#L121)

- Question : quels enregistrements a-t-on, à quelle heure locale, avec quels drapeaux ?
- Entrée : connexion, configuration, jeux gardés (`None` = tous). Sortie : la table `recordings` plus `fl` (drapeaux lus par `qc.parse_flags`), `t`, `local`, `day`, `slot`, `excluded` (`qc.is_excluded`), `campaign`, `normal`, `survey`, `family`.
- Règle : heure locale à `recorder.filename_utc_offset_h` (−3) ; durée normale à `qc.duration_tolerance_s` (1 s) de `qc.expected_duration_s` (120 s). Reprend `charger()` de `documentation/prez/scripts/presentation_4_figures.py` et les cellules d'ouverture des notebooks 05 et 06.
- Limite : décalage horaire fixe ; les drapeaux sont ceux rangés (voir `stale_flags`). La colonne s'appelle `fl` et non `flags`, parce que `rec.flags` est un attribut de pandas.
- Prez n° 4 : toutes les diapos de l'inventaire. Résultat : 96 310 lignes pour les jeux 2023 et 2026.

#### `dataset_counts(con)` — [inventaire.py:171](inventaire.py#L171)

- Question : qu'y a-t-il dans la base hors des jeux du projet ?
- Entrée : connexion. Sortie : une ligne par jeu (`recordings`, `hours`, `recorders`, `in_project`).
- Règle : `in_project` = jeu 2023 ou 2026. Limite : aucune.
- Prez n° 4 : aucune (c'est ce que la diapo 3 ne dit pas). Résultat : 2018 : 31 lignes (un ancien enregistreur, `TEST-BLANCI`, 1,6 h) ; 2023 : 66 779 ; 2026 : 29 531 ; `reference` : 8 (chants dirigés). Ce qu'on peut dire : la base a 96 349 lignes, dont 96 310 comptées par la présentation. Ce qu'on ne peut pas dire : rien de plus, le filtre est un choix.

## 1. Structure du corpus

#### `corpus_totals(rec)` — [inventaire.py:193](inventaire.py#L193)

- Question : combien d'enregistrements, d'heures, d'enregistreurs, de points ?
- Entrée : `load_recordings`. Sortie : dict `recordings`, `hours`, `recorders`, `points`, `sites`, `excluded`, et les mêmes par jeu.
- Règle : point = (jeu, site, micro) ; un enregistreur posé à deux sites compte une fois en enregistreurs, deux fois en points.
- Limite : les heures somment les durées réelles, écartés compris.
- Prez n° 4 : diapo 3. Résultat : 96 310 enregistrements, 3 206,8 h, 95 enregistreurs, 164 points, 8 sites en base (4 lieux une fois les relevés empilés sur leur site), 681 écartés. Ce qu'on peut dire : les quatre chiffres de la diapo 3 tiennent. Ce qu'on ne peut pas dire : que 3 207 h sont 3 207 h utilisables (681 écartés, voir `flag_counts`).

#### `counts_by(rec, keys)` — [inventaire.py:220](inventaire.py#L220)

- Question : combien par jeu, site, micro, jour, heure, relevé ?
- Entrée : `load_recordings`, colonnes de regroupement. Sortie : `recordings`, `hours`, `recorders`, `first_day`, `last_day`, `days`, `excluded`.
- Règle : reprend le tableau du § 2 du notebook 05 et `fig_sites`. Limite : un groupe vide n'apparaît pas (voir `empty_days`).
- Prez n° 4 : diapo 3 (heures par site), diapo 10 (Mataroni les 30-31/12), diapo 19 (écartés par point). Résultat : Mataroni les 30 et 31/12/2025 : 26 enregistrements de 20 micros ; les six points les plus touchés par les exclusions, tous à CDR : 452 écartés (2MA01489 84, 2LA04443 77, 2MA10139 75, 2LA05244 73, 2LA04185 72, 2LA05017 71). Ce qu'on peut dire : le 452 de la diapo 19 se compte par (site, micro). Ce qu'on ne peut pas dire : par micro seul, on trouve 455 (2MA01489 et 2LA04185 ont aussi des écartés ailleurs).

#### `audio_formats(rec)` — [inventaire.py:242](inventaire.py#L242)

- Question : tout est-il en 48 kHz stéréo ?
- Entrée : `load_recordings`. Sortie : une ligne par (jeu, fréquence, pistes) avec effectifs, médiane de durée, nombre de durée normale.
- Règle : valeurs lues par libsndfile à l'inventaire. Limite : ne dit rien du codage (bits).
- Prez n° 4 : aucune. Résultat : 96 194 enregistrements en 48 kHz stéréo ; les 116 autres (24 ou 32 kHz, souvent mono) ont une durée médiane de 4,6 à 20,9 s et aucun ne dure 2 min. Ce qu'on peut dire : tout le programme est en 48 kHz stéréo ; les autres formats sont des déclenchements de test, tous écartés par `duration_off`. Ce qu'on ne peut pas dire : pourquoi ces tests sont à 24 kHz (réglage par défaut de l'application, probablement, non vérifié).

#### `survey_folders(rec)` — [inventaire.py:258](inventaire.py#L258)

- Question : combien d'enregistrements vient de chaque relevé, sur quelles dates ?
- Entrée : `load_recordings`. Sortie : une ligne par (jeu, relevé) : effectifs, sites, micros, premier et dernier jour, écartés.
- Règle : relevé = deux premiers dossiers du chemin. L'ordre compte : à l'inventaire, le premier fichier d'un nom l'emporte.
- Limite : un fichier déplacé prend le relevé de son nouveau chemin.
- Prez n° 4 : aucune. Résultat : 12 relevés. Le relevé de CDR commence le 19/07/2024 et celui de Mataroni le 21/07/2024, alors que les poses sont en décembre 2025 et janvier 2026 : des fichiers de test restés sur les cartes, tous hors relevé ou de durée anormale. Le relevé 7 de Mataroni (`audio/R7_2026-05_Mataroni`, 18 enregistrements du 19/05/2026) est inventorié. Ce qu'on peut dire : l'écart entre 96 292 (notebooks 05 et 06, diapo 24) et 96 310 (diapo 3) correspond à ces 18. Ce qu'on ne peut pas dire : la date d'inventaire de chaque relevé (la base ne la garde pas).

#### `station_successions(rec, stations)` — [inventaire.py:280](inventaire.py#L280)

- Question : quand chaque micro d'une station 2023 a-t-il pris le relais, avec quel trou ?
- Entrée : `load_recordings`, `plan.stations_2023`. Sortie : une ligne par (station, micro) : premier et dernier jour, effectif, `gap_to_next_days`.
- Règle : campagne seulement. Limite : la liste des micros d'une station vient de la configuration.
- Prez n° 4 : diapo 7 (les trous). Résultat : Kaw D : SMA14079 jusqu'au 28/08/2024, SMA15429 à partir du 03/09/2024, 5 jours sans micro ; Molokoï E : SMA14157 du 14 au 21/11/2023, SMA14636 à partir du 15/01/2024, 54 jours sans micro. Ce qu'on peut dire : les deux grands trous de 2023 sont des changements d'enregistreur. Ce qu'on ne peut pas dire : pourquoi SMA14157 s'est arrêté au bout d'une semaine (voir `series_ends`).

## 2. Programme et complétude

#### `recording_programme(rec, min_share=0.5)` — [inventaire.py:325](inventaire.py#L325)

- Question : à quelles heures chaque micro enregistre-t-il ?
- Entrée : `load_recordings`, `min_share`. Sortie : une ligne par point : créneaux, premier, dernier, nombre, pas médian, étiquette, jours intérieurs.
- Règle : campagne et durée normale, jours intérieurs (ni pose ni relève), créneau présent au moins un jour sur deux. Reprend `programmes()` de `presentation_4_figures.py` (§ 4 du notebook 05). Seuil `min_share = 0,5` : choix de lecture, absent de `config/default.yaml`.
- Limite : un micro qui n'a jamais tenu son programme a pour programme ce qu'il a fait ; un point de moins de trois jours n'a pas de programme.
- Prez n° 4 : diapo 3 (droite), diapo 7. Résultat : 163 points ont un programme, tous à 30 min de pas : 117 de 6 h à 18 h 30, 46 de 5 h à 19 h 30 ; un point sans programme lu, PatawaOuest SMA14826 (14 enregistrements, posé le 30/12/2025, arrêté le 01/01/2026). Ce qu'on peut dire : le programme se lit dans les données, sans exception de cadence. Ce qu'on ne peut pas dire : le programme réglé (il n'est écrit nulle part dans les fichiers).

#### `programme_of(rec, programmes)` — [inventaire.py:363](inventaire.py#L363)

- Question : sous quel programme chaque enregistrement a-t-il été fait ?
- Entrée : `load_recordings`, `recording_programme`. Sortie : une série alignée sur `rec`, « ? » sans programme.
- Règle : jointure sur le point. Limite : celle de `recording_programme`. Prez n° 4 : diapo 3, par `programme_counts`.

#### `programme_counts(rec, programmes)` — [inventaire.py:375](inventaire.py#L375)

- Question : combien d'enregistrements viennent de micros programmés 5 h → 19 h 30, et combien de 6 h → 18 h 30 ?
- Entrée : `load_recordings`, `recording_programme`. Sortie : une ligne par (programme, jeu, site).
- Règle : reprend `fig_programmes`. Limite : un enregistrement hors programme compte sous le programme de son micro.
- Prez n° 4 : diapo 3. Résultat : 74 007 (5 h → 19 h 30) et 22 289 (6 h → 18 h 30), comme annoncé, plus 14 enregistrements d'un point sans programme. Ce qu'on peut dire : tout le suivi 2023-2024 est en 5 h → 19 h 30, et en 2026 les deux programmes coexistent (22 289 contre 7 228). Ce qu'on ne peut pas dire : que le programme suit la pose plutôt que le modèle d'enregistreur ; c'est une lecture des notes, la fonction ne la teste pas.

#### `off_programme_recordings(rec, programmes)` — [inventaire.py:389](inventaire.py#L389)

- Question : quels enregistrements ne tombent pas sur un créneau du programme de leur micro ?
- Entrée : `load_recordings`, `recording_programme`. Sortie : les lignes concernées, avec la raison.
- Règle : créneau « HH:MM » local absent du programme du point. Limite : une horloge décalée d'une heure exacte tombe sur un autre créneau et ne se voit pas ici (voir `name_start_offsets`).
- Prez n° 4 : aucune. Résultat : 129 enregistrements hors créneau et 14 d'un point sans programme ; parmi les gardés de durée normale dans la campagne, il en reste 11, tous de PatawaOuest SMA14826. Ce qu'on peut dire : à part ce micro, tout ce qui est hors programme est déjà écarté (tests). Ce qu'on ne peut pas dire : ce que sont les 11 de SMA14826 sans les écouter.

#### `mic_days(rec, programmes)` — [inventaire.py:411](inventaire.py#L411)

- Question : chaque jour, chaque micro a-t-il fait tout son programme ?
- Entrée : `load_recordings`, `recording_programme`. Sortie : une ligne par jour-micro : reçus, créneaux, bord, attendus, complétude, créneaux manquants.
- Règle : reprend `jours_micros()` de `presentation_4_figures.py` (§ 4 du notebook 05). Limite : un jour sans rien n'a pas de ligne.
- Prez n° 4 : diapo 7, par `completeness_summary`.

#### `completeness_summary(days)` — [inventaire.py:439](inventaire.py#L439)

- Question : sur quelle part des jours entre pose et relève le programme est-il tenu ?
- Entrée : `mic_days`. Sortie : jours-micros intérieurs, complets, part, partiels, et par jeu.
- Règle : jours intérieurs seulement. Limite : un jour vide n'entre pas dans la part ; la complétude est relative au programme lu.
- Prez n° 4 : diapo 7. Résultat : 3 128 jours complets sur 3 137 (99,71 %), 9 partiels ; 2023 : 2 217 jours, 99,82 % ; 2026 : 920 jours, 99,46 %. Ce qu'on peut dire : tous les chiffres de la diapo 7 tiennent. Ce qu'on ne peut pas dire : que le programme a été tenu sur les jours vides, qui ne sont pas comptés (voir `empty_days`).

#### `empty_days(rec, level="mic", stations=None)` — [inventaire.py:474](inventaire.py#L474)

- Question : où le suivi s'est-il interrompu entre la pose et la relève ?
- Entrée : `load_recordings`, niveau (« mic » ou « station »), `plan.stations_2023`. Sortie : une ligne par point qui a un jour vide : nombre, plus long trou, dates (`empty_dates`).
- Règle : campagne et durée normale, comme `fig_completude`. Limite : au niveau du micro, le trou entre deux micros d'une station ne se voit pas ; au niveau de la station, si.
- Prez n° 4 : diapo 7. Résultat : par station, Molokoï E 54 jours (21/11/2023 → 14/01/2024) et Kaw D 5 jours ; par micro, 5 points, tous à RNRT le 10/02/2026 (2LA03560, 2MA05015, 2MA07933, SMA13417, SMA14826), un jour chacun. Ce qu'on peut dire : les trois trous de la diapo 7 sont tous ceux du corpus. Ce qu'on ne peut pas dire : ce qui s'est passé à RNRT le 10/02 (cinq micros muets le même jour, puis repartis).

## 3. Noms et arborescence

#### `filename_conformity(rec)` — [inventaire.py:530](inventaire.py#L530)

- Question : quels fichiers ont un nom hors du motif `<série>_<AAAAMMJJ>_<HHMMSS>`, un préfixe qui n'est pas leur micro, une extension inattendue ?
- Entrée : `load_recordings`. Sortie : une ligne par enregistrement : motif, préfixe, préfixe = micro, suffixe, extension.
- Règle : motif `ingest.SONGMETER_NAME`. Limite : un nom conforme peut porter une heure fausse.
- Prez n° 4 : diapo 5 (le micro vient du GUANO, pas du nom). Résultat : les 96 310 noms suivent le motif, tous en `.wav`, aucun suffixe ; 893 ont un préfixe qui n'est pas le numéro de série : `LAGOTHRIX` (350), `SAIMIRI` (350) et `ALOUATTA` (193), à CDR. Ce qu'on peut dire : le choix de lire le micro dans le GUANO plutôt que dans le nom évite 893 erreurs de micro. Ce qu'on ne peut pas dire : qui a donné ces préfixes (des noms de singes, réglés dans l'application).

#### `homonyms(rec)` — [inventaire.py:556](inventaire.py#L556)

- Question : un même nom de fichier est-il rangé deux fois en base ?
- Entrée : `load_recordings` (de préférence tous les jeux). Sortie : une ligne par nom répété.
- Règle : comparaison en minuscules, sans extension, comme l'index de `ingest`. Limite : l'inventaire écarte déjà les doublons de nom à l'entrée.
- Prez n° 4 : diapo 8. Résultat : aucun homonyme, tous jeux confondus. Ce qu'on peut dire : la règle des doublons à l'entrée a tenu. Ce qu'on ne peut pas dire : qu'il n'y a pas de doublon sous un autre nom (voir `content_duplicates`).

#### `mic_folders(rec)` — [inventaire.py:575](inventaire.py#L575)

- Question : le même enregistreur est-il rangé sous plusieurs noms de dossier ?
- Entrée : `load_recordings`. Sortie : une ligne par point : dossiers, nombre, numéro de série dans tous les dossiers.
- Règle : dossier du micro = parent du fichier, ou grand-parent si le parent est `Data`. Limite : un dossier sans numéro n'est pas une erreur.
- Prez n° 4 : diapo 5 (« 2LA04185 à CDR, 2LA04185_GI19 à Mataroni »). Résultat : 7 points ont plusieurs noms de dossier, tous en 2023 (Kaw SMA14079 en a 5 : « SM D_SMA14079 », « SM D_SMA14079 disfonction », « SM4 D », « SMA14079 », « SMA14079 D ») ; 10 points ont au moins un dossier sans le numéro de série (7 en 2023, 3 en 2026). Ce qu'on peut dire : le nom du dossier ne suffit pas à identifier le micro. Ce qu'on ne peut pas dire : ce que veut dire « disfonction » sur le terrain (à demander).

## 4. En-têtes GUANO

#### `parse_guano_fields(guano)` — [inventaire.py:605](inventaire.py#L605)

- Question : que dit l'en-tête d'un fichier ?
- Entrée : le dict de `ingest.read_guano`. Sortie : marque, modèle, série, firmware, préfixe, horodatage et fuseau, durée, fréquence, gain et fréquence par piste, position, `has_fix`, température, nombre de champs.
- Règle : « Loc Position = 0.00000 0.00000 » veut dire pas de position. Les deux fichiers lus le 18/11/2026 (CDR et Molokoï) n'ont aucun champ de batterie.
- Limite : un champ absent vaut `None` ; un FLAC n'a pas de GUANO. Prez n° 4 : diapo 5 (les blocs du WAV), par les fonctions suivantes.

#### `sample_recordings(rec, per, hour=10.0, n=1, channels=None)` — [inventaire.py:661](inventaire.py#L661)

- Question : quels fichiers lire pour connaître les en-têtes sans lire les 96 000 ?
- Entrée : `load_recordings`, groupes, heure visée, nombre par groupe, pistes. Sortie : les lignes tirées.
- Règle : parmi les durées normales, les plus proches de `hour` ; même règle que le motif « jour » de `scripts/controle_qualite/echantillon_pistes.py`. Déterministe.
- Limite : un réglage changé en cours de journée ne se voit que le lendemain.
- Prez n° 4 : diapo 6. Résultat : l'échantillon des pistes a 3 607 lignes, dont 3 466 « un par micro et par jour » et 141 « bascules » (le dernier sain avant une suite étouffée, etc.). Ce qu'on peut dire : 3 607 est le nombre de fichiers lus. Ce qu'on ne peut pas dire : « un enregistrement par micro et par jour, soit 3 607 » (notes de la diapo 6) ; c'est 3 466 plus 141.

#### `read_headers(sample, raw, renamed=None, extra_roots=())` — [inventaire.py:689](inventaire.py#L689)

- Question : que disent les en-têtes des fichiers tirés ?
- Entrée : un échantillon, `paths.raw`, `audio.renamed`. Sortie : l'échantillon plus `found` et les champs de `parse_guano_fields`.
- Règle : le fichier est cherché par `core.locate.locate` ; seul le bloc `guan` est lu.
- Limite : **exige le disque**. Si aucun fichier n'est trouvé, `FileNotFoundError` ; `locate` cherche aussi dans les racines déjà trouvées pendant la session.
- Prez n° 4 : diapo 6. Résultat : un fichier par point (164) lu sur F: le 18/11/2026, tous trouvés, en 5 s. Ce qu'on peut dire : sur ce tirage, tous les champs lus par `parse_guano_fields` sont présents, en 48 kHz, fuseau « -3:00 », 2 min, et les gains redonnent 143 / 21. Ce qu'on ne peut pas dire : la même chose pour les fichiers que le tirage n'a pas lus.

#### `header_field_summary(headers, fields=None)` — [inventaire.py:720](inventaire.py#L720)

- Question : quels champs d'en-tête manquent, lesquels varient ?
- Entrée : `read_headers` ou l'échantillon des pistes. Sortie : une ligne par champ : présents, part, valeurs distinctes, valeurs les plus fréquentes.
- Règle : sur les fichiers trouvés. Limite : ne vaut que pour l'échantillon.
- Prez n° 4 : aucune. Résultat : sur les 3 607 fichiers de l'échantillon des pistes, un seul modèle (Song Meter Mini) et 7 firmwares (3.9 : 1 854 fichiers, 4.5 : 1 049, 3.7 : 445, 4.6 : 132, 4.3 : 61…). Ce qu'on peut dire : tout le parc est du même modèle, avec des firmwares différents. Ce qu'on ne peut pas dire : qu'un firmware change la réponse des micros (rien ne le mesure ici).

#### `header_outliers(headers, field, by=("dataset", "model"))` — [inventaire.py:748](inventaire.py#L748)

- Question : quel micro a un champ d'en-tête différent des autres de son groupe ?
- Entrée : `read_headers` ou l'échantillon des pistes, le champ, le groupe. Sortie : une ligne par (point, valeur) hors du mode.
- Règle : mode du groupe, sans seuil. Limite : un groupe partagé à parts égales désigne un mode arbitraire.
- Prez n° 4 : aucune. Résultat (firmware, échantillon des pistes) : en 2023, deux micros sortent du lot avec le firmware 3.7 contre 3.9 : SMA15429 et **SMA14636, le micro à l'horloge fausse** ; en 2026, le mode est 4.5 et 33 couples (point, firmware) en diffèrent. Ce qu'on peut dire : le micro dont l'en-tête était en UTC−4 a aussi un firmware plus ancien que ses voisins. Ce qu'on ne peut pas dire : que le firmware est la cause (SMA15429, même firmware, a une horloge juste).

#### `gain_settings(headers)` — [inventaire.py:770](inventaire.py#L770)

- Question : quel micro a la piste 0 au gain faible, lequel au gain fort, lequel a changé de réglage ?
- Entrée : `read_headers` ou l'échantillon des pistes. Sortie : une ligne par point : premier réglage, suite des réglages, modèle, firmware, fichiers.
- Règle : reprend le § 3 du notebook 05 et `fig_gains` ; réglage du point = le premier dans le temps.
- Limite : un changement de réglage n'est daté qu'au jour près.
- Prez n° 4 : diapo 6. Résultat : 143 points à 6 / 18 dB et 21 à 18 / 6 dB (18 enregistreurs 2MA et 3 SMA), sur l'échantillon des pistes comme sur le tirage d'un fichier par point lu le 18/11 ; aucun point ne change de réglage. Ce qu'on peut dire : les chiffres de la diapo 6 sont recomptés (le fichier `echantillon_pistes.csv` était lisible sur F: le 18/11, contrairement à ce que disent les notes). Ce qu'on ne peut pas dire : quel réglage avaient les fichiers non lus.

## 5. Horloge

#### `name_start_offsets(rec)` — [inventaire.py:806](inventaire.py#L806)

- Question : pour quels enregistrements l'heure rangée en base ne tombe-t-elle pas sur l'heure du nom ? (sans lire le disque)
- Entrée : `load_recordings`. Sortie : heure du nom, heure rangée, écart en minutes, `clock_corrected`.
- Règle : la comparaison de `clock_off` (`qc.metadata_flags`), signée. Les enregistrements corrigés ont un écart nul par construction.
- Limite : une horloge qui écrit la même fausse heure dans le nom et l'en-tête ne se voit pas.
- Prez n° 4 : diapos 9 et 11 (`clock_off`). Résultat : 78 enregistrements à plus de 5 min (les 78 `clock_off`) : 72 à −180 min (en-tête en UTC, déclenchements de test de CDR et Mataroni), 4 en 2023 (−180 à −300 min), 2 à +60 min (deux tests de CDR). Ce qu'on peut dire : hors Molokoï, tous les écarts d'horloge en base sont des tests, déjà écartés pour leur durée. Ce qu'on ne peut pas dire : l'écart des 1 521 enregistrements corrigés, effacé de la base par la correction (voir `clock_offsets`).

#### `clock_offsets(headers, utc_offset_h)` — [inventaire.py:828](inventaire.py#L828)

- Question : de combien l'en-tête d'un fichier s'écarte-t-il de son nom, et sous quel fuseau ?
- Entrée : `read_headers` (ou l'échantillon des pistes, colonne `guano_timestamp` lue par `parse_guano_timestamp`), `recorder.filename_utc_offset_h`. Sortie : heure du nom, heure de l'en-tête, écart, fuseau.
- Règle : nom en heure locale ; en-tête avec son fuseau. Limite : ne voit que les fichiers lus.
- Prez n° 4 : diapo 9. Résultat : dans l'échantillon des pistes, 3 556 en-têtes en « -3:00 » et 51 en « -4:00 », tous de Molokoï SMA14636, tous à +60 min ; le tirage d'un fichier par point du 18/11 n'en a aucun. Ce qu'on peut dire : l'heure d'avance de SMA14636 se relit dans les en-têtes, et c'est le seul micro concerné dans l'échantillon. Ce qu'on ne peut pas dire : la date exacte du début et de la fin (l'échantillon a un fichier par jour ; les bornes de `recorder.clock_corrections` viennent du nom des fichiers).

#### `clock_summary(offsets, tolerance_min)` — [inventaire.py:853](inventaire.py#L853)

- Question : un micro décalé l'est-il d'un bloc, ou en dérive ?
- Entrée : `clock_offsets` ou `name_start_offsets`, `qc.clock_tolerance_min` (5 min). Sortie : une ligne par suite de fichiers au même décalage, avec la pente de dérive du point.
- Règle : nouvelle suite quand le décalage arrondi ou le fuseau change ; pente de Theil-Sen, insensible à quelques fichiers de test.
- Limite : une pente sur une série coupée par un saut d'une heure ne veut rien dire ; lire les suites.
- Prez n° 4 : diapo 9. Résultat : sur la base, les 78 suites hors tolérance sont les fichiers de `name_start_offsets`. Ce qu'on peut dire : aucun micro ne montre de dérive lente visible en base. Ce qu'on ne peut pas dire : qu'il n'y a pas de dérive de moins de 5 min (la base n'a qu'une heure au fichier près).

#### `clock_corrected_counts(rec)` — [inventaire.py:886](inventaire.py#L886)

- Question : combien d'enregistrements la correction d'horloge touche-t-elle ?
- Entrée : `load_recordings`. Sortie : une ligne par point : effectif, premier, dernier.
- Règle : drapeau `clock_corrected`, posé d'après `recorder.clock_corrections`. Limite : à jour après `blanci flag`.
- Prez n° 4 : diapos 9 et 11. Résultat : 1 521, Molokoï SMA14636, du 20/02/2024 16 h 30 au 11/04/2024 11 h 30. Ce qu'on peut dire : le 1 521 est recompté.

#### `clock_lag_correlation(rec, target, reference, site=None, lags_h, header_shift_h=1.0)` — [inventaire.py:900](inventaire.py#L900)

- Question : à quel décalage le niveau sonore du micro douteux se superpose-t-il à celui d'un voisin sain ?
- Entrée : `load_recordings` (RMS de la piste 0 rangé par `blanci qc`), micro douteux, micro de référence, site, décalages, avance de l'en-tête sur le nom (+1 h). Sortie : décalage, corrélation de Pearson, créneaux communs.
- Règle : reprend `molokoi_niveaux()` (`presentation_4_figures.py`) et le § 6 du notebook 05 ; résidu = RMS − médiane du même micro à la même heure (cycle jour-nuit retiré).
- Limite : une seule paire et une seule période ; `header_shift_h` est fourni, pas mesuré (l'heure d'en-tête n'est plus en base) ; Pearson pèse surtout les orages.
- Prez n° 4 : diapo 9. Résultat : 0,566 à −1 h, 0,342 sans décalage, maximum à −1 h, 1 521 créneaux communs à −1 h. Ce qu'on peut dire : les quatre chiffres de la diapo 9 tiennent. Ce qu'on ne peut pas dire : qu'une autre paire de micros donnerait la même conclusion (une seule a été mesurée).

## 6. Position

#### `mic_positions(headers)` — [inventaire.py:957](inventaire.py#L957)

- Question : où est chaque micro d'après son en-tête, a-t-il bougé ?
- Entrée : `read_headers` ou l'échantillon des pistes. Sortie : une ligne par point : fichiers, sans position, latitude et longitude médianes, plus grand écart à la médiane.
- Règle : distance de haversine. Limite : la précision du GPS n'est pas connue.
- Prez n° 4 : diapo 5 (« le site vient du dossier, alors que la position GPS du GUANO le pourrait »), diapo 21 (« le critère qui manque »). Résultat (échantillon des pistes) : tous les points ont une position, mais 7 points de 2023 « bougent » de 0,7 à 48 km au fil de l'année (5 de 43 à 48 km). Au premier relevé de 2023, plusieurs stations de Kaw et de Molokoï portent la même position (4,90323 ; −52,27165). En 2026, la position médiane des micros de CDR, de Mataroni, de PatawaOuest et de RNRT est à 2,6 à 4,2 km du centre de Cayenne (4,9372 ; −52,3260). Ce qu'on peut dire : la position de l'en-tête n'est pas celle de la pose pour une grande partie des fichiers. Ce qu'on ne peut pas dire : qu'elle est celle du lieu de réglage de l'enregistreur ; l'hypothèse est plausible mais non vérifiée.

#### `position_outliers(positions, max_km)` — [inventaire.py:982](inventaire.py#L982)

- Question : un micro rangé sous un site est-il vraiment à ce site ?
- Entrée : `mic_positions`, `max_km`. Sortie : une ligne par point : centre des autres micros du site, distance, `outlier`.
- Règle : la configuration n'a pas de coordonnées de site, donc le centre est celui des autres micros. `max_km` est absent de toute configuration : il est choisi à l'appel, et il faut le dire.
- Limite : si la moitié des micros d'un site est mal placée, le centre est faux.
- Prez n° 4 : diapo 21. Résultat (à 2 km) : 58 points sur 164 sont loin du centre de leur site. Les enregistreurs 2MA écrivent la longitude **sans signe moins** (52,29 au lieu de −52,29), ce qui les met à 11 568 km. Ce qu'on peut dire : en l'état, le GPS de l'en-tête ne peut pas servir à vérifier le site. Ce qu'on ne peut pas dire : quels points sont vraiment mal rangés ; il faudrait des coordonnées de pose du terrain.

## 7. Température et fin de série

#### `temperature_series(headers, by)` — [inventaire.py:1013](inventaire.py#L1013)

- Question : quelles températures l'enregistreur a-t-il vues ?
- Entrée : `read_headers` ou l'échantillon des pistes. Sortie : une ligne par groupe : fichiers, minimum, médiane, maximum.
- Règle : `Temperature Int` du GUANO. Limite : température du boîtier (soleil compris), un fichier par jour vers 10 h.
- Prez n° 4 : aucune. Résultat (échantillon des pistes, par site) : médianes de 24,75 à 26,25 °C, extrêmes de 22,25 (CDR) à 34,00 °C (Kaw). Ce qu'on peut dire : aucun boîtier n'a vu de température extrême à 10 h. Ce qu'on ne peut pas dire : la température de l'après-midi, ni un lien avec les arrêts.

#### `series_ends(rec, days, cfg_qc)` — [inventaire.py:1032](inventaire.py#L1032)

- Question : quels micros se sont arrêtés avant les autres de leur site ?
- Entrée : `load_recordings`, `mic_days`, `qc`. Sortie : une ligne par point : dernier enregistrement, complétude du dernier jour, durée du dernier fichier, `cut_at_end`, dernier jour du site, `stops_early`.
- Règle : sans champ de batterie dans le GUANO, l'arrêt en fin de batterie se lit indirectement (fin anticipée de plus d'un jour, dernier jour partiel, dernier fichier coupé).
- Limite : un indice, pas une preuve ; une carte pleine ou un retrait donnent la même signature.
- Prez n° 4 : diapo 10 (« coupés en fin de batterie »). Résultat : 3 points s'arrêtent avant leur site : Molokoï SMA14157 (21/11/2023, à midi, dernier jour à moitié), Kaw SMA14079 (28/08/2024, dernier jour complet : un retrait plutôt qu'une panne) et PatawaOuest SMA14826 (01/01/2026) ; 20 points ont un dernier fichier coupé. Ce qu'on peut dire : SMA14157 a la signature d'une panne (arrêt en milieu de journée, une semaine après la pose). Ce qu'on ne peut pas dire : que c'est la batterie.

## 8. Durées et tailles

#### `duration_classes(rec, cfg_qc, test_max_s=30.0)` — [inventaire.py:1070](inventaire.py#L1070)

- Question : parmi les fichiers qui ne durent pas 2 min, lesquels sont des tests, lesquels du vrai terrain coupé ?
- Entrée : `load_recordings`, `qc`, `test_max_s`. Sortie : les fichiers de durée anormale, avec leur classe (test, coupé, long) et `on_slot`.
- Règle : durée anormale = règle de `duration_off` (1 s autour de 120 s) ; reprend le § 8 du notebook 05. `test_max_s = 30 s` : choix de lecture, absent de la configuration.
- Limite : un test lancé pile sur un créneau passe pour du programme.
- Prez n° 4 : diapo 10. Résultat : 173 = 125 tests + 34 coupés + 14 longs ; parmi les 34 coupés, 3 ont un score BlanciNet ≥ 0,5. Ce qu'on peut dire : tous les chiffres de la boîte « durée anormale » tiennent. Ce qu'on ne peut pas dire : combien de chants sont perdus dans les coupés (un seul écouté).

#### `expected_size(duration_s, sample_rate, channels, bits=16, header_bytes=44)` — [inventaire.py:1095](inventaire.py#L1095)

- Question : combien un WAV de cette durée doit-il peser ?
- Entrée : durée, fréquence, pistes, bits, en-tête. Sortie : des octets.
- Règle : en-tête + durée × fréquence × pistes × bits / 8 ; 2 min en 48 kHz stéréo 16 bits = 23 040 044 octets.
- Limite : le bloc GUANO s'ajoute ; ne vaut pas pour le FLAC. Prez n° 4 : aucune.

#### `file_size_check(sample, raw, renamed=None, bits=16, tolerance_bytes=65536)` — [inventaire.py:1108](inventaire.py#L1108)

- Question : un fichier pèse-t-il ce que sa durée annonce ?
- Entrée : un échantillon, `paths.raw`, `audio.renamed`. Sortie : trouvé, taille, attendue, écart, `is_empty`, `size_off`.
- Règle : `expected_size`, taille lue par `stat` ; tolérance de 64 Kio, choix de lecture.
- Limite : **exige le disque** ; un en-tête qui annonce la durée tronquée ne se voit pas.
- Prez n° 4 : aucune. Résultat : un fichier par point de 2026 (156), lu sur F: le 18/11 : aucun vide, aucun hors tolérance, et 1 142 à 1 146 octets de plus que la formule partout (le bloc GUANO). Ce qu'on peut dire : sur ce tirage, aucun fichier tronqué. Ce qu'on ne peut pas dire : la même chose pour 2023 (disque de la phénologie incomplet sur F:) ni pour les fichiers non tirés.

## 9. Doublons et rapports d'inventaire

#### `duplicate_report_summary(duplicates)` — [inventaire.py:1152](inventaire.py#L1152)

- Question : combien de fichiers l'inventaire a-t-il écartés comme copies, et entre quels relevés ?
- Entrée : les rapports `paths.reports/ingest_duplicates_*.csv`. Sortie : une ligne par (relevé écarté, relevé conservé).
- Règle : reprend le § 5 du notebook 05. Limite : le rapport dit ce que l'inventaire a décidé, pas que le contenu concorde.
- Prez n° 4 : diapo 8. Résultat : 3 673 fichiers, tous de Mataroni, chacun avec son original dans le relevé de CDR. Ce qu'on peut dire : le 3 673 est recompté depuis le rapport (lisible sur F: le 18/11). Ce qu'on ne peut pas dire : que les 3 673 ont le même contenu que leur original (vérification par le contenu non faite, disque d'origine absent).

#### `duplicate_verification_summary(verification)` — [inventaire.py:1167](inventaire.py#L1167)

- Question : les paires de même nom ont-elles le même contenu ?
- Entrée : le rapport de `scripts/controle_qualite/verifier_doublons.py`. Sortie : une ligne par verdict.
- Règle : `ingest.same_recording`, lancé par le script. Limite : « introuvable » = non vérifié.
- Prez n° 4 : diapo 8. Résultat : le seul rapport présent (disque de la phénologie) a une paire, « identiques ». Ce qu'on peut dire : sur les disques du stage, un seul nom répété, de même contenu. Ce qu'on ne peut pas dire : rien sur les 3 673 de Mataroni.

#### `content_duplicates(rec)` — [inventaire.py:1182](inventaire.py#L1182)

- Question : la base contient-elle deux fois le même contenu sous deux identifiants ?
- Entrée : `load_recordings`. Sortie : dict : lignes avec empreinte, part, groupes.
- Règle : SHA-256 du fichier entier, calculé à l'inventaire quand il est demandé.
- Limite : presque aucune ligne n'a d'empreinte.
- Prez n° 4 : aucune. Résultat : 57 lignes sur 96 349 ont une empreinte (0,06 %) ; un groupe, deux enregistrements dirigés de 2021 copiés sous deux noms dans le dossier de Trésor (jeu `reference`). Ce qu'on peut dire : le test marche. Ce qu'on ne peut pas dire : qu'il n'y a pas de doublon de contenu dans le corpus (rien n'a été mesuré sur 99,94 % de la base).

#### `ingest_errors_summary(errors)` — [inventaire.py:1202](inventaire.py#L1202)

- Question : combien de fichiers sont restés hors de la base parce qu'illisibles ?
- Entrée : les rapports `paths.reports/ingest_errors_*.csv`. Sortie : une ligne par (relevé, type d'erreur).
- Règle : type = texte avant le premier « : ». Limite : seulement les rapports gardés.
- Prez n° 4 : diapo 21 (« le fichier illisible »). Résultat : 26 fichiers, tous `LibsndfileError` : 2 au premier relevé de 2023, 7 à CDR, 8 à Patawa, 9 à Mataroni. Ce qu'on peut dire : 26 fichiers ne sont pas dans la base, et on sait lesquels. Ce qu'on ne peut pas dire : ce qu'ils contenaient.

## 10. Pistes

#### `track_levels(rec)` — [inventaire.py:1221](inventaire.py#L1221)

- Question : de combien la piste 0 est-elle plus forte ou plus faible que la piste 1 ?
- Entrée : `load_recordings`. Sortie : RMS et indice d'aigus des deux pistes, écart.
- Règle : indices rangés par `blanci qc` ; même mesure que `level_diff_db` de l'échantillon des pistes.
- Limite : **`blanci qc` ne lit la piste 1 que pour les candidats au son étouffé et leurs voisins** : 1 404 enregistrements sur 96 135 contrôlés. Ce sous-ensemble est choisi par l'indice : il n'est pas représentatif.
- Prez n° 4 : diapos 6 et 20, mais elles se lisent sur l'échantillon des pistes, pas ici. Résultat : écart médian −12,3 dB sur les 1 404. Ce qu'on peut dire : la piste 1 n'est en base que là où le contrôle en avait besoin. Ce qu'on ne peut pas dire : l'écart entre pistes sur tout le corpus sans relire le disque.

#### `track_gap_by_point(levels, settings, gap_col="gap_db")` — [inventaire.py:1247](inventaire.py#L1247)

- Question : quel point a un micro qui n'entend plus comme l'autre ?
- Entrée : `track_levels` ou l'échantillon des pistes (`level_diff_db`), `gain_settings`. Sortie : une ligne par point : réglage, écart réglé, écart médian mesuré, déviation.
- Règle : médiane. Limite : une dégradation de deux mois dans une série d'un an ne déplace pas la médiane.
- Prez n° 4 : diapo 20. Résultat (échantillon des pistes) : 5 points à plus de 6 dB de l'écart réglé. Ce qu'on peut dire : le 5 de la diapo 20 tient. Ce qu'on ne peut pas dire : la perte de 30 dB de Kaw D de mai à juillet 2024 (série dans le temps, non rejouée ici).

#### `track_flags_summary(rec, cfg_qc)` — [inventaire.py:1269](inventaire.py#L1269)

- Question : que deviennent les pistes sous le seuil du son étouffé ?
- Entrée : `load_recordings`, `qc`. Sortie : une ligne par (catégorie, écarté).
- Règle : « retenue » = `muffled_0` / `muffled_1` rangés ; « candidat » = `qc.muffled_candidate` aux seuils actuels (`silent_dbfs` = −70 dBFS, `muffled_hf_ratio` = 0,05).
- Limite : « écarté » mêle le calcul et les confirmations à l'écoute.
- Prez n° 4 : diapo 17. Résultat :
  - deux pistes retenues : 399, tous écartés ;
  - piste 0 retenue seule : 407 gardés et 85 écartés ;
  - piste 1 retenue seule : 46 (38 gardés, 8 écartés) ;
  - candidat isolé sur la piste 0 sans piste retenue : 268 (263 gardés) ;
  - candidat isolé sur la piste 1 seule : 4 ;
  - 175 sans indices, tous écartés et jamais contrôlés : les 173 de durée anormale et 2 fichiers de 2 min hors relevé.
- Ce qu'on peut dire : 399, 407, 85 et 268 tiennent. Ce qu'on ne peut pas dire : « une quarantaine » de pistes 1 seules ; c'est 46.

## 11. Drapeaux

#### `exclusion_cause(flags)` — [inventaire.py:1313](inventaire.py#L1313)

- Question : pourquoi cet enregistrement est-il écarté ?
- Entrée : un dict `qc_flags` lu. Sortie : une cause dans un ordre fixe (étouffé par le calcul, confirmé, noté à l'écoute ; durée et hors relevé ; durée ; hors relevé ; silencieux ; autre).
- Règle : ordre du § 1 du notebook 06. Limite : une seule cause par enregistrement. Prez n° 4 : diapos 9 ter et 24, par `flag_counts`.

#### `flag_counts(rec)` — [inventaire.py:1343](inventaire.py#L1343)

- Question : combien sont écartés, pour quelle cause, combien portent chaque drapeau ?
- Entrée : `load_recordings`. Sortie : écartés, gardés, gardés grâce à `keep`, drapeaux levés, causes.
- Règle : `qc.is_excluded`, comme partout dans la chaîne. Limite : compte ce qui est rangé.
- Prez n° 4 : diapos 9 ter, 11, 24. Résultat :
  - 681 écartés, 95 629 gardés, 19 gardés grâce à `keep` ;
  - causes : étouffé par le calcul 399, étouffé confirmé 107 (soit 506 pour le son étouffé), durée seule 86, durée et hors relevé 87, hors relevé seul 2 ;
  - drapeaux levés : `clock_off` 78, `clock_corrected` 1 521, `rain` 13, `silent` 0.
- Ce qu'on peut dire : 681, 506, 399, 173 et 2 tiennent. Ce qu'on ne peut pas dire :
  - « 86 hors relevé aussi de durée anormale » sur la base telle qu'elle est : c'est 87, à cause d'un drapeau périmé (voir `stale_flags`) ;
  - « 95 611 sur 96 292 » (diapo 24) : c'est 95 629 sur 96 310 depuis l'inventaire du relevé 7.

#### `campaign_blocks(rec, cfg_qc)` — [inventaire.py:1364](inventaire.py#L1364)

- Question : comment la règle du hors relevé découpe-t-elle chaque série ?
- Entrée : `load_recordings`, `qc`. Sortie : une ligne par bloc : principal ou non, effectif, de durée normale, dates, écartés.
- Règle : celle de `off_campaign` (trou de plus de `campaign_gap_days` = 7 jours) ; reprend le § 7 du notebook 05.
- Limite : un micro reposé au même site après plus de 7 jours fait deux vrais blocs, et le petit passe hors relevé ; le seuil n'est pas estimé.
- Prez n° 4 : diapo 10. Résultat : 238 blocs, 74 petits, 106 enregistrements hors relevé dont 20 de 2 min, 88 écartés. Ce qu'on peut dire : tous les chiffres de la boîte « hors relevé » tiennent. Ce qu'on ne peut pas dire : la date de pose réelle de chaque micro.

#### `near_campaign_orphans(rec, cfg_qc, min_gap_days=3.0)` — [inventaire.py:1391](inventaire.py#L1391)

- Question : quels enregistrements sont isolés de leur série, mais pas assez pour être hors relevé ?
- Entrée : `load_recordings`, `qc`, `min_gap_days`. Sortie : les blocs séparés de la campagne par 3 à 7 jours.
- Règle : découpage refait à `min_gap_days` (3 jours, choix de lecture, absent de la configuration).
- Limite : ne dit pas ce que sont ces fichiers.
- Prez n° 4 : diapos 4 et 10. Résultat : 19 blocs, 25 enregistrements de 19 micros, tous à Mataroni les 30 et 31/12/2025, séparés de la pose par 5,8 à 6,9 jours. Avec le 26ᵉ, déjà hors relevé (2LA03555, 0,8 s), on retrouve les 26 enregistrements de 20 micros de la diapo 10. Ce qu'on peut dire : le « 25 enregistrements de 12 micros » des notes de la diapo 4 est faux sur les micros (19, ou 20 avec le 26ᵉ). Ce qu'on ne peut pas dire : si ce sont des tests ou des restes de carte (il faut écouter ; ils sont de toute façon écartés pour leur durée).

#### `stale_flags(rec, cfg)` — [inventaire.py:1426](inventaire.py#L1426)

- Question : les drapeaux d'inventaire rangés sont-ils ceux que donnerait la configuration actuelle ?
- Entrée : `load_recordings` (tous les jeux), configuration. Sortie : une ligne par drapeau : rangés, recalculés, différents.
- Règle : `qc.metadata_flags`, la fonction même de `blanci flag`, sur une copie ; rien n'est écrit.
- Limite : les drapeaux audio ne se recalculent pas ici.
- Prez n° 4 : diapos 9 ter et 10. Résultat : `duration_off` et `clock_off` à jour ; `off_campaign` diffère pour 1 enregistrement (Kaw 2023, `f670702208cbfa39`, 18 s), rangé hors relevé alors que la règle actuelle ne l'y met plus. Ce qu'on peut dire : la base a un drapeau périmé, que `blanci flag` corrigerait ; c'est lui qui fait 87 au lieu de 86. Ce qu'on ne peut pas dire : depuis quand.

## 12. Base et disque

#### `disk_inventory(root, suffixes)` — [inventaire.py:1452](inventaire.py#L1452)

- Question : qu'y a-t-il sur ce disque ?
- Entrée : un dossier, les extensions. Sortie : chemin relatif, nom, taille.
- Règle : `ingest.iter_audio_files`. Limite : **exige le disque** (`FileNotFoundError`).
- Prez n° 4 : aucune. Résultat : F: le 18/11/2026, 31 414 fichiers audio, parcourus en 27 s.

#### `base_vs_disk(rec, disk, renamed=None)` — [inventaire.py:1469](inventaire.py#L1469)

- Question : la base et le disque disent-ils la même chose ?
- Entrée : `load_recordings`, `disk_inventory`, `audio.renamed`. Sortie : une ligne par chemin, avec son statut.
- Règle : chemin cherché sous son nom et sous chaque dossier renommé (`core.locate.variants`).
- Limite : un « fichier sans ligne » peut être un doublon, un illisible, un relevé non inventorié ou un dossier hors corpus.
- Prez n° 4 : diapo 9 (« ce que l'inventaire ne voit pas »). Résultat (F:, 18/11) :
  - 29 764 lignes de la base retrouvées (29 746 sous un dossier renommé, 18 sous leur chemin : le relevé 7) ;
  - 66 546 absentes, toutes de la phénologie 2023-2024, dont F: n'a qu'une copie partielle ;
  - 1 650 fichiers sans ligne : 1 631 dans `data/` (hors corpus) et 19 dans `audio/`, qui sont des illisibles des rapports d'inventaire (7 à CDR, 8 à Patawa, 4 à Mataroni).
- Ce qu'on peut dire : pour 2025-2026, tout fichier audio lisible de F: est en base. Ce qu'on ne peut pas dire : la même chose pour 2023-2024 sans le disque de la phénologie (D:).

## 13. Annotations et scores

#### `listened_recordings(con)` — [inventaire.py:1508](inventaire.py#L1508)

- Question : quels enregistrements ont été écoutés, combien de secondes, A. blanci y a-t-il été entendu ?
- Entrée : connexion. Sortie : une ligne par enregistrement écouté : secondes (union des extraits), fenêtres étiquetées, positif.
- Règle : union des extraits ; `qc.positive_recordings`. Limite : une fenêtre étiquetée compte sans durée.
- Prez n° 4 : aucune directement. Résultat : 941 enregistrements écoutés, 186 positifs (tous jeux).

#### `annotation_coverage_by(rec, listened, by=("dataset", "site"))` — [inventaire.py:1540](inventaire.py#L1540)

- Question : quelle part de l'inventaire a été écoutée, par site ?
- Entrée : `load_recordings`, `listened_recordings`, regroupement. Sortie : effectifs, écoutés, part, heures, positifs, écartés écoutés.
- Règle : jointure sur l'identifiant. Limite : écouté ne veut pas dire écouté en entier.
- Prez n° 4 : aucune. Résultat :
  - parts écoutées : 0,35 à 0,50 % par site en 2023, 1,1 à 2,9 % en 2026 ;
  - positifs : 80 à Mataroni, 39 à Molokoï, 0 à RNRT et à PatawaOuest ;
  - à CDR, 64 écoutés sont écartés (les files ciblées du son étouffé).
- Ce qu'on peut dire : RNRT, tenu à l'écart pour le test (`plan.level2_site`), a 83 écoutés et aucun positif. Ce qu'on ne peut pas dire : qu'A. blanci est absent de RNRT.

#### `score_coverage(con, model_ids=None)` — [inventaire.py:1565](inventaire.py#L1565)

- Question : quels enregistrements ont été passés par quel modèle ?
- Entrée : connexion, modèles. Sortie : fenêtres et enregistrements par modèle.
- Règle : jointure `scores` → `windows`. Limite : sans filtre, la jointure parcourt 27 millions de lignes.
- Prez n° 4 : diapo 10. Résultat (`blancinet`) : 74 785 fenêtres, 8 349 enregistrements, tous de 2026. Les modèles birdnet n'ont pas été comptés (jointure trop longue pour ce rejeu).

#### `recording_detections(con, model_id)` — [inventaire.py:1584](inventaire.py#L1584)

- Question : combien de détections et quel score maximal par enregistrement ?
- Entrée : connexion, modèle. Sortie : détections, score maximal.
- Règle : requête des notebooks 05 (§ 8) et 06 (§ 1). Limite : BlanciNet est un témoin, pas une vérité.
- Prez n° 4 : diapo 10. Résultat : 8 349 enregistrements de 2026 scorés, comme annoncé.

#### `excluded_with_detections(rec, detections)` — [inventaire.py:1599](inventaire.py#L1599)

- Question : parmi les écartés, lesquels un détecteur signale-t-il ?
- Entrée : `load_recordings`, `recording_detections`. Sortie : les écartés détectés, avec leur cause.
- Règle : reprend le § 1 du notebook 06. Limite : on ne voit que ce que le détecteur a scoré.
- Prez n° 4 : aucune (notebook 06). Résultat : 108 écartés ont une détection, 19 à 0,5 ou plus. Ce qu'on peut dire : ces chiffres du notebook 06 tiennent.

#### `protected_positives(rec, positives)` — [inventaire.py:1613](inventaire.py#L1613)

- Question : un enregistrement où A. blanci a été entendu est-il écarté ?
- Entrée : `load_recordings`, `qc.positive_recordings(con)`. Sortie : positifs, positifs portant un drapeau écartant, positifs écartés.
- Règle : `qc.is_excluded`. Limite : ne protège que ce qui a été écouté.
- Prez n° 4 : diapo 11 (« un enregistrement où A. blanci a été entendu n'est jamais écarté »). Résultat : 160 positifs dans les jeux 2023 et 2026, **dont 1 écarté** : Mataroni SMA13417, 06/01/2026 11 h 30, 104 s (`9e453c33941c80ce`), celui de la diapo 10, écouté le 12/11 avec A. blanci de 0 à 67 s. Il porte `confirmed_duration_off` (confirmation en bloc) et pas `keep`. Ce qu'on peut dire : la règle de la diapo 11 est fausse sur la base d'aujourd'hui pour cet enregistrement. Ce qu'on ne peut pas dire : qu'il n'y en a pas d'autres parmi les écartés jamais écoutés.

## 14. Comparaisons

#### `compare_groups(rec, by="dataset", programmes=None)` — [inventaire.py:1632](inventaire.py#L1632)

- Question : en quoi 2023-2024 et 2025-2026 (ou deux sites) diffèrent-ils ?
- Entrée : `load_recordings`, regroupement, programmes. Sortie : une ligne par groupe sur les mêmes indicateurs.
- Règle : effectifs bruts, sans test statistique. Limite : les jeux ne sont pas comparables en saison.
- Prez n° 4 : diapo 4. Résultat :

| | 2023 | 2026 |
|---|---|---|
| Enregistrements | 66 779 | 29 531 |
| Heures | 2 225,8 | 980,9 |
| Points | 8 | 156 |
| Jours enregistrés | 393 | 84 |
| Enregistrements par point et par jour | 29,7 | 22,1 |
| Part écartée | 0,04 % | 2,2 % |

Ce qu'on peut dire : 2023 donne la durée (un an, 8 points), 2026 la diversité (156 points) ; l'essentiel des écartés vient de 2026. Ce qu'on ne peut pas dire : que le matériel de 2026 est moins fiable (les deux années n'ont ni les mêmes micros ni les mêmes poses).

#### `stratum_effort(rec, tranches, periods=None, kept_only=True)` — [inventaire.py:1670](inventaire.py#L1670)

- Question : combien d'enregistrements le tirage stratifié a-t-il dans chaque strate ?
- Entrée : `load_recordings`, `plan.tranches`, `plan.periods_2023`. Sortie : une ligne par (jeu, site, tranche[, période]).
- Règle : tranches et périodes de la configuration, sans changement. Limite : la nuit n'apparaît pas.
- Prez n° 4 : diapo 3 (« l'aube et le soir n'existent que pour une partie des sites »). Résultat : toutes les tranches (aube 5 h-7 h, pic du matin, journée, pic du soir, soir 17 h-20 h) ont des enregistrements à tous les sites ; à 5 h et à 19 h, seuls les micros programmés 5 h → 19 h 30 enregistrent. Ce qu'on peut dire : aucune strate n'est vide. Ce qu'on ne peut pas dire : que les strates sont équilibrées entre sites (Mataroni a 994 enregistrements d'aube pour 6 025 de journée).

## 15. Rejeu de la présentation n° 4

#### `presentation_4_numbers(con, cfg, lieux=None, pistes=None, duplicates=None)` — [inventaire.py:1717](inventaire.py#L1717)

- Question : chaque chiffre annoncé tient-il encore, et d'où vient-il ?
- Entrée : connexion, configuration, et selon ce qui est lisible : `lieux`, `echantillon_pistes.csv`, `ingest_duplicates_*.csv`.
- Sortie : une ligne par chiffre : diapo, chiffre, annoncé, recalculé, fonction, statut.
- Règle : chaque valeur vient d'une fonction de ce module ; aucune n'est recopiée des notes orateur.
- Limite : ce qui demande l'audio ou les files d'écoute n'y est pas (voir la section 16).
- Script : `scripts/verification/chiffres_presentation_4.py`. Résultat du 18/11/2026 : 67 chiffres, dont 58 « ok », 6 « différents » et 3 sans chiffre annoncé.

## 16. Chiffres de la présentation n° 4 : ce qui se rejoue, ce qui ne se rejoue pas

**Rejouables par une fonction, et justes sur la base du 18/11/2026** : diapo 3 (96 310 ; 3 207 h ; 164 points ; 95 enregistreurs ; 4 lieux ; 74 007 ; 22 289), diapo 4 (25 enregistrements des 30-31/12), diapo 5 bis (6,7 Go), diapo 6 (143 ; 21 ; 18 2MA et 3 SMA ; 3 607 ; 164), diapo 7 (3 137 ; 3 128 ; 99,7 % ; 9 ; 2 217 et 99,8 % ; 920 et 99,5 % ; 54 jours ; 5 jours ; 5 micros le 10/02), diapo 9 (1 521 ; 0,57 ; 0,34 ; −1 h), diapos 9 ter et 11 (681 ; 506 ; 399 ; 173 ; 2), diapo 10 (238 blocs ; 74 ; 106 ; 20 ; 88 ; 173 = 125 + 34 + 14 ; 8 349 ; 3 coupés à 0,5 ou plus ; 26 enregistrements de 20 micros), diapo 8 (3 673, depuis le rapport), diapo 17 (399 ; 407 ; 85 ; 268), diapo 19 (452), diapo 20 (5 points à plus de 6 dB).

**Rejouables, mais différents** :
- diapo 4 : « 25 enregistrements de 12 micros » ; c'est 19 micros (`near_campaign_orphans`) ;
- diapo 6 : « un par micro et par jour, soit 3 607 » ; 3 607 = 3 466 par jour + 141 bascules (`sample_recordings`) ;
- diapo 9 ter : « 86 hors relevé aussi de durée » ; 87 en base, à cause d'un drapeau périmé (`stale_flags`) ;
- diapo 17 : « une quarantaine » de pistes 1 seules ; 46 (`track_flags_summary`) ;
- diapo 24 : « 95 611 sur 96 292 » ; 95 629 sur 96 310 depuis l'inventaire du relevé 7 (`flag_counts`) ;
- diapo 11 : « jamais écarté s'il a été entendu » ; un positif est écarté (`protected_positives`).

**Pas encore rejouables par ce module** :
- diapo 9, les spectrogrammes : ils demandent les fichiers de Molokoï du 03/04/2024 (`fig_molokoi_spectros`) ;
- diapo 8, la vérification des 3 673 par le contenu : elle demande le disque d'origine (Biophonia) ;
- diapo 13, les ~5 600 morceaux de Welch : c'est un calcul, pas un comptage ;
- diapo 15 (761 et 1 563 pistes selon le seuil) : `blanci qc-calibrate` et le notebook 06 ;
- diapo 16 (1,2 %, 0,002, « une cinquantaine de suites ») : notebook 06 ;
- diapo 18 (674 écoutés, 6 inutilisables, 0,3 à 1,9 %) : ils dépendent des files d'écoute et des commentaires (notebook 06, § 4), que ce module ne lit pas ;
- diapo 19 (9 pistes étouffées dès la pose, 14 pistes de 10 enregistreurs dans la nuit du 24 au 25/12) : notebook 06 ;
- diapo 20 (Kaw D, −30 dB du 23/05 au 23/07/2024 ; 1 871 encodés sur la piste sourde ; six dans le jeu de test) : notebook 06 ;
- diapo 22 (31 et 29 écoutés) et diapo 23 (4 étouffés dans le jeu de test, 3 dans le lot, 68 extraits dont 15 positifs) : files d'écoute et paquets d'annotation.

Les chiffres du contrôle qualité qui viennent du notebook 06 sont rejouables en relançant le notebook, pas par une fonction testée ; les passer dans ce module (ou dans un module voisin du contrôle qualité) est la suite naturelle.

## 17. Écarts relevés

1. **Un enregistrement entendu positif est écarté** : Mataroni SMA13417, 06/01/2026 11 h 30 (`9e453c33941c80ce`). Il porte `confirmed_duration_off` et pas `keep`. Il faudrait poser `keep` (`qc.keep_flags`, depuis la chaîne, pas d'ici).
2. **Un drapeau `off_campaign` périmé** : Kaw 2023, `f670702208cbfa39`. `blanci flag` le recalculerait.
3. **Les longitudes des enregistreurs 2MA n'ont pas de signe** (52,29 au lieu de −52,29). Et beaucoup de positions sont à quelques kilomètres de Cayenne, quel que soit le site (`mic_positions`). Le GPS de l'en-tête ne peut donc pas vérifier le site, ce qui rend caduc le « critère qui manque » de la diapo 21 tant qu'on n'a pas les coordonnées de pose.
4. **La piste 1 n'est en base que pour 1 404 enregistrements**. L'écart entre pistes se mesure sur l'échantillon des pistes, pas sur la base.
5. **Deux totaux circulent dans la présentation** (96 310 et 96 292), et les notebooks 05 et 06 affichent encore 96 292. Leurs sorties datent d'avant l'inventaire du relevé 7.
6. **PatawaOuest SMA14826** : 14 enregistrements, aucun programme lisible, 11 gardés hors des créneaux. Il est arrêté le 01/01/2026, trois jours avant la relève.
