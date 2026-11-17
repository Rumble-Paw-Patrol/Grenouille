# `blanci/inputs/` : le code fonction par fonction

Fiche vérifiée le 10/11/2026 sur le commit `9473508` du dépôt public (`git log -1 --format=%h`), arbre de travail sans modification dans ce dossier (`qc.py` et `ingest.py` relus juste avant d'écrire). Pour savoir si elle est périmée : `git diff --stat 9473508 -- blanci/inputs` ; une sortie vide veut dire que le code de ce dossier n'a pas bougé.

`inputs/` est l'étape 1 : 6 fichiers, 2 245 lignes. Il range les enregistrements en base, pose les drapeaux de contrôle qualité, définit les labels, et fabrique les jeux d'apprentissage à partir des annotations.

| Fichier | Lignes | Rôle |
|---|---|---|
| [`ingest.py`](ingest.py) | 406 | l'inventaire : un fichier audio → une ligne de `recordings` |
| [`qc.py`](qc.py) | 623 | les drapeaux : qui est écarté du corpus, et pourquoi |
| [`labels.py`](labels.py) | 215 | la liste des labels, la lecture des commentaires, les tableaux d'un outil externe |
| [`dataset.py`](dataset.py) | 896 | des annotations aux fenêtres étiquetées, négatifs appariés, folds |
| [`frozen.py`](frozen.py) | 97 | le jeu gelé (jeu de test) |
| [`__init__.py`](__init__.py) | 8 | une docstring, aucun code |

## La chaîne

Lecture : `a → b` veut dire « `a` appelle `b` » ; un point-virgule sépare les appels successifs d'une même fonction. Les noms sans préfixe sont dans `inputs/`.

- **`blanci ingest`** : `cli.ingest` → `ingest` → `iter_audio_files` → `natural_key` ; pour un nom déjà inventorié ailleurs : `same_recording` → `audio_signature` → `read_guano` ; sinon `describe_recording` → `read_guano`, `parse_songmeter_name`, `audio_qc` (→ `qc_indices` → `_welch` ; `qc_flags` → `muffled_candidate`), `clock_corrected`, `core.recording_id_for`, `start_utc` (→ `parse_guano_timestamp`, `parse_songmeter_name`). Une fois tous les fichiers passés : `apply_clock_corrections` ; `apply_metadata_flags` → `metadata_flags` ; `apply_audio_flags` → `qc_flags`, `muffled_runs`, `_write_flags` ; `apply_annotation_flags` → `annotation_flags` → `_current_spans`.
- **`blanci qc`** : `cli.qc_command` → `embed.select_recordings` → `is_excluded`, `positive_recordings` ; `check_recordings` → `positive_recordings` ; `_unread_neighbours` ; `_read_tracks` → `core.ahead` → `prepare` (dans les fils) → `core.variants`, `audio_qc` ; puis dans le fil principal `merge_audio_flags` → `parse_flags`, `_write_flags` ; `is_excluded`. Enfin `apply_audio_flags`.
- **`blanci flag`** : `cli.flag` → `apply_clock_corrections` → `clock_corrected`, `start_utc`, `parse_flags` ; `apply_metadata_flags` ; `apply_audio_flags` ; `apply_annotation_flags`. Aucune lecture d'audio.
- **`blanci embed`** : `cli.embed` → `embed.select_recordings` → `is_excluded`, `positive_recordings` ; avec `--subset benchmark` : `benchmark_subset` → `pairing_options`, `benchmark_recordings` → `recordings_table`, `current_labels`, `load_spans`, `_positive_intervals`, `is_excluded`, `_pairing_pools`. Pendant l'encodage, `embed.embed_recordings` → `parse_flags`, `qc_indices`, `qc_flags`, `is_excluded`, `merge_audio_flags`, et à la fin `apply_audio_flags`.
- **`blanci train`** (et tout ce qui entraîne ou évalue une tête) : `service.train_and_register` → `frozen_recordings` → `frozen_versions` → `frozen_directory` ; `pairing_options` ; `embedded_training_set` → `training_set` → `current_labels` ; `load_spans` → `annotator_filter` ; `interval_labels` → `current_intervals`, `_within_union` ; `transfer_labels` ; `_merge_labelled` ; `positive_annotations` ; `_positive_intervals` ; `suspect_false_negatives` ; `recordings_table` ; `paired_negatives` → `local_minutes`, `local_days`, `_pairing_pools`. Puis `folds_for` → `benchmark_folds`.
- **Poste d'annotation** : `workbench.save_span` et `workbench.save_answer` → `comment_fields` → `normalize`, `find_species` ; puis `service.append_span` et `service.append_label` → `apply_annotation_flags`.
- **`blanci export-labels`** : `cli.export_labels` → `service.export_annotations` → `annotated_spans` (→ `recordings_info`, `load_spans`, `current_intervals`) ; `add_versions` → `_last_version` ; `reannotations`.
- **`blanci clips-import`** : `cli.clips_import` → `selection.import_clip_labels` → `read_annotation_table` ; `detect_columns` → `column_key`, `_contains_words`.
- **`blanci freeze`** : `cli.freeze` → `freeze` → `frozen_directory`, `recordings_table`, `current_labels`, `load_spans`, `current_intervals`, `frozen_recordings`.
- **`blanci status`** : `cli.status` → `parse_flags`.
- **Script `scripts/son_etouffe/confirmer_drapeaux.py`** : `main` → `positive_recordings`, `parse_flags`, `flag_raised`, `confirm_flags`, `keep_flags`.

## `ingest.py` : l'inventaire

Parcourt un dossier d'enregistrements et écrit une ligne par fichier dans `recordings`. Lit : les fichiers audio sous `paths.raw` (en-tête, bloc GUANO, et tout le fichier si le contrôle qualité ou l'empreinte sont demandés). Écrit : la table `recordings` seulement ; la docstring du fichier le dit, la racine audio n'est jamais modifiée. Utilisé par `blanci ingest`, `blanci flag`, `evaluation/anuraset.prepare` et `scripts/controle_qualite/verifier_doublons.py`.

#### `ingest(con, root, dataset, cfg, run_qc=True, hash_file=True, force=False, progress_every=500, scan=None, site=None)` — [ingest.py:290](ingest.py#L290)

- Quand : c'est la commande `blanci ingest` ([`cli.ingest`](../cli.py#L135), sous le nom `run_ingest`) ; aussi `evaluation/anuraset.prepare` pour AnuraSet.
- Reçoit : la connexion, `paths.raw`, le nom du jeu (`2023`, `2026`), la configuration, et au besoin le dossier à parcourir (`scan`, par défaut `<racine>/<jeu>`) et le site. Renvoie : un `IngestReport` (ajoutés, sautés, déplacés, doublons, conflits, erreurs, drapeaux).
- Fait : 1. refuse un dossier qui n'est pas sous `paths.raw` (les chemins rangés lui sont relatifs) ; 2. charge les chemins déjà en base et un index par nom de fichier en minuscules ; 3. pour chaque fichier de `iter_audio_files` : un chemin déjà connu est sauté (sauf `force`) ; un nom déjà connu sous un autre chemin passe par les trois cas ci-dessous ; sinon `describe_recording` puis écriture ; 4. valide environ tous les 500 fichiers parcourus ; 5. à la fin, recalcule tous les drapeaux (`apply_clock_corrections`, `apply_metadata_flags`, `apply_audio_flags` si le contrôle audio a tourné, `apply_annotation_flags`).
- Les trois cas d'un nom déjà connu : l'autre fichier existe encore et `same_recording` dit que le contenu concorde → **doublon**, écarté ; il existe encore et le contenu diffère (ou il est illisible) → **conflit**, écarté et signalé avec la raison ; il n'existe plus sous la racine → **déplacement**, la ligne prend le nouveau chemin, sauf si le site a changé, auquel cas c'est un doublon (DECISIONS n° 144).
- Détail qui ne se voit pas : l'écriture est un `INSERT … ON CONFLICT DO UPDATE` où `sha256` et `qc_flags` gardent leur ancienne valeur si la nouvelle est vide. Un réinventaire lancé sans contrôle qualité ni empreinte n'efface donc pas ceux déjà calculés.
- Pourquoi (docstring) : les cartes SD rapportées d'un relevé contiennent encore les fichiers du relevé précédent, qui compteraient deux fois sous deux sites. Le premier inventorié l'emporte : il faut inventorier les relevés dans l'ordre chronologique. Un déplacement garde labels, fenêtres et embeddings parce que l'identifiant ne dépend que du nom du fichier.
- `IngestReport` ([ingest.py:43](ingest.py#L43)) : le bilan rendu, une simple structure de compteurs et de listes.

#### `iter_audio_files(directory, suffixes=(".wav", ".flac"))` — [ingest.py:54](ingest.py#L54)

- Quand : au début de `ingest` ; aussi `repeated_names` dans `scripts/controle_qualite/verifier_doublons.py`.
- Reçoit : un dossier. Renvoie : un par un, les fichiers audio de ce dossier et de ses sous-dossiers.
- Fait : trie dans l'ordre naturel (« RELEVE 2 » avant « RELEVE 10 »), ne garde que les extensions voulues, ignore les fichiers cachés, dont les `._*` que macOS sème sur un disque externe.
- Pourquoi (docstring) : pour que la règle « le premier inventorié l'emporte » suive l'ordre des relevés (DECISIONS n° 144, 2ᵉ point).
- `natural_key(path)` ([ingest.py:66](ingest.py#L66)) : la clé de tri ; découpe le chemin en morceaux de texte et de chiffres, et compare les chiffres comme des nombres.

#### `describe_recording(path, root, dataset, cfg, run_qc, hash_file, site=None)` — [ingest.py:233](ingest.py#L233)

- Quand : pour chaque fichier nouveau ou déplacé ; un seul appelant, `ingest`.
- Reçoit : le fichier, la racine, le jeu, la configuration, deux interrupteurs. Renvoie : un dictionnaire, la ligne de `recordings` (identifiant, chemin relatif, jeu, site, micro, heure UTC, durée, fréquence, canaux, SHA-256, drapeaux).
- Fait : 1. si le contrôle qualité ou l'empreinte sont demandés, lit tout le fichier une seule fois en mémoire et tout le reste travaille sur cette copie ; sinon n'ouvre que l'en-tête ; 2. lit le bloc GUANO et le nom du fichier ; 3. micro = numéro de série GUANO, sinon préfixe du nom, sinon dossier ; site = celui donné, sinon le dossier sous `<racine>/<jeu>/` ; 4. lance `audio_qc` si demandé ; 5. si l'horloge de ce micro est corrigée dans la configuration, oublie l'heure GUANO pour que le nom fasse foi ; 6. calcule l'identifiant et l'heure UTC.
- Pourquoi : sans contrôle qualité ni empreinte, l'en-tête suffit et l'inventaire d'un disque entier prend des minutes (commentaire : un fichier pèse 23 Mo). Le numéro de série est préféré parce que le même enregistreur s'appelle « SM4 A », « SM A_SMA13417 » ou « SMA13417 » selon le relevé (docstring).

#### `read_guano(source)` — [ingest.py:71](ingest.py#L71)

- Quand : dans `describe_recording` et `audio_signature` ; aussi `guano_fields` dans `scripts/controle_qualite/echantillon_pistes.py`.
- Reçoit : un chemin ou un fichier ouvert. Renvoie : les métadonnées GUANO en dictionnaire (`Serial`, `Timestamp`…), ou `{}`.
- Fait : vérifie l'en-tête `RIFF … WAVE`, puis saute de bloc en bloc avec `seek` jusqu'au bloc `guan`, sans lire le son. Un FLAC, ou un WAV sans ce bloc, rend `{}`.
- Détail : c'est ce saut qui rend l'inventaire rapide quand on ne demande ni contrôle qualité ni empreinte. La première ligne du bloc (la version GUANO) est ignorée ; chaque ligne suivante est coupée au premier `:`.

#### `start_utc(guano, stem, utc_offset_h)` — [ingest.py:129](ingest.py#L129)

- Quand : dans `describe_recording` et `apply_clock_corrections`.
- Reçoit : le GUANO, le nom du fichier sans extension, le décalage horaire de la configuration (`recorder.filename_utc_offset_h`, −3 en Guyane). Renvoie : l'heure de début en UTC, en texte, ou `None`.
- Fait : prend l'horodatage GUANO s'il est lisible, sinon celui du nom du fichier ; un horodatage sans fuseau est supposé à l'heure locale ; convertit en UTC.
- `parse_guano_timestamp(value)` ([ingest.py:116](ingest.py#L116)) : lit « 2025-12-27 15:30:00-3:00 » en ajoutant le zéro que le Song Meter omet dans « -3:00 », que Python refuse sinon. Pourquoi (docstring) : le fuseau varie d'un enregistreur à l'autre ; le perdre fausse l'heure de 3 h sans le dire.
- `parse_songmeter_name(stem)` ([ingest.py:100](ingest.py#L100)) : `<préfixe>_<AAAAMMJJ>_<HHMMSS>` → (préfixe, date et heure sans fuseau), ou `None`.

#### `same_recording(a, b)` — [ingest.py:166](ingest.py#L166)

- Quand : dans `ingest`, quand un nom de fichier est déjà inventorié sous un autre chemin encore présent ; aussi dans `scripts/controle_qualite/verifier_doublons.py`.
- Reçoit : deux fichiers. Renvoie : (même enregistrement ou non, raison du désaccord).
- Fait : compare le nombre d'échantillons, la fréquence, les pistes, le numéro de série et l'horodatage GUANO (ces deux derniers seulement s'ils sont présents des deux côtés), puis la première seconde de son, échantillon par échantillon, à quelques pas de quantification près (`SIGNATURE_TOLERANCE`).
- Pourquoi : un doublon se reconnaît au contenu, pas au seul nom. Le son est comparé décodé, pas en octets : un WAV et le FLAC tiré du même enregistrement donnent le même son à un pas d'arrondi près (docstring de `audio_signature`).
- `audio_signature(path)` ([ingest.py:147](ingest.py#L147)) : ce qui est comparé, pour un fichier.

#### `apply_clock_corrections(con, recorder)` — [ingest.py:194](ingest.py#L194)

- Quand : à la fin de `ingest`, et au début de `blanci flag` ([`cli.flag`](../cli.py#L255)).
- Reçoit : la connexion et la section `recorder` de la configuration. Écrit : `start_utc` et `qc_flags` des enregistrements concernés. Renvoie : leur nombre.
- Fait : pour chaque micro cité dans `recorder.clock_corrections`, recalcule l'heure de début depuis le nom du fichier et pose la remarque `clock_corrected`. Un enregistrement qui n'est plus dans la période perd la remarque, pas l'heure corrigée.
- Détail : la fonction peut tourner autant de fois qu'on veut, l'heure est toujours recalculée depuis le nom, jamais depuis l'heure déjà rangée (docstring).
- `clock_corrected(stem, mic_id, corrections)` ([ingest.py:180](ingest.py#L180)) : vrai si ce micro a une correction dont la période `[from, to[` contient l'horodatage du nom.

## `qc.py` : les drapeaux

Un drapeau est une remarque rangée dans la colonne `recordings.qc_flags` (du JSON) ; le fichier audio n'est jamais touché. Lit : l'audio (les deux pistes) pour les drapeaux audio, les tables `recordings`, `labels`, `windows`, `spans`. Écrit : seulement `recordings.qc_flags`. Utilisé par l'inventaire, `blanci qc`, `blanci flag`, l'encodage, le poste d'annotation, les scripts de `scripts/son_etouffe/`.

Les clés que ce code écrit dans `qc_flags` :

| Clé | Posée par | Écarte du corpus |
|---|---|---|
| `silent` | `qc_flags` (niveau sous `silent_dbfs`) | oui |
| `muffled_0`, `muffled_1` | `qc_flags` (candidat), confirmés par `apply_audio_flags` | non |
| `muffled` | `apply_audio_flags` (les deux pistes) | oui |
| `indices`, `indices_1` | `qc_flags` (les mesures de chaque piste) | non |
| `duration_off`, `off_campaign` | `apply_metadata_flags` | oui |
| `clock_off` | `apply_metadata_flags` | non (une remarque, DECISIONS n° 191) |
| `clock_corrected` | `ingest.apply_clock_corrections` | non |
| `annotated` (liste : `muffled`, `rain`) | `apply_annotation_flags` | `muffled` oui, `rain` non |
| `confirmed_muffled`, `confirmed_duration_off`, `confirmed_off_campaign` | `confirm_flags` | oui |
| `keep` | `keep_flags` | non : annule tous les autres |

#### `parse_flags(qc)` — [qc.py:55](qc.py#L55)

- Quand : partout où `qc_flags` est lu : `is_excluded`, `apply_audio_flags`, `embed.embed_recordings`, `cli.status`, `workbench.flagged_candidates`, quatre scripts.
- Reçoit : le texte JSON de la base, un dictionnaire déjà lu, ou rien. Renvoie : un dictionnaire.
- Détail : l'ancien nom `in_bag` est lu comme `muffled`, dans les clés et dans la liste `annotated` (`LEGACY_FLAGS`, [qc.py:52](qc.py#L52)). Pourquoi (commentaire) : tant que `blanci qc` n'a pas recalculé la base, l'enregistrement reste écarté.

#### `is_excluded(qc, keys=EXCLUDING_FLAGS)` — [qc.py:78](qc.py#L78)

- Quand : chaque fois qu'on décide si un enregistrement est dans le corpus : [`embed.select_recordings`](../embedding/embed.py#L79) (ce qui sera encodé), `embed.embed_recordings`, `_read_tracks`, `benchmark_recordings`, `annotation/plan.eligible_recordings`, `workbench.flagged_candidates`, `service._qc_flagged`, `explore.py`.
- Reçoit : les drapeaux, et la liste des clés qui écartent. Renvoie : vrai si l'enregistrement est écarté.
- Fait : si `keep` est posé, rend faux ; sinon vrai dès qu'une des clés est levée.
- `flag_raised(flags, key)` ([qc.py:73](qc.py#L73)) : un drapeau est levé s'il est vrai, ou s'il figure dans la liste `annotated` (posé à l'écoute).
- `EXCLUDING_FLAGS` ([qc.py:48](qc.py#L48)) : `muffled`, `silent`, `duration_off`, `off_campaign` et les trois `confirmed_…`. Le commentaire donne la conséquence : jamais encodé, donc ni négatif apparié, ni candidat à écouter, ni score.

#### `positive_recordings(con)` — [qc.py:127](qc.py#L127)

- Quand : avant tout tri par drapeaux : `embed.select_recordings`, `embed.embed_recordings`, `check_recordings`, deux scripts de `scripts/son_etouffe/`.
- Renvoie : l'ensemble des enregistrements où A. blanci a été entendu : le dernier label d'une de leurs fenêtres est positif, ou un intervalle positif y est encore valable.
- Pourquoi (docstring du fichier) : un enregistrement où A. blanci a été entendu n'est jamais écarté, l'écoute prime sur le calcul.

#### `audio_qc(source, thresholds)` — [qc.py:215](qc.py#L215)

- Quand : à l'inventaire (`ingest.describe_recording`) et dans `blanci qc` (`prepare` de `check_recordings`).
- Reçoit : un chemin ou un fichier en mémoire, et la section `qc` de la configuration. Renvoie : les drapeaux audio et les indices des deux pistes.
- Fait : lit tout le fichier, les deux pistes en une fois, mesure chaque piste avec `qc_indices`, puis `qc_flags`.

#### `qc_indices(wav, sr)` — [qc.py:163](qc.py#L163)

- Quand : dans `audio_qc` ; pendant l'encodage sur la piste lue (`embed.embed_recordings`) ; dans `evaluation/qc_calibration.py` et deux scripts.
- Reçoit : une piste et sa fréquence. Renvoie : cinq mesures : `rms_dbfs` (niveau moyen), `peak`, `clip_fraction` (part d'échantillons au maximum), `hf_ratio` (part de l'énergie au-dessus de 2 kHz, sur l'énergie au-dessus de 100 Hz), `flatness_1_10k` (platitude du spectre entre 1 et 10 kHz).
- `_welch(x, sr, nperseg=2048)` ([qc.py:143](qc.py#L143)) : le spectre moyen, comme `scipy.signal.welch` mais calculé par paquets de 256 segments. Pourquoi (docstring) : trois à quatre fois plus vite sur un enregistrement de 2 min, pour un écart relatif de 1e-7.

#### `qc_flags(indices, thresholds, indices_1=None)` — [qc.py:191](qc.py#L191)

- Quand : dans `audio_qc`, dans `apply_audio_flags` (sur les indices déjà rangés), pendant l'encodage, dans `evaluation/qc_calibration.current_flags`.
- Reçoit : les indices de la piste 0, les seuils, et les indices de la piste 1 d'un fichier stéréo. Renvoie : `silent`, `muffled_0`, `indices`, et `muffled_1`, `indices_1` s'il y a une seconde piste.
- Détail : les indices sont gardés avec les drapeaux ; un seuil changé se réapplique sans relire l'audio. Tous les indices mesurés sont rangés, y compris ceux qu'aucun drapeau ne lit (saturation, platitude). `muffled_0` et `muffled_1` ne sont ici que des candidats.
- `muffled_candidate(indices, thresholds)` ([qc.py:182](qc.py#L182)) : une piste est candidate si elle n'est pas silencieuse et si `hf_ratio` est sous `muffled_hf_ratio` (0,05 dans `default.yaml`).

#### `check_recordings(con, recordings, raw_root, thresholds, workers=4, commit_every=100, progress_every=1000, renamed=None)` — [qc.py:321](qc.py#L321)

- Quand : c'est `blanci qc` ([`cli.qc_command`](../cli.py#L211)), à lancer avant le premier encodage.
- Reçoit : les enregistrements retenus, `paths.raw`, les seuils, le nombre de fils, `audio.renamed`. Renvoie : un bilan (contrôlés, écartés, déjà faits, illisibles). Écrit : `qc_flags`.
- Fait : 1. saute ce qui est déjà mesuré ; 2. lit le reste dans `workers` fils (`_read_tracks`) ; 3. recommence avec les voisins des candidats de la piste 1 dont la piste 1 n'a pas été lue (`_unread_neighbours`), de proche en proche, jusqu'à ce qu'il n'y en ait plus.
- Détail : « déjà mesuré » veut dire que les indices de la piste 0 sont rangés, et que la piste 1 l'est aussi, ou que le fichier est mono, ou que la piste 0 n'est pas candidate. Le commentaire donne la raison : sans piste 0 candidate, l'enregistrement ne peut pas être `muffled`. Le fichier est cherché sous `paths.raw` seulement, sous son nom d'origine puis sous les noms renommés.
- `_read_tracks(…)` ([qc.py:422](qc.py#L422)) : la boucle de lecture : range les drapeaux avec `merge_audio_flags`, valide tous les 100, affiche l'avancement tous les 1 000 ; si un fichier est illisible et que `paths.raw` n'existe plus, s'arrête avec « n'est plus accessible ; relancer » au lieu de compter tout le disque en erreur.
- `_unread_neighbours(con, known, thresholds)` ([qc.py:387](qc.py#L387)) : les enregistrements stéréo dont la piste 1 n'est pas lue et qui précèdent ou suivent, sans trou de plus de `muffled_max_gap_min`, un candidat piste 1 du même micro.

#### `merge_audio_flags(con, recording_id, audio)` — [qc.py:309](qc.py#L309)

- Quand : pour chaque enregistrement contrôlé, dans `_read_tracks` et dans `embed.embed_recordings`.
- Fait : relit les drapeaux de la ligne, pose les drapeaux audio par-dessus sans effacer les autres, écrit, renvoie le tout.
- `_write_flags(con, updates)` ([qc.py:303](qc.py#L303)) : l'écriture groupée de `qc_flags`, dans une transaction ; ne fait rien si la liste est vide.

#### `apply_audio_flags(con, thresholds)` — [qc.py:489](qc.py#L489)

- Quand : à la fin de `blanci qc`, de `blanci ingest` et d'un encodage qui a contrôlé des fichiers ; dans `blanci flag`.
- Reçoit : les seuils. Écrit : `silent`, `muffled_0`, `muffled_1`, `muffled`. Renvoie : le nombre d'enregistrements par drapeau.
- Fait : 1. recalcule les candidats de chaque piste depuis les indices déjà rangés, aux seuils actuels, sans relire l'audio ; 2. ne garde un candidat que dans une suite (`muffled_runs`) ; 3. lève `muffled` si les deux pistes sont retenues ; pour un fichier mono, la piste 0 compte pour les deux.
- Détails : un enregistrement sans indices ne change pas. Un fichier stéréo dont la piste 1 n'a pas été lue et dont la piste 0 est candidate garde son `muffled` d'avant. Une ligne n'est réécrite que si un drapeau a changé.
- Pourquoi : la règle des suites est le n° 194, la règle des deux pistes le n° 203 (docstrings).
- `muffled_runs(series, min_run=4, max_gap_min=60.0)` ([qc.py:466](qc.py#L466)) : garde les candidats qui forment une suite d'au moins `min_run` enregistrements consécutifs du même micro, sans trou de plus de `max_gap_min` minutes. Pourquoi (docstring) : un micro dans un sac donne une série continue ; un enregistrement grave isolé (singes hurleurs, chœur, orage) n'en est pas une. Aussi appelée par `evaluation/qc_calibration.py` et `scripts/son_etouffe/qc_deux_pistes.py`.

#### `apply_metadata_flags(con, thresholds, utc_offset_h=-3.0)` — [qc.py:277](qc.py#L277)

- Quand : à la fin de `ingest`, et dans `blanci flag`.
- Écrit : `duration_off`, `off_campaign`, `clock_off` de tous les enregistrements, sans toucher aux autres clés. Renvoie : le nombre d'enregistrements par drapeau.
- Pourquoi (commentaire de `ingest`) : ces drapeaux dépendent de toute la série d'un micro, ils sont donc recalculés sur toute la base à chaque fois.
- `metadata_flags(recordings, thresholds, utc_offset_h=-3.0)` ([qc.py:227](qc.py#L227)) : le calcul. `duration_off` : durée à plus de `duration_tolerance_s` (1 s) de `expected_duration_s` (120 s). `off_campaign` : pour chaque (jeu, site, micro), les dates sont coupées en blocs là où un trou dépasse `campaign_gap_days` (7 jours) ; le plus gros bloc est le relevé, les autres en sortent. `clock_off` : l'heure du nom de fichier et celle de l'en-tête diffèrent de plus de `clock_tolerance_min` (5 min).

#### `apply_annotation_flags(con, recording_ids=None)` — [qc.py:601](qc.py#L601)

- Quand : après chaque annotation ([`service.append_label`](../service.py#L616), [`append_span`](../service.py#L673), [`append_labels_bulk`](../service.py#L748), `_import_set`), au début de `service.retrain`, à la fin de `ingest`, dans `blanci flag`.
- Écrit : la clé `annotated` des enregistrements donnés (ou de tous). Renvoie : le nombre d'enregistrements annotés et, par drapeau, le nombre de signalés.
- Fait : réécrit `annotated` d'après les derniers labels et les extraits encore valables ; un enregistrement qui n'est plus annoté perd la clé. Un label corrigé ou un extrait réécouté retire donc le drapeau qu'il avait posé.
- `annotation_flags(con, recording_ids=None)` ([qc.py:577](qc.py#L577)) : le calcul : le label `artefact_muffled` (ou l'ancien `artefact_in_bag`) pose `muffled`, le label `rain` ou une mention de pluie dans le commentaire pose `rain`. Un enregistrement annoté sans rien de tel a une liste vide.
- `_current_spans(con)` ([qc.py:554](qc.py#L554)) : les extraits encore valables, c'est-à-dire non recouverts en entier par un extrait plus récent.

#### `confirm_flags(con, recording_ids, flags)` et `keep_flags(con, recording_ids)` — [qc.py:86](qc.py#L86)

- Quand : aucune commande `blanci` ne les appelle ; seul `scripts/son_etouffe/confirmer_drapeaux.py` le fait, après écoute.
- `confirm_flags` pose `confirmed_<drapeau>` (un des trois de `CONFIRMED_FLAGS`, sinon `ValueError`). `keep_flags` ([qc.py:110](qc.py#L110)) pose `keep`. Les deux renvoient le nombre d'enregistrements modifiés et lèvent `KeyError` pour un enregistrement absent de la base.
- Pourquoi (commentaires) : ce sont des clés propres, que ni l'inventaire ni le contrôle audio ne réécrivent. `keep` veut dire « écouté et gardé malgré un drapeau » et l'emporte sur tous les autres.

## `labels.py` : les labels et les commentaires

Des listes de référence et des fonctions de lecture de texte. Ne lit pas la base ; `read_annotation_table` lit un fichier CSV ou Excel. N'écrit rien. Utilisé par `service.py` (contrôle de ce qu'on enregistre), le poste d'annotation, `qc.py`, `dataset.py`.

#### `POSITIVE_LABELS`, `LABELS`, `QUALITIES`, `SOURCES` — [labels.py:14](labels.py#L14)

- `POSITIVE_LABELS` : `blanci`, `blanci_solo`, `blanci_chorus`. C'est la définition de « positif » pour tout le projet (19 fonctions hors tests y font référence).
- `LABELS` : les 15 labels acceptés. `service.append_label`, `append_span`, `append_labels_bulk` et `workbench.save_answer` s'en servent pour vérifier ce qu'ils reçoivent. `artefact_in_bag` est l'ancien nom de `artefact_muffled`, encore lu.
- `QUALITIES` : `A`, `B`, `C`. `SOURCES` : les 14 origines possibles d'un label, c'est-à-dire la méthode qui a proposé la fenêtre (DECISIONS n° 99). Le commentaire explique les moins évidentes : `bulk`, `gap`, `flag`, `external`, `plan`.

#### `comment_fields(comment)` — [labels.py:146](labels.py#L146)

- Quand : à chaque réponse du poste qui porte un commentaire ([`workbench.save_answer`](../annotation/workbench.py#L584), [`workbench.save_span`](../annotation/workbench.py#L636)), et à l'import d'annotations (`service._import_set`).
- Reçoit : le commentaire libre. Renvoie : `{"tags": […], "co_occurring": […]}`, ou `{}` pour un commentaire vide.
- Fait : normalise le texte, cherche les mots de `CONDITION_TAGS` (second plan, lointain, pluie, bruit, point d'interrogation) et les espèces citées. Ne décide jamais du label (docstring).
- `normalize(text)` ([labels.py:58](labels.py#L58)) : minuscules, sans accents ni espaces multiples.
- `find_species(norm)` ([labels.py:129](labels.py#L129)) : les espèces de `SPECIES_RULES` citées, dans l'ordre du texte ; un « Hyalinobatrachium sp. » est retiré si une espèce de ce genre est citée.
- `SpeciesRule`, `SPECIES_RULES` ([labels.py:75](labels.py#L75)) : 21 règles (12 oiseaux, 9 amphibiens), chacune un motif, une famille et un nom. Le commentaire les présente comme les faux amis relevés à l'annotation ; les commentaires y portent trois mentions « [À VÉRIFIER] » (les noms d'oiseaux, « piau », « A. hahneli »).

#### `read_annotation_table(path)` et `detect_columns(df, candidates)` — [labels.py:165](labels.py#L165)

- Quand : dans `blanci clips-import` seulement ([`selection.import_clip_labels`](../annotation/selection.py#L770)), pour relire les réponses d'un outil externe sur des extraits exportés.
- `read_annotation_table` lit un fichier Excel, ou un CSV dont le séparateur est deviné, en UTF-8 puis en cp1252.
- `detect_columns` ([labels.py:188](labels.py#L188)) retrouve quelle colonne du tableau correspond à chaque champ attendu : d'abord les noms exactement égaux, ensuite les libellés composés (« Nom de l'enregistrement » pour `fichier`). Une colonne ne sert qu'à un champ.
- `column_key(name)` ([labels.py:176](labels.py#L176)) : « Début (s) » → `debut`. `_contains_words(column, candidate)` ([labels.py:182](labels.py#L182)) : comparaison par mots entiers.

## `dataset.py` : des annotations aux fenêtres étiquetées

Le plus gros fichier du dossier. Lit : les tables `labels`, `windows`, `spans`, `intervals`, `recordings`, et un stock d'embeddings (`embedded_training_set`). N'écrit rien : les négatifs présumés ne sont jamais rangés dans `labels` (docstring). Utilisé par `service.py`, `evaluation/`, `combination/`, `annotation/`, `heads/baselines.py`, `exploration/`. La docstring en tête du fichier donne les règles en détail ; ce qui suit dit qui fait quoi.

### Lire les annotations

#### `current_labels(con)` — [dataset.py:94](dataset.py#L94)

- Quand : dès qu'on a besoin des labels de fenêtres : `training_set`, `benchmark_recordings`, `benchmark_folds`, `frozen.freeze`, `workbench.annotation_status`, `service.export_annotations`, `explore.py`, le tableau de bord.
- Renvoie : un tableau, le dernier label de chaque fenêtre annotée (une correction est une ligne plus récente), avec l'enregistrement, le décalage, la durée, la qualité, la source et le commentaire.

#### `load_spans(con, annotator=None)` — [dataset.py:124](dataset.py#L124)

- Quand : dès qu'on a besoin de l'annotation par intervalles : mêmes appelants que `current_labels`, plus `qc.positive_recordings` et `evaluation/qc_calibration.labelled_windows`.
- Renvoie : deux tableaux, les extraits écoutés et les intervalles ; au besoin ceux d'un seul annotateur ou d'une liste.
- `annotator_filter(annotator, column="annotator")` ([dataset.py:108](dataset.py#L108)) : la condition SQL correspondante ; `None` dans la liste veut dire « les lignes sans annotateur ». Aussi appelée par trois fonctions de `workbench.py`.

#### `current_intervals(spans, intervals)` — [dataset.py:287](dataset.py#L287)

- Quand : avant d'utiliser des intervalles : `interval_labels`, `_positive_intervals`, `annotated_spans`, `frozen.freeze`, `selection.gap_candidates`, `workbench.annotated_recordings`.
- Renvoie : les intervalles encore valables. Un intervalle est remplacé dès qu'un extrait plus récent le couvre en entier : c'est la correction en ajout seul, comme pour les labels.

### Donner un label à chaque fenêtre de la grille

#### `interval_labels(spans, intervals, grid, min_overlap=0.5, start_s=0.07, end_s=0.2)` — [dataset.py:329](dataset.py#L329)

- Quand : dans `training_set` ; au poste, pour dire ce qui est déjà écouté (`workbench.annotation_status`, `workbench.progress`) ; dans `evaluation/qc_calibration.labelled_windows`.
- Reçoit : les extraits, les intervalles, et une grille de fenêtres (identifiant, enregistrement, décalage, durée), quelle que soit sa durée de fenêtre. Renvoie : un tableau avec le label de chaque fenêtre, `y` (1 ou 0), la qualité, le commentaire.
- Fait, pour chaque fenêtre, dans cet ordre : 1. **positive** (`blanci`, ou `blanci_chorus` si un des intervalles l'est) si elle recouvre assez un intervalle sûr ; 2. sinon `blanci_uncertain` si elle touche un intervalle « A. blanci ? » ; 3. sinon `edge` si elle ne fait qu'effleurer un intervalle ; 4. sinon `false_friend` si elle recouvre assez un intervalle « faux ami » ; 5. sinon le label du reste de l'extrait (`other_label`) si elle est entièrement écoutée ; 6. sinon pas de ligne.
- « Recouvrir assez » : la durée commune atteint le plus petit de trois seuils : la moitié de l'intervalle, la moitié de la fenêtre, et 0,07 s si la fenêtre contient le début de l'intervalle, 0,2 s sinon.
- Pourquoi (commentaire, DECISIONS n° 182, 183, 201) : avec des fenêtres qui se recouvrent de moitié, la règle de la moitié garantit au moins une fenêtre positive par intervalle. Les deux seuils de bord suivent la précision du tracé : le début d'un intervalle est posé sur l'attaque de la note, la fin dans sa décroissance, bien moins nette.
- Détail : la qualité est celle de l'intervalle s'il en porte une, sinon celle de son extrait. Les fenêtres `blanci_uncertain` et `edge` sortent avec `y = 0` : c'est à l'appelant de les écarter (`EXCLUDED_LABELS`, [dataset.py:62](dataset.py#L62)).
- `_within_union(w0, w1, s0, s1)` ([dataset.py:310](dataset.py#L310)) : dit si une fenêtre est entièrement couverte par la réunion des extraits ; deux extraits qui se touchent (4 × 30 s d'un enregistrement) comptent pour une seule plage.

#### `transfer_labels(annotations, grid)` — [dataset.py:492](dataset.py#L492)

- Quand : dans `training_set`, pour les annotations par fenêtres (celles d'avant les intervalles) ; aussi `explore.recording_windows`.
- Reçoit : les annotations (enregistrement, décalage, durée, label) et la grille. Renvoie : les fenêtres de la grille qui héritent d'un label, avec `y`.
- Fait : une fenêtre hérite si elle contient l'annotation (annotation courte de 3 s) ou si elle y est contenue (annotation d'enregistrement entier) ; celles qui ne font que chevaucher sont écartées (DECISIONS n° 4). Une fenêtre qui hérite à la fois d'un positif et d'un négatif est retirée.

### Les négatifs appariés

#### `paired_negatives(grid, recordings, positive_recording_ids, per_positive, …, strategy="nearest", …)` — [dataset.py:515](dataset.py#L515)

- Quand : dans `training_set` quand `per_positive > 0` ; aussi `explore.paired_for_recording`.
- Reçoit : la grille, la table des enregistrements, les enregistrements positifs, le nombre de négatifs voulus par positif, la stratégie. Renvoie : des fenêtres au label `background_presumed`, `y = 0`, avec une colonne `pairing` (`same_recording`, `same_day`, `other_day`).
- Fait, pour chaque enregistrement positif, selon la stratégie (DECISIONS n° 88, 101) : `nearest` prend les fenêtres du même enregistrement les plus proches d'une annotation positive, puis le même jour, puis un autre jour ; `other_day` le même créneau horaire un autre jour ; `same_day` les enregistrements les plus proches du même jour, à au moins `min_gap_min` ; `mixed` moitié autre jour, moitié même jour.
- Détails : toujours le même micro. Avec `nearest`, une fenêtre qui chevauche une annotation positive n'est jamais tirée, ni une fenêtre encadrée d'annotations positives (faux négatif suspect, n° 102). Le tirage au hasard dépend de `seed`. Une fenêtre n'est tirée qu'une fois.
- `_pairing_pools(rec, rid, strategy, …)` ([dataset.py:459](dataset.py#L459)) : pour un positif, les enregistrements du même micro éligibles : ceux d'un autre jour au même créneau, et ceux du même jour rangés par distance.
- `local_minutes(start_utc, utc_offset_h)` ([dataset.py:447](dataset.py#L447)) et `local_days` ([dataset.py:453](dataset.py#L453)) : minute du jour et jour calendaire en heure locale.

#### `pairing_options(cfg)` — [dataset.py:75](dataset.py#L75)

- Quand : chaque fois qu'une fonction de `service.py`, `evaluation/`, `combination/` ou `heads/baselines.py` appelle `embedded_training_set` ou `benchmark_recordings` (16 fonctions).
- Renvoie : les cinq réglages des négatifs appariés lus dans la configuration, à passer tels quels avec `**`.

#### `suspect_false_negatives(windows, grid, positive_windows, radius)` — [dataset.py:640](dataset.py#L640)

- Quand : dans `training_set`, seul appelant.
- Renvoie : pour chaque fenêtre négative, vrai si des annotations positives l'encadrent à moins de `radius` dans le même enregistrement. La fenêtre reste négative ; la colonne `suspect_fn` sert à la proposer à la réécoute (docstring de `training_set`, n° 85 et 102).
- `positive_annotations(labels, exclude_recordings=None)` ([dataset.py:630](dataset.py#L630)) : les annotations positives courantes (enregistrement, décalage, durée), jeu gelé exclu.

### Assembler le jeu

#### `training_set(con, grid, per_positive=0, …, exclude_recordings=None, …)` — [dataset.py:691](dataset.py#L691)

- Quand : dans `embedded_training_set` ; directement dans [`service.evaluate_frozen`](../service.py#L1339) et `heads/baselines.evaluation_windows`.
- Reçoit : la connexion, la grille d'un stock, le nombre de négatifs par positif, les enregistrements à exclure (le jeu gelé). Renvoie : un tableau de fenêtres étiquetées, avec `y`, `presumed`, `suspect_fn`, `row` (le rang dans la grille), `point` (le groupe des folds) et `site`.
- Fait : 1. retire le jeu gelé de la grille, des labels et des extraits ; 2. calcule les labels venus des intervalles et ceux venus des annotations par fenêtres, et les réunit ; 3. met de côté les fenêtres douteuses (`EXCLUDED_LABELS`) pour qu'elles ne soient ni étiquetées ni tirées comme négatifs ; 4. marque les faux négatifs suspects ; 5. tire les négatifs appariés parmi les fenêtres restantes.
- `_merge_labelled(*parts)` ([dataset.py:666](dataset.py#L666)) : la réunion de l'étape 2 ; une fenêtre que les deux sources étiquettent en désaccord est écartée.
- `_positive_intervals(spans, intervals)` ([dataset.py:678](dataset.py#L678)) : les intervalles positifs valables, au même format que `positive_annotations`. Aussi appelée par `qc.positive_recordings`.

#### `embedded_training_set(con, store, window_s, per_positive=0, …)` — [dataset.py:760](dataset.py#L760)

- Quand : avant tout entraînement ou toute évaluation sur un stock : [`service.train_and_register`](../service.py#L135), `evaluate_holdout`, `similarity_search`, `run_clustering`, `compute_tokens`, `upstream_bench`, `evaluation/benchmark.benchmark_encoder`, `evaluation/head_benchmark.benchmark_data`, `combination/stacking.py`, `annotation/selection.py`, `explore.corpus_prototypes` (13 fonctions).
- Reçoit : la connexion, le stock d'embeddings, la durée de fenêtre. Renvoie : `(data, X)` : le tableau de `training_set` avec une colonne `gated`, et les embeddings alignés ligne à ligne.
- Pourquoi (docstring) : c'est le point d'entrée commun du benchmark et de l'entraînement, pour que les deux voient exactement les mêmes labels et les mêmes négatifs appariés.

### Pour le benchmark

#### `benchmark_recordings(con, …, strategy="nearest", …)` et `benchmark_subset(con, cfg)` — [dataset.py:800](dataset.py#L800)

- Quand : `benchmark_subset` ([dataset.py:848](dataset.py#L848)) dans `blanci embed --subset benchmark` et `blanci onsets --subset benchmark` ; `benchmark_recordings` aussi dans `heads/baselines.evaluation_windows`.
- Renvoie : les enregistrements dont le benchmark a besoin : les annotés (`role = labelled`) et les candidats aux négatifs appariés, non écartés par un drapeau (`paired_candidate`).
- Détail : `benchmark_subset` force la stratégie `mixed`, qui réunit les candidats « autre jour » et « même jour ». Pourquoi (docstring) : changer `benchmark.pairing` ne demande alors pas de nouvel encodage.

#### `benchmark_folds(con, n_splits=5, seed=0, exclude_recordings=None)` et `folds_for(con, cfg)` — [dataset.py:855](dataset.py#L855)

- Quand : `folds_for` ([dataset.py:890](dataset.py#L890)) avant toute validation croisée : `service.train_and_register`, `evaluate_holdout`, `upstream_bench`, `evaluation/benchmark.py`, `evaluation/head_benchmark.py`, `combination/`, `heads/baselines.run_baselines`, `heads/detectors/base.evaluate_detector` (9 fonctions).
- Renvoie : un dictionnaire point d'écoute → numéro de fold.
- Pourquoi (docstring, DECISIONS n° 91) : les folds sont calculés sur les enregistrements annotés, jamais sur les fenêtres d'un encodeur. Deux encodeurs, une baseline ou une fusion voient donc exactement les mêmes micros tenus à l'écart dans chaque fold.
- Détail : `folds_for` lit le nombre de folds et la graine dans `cfg["head"]` et exclut le jeu gelé. Le calcul lui-même est dans `evaluation/evaluate.fold_assignment`.

### Tables et export

#### `recordings_table(con)` — [dataset.py:437](dataset.py#L437)

- Quand : dès qu'on a besoin de la table des enregistrements : 25 fonctions hors tests.
- Renvoie : toute la table `recordings`, avec une colonne `point` = `site/micro`. C'est ce `point` qui sert de groupe aux folds.
- `recordings_info(con, recording_ids)` ([dataset.py:425](dataset.py#L425)) : la même chose pour quelques enregistrements seulement. Pourquoi (docstring) : lire toute la table coûtait 0,7 s à chaque export du poste.

#### `annotated_spans(con)`, `add_versions(extracts)`, `reannotations(extracts)` — [dataset.py:151](dataset.py#L151)

- Quand : dans `blanci export-labels` ([`service.export_annotations`](../service.py#L794)) ; les deux premières aussi dans `annotation/verification.verification_sample`.
- `annotated_spans` renvoie deux tableaux lisibles (un extrait par ligne, un intervalle par ligne), avec le fichier, le site, le micro, l'heure, et `valable` faux pour un intervalle remplacé.
- `add_versions` ([dataset.py:212](dataset.py#L212)) repère les versions successives d'un même extrait : même enregistrement, même annotateur (casse ignorée), recouvrement d'au moins la moitié du plus court (`SAME_EXTRACT_OVERLAP`). Ajoute `version`, `n_versions`, `remplace_par`.
- `reannotations` ([dataset.py:247](dataset.py#L247)) met côte à côte la première et la dernière version de chaque extrait réécouté.
- `_last_version(i, replaced_by)` ([dataset.py:206](dataset.py#L206)) : suit la chaîne des remplacements jusqu'à la dernière version.

## `frozen.py` : le jeu gelé

Le jeu de test : une liste d'enregistrements figée, versionnée, qui ne sert jamais à entraîner ni à régler. Lit : les fichiers `jeu_gele_<version>.csv` de `paths.frozen_test`, et la base pour le bilan d'un gel. Écrit : un nouveau fichier `jeu_gele_<version>.csv`, mis en lecture seule. Utilisé par `service.py`, `evaluation/`, `combination/`, `annotation/selection.py`, `heads/baselines.py`, `exploration/`, le tableau de bord.

#### `frozen_recordings(cfg, versions=None)` — [frozen.py:52](frozen.py#L52)

- Quand : avant tout entraînement, benchmark ou sélection, pour exclure le jeu gelé : 18 fonctions, dont `service.train_and_register`, `evaluate_holdout`, `dataset.folds_for`.
- Renvoie : l'ensemble des identifiants gelés, toutes versions confondues par défaut ; une version inconnue lève `ValueError`.
- `frozen_versions(cfg)` ([frozen.py:41](frozen.py#L41)) : `{version: tableau}` de tous les gels, relus du disque à chaque appel. `frozen_directory(cfg)` ([frozen.py:37](frozen.py#L37)) : le dossier `paths.frozen_test`.

#### `freeze(con, cfg, source, version)` — [frozen.py:62](frozen.py#L62)

- Quand : c'est `blanci freeze` ([`cli.freeze`](../cli.py#L996), sous le nom `freeze_recordings`).
- Reçoit : une file CSV qui a une colonne `recording_id`, et un nom de version. Renvoie : le chemin du fichier écrit et un bilan.
- Fait : 1. refuse d'écraser une version existante ; 2. refuse un enregistrement inconnu de l'inventaire ; 3. écrit la liste avec chemin, site, micro, heure et date du gel ; 4. met le fichier en lecture seule.
- Le bilan compte ce qui sort de l'entraînement : labels, labels positifs, extraits, intervalles valables, intervalles positifs, et les enregistrements déjà gelés ailleurs.
- Pourquoi (docstring du fichier) : sans jeu gelé, chaque mesure se fait sur des données qui ont aidé à construire le modèle, directement ou en guidant les choix ; elle est optimiste, de plus en plus à mesure que les tours d'annotation s'enchaînent.

## `__init__.py`

Une docstring qui présente les cinq fichiers, aucun code.

## Écarts relevés

Signalés, pas corrigés.

1. **`__init__.py` décrit `labels` comme « import des annotations reçues ».** `labels.py` ne fait plus cet import (retiré, DECISIONS n° 188) ; sa propre docstring dit : schéma de labels, analyse des commentaires, lecture des tableaux d'un outil externe.
2. **Aucune clé `saturation` ni `rain` calculée.** Le commentaire du schéma (`core/db.py`, colonne `qc_flags`) cite « rain, saturation, muffled, silent, indices ». Le code n'écrit jamais de clé `saturation` ; `rain` n'existe que dans la liste `annotated`, posée à l'écoute. La docstring d'`apply_metadata_flags` parle aussi d'« indices audio déjà calculés (pluie, saturation…) » : il n'y a pas d'indice de pluie parmi les cinq de `qc_indices`. `service._qc_flagged` garde `("saturation", "rain", "muffled")` comme clés par défaut.
3. **`blanci/README.md` ne cite pas `dataset.py`.** Il décrit `inputs/` par « inventaire, contrôle qualité, labels, jeu gelé » ; le plus gros fichier du dossier (896 lignes : jeux d'apprentissage, négatifs appariés, folds) n'y est pas.
4. **`README.md` range `flag` avec l'inventaire.** Le tableau des étapes donne `ingest`, `flag`, `status` pour `inputs/ingest.py` et `qc` pour `inputs/qc.py`. `blanci flag` appelle une fonction d'`ingest.py` (`apply_clock_corrections`) et trois de `qc.py`.
5. **L'étape 1 dépend des étapes suivantes.** `dataset.py` importe `blanci.embedding.store` (étape 2) et `blanci.heads.signal_processing` (étape 3) ; `benchmark_folds` importe `blanci.evaluation.evaluate` (étape 5). `qc.positive_recordings` importe `dataset.py` à l'intérieur de la fonction, `dataset.py` important déjà `qc.py`.
6. **Chiffre daté dans une docstring.** `benchmark_recordings` parle d'éviter de « passer les 29 000 enregistrements dans chaque encodeur » ; `README.md` en compte 96 292 (29 513 pour 2026 seul).
7. **`confirm_flags` et `keep_flags` n'ont pas de commande.** Elles ne sont appelées que par `scripts/son_etouffe/confirmer_drapeaux.py`. Aucune fonction du dossier n'est sans appelant.
