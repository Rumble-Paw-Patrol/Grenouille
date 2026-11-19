# `blanci/embedding/` : le code fonction par fonction

Fiche vérifiée le 10/11/2026 sur le commit `9473508` du dépôt public (`git log -1 --format=%h`), arbre de travail sans modification dans ce dossier ; mise à jour le 19/11/2026 pour `channels.py` (piste par micro et par période) et ce qu'il change dans `embed.py`. Pour savoir si elle est périmée : `git diff --stat 9473508 -- blanci/embedding` ; une sortie vide veut dire que le code de ce dossier n'a pas bougé.

`embedding/` est l'étape 2 : 6 fichiers, 1 282 lignes, plus le sous-dossier `encoders/` (10 fichiers, 1 129 lignes), qui a sa propre fiche : [`encoders/FONCTIONS.md`](encoders/FONCTIONS.md). Ce dossier choisit les enregistrements à encoder, les découpe en fenêtres, les passe dans un encodeur gelé, range les vecteurs (embeddings) dans des fichiers Parquet, et sait y chercher les fenêtres qui ressemblent à un exemple.

| Fichier | Lignes | Rôle |
|---|---|---|
| [`embed.py`](embed.py) | 551 | l'encodage : enregistrements → fenêtres → encodeur → stock ; identité d'un stock |
| [`channels.py`](channels.py) | 339 | la piste lue pour chaque enregistrement (règles par micro et par période, repli du son étouffé), et le retrait d'un stock de ce qui est encodé sur une autre |
| [`grid.py`](grid.py) | 88 | la grille de fenêtres d'un enregistrement (décalages, pas, chevauchement) |
| [`store.py`](store.py) | 220 | le stock d'embeddings : fichiers Parquet par encodeur, jeu, site et mois |
| [`index.py`](index.py) | 76 | la recherche par similarité cosinus dans un stock |
| [`__init__.py`](__init__.py) | 8 | une docstring, aucun code |

## La chaîne

Lecture : `a → b` veut dire « `a` appelle `b` » ; un point-virgule sépare les appels successifs d'une même fonction. Les noms sans préfixe sont dans `embedding/`, les autres portent leur fichier. Chaque ligne a été remontée depuis `cli.py` et `service.py` en cherchant les appelants dans le code.

- **`blanci embed`** : `cli.embed` → `overlap_from_cfg` ; `select_recordings` → `inputs/qc.is_excluded`, `inputs/qc.positive_recordings` ; avec `--subset benchmark` : `inputs/dataset.benchmark_subset` ; avec `--recordings` : la liste lue dans un CSV ; `cli.upstream_chain` → `signal_processing.upstream_from_cfg` ; `encoders.get_encoder` (voir la fiche d'`encoders/`) ; `cli._echo_grid` → `hop_for_overlap`, `window_grid`, `overlap_of` ; `embed_recordings`.
- **Dans `embed_recordings`, avant la boucle** : `encoders/base.stock_id` → `encoder_id` ; `resample_mode` ; `channels.channel_rules` ; `stock_identity` ; `check_stock_identity` ; `channels.check_stock_channels` → `channel_plan` → `stock_contents`, `recorded_channels`, `legacy_channels`, `expected_channels` ; `record_channels` ; `inputs/qc.positive_recordings` ; `hop_for_overlap` ; `_register_identity` → `_static_params` → `overlap_of` ; `month_of` ; `_local_disks` → `core/locate.mounted_disks`.
- **Pour chaque partition (jeu, site, mois)** : `EmbeddingStore.consolidate` ; `EmbeddingStore.read_meta` ; `core/ahead.ahead` → `prepare` (dans le fil de lecture) → `inputs/qc.parse_flags` ; `find` → `core/locate.variants` ; `channels.recording_channel` → `rule_channel` → `matching_rule` ; `core/audio.load_audio` ; `inputs/qc.qc_indices`, `qc_flags`, `is_excluded` ; `window_grid` ; `encoder_windows` → `core/audio.resample`, `cut_windows`.
- **Pour chaque enregistrement lu, dans le fil principal** : `inputs/qc.merge_audio_flags` ; avec des portes : `signal_processing.load_onsets`, `recording_onsets`, `gate_values`, `Upstream.passes` ; `encoder.embed` ; `core/db.window_id_for` ; `_store_logits` ; la piste lue dans `stock_channels` (`channels.channel_key`) ; tous les 50 enregistrements `_flush` → `EmbeddingStore.append` → `_table`, `_write_atomic`. En fin de partition : `_flush` ; `EmbeddingStore.consolidate`.
- **À la fin du passage** : `inputs/qc.apply_audio_flags` (si un contrôle audio a eu lieu) ; `register_encoder` → `stock_identity`, `_static_params`, `_add_totals`.
- **`blanci pistes`** : `cli.pistes` → `cli._stock_dir` ; `channels.channel_rules` ; `channel_plan` ; avec `--apply` : `drop_recordings` → `drop_from_partition` → `EmbeddingStore.consolidate`.
- **`scripts/encodage/file_pistes.py`** (la file « bonnes pistes ») : `channel_plan`, `expected_channels`, `matching_rule`, `select_recordings`, `window_grid`, sur une base ouverte en lecture seule.
- **`blanci tokens`** : `cli.tokens` → `overlap_from_cfg` ; `encoders.get_encoder` ; `service.compute_tokens` → `encoders/base.stock_id` ; `service.store_for` → `EmbeddingStore` ; `inputs/dataset.embedded_training_set` → `EmbeddingStore.load` → `fragments`, `read` ; `encoder_windows` ; `encoder.embed_tokens`. Les jetons eux-mêmes sont rangés par `heads/attentive.TokenStore`, pas par ce dossier.
- **`blanci qc`** et **`blanci onsets`** : `cli.qc_command` et `cli.onsets` → `select_recordings` seulement (la liste des enregistrements à lire) ; rien d'autre de ce dossier.
- **`blanci check-grid`** : `cli.check_grid` → `max_hop_without_cut` ; `window_grid` ; `containing_windows`.
- **`blanci select --method similarity`** : `cli.select` → `annotation/selection.select_candidates` → `similarity_candidates` → `service.similarity_search` → `store_for` ; `inputs/dataset.embedded_training_set` ; `search` → `l2_normalize` ; `EmbeddingStore.fragments`, `read` ; `gated_mask` ; `_top_k`. `--method negative_mining` : `selection.negative_mining_candidates` → `search`. Le poste d'annotation passe par la même `select_candidates` (`annotation/app.py`).
- **Tout ce qui entraîne, note ou évalue une tête** lit le stock sans passer par `embed.py` : `service.store_for` → `EmbeddingStore`, puis `load` (`inputs/dataset.embedded_training_set`, `service.evaluate_frozen`), `fragments` + `read` (`service.score_and_decide`, `combination/stacking.store_rows`, `heads/regularization/`), `sample` (`service.run_clustering`, `service.run_pca_curve`, `annotation/selection.py`).
- **`blanci anuraset campaign`** et **`anuraset/encodage/encoder.py`** : `select_recordings(con, exclude_flags=())` (aucun drapeau n'écarte un enregistrement d'AnuraSet) ; `encoders.get_encoder` ; `embed_recordings`, sans contrôle audio ni portes.
- **`blanci throughput`** : `evaluation/throughput.measure_by_name` → `encoders.get_encoder` ; `overlap_from_cfg`. Aucun enregistrement n'est lu.
- **Tableau de bord** : `documentation/tableau-de-bord/construire.encodage` → `select_recordings` (le nombre d'enregistrements encodables), et relit les `totals` écrits par `register_encoder`.

## `embed.py` : l'encodage

Passe les enregistrements dans un encodeur et range le résultat. Lit : la table `recordings`, les fichiers audio (sous `paths.raw`, puis sur les disques branchés), la table `onsets` si des portes sont actives, la table `models`. Écrit : le stock Parquet sous `paths.embeddings/<stock>/` ; les tables `windows` (une ligne par fenêtre, `INSERT OR IGNORE`), `models` (une ligne par stock, sorte `encoder`), `scores` (seulement les logits de classes d'un encodeur qui en garde) ; `recordings.qc_flags` quand le contrôle audio se fait au passage. N'écrit jamais sur les disques des enregistrements. Utilisé par `blanci embed`, `blanci anuraset campaign` et `anuraset/encodage/encoder.py`. Tests : [`tests/embedding/test_embed.py`](../../tests/embedding/test_embed.py), [`tests/embedding/test_embed_identity.py`](../../tests/embedding/test_embed_identity.py), [`tests/embedding/test_upstream.py`](../../tests/embedding/test_upstream.py).

#### `select_recordings(con, dataset=None, site=None, peak_hours=None, utc_offset_h=-3, exclude_flags=EXCLUDING_FLAGS)` — [embed.py:86](embed.py#L86)

- Quand : pour dresser la liste des enregistrements d'une commande qui lit l'audio ou tire des candidats : [`cli.embed`](../cli.py#L433), [`cli.qc_command`](../cli.py#L211), [`cli.onsets`](../cli.py#L1024), [`cli.anuraset_campaign`](../cli.py#L1760), `annotation/selection.phenology_candidates`, `annotation/workbench.random_candidates` et `recording_candidates`, `anuraset/encodage/encoder.py`, `documentation/tableau-de-bord/construire.encodage` ; et dans 23 fonctions de tests.
- Reçoit : la connexion et des filtres facultatifs. Renvoie : un tableau pandas, une ligne par enregistrement retenu, toutes les colonnes de `recordings`, trié par chemin.
- Fait : 1. lit toute la table `recordings` ; 2. garde le jeu demandé (égalité stricte) et le site demandé (sans tenir compte de la casse) ; 3. avec `peak_hours`, garde les enregistrements dont l'heure locale (`start_utc` + `utc_offset_h`) tombe dans une des plages `[début, fin[` ; 4. retire ceux qu'un des drapeaux `exclude_flags` écarte (`is_excluded`, qui laisse passer un enregistrement marqué `keep`), sauf ceux où A. blanci a été entendu (`positive_recordings`).
- Détails : `cli.embed` passe `recorder.filename_utc_offset_h` (−3, la Guyane) et, avec `--peak-hours`, `peak_hours_local` (7–9 h et 15–17 h dans `config/default.yaml`). `cli.qc_command` retire `muffled` des drapeaux qui écartent, pour relire les enregistrements déjà jugés étouffés. `exclude_flags=()` ne retire rien (AnuraSet).
- Pourquoi (docstring) : un enregistrement signalé n'est jamais encodé, il ne peut donc devenir ni négatif apparié, ni candidat d'une file de vérification, ni score ; le fichier, lui, n'est pas touché.

#### `embed_recordings(con, encoder, recordings, raw_root, store_root, overlap=0.5, flush_every=50, progress_every=100, channel="mean", qc_thresholds=None, gates=None, ahead=2, resample="recording", renamed=None, rules=None)` — [embed.py:206](embed.py#L206)

- Quand : c'est le cœur de `blanci embed` ([`cli.embed`](../cli.py#L433)) ; aussi [`cli.anuraset_campaign`](../cli.py#L1760) et `anuraset/encodage/encoder.py` ; 32 fonctions de tests.
- Reçoit : la connexion, l'encodeur (rendu par `get_encoder`), le tableau de `select_recordings`, `paths.raw`, `paths.embeddings`, le chevauchement, le canal (`audio.channel`), les seuils du contrôle audio (`cfg["qc"]`, ou `None` pour ne pas contrôler), les portes du traitement du signal en amont, le mode de rééchantillonnage (`encoders.resample`), la table des dossiers renommés (`audio.renamed`), les règles de piste (`audio.channel_rules`). Renvoie : un `EmbedReport`.
- Fait, avant la boucle : 1. calcule le nom du stock : `stock_id(encoder, overlap)`, suivi de `+<étiquette des portes>` si des portes sont actives (sinon `gates` est mis à `None`) ; 2. refuse de continuer si ce stock existe déjà avec d'autres réglages (`check_stock_identity`), ou s'il contient des enregistrements encodés sur une autre piste que celle de la règle (`channels.check_stock_channels`, qui range au passage la piste des enregistrements encodés avant la table `stock_channels`) ; 3. range l'identité du stock dans `models` tout de suite (`_register_identity`) ; 4. dresse une fois la liste des racines où chercher l'audio : `raw_root`, puis les disques de `_local_disks`.
- Fait, pour chaque groupe (jeu, site, mois) : 5. fond dans le fichier de la partition les morceaux laissés par un passage interrompu (`consolidate`) et lit les `recording_id` déjà présents ; ceux-là sont sautés (`skipped`) ; 6. lit les autres en avance dans un fil à part (`core.ahead`, `prepare` ci-dessous) ; 7. un fichier illisible compte dans `errors` et n'arrête pas le passage, sauf si le dernier disque qui a donné un fichier n'existe plus : `RuntimeError` « n'est plus accessible ; relancer » ; 8. si le contrôle audio a été fait au passage, range ses drapeaux (`merge_audio_flags`) et saute l'enregistrement s'il est écarté et sans positif (`qc_excluded`) ; 9. avec des portes, calcule leurs valeurs par fenêtre (débuts de notes relus de la table `onsets`, sinon recalculés sans être rangés) et marque les fenêtres arrêtées ; 10. appelle `encoder.embed` sur les seules fenêtres qui passent ; une fenêtre arrêtée garde un embedding nul ; 11. écrit les fenêtres dans `windows`, les logits éventuels dans `scores` (`_store_logits`), la piste lue dans `stock_channels`, et valide tout de suite ; 12. garde les lignes en mémoire et les écrit dans un morceau du stock tous les `flush_every` enregistrements (`_flush`) ; 13. en fin de groupe, écrit le reste et fond les morceaux (`consolidate`).
- Fait, à la fin : 14. si un contrôle audio a eu lieu, `apply_audio_flags` rejuge le son étouffé par suites sur toute la base ; 15. `register_encoder` range le débit et le cumul.
- Détails qui ne se voient pas : `encode_s` ne compte que le temps passé dans `encoder.embed` (c'est lui qui donne les fenêtres par seconde), `wall_s` tout le passage. La reprise est guidée par le stock, pas par la base : un enregistrement est « fait » s'il a une ligne dans le fichier de sa partition. Le compteur de `flush_every` est celui du passage entier, pas de la partition. La durée de fenêtre est arrondie au centième avant le calcul du pas.
- À savoir : un enregistrement déjà contrôlé (ses drapeaux ont une clé `indices`) n'est pas rejugé ici ; s'il est écarté, c'est `select_recordings` qui l'a retiré avant. Pendant l'encodage, seul `silent` peut écarter un enregistrement fraîchement contrôlé : le candidat `muffled_0` est retiré des drapeaux rangés, le son étouffé se juge sur deux pistes et par suites dans `blanci qc` (docstring, DECISIONS n° 194 et 203).
- Pourquoi : la validation après chaque enregistrement ne garde le verrou d'écriture que le temps des insertions, pour que le poste d'annotation puisse écrire pendant un encodage (commentaire) ; l'identité est rangée avant le premier lot pour qu'un premier passage interrompu ne laisse pas un stock sans identité, qu'une reprise avec d'autres réglages compléterait sans refus (commentaire, DECISIONS n° 143) ; les débuts de notes ne se calculent plus ici (DECISIONS n° 178).
- `find(path)` ([embed.py:277](embed.py#L277)), fonction interne : essaie chaque nom de `core.locate.variants` sous chaque racine ; la racine qui donne un fichier passe en tête de liste et devient `last_root`. Si rien n'est trouvé, rend `raw_root/<chemin d'origine>`, dont la lecture échouera. Pourquoi une recherche à part (commentaire) : `core.locate.locate` reparcourt les disques à chaque fichier absent, ce qui arrête l'encodage quand un disque n'est pas branché.
- `prepare(rec)` ([embed.py:287](embed.py#L287)), fonction interne, tourne dans le fil de lecture : lit le fichier sur le canal de `channels.recording_channel` (règle du micro, puis repli du son étouffé), gardé dans `_Audio.channel` ; une erreur de lecture rend un `_Audio` marqué `failed` ; si le contrôle reste à faire, calcule `qc_flags(qc_indices(...))` sur la piste lue ; si l'enregistrement sera écarté, s'arrête là ; sinon calcule la grille (`window_grid`) et les fenêtres prêtes pour l'encodeur (`encoder_windows`). Elle ne touche ni à la base ni à l'encodeur.
- `_Audio` ([embed.py:121](embed.py#L121)) : ce que `prepare` rend : signal, fréquence, `failed`, drapeaux audio, grille, fenêtres découpées et leur fréquence, piste lue.
- `_local_disks()` ([embed.py:196](embed.py#L196)) : `core.locate.mounted_disks`, moins les lecteurs réseau sous Windows (`GetDriveTypeW` = 4).
- `_flush(store, con, metas, embs, dataset, site, month)` ([embed.py:158](embed.py#L158)) : écrit le tampon dans un morceau de sa partition (`EmbeddingStore.append`), valide la base, vide le tampon ; sans effet si le tampon est vide.
- `_store_logits(con, encoder, encoder_id, window_ids)` ([embed.py:176](embed.py#L176)) : si l'encodeur a une méthode `pop_logits` et garde des classes (`logit_classes` : `birdnet_v3` dans la configuration par défaut), range un score par (fenêtre, classe) sous le modèle `<stock>:logit:<classe>`, en `INSERT OR REPLACE` ; sans effet sinon, ou si le nombre de lignes ne correspond pas à celui des fenêtres.

#### `EmbedReport` — [embed.py:64](embed.py#L64)

- Le compte rendu d'un passage : nom du stock (`encoder_id`), enregistrements encodés, sautés (déjà faits), fenêtres, secondes d'audio, secondes dans l'encodeur, fichiers illisibles, contrôles audio faits et écartés, fenêtres arrêtées par les portes, durée réelle.
- Deux valeurs calculées : `windows_per_s` (fenêtres ÷ temps dans l'encodeur) et `realtime_factor` (secondes d'audio ÷ temps dans l'encodeur), toutes deux `nan` si rien n'a été encodé. `cli.embed` les affiche ; `register_encoder` les range dans `models`.

#### `encoder_windows(encoder, wav, sr, windows, mode)` — [embed.py:148](embed.py#L148)

- Quand : pour découper un enregistrement en fenêtres à donner à l'encodeur : `prepare` et `embed_recordings`, et [`service.compute_tokens`](../service.py#L1595), qui relit le mode rangé avec le stock.
- Reçoit : l'encodeur, le signal, sa fréquence, la grille, le mode. Renvoie : `(tableau de fenêtres, fréquence de ces fenêtres)`.
- Fait : en mode `recording`, si la fréquence du fichier n'est pas celle de l'encodeur, rééchantillonne l'enregistrement entier puis découpe ; sinon découpe à la fréquence d'origine (l'encodeur rééchantillonnera chaque fenêtre, voir `BaseEncoder._prepare`).

#### `resample_mode(encoder, mode="recording")` — [embed.py:137](embed.py#L137)

- Quand : une fois au début de `embed_recordings`, seul appelant.
- Reçoit : l'encodeur et la valeur de `encoders.resample`. Renvoie : `"recording"` ou `"window"` ; une autre valeur lève `ValueError`.
- Fait : un encodeur enveloppé (il a un attribut `inner` : passe-bas, transformations en amont) reste toujours à `"window"`, quel que soit le réglage.
- Pourquoi (docstring, DECISIONS n° 176) : en mode `recording`, chaque seconde n'est convertie qu'une fois et le filtre voit le vrai son de part et d'autre de chaque fenêtre ; un encodeur enveloppé transforme le son à sa fréquence d'origine, il faut donc lui donner des fenêtres non rééchantillonnées. `RESAMPLE_MODES` ([embed.py:134](embed.py#L134)) liste les deux valeurs admises.

#### `stock_identity(encoder, channel, resample="window", gates=None, rules=None)` — [embed.py:411](embed.py#L411)

- Quand : au début de `embed_recordings` (pour le contrôle) et dans `register_encoder` (pour le rangement).
- Renvoie : un dictionnaire des réglages qui ne sont pas dans le nom du stock mais qui changent les embeddings : `channel`, `muffled_0_channel` (toujours 1), `channel_rules` (les règles de piste, sous la forme de `channels.channel_rules`), `resample`, `window_s` ; et, s'ils existent : `checkpoint` (de l'encodeur ou de celui qu'il enveloppe), `openvino_precision` (si l'encodeur garde des réglages `openvino`), `transforms` (transformations en amont, avec tous leurs réglages), `gate_signal` (la section `signal` de la configuration, si des portes sont actives).
- Détail : le dictionnaire passe par un aller-retour JSON pour ressembler à ce qui sera relu de la base (listes au lieu de tuples) ; sans cela la comparaison de `check_stock_identity` échouerait à tort.

#### `check_stock_identity(con, eid, identity)` — [embed.py:449](embed.py#L449)

- Quand : au début de `embed_recordings`, avant toute lecture d'audio.
- Fait : relit `params_json` du stock dans `models`. Stock inconnu : rien. Sinon, pour chaque clé de l'identité du passage en cours, compare à la valeur rangée ; une différence lève `ValueError` (« les embeddings ne se comparent pas ; reprendre avec le même réglage, ou encoder sous un autre nom »).
- Détails : une clé absente de la base prend la valeur de `LEGACY_IDENTITY` ([embed.py:405](embed.py#L405)) : `resample` = `window`, `openvino_precision` = `f32` (les stocks rangés avant que ces réglages n'existent) ; une clé absente de la base et de `LEGACY_IDENTITY` n'est pas comparée. Seules les clés du passage en cours sont parcourues : une clé rangée que le nouveau passage n'a pas n'est pas comparée non plus. Les réglages de piste (`PER_RECORDING_KEYS` ([embed.py:408](embed.py#L408)) : `muffled_0_channel`, `channel_rules`) ne sont pas comparés ici : une règle changée ne compte que pour les enregistrements qu'elle touche, et c'est `channels.check_stock_channels` qui les juge un par un.
- Pourquoi (DECISIONS n° 143) : reprendre `embed` avec un autre canal, un autre checkpoint ou d'autres transformations ajoutait au même stock des embeddings qui ne se comparent pas.

#### `register_encoder(con, encoder, hop_s, report, channel="mean", gates=None, resample="window", rules=None)` — [embed.py:528](embed.py#L528)

- Quand : à la fin de `embed_recordings`, seul appelant.
- Écrit : la ligne du stock dans `models` (sorte `encoder`), puis valide. `params_json` contient les paramètres fixes (`_static_params`), `last_run` (le compte rendu du passage, avec `windows_per_s` et `realtime_factor`) et `totals`.
- Détail : si la ligne existe (c'est toujours le cas, `_register_identity` l'a créée), seul `params_json` est remplacé ; le nom, la version et la date de création restent ceux du premier passage. Cette fonction écrit dans `models` avec sa propre requête, sans passer par `core/db.register_model`.
- `_static_params(encoder, hop_s, identity, gates)` ([embed.py:472](embed.py#L472)) : `sample_rate`, `window_s`, `hop_s`, `overlap` (4 décimales), l'identité, `dim`, `has_tokens`, et `gates` (`thresholds` et `combine`, ou `None`). Ce sont ces valeurs que `core/db.encoder_params` rend aux autres étapes.
- `_register_identity(con, encoder, hop_s, identity, gates, eid)` ([embed.py:488](embed.py#L488)) : crée la ligne du stock avec les paramètres fixes si elle n'existe pas (`ON CONFLICT DO NOTHING`).
- `_add_totals(con, report)` ([embed.py:512](embed.py#L512)) : ajoute ce passage au cumul déjà rangé : durée réelle, enregistrements, fenêtres, nombre de passages. Pourquoi (docstring) : `last_run` ne garde que le dernier passage, et un encodage du corpus se fait en plusieurs ; le cumul donne le temps d'encodage affiché par le tableau de bord.

#### `month_of(start_utc)` — [embed.py:116](embed.py#L116)

- Quand : pour ranger un enregistrement dans sa partition : `embed_recordings`, et `exploration/explore.recording_embeddings`, qui retrouve ainsi le fichier d'un enregistrement.
- Reçoit : l'heure UTC en texte. Renvoie : `AAAAMM` (les 7 premiers caractères sans le tiret), ou `unknown` si l'heure manque.

## `channels.py` : la piste lue

La piste par défaut est `audio.channel` (0 : le micro réglé à 6 dB). Une règle de `audio.channel_rules` la change pour un micro (et, s'ils sont dits, un jeu, un site, une période) ; puis le repli du son étouffé prend l'autre piste si celle-ci est étouffée et pas l'autre (DECISIONS n° 203). Chaque stock range la piste lue de chaque enregistrement (table `stock_channels`, migration 6 de `core/db.py`) ; `embed` refuse un stock qui contient des enregistrements encodés sur une autre piste que celle de la règle, et `blanci pistes` les retire.

#### `channel_rules(raw)` — [channels.py:47](channels.py#L47)

- Quand : au début de `embed_recordings`, dans `stock_identity`, dans `cli.pistes` et `file_pistes.py`.
- Reçoit : la liste `audio.channel_rules` (ou `None`). Renvoie : les règles sous une forme fixe : `mic_id` en liste triée, `dataset` et `site` en texte, `from` et `to` en UTC au format de `start_utc` (`_utc` ([channels.py:39](channels.py#L39)) : une heure sans fuseau est lue en UTC), `reason` retirée. Une clé inconnue, une piste autre que 0 ou 1, un `from` qui ne précède pas `to` lèvent `ValueError`.
- Pourquoi : c'est cette forme qui entre dans l'identité du stock ; la fonction est idempotente (une règle relue de la base garde sa forme), et un commentaire changé (`reason`) ne change pas l'identité.

#### `matching_rule(rec, rules)` — [channels.py:77](channels.py#L77)

- Renvoie : le rang de la première règle qui s'applique à l'enregistrement, `None` sinon. Une règle s'applique si `mic_id` est dans sa liste, si `dataset` et `site` (quand la règle les dit) sont ceux de l'enregistrement, et si le début est dans [`from`, `to`[ ; un enregistrement sans date n'est pas touché par une règle datée.
- `rule_channel(rec, rules, default)` ([channels.py:95](channels.py#L95)) : la piste de cette règle, ou `default`.

#### `recording_channel(flags, channel, rec=None, rules=None, fallback=True)` — [channels.py:103](channels.py#L103)

- Quand : dans `prepare` (`embed_recordings`), pour chaque enregistrement ; via `expected_channels` et `legacy_channels`.
- Renvoie : la piste de la règle (sinon `channel`), puis, avec `fallback`, l'autre piste si celle-ci est étouffée seule : piste 0 étouffée et pas la 1 → `MUFFLED_0_CHANNEL` ([channels.py:35](channels.py#L35)), c'est-à-dire 1 ; piste 1 choisie, étouffée et pas la 0 → 0. Avec `channel="mean"`, seul le premier cas joue (comme avant les règles).
- Pourquoi : un enregistrement dont seule la piste lue est étouffée reste dans le corpus et s'encode sur l'autre (n° 203) ; un micro sourd (Kaw D SMA14079, 2024) ou réglé à 18 dB sur la piste 0 (21 points de Patawa Est et RNRT) se lit sur la piste 1.

#### `channel_key(channel, channels=2)` — [channels.py:122](channels.py#L122)

- Renvoie : la piste telle que rangée dans `stock_channels` : « 0 », « 1 » ou « mean » ; « 0 » pour un fichier mono (`recordings.channels` = 1), qui n'a qu'une piste quel que soit le canal demandé.

#### `expected_channels(recordings, channel, rules)` et `legacy_channels(recordings, stored, channel)` — [channels.py:145](channels.py#L145), [channels.py:152](channels.py#L152)

- Renvoient : une piste par enregistrement (index : `recording_id`), passée par `channel_key`. `expected_channels` : la piste de la règle d'aujourd'hui, repli compris. `legacy_channels` : la piste qu'a lue un enregistrement encodé avant la table `stock_channels`, d'après les réglages rangés avec son stock : son canal, ses règles (aucune avant leur création), le repli du son étouffé seulement si `muffled_0_channel` est rangé (depuis le n° 203).
- Limite : les drapeaux sont ceux d'aujourd'hui ; un `muffled_0` posé après l'encodage d'un enregistrement ancien n'est pas vu. Une fois sa piste rangée (`check_stock_channels`), un drapeau changé se juge contre elle.
- `_channels_of(recordings, channel, rules, fallback)` ([channels.py:132](channels.py#L132)) : le calcul commun.

#### `stock_contents(store)` — [channels.py:169](channels.py#L169)

- Renvoie : les enregistrements du stock et leurs fenêtres, partition par partition (morceaux d'un passage interrompu compris) : `recording_id`, `dataset`, `site`, `month`, `windows`. Ne lit que la colonne `recording_id`.

#### `recorded_channels(con, encoder_id)` et `record_channels(con, encoder_id, channels, replace=True)` — [channels.py:185](channels.py#L185), [channels.py:197](channels.py#L197)

- Lisent et écrivent la table `stock_channels`. `recorded_channels` rend une série vide si la table n'existe pas (base ouverte en lecture seule d'avant la migration 6, comme dans `file_pistes.py`). `replace=False` garde une piste déjà rangée.

#### `channel_plan(con, store, channel, rules, contents=None)` — [channels.py:216](channels.py#L216)

- Quand : dans `check_stock_channels`, `cli.pistes` et `file_pistes.py`.
- Renvoie : une ligne par (enregistrement, partition) du stock : `encoded` (la piste rangée, sinon `legacy_channels`), `expected` (`expected_channels`), `recorded`, `stale` (les deux diffèrent), avec le micro, la date et les fenêtres. Un enregistrement du stock absent de `recordings` est laissé de côté.
- `_stored_identity(con, encoder_id)` ([channels.py:209](channels.py#L209)) : `params_json` du stock, `{}` s'il n'est pas rangé.

#### `check_stock_channels(con, store, channel, rules, command="blanci pistes")` — [channels.py:251](channels.py#L251)

- Quand : au début de `embed_recordings`, après `check_stock_identity`.
- Fait : `ValueError` si le plan a des lignes `stale` (le message dit combien, de quelle piste à quelle piste, combien de fenêtres, et la commande à lancer) ; sinon range la piste des enregistrements encodés avant la table (`record_channels`, sans remplacer). Rend le plan.
- Pourquoi : la reprise saute un enregistrement déjà présent dans le stock ; sans ce refus, un stock garderait des embeddings d'une piste et en recevrait de l'autre.

#### `drop_recordings(con, store, plan)` — [channels.py:323](channels.py#L323)

- Quand : `blanci pistes --apply`, sur les lignes `stale` du plan.
- Fait : pour chaque partition, `drop_from_partition` ; puis efface la piste rangée de ces enregistrements dans `stock_channels`. Rend, par partition, les enregistrements et fenêtres retirés.
- Ce qui n'est pas touché : la table `windows`. Ses lignes ne dépendent pas de la piste, servent aux autres stocks de même grille et portent les labels ; la reprise de `embed` les réinsère à l'identique (`INSERT OR IGNORE`). Les scores déjà calculés sur ces fenêtres (table `scores`) ne sont pas touchés non plus.
- `drop_from_partition(store, dataset, site, month, recording_ids)` ([channels.py:285](channels.py#L285)) : fond d'abord les morceaux (`consolidate`), recopie la partition groupe de lignes par groupe de lignes sans les fenêtres retirées dans un fichier temporaire qui la remplace d'un coup (interrompue, la partition reste entière) ; une partition vidée est supprimée. Rend le nombre de fenêtres retirées.

## `grid.py` : la grille de fenêtres

Des calculs seulement : ne lit et n'écrit rien. Une fenêtre est un couple `(décalage, durée)` en secondes (`Window`, [grid.py:11](grid.py#L11)) ; les décalages sont arrondis au centième, la précision de `window_id`. Utilisé par l'encodage, le poste d'annotation, les baselines, l'exploration, la calibration du contrôle qualité. Tests : [`tests/embedding/test_grid.py`](../../tests/embedding/test_grid.py).

#### `window_grid(duration_s, window_s, hop_s, tol_s=0.05)` — [grid.py:50](grid.py#L50)

- Quand : chaque fois qu'il faut les fenêtres d'un enregistrement : à l'encodage (`prepare`, `embed_recordings`), et dans 9 fonctions hors de ce dossier : [`cli.check_grid`](../cli.py#L399), [`cli._echo_grid`](../cli.py#L714), `service.estimate_prevalence`, `annotation/selection.phenology_candidates`, `annotation/workbench.random_candidates`, `evaluation/qc_calibration._span_grid`, `exploration/explore.recording_windows` et `paired_for_recording`, `heads/baselines.evaluation_windows`.
- Reçoit : la durée de l'enregistrement, la durée de fenêtre, le pas. Renvoie : la liste des fenêtres.
- Fait : 1. refuse une durée ou un pas nuls ou négatifs, un pas plus grand que la fenêtre (« la grille aurait des trous »), un pas qui n'est pas un multiple de 0,01 s ; 2. un enregistrement plus court qu'une fenêtre (à `tol_s` près) donne une seule fenêtre à 0 s ; 3. sinon une fenêtre tous les `hop_s`, tant qu'elle ne dépasse pas la fin de plus de `tol_s` (5 centièmes : l'encodeur complétera par des zéros) ; 4. si la fin n'est pas couverte, une dernière fenêtre est collée à la fin (décalage arrondi au centième inférieur).
- Exemple vérifié sur la formule : 120 s, fenêtre de 5 s, pas de 2,5 s → 47 fenêtres.
- À savoir : l'encodage prend la durée de fenêtre de l'encodeur et le pas de `encoders.overlap` ; les autres appelants (poste, baselines, calibration, `check-grid`) prennent `window_s` et `hop_s` dans la section `grids` de la configuration (`w3` : 3 s / 1,5 s ; `w5` : 5 s / 2,5 s).

#### `hop_for_overlap(window_s, overlap)` — [grid.py:17](grid.py#L17)

- Quand : pour passer du chevauchement demandé au pas : `embed_recordings`, [`cli._echo_grid`](../cli.py#L714), `exploration/explore.recording_windows`.
- Renvoie : `fenêtre × (1 − chevauchement)`, arrondi au centième, jamais moins de 0,01 s. Un chevauchement négatif ou au-dessus de `MAX_OVERLAP` (0,99, [grid.py:14](grid.py#L14)) lève `ValueError`.
- Détail (docstring) : à cause de l'arrondi, le chevauchement réel peut différer un peu du demandé ; à 0 % une note à cheval sur deux fenêtres est coupée, à 99 % il y a 50 fois plus de fenêtres qu'à 50 %.

#### `overlap_of(window_s, hop_s)` — [grid.py:31](grid.py#L31)

- Quand : pour afficher ou ranger le chevauchement réel : [`cli._echo_grid`](../cli.py#L714) et `embed._static_params`.
- Renvoie : `1 − pas ÷ fenêtre`.

#### `overlap_from_cfg(cfg)` — [grid.py:36](grid.py#L36)

- Quand : quand la commande n'a pas reçu `--overlap` : [`cli.embed`](../cli.py#L433), [`cli.tokens`](../cli.py#L1529), [`cli.anuraset_campaign`](../cli.py#L1760), `evaluation/throughput.measure_by_name`, `anuraset/encodage/encoder.py`.
- Renvoie : `1 − encoders.grid_hop_ratio` si cet ancien réglage est fixé ; sinon `encoders.overlap` ; sinon `DEFAULT_OVERLAP` (0,5, [grid.py:13](grid.py#L13)).
- Pourquoi (docstring) : `encoders.overlap` est toujours défini dans `config/default.yaml` ; sans cette priorité, un `grid_hop_ratio` mis dans une configuration locale ne serait jamais lu.

#### `containing_windows(start_s, end_s, windows, eps=1e-6)` — [grid.py:75](grid.py#L75)

- Quand : dans [`cli.check_grid`](../cli.py#L399) seulement.
- Renvoie : les indices des fenêtres qui contiennent en entier l'intervalle `[start_s, end_s]`.

#### `max_hop_without_cut(window_s, note_s)` — [grid.py:86](grid.py#L86)

- Quand : dans [`cli.check_grid`](../cli.py#L399) seulement.
- Renvoie : `fenêtre − durée de la note` : le plus grand pas pour lequel une note de cette durée tient toujours entière dans au moins une fenêtre.

## `store.py` : le stock d'embeddings

Un stock est un dossier `paths.embeddings/<stock>/<jeu>/<site>/<AAAAMM>.parquet` : un fichier Parquet par partition. Colonnes : `window_id`, `recording_id`, `offset_s`, `emb` (liste de taille fixe, en `float16`), et `gated` (booléen) dans un stock encodé avec des portes. Lit et écrit seulement ces fichiers ; ne touche pas à la base. Écrit par `embed.py` ; lu par presque toutes les étapes suivantes. Tests : dans [`tests/embedding/test_embed.py`](../../tests/embedding/test_embed.py) et [`tests/embedding/test_index.py`](../../tests/embedding/test_index.py) ; il n'y a pas de fichier `test_store.py`.

#### `EmbeddingStore(root, encoder_id)` — [store.py:73](store.py#L73)

- Quand : construit par `embed.embed_recordings`, par [`service.store_for`](../service.py#L99) (dont se servent les fonctions de `service.py`), `annotation/selection._store`, `combination/ensemble.concat_training_set`, `evaluation/anuraset._encoder_windows`, `evaluation/benchmark.benchmark_encoder`, `evaluation/head_benchmark.py`, `exploration/explore.py`, `anuraset/encodage/jetons.py`, `anuraset/calcul/rassembler_global.py` ; et dans 26 fonctions de tests.
- Reçoit : `paths.embeddings` et le nom du stock. Ne lit rien à la construction : c'est une simple paire de valeurs.
- `directory` ([store.py:78](store.py#L78)) : le dossier du stock. `partition_path(dataset, site, month)` ([store.py:81](store.py#L81)) : le fichier d'une partition ; hors tests, appelée seulement depuis ce fichier.

#### `EmbeddingStore.append(meta, emb, dataset, site, month)` — [store.py:104](store.py#L104)

- Quand : pendant l'encodage, par `embed._flush`, tous les 50 enregistrements.
- Reçoit : le tableau des fenêtres et leurs embeddings, alignés ligne à ligne (sinon `ValueError`). Écrit : un nouveau morceau `<AAAAMM>.part-<n>.parquet` à côté du fichier de la partition, `n` sur 6 chiffres, un de plus que le dernier morceau présent. Renvoie son chemin.
- Pourquoi (docstring) : `write` relit et réécrit toute la partition : 42 s et 8 Go de mémoire pour ajouter 50 enregistrements à une partition de 0,9 Go (Mataroni, janvier 2026, à mi-encodage), contre 0,2 s ici.
- `part_paths(dataset, site, month)` ([store.py:99](store.py#L99)) : les morceaux d'une partition, triés par nom, donc dans l'ordre d'écriture.
- `_table(meta, emb)` ([store.py:45](store.py#L45)) : assemble le tableau Arrow, les embeddings en liste de taille fixe `float16`.
- `_write_atomic(table, path)` ([store.py:52](store.py#L52)) : écrit dans `<nom>.parquet.tmp` puis renomme, pour ne jamais laisser un fichier à moitié écrit ; crée le dossier au besoin. `WRITE_OPTIONS` ([store.py:29](store.py#L29)) : encodage par dictionnaire pour les deux colonnes d'identifiants seulement ; sur les embeddings il ralentit l'écriture d'un tiers et la lecture d'un facteur 4 sans rien gagner en taille (commentaire).

#### `EmbeddingStore.consolidate(dataset, site, month)` — [store.py:127](store.py#L127)

- Quand : dans `embed_recordings`, au début de chaque partition (morceaux d'un passage interrompu) et à sa fin.
- Fait : sans morceau, rend le chemin de la partition sans rien faire. Sinon lit les `window_id` du fichier et des morceaux : s'ils sont tous distincts et que les fichiers ont le même schéma, recopie les groupes de lignes l'un après l'autre dans un fichier temporaire, sans charger la partition en mémoire ; sinon charge tout et fusionne avec `_merged` (la dernière écriture d'une fenêtre l'emporte). Le fichier est remplacé d'un coup, puis les morceaux sont supprimés.
- Pourquoi (docstring) : interrompue, la fonte se refait à l'identique, puisque les morceaux ne sont supprimés qu'après le remplacement.
- `_merged(old, new)` ([store.py:60](store.py#L60)) : `old` suivi de `new`, sans les fenêtres de `old` reprises dans `new` ; remet la colonne `gated` en booléen (une fenêtre sans valeur vaut `False`).

#### `EmbeddingStore.read_meta(path)` — [store.py:161](store.py#L161)

- Quand : dans `embed_recordings`, pour savoir quels enregistrements d'une partition sont déjà faits.
- Renvoie : les trois colonnes de `META_COLUMNS` ([store.py:23](store.py#L23)), sans charger les embeddings (ni la colonne `gated`).

#### `EmbeddingStore.read(path)` — [store.py:165](store.py#L165)

- Quand : pour lire un fichier du stock : `load`, `sample`, `write`, `consolidate`, [`index.search`](index.py#L28), [`service.score_and_decide`](../service.py#L438), `combination/stacking.store_rows`, `heads/regularization/selection.sample_pool`, `heads/regularization/windows.store_domain_statistics`.
- Renvoie : `(tableau des colonnes sauf emb, tableau numpy des embeddings)`, en `float16`, une ligne par fenêtre.

#### `EmbeddingStore.fragments(filters=None)` — [store.py:172](store.py#L172)

- Quand : pour parcourir le stock fichier par fichier sans tout charger : `load`, `sample`, `index.search`, `service.score_and_decide`, `combination/stacking.store_rows`, `exploration/explore.recording_embeddings`, `heads/regularization/`.
- Reçoit : des filtres `{dataset, site, month}`, chacun une valeur ou une liste ; une autre clé lève `ValueError`. Rend, un à un et triés, les fichiers `*.parquet` des partitions retenues.
- Détail : un morceau `.part-<n>` est rendu comme sa partition (`PART_MARK`, [store.py:26](store.py#L26)). Un stock en cours d'encodage se lit donc entier, morceaux compris.
- `_as_set(value)` ([store.py:39](store.py#L39)) : une valeur ou une liste → un ensemble ; `None` reste `None` (pas de filtre).

#### `EmbeddingStore.load(filters=None)` — [store.py:186](store.py#L186)

- Quand : pour charger tout un stock, ou une partie, en mémoire : [`inputs/dataset.embedded_training_set`](../inputs/dataset.py#L760), `service.evaluate_frozen`, `combination/ensemble.concat_training_set`, `evaluation/anuraset._encoder_windows`, `anuraset/encodage/jetons.py`, `anuraset/calcul/rassembler_global.py`.
- Renvoie : `(tableau, embeddings)` de tous les fichiers retenus, mis bout à bout ; un tableau vide et une matrice `(0, 0)` si rien n'est retenu.

#### `EmbeddingStore.sample(n, filters=None, seed=0)` — [store.py:195](store.py#L195)

- Quand : pour le clustering, l'ACP et les candidats tirés d'un grand stock : `service.run_clustering`, `service.run_pca_curve`, `annotation/selection.coverage_candidates`, `cluster_candidates`, `embedding_map`.
- Renvoie : `n` fenêtres tirées au hasard (ou tout le stock s'il en a moins), avec leurs embeddings.
- Fait : lit le nombre de lignes de chaque fichier dans son en-tête Parquet ; répartit `n` au prorata des tailles (partie entière, puis le reste distribué au hasard, les gros fichiers ayant plus de chances) ; lit un fichier à la fois et y tire sa part sans remise.
- Pourquoi (docstring) : une partition à la fois en mémoire, pour que l'échantillon d'un stock de plusieurs millions de fenêtres tienne dans la mémoire du portable.

#### `EmbeddingStore.write(meta, emb, dataset, site, month)` — [store.py:84](store.py#L84)

- Quand : dans les tests seulement (ils fabriquent ainsi leurs stocks) ; `embed.py` passe par `append` et `consolidate`.
- Fait : ajoute des fenêtres au fichier de la partition ; s'il existe, le relit en entier, remplace les fenêtres déjà présentes (`_merged`) et réécrit le tout.

#### `gated_mask(meta)` — [store.py:32](store.py#L32)

- Quand : chaque fois qu'on lit un stock et qu'il faut écarter les fenêtres sans embedding : 14 fonctions, dans `service.py` (5 : `score_and_decide`, `run_clustering`, `run_pca_curve`, `evaluate_frozen`, `fused_scores`), `annotation/selection.py` (3), `combination/stacking.py` (2), `index.search`, `heads/regularization/` (2), `inputs/dataset.embedded_training_set`.
- Reçoit : le tableau d'un stock. Renvoie : un tableau de booléens, vrai pour une fenêtre arrêtée par une porte ; tout à faux si le stock n'a pas de colonne `gated` (`GATED`, [store.py:24](store.py#L24)).
- Pourquoi (docstring du fichier) : une fenêtre arrêtée a un embedding nul, n'est jamais apprise et reçoit le score le plus bas.

## `index.py` : la recherche par similarité

Cherche dans un stock les fenêtres les plus proches d'exemples donnés, en parcourant tout le stock (pas d'index construit d'avance). Lit : les fichiers du stock. N'écrit rien. Utilisé par deux méthodes de sélection de candidats ; `l2_normalize` sert bien au-delà. Tests : [`tests/embedding/test_index.py`](../../tests/embedding/test_index.py).

#### `search(queries, store, k, filters=None, negatives=None, chunk_rows=200_000)` — [index.py:28](index.py#L28)

- Quand : [`service.similarity_search`](../service.py#L1022) (requêtes = embeddings des positifs annotés, négatifs = négatifs appariés) et `annotation/selection.negative_mining_candidates`.
- Reçoit : les embeddings des exemples cherchés, le stock, le nombre de résultats, des filtres de partition, et au besoin des exemples négatifs. Renvoie : un tableau des `k` meilleures fenêtres, colonnes `rank`, `window_id`, `recording_id`, `offset_s`, `score`, `sim_pos`, `sim_neg`, `query`.
- Fait : 1. normalise requêtes et négatifs ; 2. pour chaque fichier du stock, par tranches de 200 000 lignes : similarité cosinus de chaque fenêtre à chaque requête ; `sim_pos` = la plus grande, `query` = le numéro de la requête qui la donne ; `sim_neg` = la plus grande similarité aux négatifs (0 sans négatifs) ; `score` = `sim_pos − sim_neg` ; 3. une fenêtre arrêtée par une porte reçoit `−inf` ; 4. garde les `k` meilleures de la tranche ; 5. à la fin, trie tout, garde `k` lignes, numérote le rang à partir de 1.
- Détail : si la partie du stock retenue a moins de `k` fenêtres ouvertes, des fenêtres arrêtées peuvent sortir, avec un score de `−inf`. Un stock vide rend un tableau vide avec les mêmes colonnes.
- `_top_k(scores, k)` ([index.py:22](index.py#L22)) : les indices des `k` plus grands scores, sans ordre ; tous les indices si `k` dépasse le nombre de scores.

#### `l2_normalize(x, eps=1e-12)` — [index.py:15](index.py#L15)

- Quand : avant tout calcul de similarité cosinus. Appelée par `search`, et par 16 fonctions hors de ce dossier : `heads/head.py` (6), `exploration/explore.py` (3), `annotation/selection.py` (2), `heads/cluster.py` (2), `combination/ensemble.concat_training_set`, `evaluation/benchmark.knn_top1`, `heads/regularization/assembly.Regularizer.window_transform`.
- Reçoit : un vecteur ou une matrice. Renvoie : une matrice `float32` dont chaque ligne est de norme 1 ; un vecteur seul devient une matrice d'une ligne ; une ligne nulle reste nulle (division par `eps`).

## `__init__.py`

Une docstring qui liste `grid`, `encoders/`, `embed`, `store`, `index`. Aucun code, aucun import.

## Écarts relevés

Signalés, pas corrigés.

1. **Docstring du fichier `embed.py` et son étouffé.** Elle dit qu'un enregistrement « silencieux ou au son étouffé est alors écarté avant d'être encodé », et l'aide de `cli.embed` le répète. La docstring de `embed_recordings` et le code disent l'inverse : pendant l'encodage, seul `silent` écarte ; le candidat `muffled_0` est retiré des drapeaux rangés, et le son étouffé n'est jugé qu'en fin de passage par `apply_audio_flags`, après que les enregistrements ont été encodés.
2. **Lecteurs réseau filtrés deux fois.** `_local_disks` retire les lecteurs de type 4 (réseau) de la liste de `core/locate.mounted_disks`, qui ne garde déjà que les types 2, 3 et 6 sous Windows : ce second filtre ne retire plus rien. Déjà relevé dans la fiche de `core/` (n° 5).
3. **`embed_recordings` a sa propre recherche de fichiers.** `find` n'appelle pas `core/locate.locate` : elle ne voit ni les racines collées au poste d'annotation (`racines_audio.txt`), ni les dossiers de premier niveau des disques, que `locate` essaie. Le commentaire donne la raison (vitesse quand un disque manque). Déjà relevé dans la fiche de `core/` (n° 4).
4. **`anuraset/encodage/encoder.py` ne passe pas `renamed`.** `cli.embed` et `cli.anuraset_campaign` passent `audio.renamed` à `embed_recordings` ; ce script ne le passe pas. À reprendre dans la fiche d'`anuraset/encodage/`.
5. **Valeurs par défaut différentes d'une fonction à l'autre.** `embed_recordings` a `resample="recording"` et `channel="mean"` ; `stock_identity` et `register_encoder` ont `resample="window"`. Sans effet aujourd'hui : `embed_recordings` leur passe toujours ses propres valeurs, et les commandes passent `audio.channel` (0) et `encoders.resample`.
6. **`EmbeddingStore.write` n'a plus d'appelant hors tests.** Depuis `append` et `consolidate`, l'encodage n'y passe plus ; les tests s'en servent pour fabriquer leurs stocks (19 appels dans 9 fichiers de `tests/`). `partition_path`, `part_paths` et `read_meta` ne sont appelées que depuis `store.py` et `embed.py`.
7. **Fonctions publiques à un seul usage.** `containing_windows` et `max_hop_without_cut` ne servent qu'à `blanci check-grid` ; `resample_mode`, `stock_identity`, `check_stock_identity` et `register_encoder` ne sont appelées que depuis `embed.py` (et ses tests). Aucune fonction de ce dossier n'est sans appelant.
8. **Renvois à d'anciens numéros de section.** `embed.py`, `grid.py` et `index.py` renvoient à « §4 », `store.py` à « §13.3 », `sample` à « §5 bis » : ce sont les sections du cadre repris en tête de `DECISIONS.md` (« §4. Architecture », « 13.3 Schéma SQLite », « §5 bis »), pas des numéros de décision.
9. **Le README dit que l'encodage produit aussi les jetons.** `README.md` range `embed` et `tokens` sous `embedding/`. Le code de `blanci tokens` est dans `service.compute_tokens`, et les jetons sont rangés par `heads/attentive.TokenStore` ; de ce dossier, `tokens` n'utilise que `encoder_windows`, `stock_id` et le stock d'embeddings. `service.compute_tokens` lit de plus l'audio sous `paths.raw/<chemin d'origine>`, sans `variants` ni recherche sur les disques (déjà relevé dans la fiche de `core/`, n° 4).
10. **Grille de l'encodage et grilles de la configuration.** `blanci check-grid` vérifie les grilles de la section `grids` (`w3`, `w5`), pas la grille réellement encodée, qui vient de la durée de fenêtre de l'encodeur et de `encoders.overlap`. Les deux coïncident pour un encodeur à fenêtre de 3 ou 5 s et un chevauchement de 50 %.
