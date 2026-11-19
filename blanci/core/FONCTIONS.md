# `blanci/core/` : le code fonction par fonction

Fiche vérifiée le 10/11/2026 sur le commit `9473508` du dépôt public (`git log -1 --format=%h`), arbre de travail sans modification. Pour savoir si elle est périmée : `git diff --stat 9473508 -- blanci/core` ; une sortie vide veut dire que le code de ce dossier n'a pas bougé.

`core/` est le socle : 6 fichiers, 553 lignes. Aucune commande n'a son code ici, mais toutes y passent : la configuration, la base SQLite, les identifiants, la lecture de l'audio, la recherche des fichiers sur les disques, et la lecture en avance dans des fils à part.

| Fichier | Lignes | Rôle |
|---|---|---|
| [`config.py`](config.py) | 57 | lire `config/default.yaml` et le fichier de l'utilisateur, résoudre les chemins |
| [`db.py`](db.py) | 270 | schéma et migrations de la base, registre des modèles, identifiants des enregistrements et des fenêtres |
| [`locate.py`](locate.py) | 132 | retrouver un enregistrement sur les disques branchés |
| [`audio.py`](audio.py) | 54 | décoder un fichier en mono, rééchantillonner, découper en fenêtres |
| [`ahead.py`](ahead.py) | 33 | préparer les éléments suivants dans des fils à part pendant que le fil principal travaille |
| [`__init__.py`](__init__.py) | 7 | une docstring, aucun code |

## La chaîne

Lecture : `a → b` veut dire « `a` appelle `b` » ; un point-virgule sépare les appels successifs d'une même fonction. Les noms sans préfixe sont dans `core/`, les autres portent leur fichier. Chaque ligne a été remontée depuis `cli.py` en cherchant les appelants dans le code.

- **Toute commande `blanci …`** : `cli.main` → `default_user_config` ; `load_config` → `_merge`. Puis la commande elle-même : `config_path` → `project_path` ; `connect` → `migrate` → `_user_version`. 44 commandes de `cli.py` ouvrent la base ainsi ; `service.py` n'ouvre jamais la base, il reçoit la connexion.
- **`blanci ingest`** : `cli.ingest` → `inputs/ingest.ingest` → `describe_recording` → `recording_id_for` → `recording_key`.
- **`blanci qc`** : `cli.qc_command` → `inputs/qc.check_recordings` → `_read_tracks` → `ahead` → `prepare` (dans les fils) → `variants`.
- **`blanci embed`** : `cli.embed` → `embedding/embed.embed_recordings` → `_register_identity` → `utc_now` ; `_local_disks` → `mounted_disks` → `_local_drive_letters` ; `ahead` → `prepare` (dans le fil de lecture) → `find` → `variants` ; `load_audio` ; `encoder_windows` → `resample` ; `cut_windows`. Puis, dans le fil principal : `encoder.embed` (pour un encodeur bâti sur `BaseEncoder` : `BaseEncoder.embed` → `_prepare` → `resample`) ; `window_id_for` ; `register_encoder` → `utc_now`.
- **`blanci onsets`** : `cli.onsets` → `heads/signal_processing.compute_onsets` → `ahead` → `prepare` → `load_audio` ; `store_onsets` → `utc_now`.
- **`blanci tokens`** : `cli.tokens` → `service.compute_tokens` → `model_params` ; `load_audio` ; `encoder_windows` → `resample` ; `cut_windows`.
- **`blanci train`** : `cli.train` → `service.train_and_register` → `encoder_params` → `model_params` ; `next_version` ; `utc_now` ; `register_model` (deux fois : la tête, puis son seuil).
- **`blanci score`** : `cli.score` → `service.score_and_decide` → `load_head` → `model_params` ; avec `--fusion` : `model_params`, `encoder_params` ; `utc_now`.
- **`blanci retrain`** : `cli.retrain` → `service.retrain` → `train_and_register` (voir `train`) ; `adopt_head` → `register_model`.
- **`blanci annotate`** : `cli.annotate` lance `streamlit run annotation/app.py`, puis `app.main` → `_config_file` → `default_user_config` ; `_setup` → `_cached_config` → `load_config` ; `connect` ; `_recordings_root` → `remembered_roots` → `remembered_file` ; `locate` → `variants` ; `candidate_roots` → `mounted_disks`, `_children`. Si le fichier est introuvable : `_missing_recording` → `variants` ; `remember_root`. Pour les réponses : `window_id_for` (dans `app.main`, `workbench._window_ids`, `workbench.ensure_window`).
- **`blanci package`** : `cli.package` → `annotation/paquet.build_package` → `locate`.
- **`blanci label`** : `cli.label_window` → `service.append_label` → `utc_now`.
- **Scripts** : `scripts/controle_qualite/echantillon_pistes.py`, `scripts/controle_qualite/verifier_doublons.py` et `scripts/son_etouffe/qc_deux_pistes.py` font `main` → `default_user_config` ; `load_config` ; `config_path`, puis cherchent chaque fichier avec `locate` (depuis `main` pour les deux premiers, depuis `measure` pour le troisième). `qc_deux_pistes.measure` passe aussi par `ahead`.

## `config.py` : la configuration

Lit `config/default.yaml`, puis par-dessus un fichier de l'utilisateur (`config/local.yaml` ou celui passé à `--config`). N'écrit rien. Utilisé partout : `cli.py`, `service.py`, le poste d'annotation, les scripts, `anuraset/`, le tableau de bord, les tests. Tests : [`tests/core/test_config.py`](../../tests/core/test_config.py).

#### `PROJECT_ROOT`, `DEFAULT_CONFIG`, `LOCAL_CONFIG` — [config.py:13](config.py#L13)

- Trois constantes calculées à l'import : la racine du projet (deux dossiers au-dessus de `config.py`, donc `Grenouille/`), `config/default.yaml`, `config/local.yaml`.
- `PROJECT_ROOT` sert aussi à `annotation/bandeau.py` ; `DEFAULT_CONFIG` à `annotation/app._setup`, qui regarde sa date de modification pour savoir s'il faut relire la configuration.

#### `default_user_config()` — [config.py:18](config.py#L18)

- Quand : au tout début de chaque commande, appelée par [`cli.main`](../cli.py#L117), par [`app._config_file`](../annotation/app.py#L118) (poste d'annotation), par `evaluation/throughput.main` et par le `main` de six scripts et de deux programmes du tableau de bord (`poste.py`, `spectrogramme.py`).
- Reçoit : rien. Renvoie : le chemin de `config/local.yaml` s'il existe, sinon `None`.
- Détail : dans `cli.main`, l'appel est `load_config(config or default_user_config())`. Un `--config autre.yaml` remplace donc `local.yaml`, il ne s'y ajoute pas : avec `--config`, `local.yaml` n'est pas lu.
- Pourquoi : la docstring dit que la commande et le poste prennent `local.yaml` sans qu'on ait à écrire `--config`.

#### `load_config(path=None)` — [config.py:38](config.py#L38)

- Quand : une fois par commande ([`cli.main`](../cli.py#L117)) ; à chaque clic du poste si un des deux YAML a changé ([`app._cached_config`](../annotation/app.py#L140), qui garde le résultat en cache sinon) ; dans 19 fonctions hors tests en tout, et 32 fonctions de tests.
- Reçoit : le chemin du fichier de l'utilisateur, ou `None`. Renvoie : un dictionnaire, la configuration complète.
- Fait : 1. lit `config/default.yaml` en entier, à chaque appel (pas de cache ici) ; 2. si `path` est donné, lit ce fichier (un fichier vide vaut `{}`) ; 3. si ce fichier a une section `sequential`, elle est renommée `signal_processing`, sauf si une section `signal_processing` y est déjà (elle gagne, `sequential` est jetée) ; 4. fusionne avec `_merge`.
- Pourquoi : `sequential` est l'ancien nom de la section (DECISIONS n° 175), il reste lu dans un fichier d'utilisateur.
- `_merge(base, override)` ([config.py:23](config.py#L23)) : pose `override` sur `base`, section par section et en descendant dans les sous-sections. Une liste ou une valeur simple remplace celle par défaut en entier. Une section écrite vide dans le fichier de l'utilisateur (`qc:` suivi seulement de commentaires, que YAML lit `None`) garde la section par défaut au lieu de la remplacer par `None` (DECISIONS n° 144, 4ᵉ point).

#### `project_path(value)` — [config.py:48](config.py#L48)

- Quand : chaque fois qu'un chemin de la configuration est transformé en vrai chemin ; appelée par `config_path` et directement par 16 autres fonctions ou modules (`cli.export_labels`, `cli.import_labels`, les commandes `anuraset_*`, `annotation/app.py`, `annotation/etiquettes.py`, `annotation/workbench.py`, `evaluation/anuraset.py`, les programmes de `anuraset/`).
- Reçoit : un chemin (texte ou `Path`). Renvoie : le même s'il est absolu, sinon ce chemin posé sous `PROJECT_ROOT`.
- Pourquoi : un chemin relatif comme `data/db/blanci.sqlite` part de la racine du projet, pas du dossier d'où la commande est lancée. Lancée depuis `D:\`, une commande créait une base vide sur le disque externe (DECISIONS n° 144, 3ᵉ point).

#### `config_path(cfg, key)` — [config.py:56](config.py#L56)

- Quand : partout où le code a besoin d'un dossier ou d'un fichier de la section `paths` (`raw`, `db`, `embeddings`, `tokens`, `models`, `frozen_test`, `reports`…). C'est la fonction la plus appelée du dossier : 151 appels, 97 fonctions hors tests, dont 77 appels dans `cli.py`.
- Reçoit : la configuration et une clé de `paths`. Renvoie : `project_path(cfg["paths"][key])`. Une clé absente lève `KeyError`.

## `db.py` : la base SQLite

Définit le schéma de `data/db/blanci.sqlite` (chemin `paths.db`), l'ouvre, la migre, et fournit les identifiants. Écrit elle-même : le fichier de la base et son dossier (`connect`), les tables à la migration, et la table `models` (`register_model`). Tout le reste de la base est écrit par les autres dossiers avec la connexion qu'elle renvoie. Il n'y a pas de fichier de tests pour `db.py` dans `tests/core/` : `next_version` et `register_model` sont testées dans `tests/interfaces/test_service.py`, `recording_id_for` dans `tests/inputs/test_real_format.py`, `window_id_for` dans `tests/annotation/test_workbench.py`, l'ajout seul des labels dans `tests/inputs/test_ingest.py`.

#### `MIGRATIONS` — [db.py:12](db.py#L12)

- Une liste de 5 scripts SQL, un par version du schéma. Le commentaire en tête donne la règle : ne jamais modifier une entrée publiée, en ajouter une.
- Ce que chaque migration crée, et qui écrit ensuite dans la table (d'après une recherche des `INSERT`, `UPDATE` et `DELETE` dans les `.py` hors tests) :

| Table | Migration | Contenu | Écrite par |
|---|---|---|---|
| `recordings` | 1 | un enregistrement : chemin relatif à `paths.raw`, jeu, site, micro, heure UTC, durée, fréquence, canaux, SHA-256, drapeaux de contrôle qualité (JSON) | `inputs/ingest.py` (ajout, mise à jour), `inputs/qc.py` (mise à jour) |
| `windows` | 1 | une fenêtre : enregistrement, décalage, durée | `embedding/embed.py`, `annotation/workbench.py`, `service.py`, `anuraset/encodage/importer_stock.py`, tous en `INSERT OR IGNORE` |
| `labels` | 1 | un label posé sur une fenêtre, avec annotateur, source et date | `service.py` |
| `models` | 1 | le registre : encodeurs, têtes, seuils, fusions, adoptions, avec leurs paramètres en JSON | `core/db.py` (`register_model`), `embedding/embed.py`, `anuraset/encodage/importer_stock.py` |
| `scores` | 1 | un score par (fenêtre, modèle) | `service.py`, `embedding/embed.py` (logits de classes) |
| `decisions` | 1 | la décision par enregistrement pour un triplet (encodeur, tête, seuil) | `service.py` (seule table où le code supprime des lignes) |
| `imports` | 1, supprimée à la 4 | les fichiers d'annotation déjà importés | plus personne (DECISIONS n° 188) |
| `onsets` | 2 | les débuts de notes d'un enregistrement, calculés une fois | `heads/signal_processing.py` |
| `spans`, `intervals` | 3 ; colonne `quality` de `intervals` à la 5 | un extrait écouté, et les intervalles où A. blanci chante dedans (DECISIONS n° 182 et 186) | `service.py` |

- Détail qui ne se voit pas : `labels`, `spans` et `intervals` portent des déclencheurs SQL qui refusent toute modification et toute suppression (`labels are append-only`…). C'est la base elle-même, pas le code Python, qui garantit qu'une annotation n'est jamais écrasée : une correction est une nouvelle ligne.

#### `connect(path)` — [db.py:152](db.py#L152)

- Quand : au début de 44 commandes de `cli.py` (`connect(config_path(cfg, "db"))`), dans le poste d'annotation ([`app._setup`](../annotation/app.py#L129) à chaque rafraîchissement, [`app._export_job`](../annotation/app.py#L750)), dans les programmes de `anuraset/`, et dans 61 fonctions de tests.
- Reçoit : le chemin de la base. Renvoie : une connexion `sqlite3` prête, schéma à jour.
- Fait : 1. crée le dossier de la base s'il manque, et SQLite crée le fichier s'il manque ; 2. ouvre avec une attente du verrou de 30 s ; 3. rend les lignes lisibles par nom de colonne (`row["version"]`) ; 4. active les clés étrangères ; 5. passe le journal en mode WAL ; 6. appelle `migrate`.
- Pourquoi : `embed` et le poste d'annotation écrivent dans la même base ; on attend le verrou d'écriture plutôt que d'échouer au bout des 5 s par défaut (commentaire du code). Le code ne dit pas pourquoi le mode WAL.
- À savoir : même une commande qui ne fait que lire (`status`) passe par l'étape 1. Si `paths.db` pointe au mauvais endroit, une base vide y est créée : c'est le cas décrit au n° 144, que `project_path` évite.

#### `migrate(con)` — [db.py:165](db.py#L165)

- Quand : à chaque `connect`, seul appelant.
- Reçoit : la connexion. Renvoie : rien ; met le schéma à la dernière version.
- Fait : tant que la version de la base (`PRAGMA user_version`) est sous le nombre de migrations, exécute la migration suivante et le nouveau numéro de version dans une seule transaction (`BEGIN IMMEDIATE … COMMIT`). Si SQLite refuse, annule ; si la version a bougé entre-temps (un autre processus a migré), repart de cette version ; sinon l'erreur remonte.
- Pourquoi (docstring) : un échec ne laisse pas de demi-migration, et deux processus qui ouvrent la base en même temps ne migrent pas deux fois.
- `_user_version(con)` ([db.py:181](db.py#L181)) : lit `PRAGMA user_version`.

#### `model_params(con, model_id, kind=None)` — [db.py:185](db.py#L185)

- Quand : pour relire ce qu'on a rangé sur un modèle ; appelée par `encoder_params` et par quatre fonctions de `service.py` : [`load_head`](../service.py#L398) (paramètres de la tête, dont le dossier où elle est sauvée), [`score_and_decide`](../service.py#L438) (paramètres de la fusion), [`compute_tokens`](../service.py#L1595) (réglage `resample` du stock), `activity_curves`.
- Reçoit : l'identifiant du modèle, et au besoin sa sorte (`kind`). Renvoie : le dictionnaire relu de `params_json`.
- Fait : une requête sur `models` ; aucune ligne lève `ValueError` (« inconnu dans la table models »).

#### `encoder_params(con, encoder_id)` — [db.py:199](db.py#L199)

- Quand : dès qu'une étape a besoin de la durée de fenêtre, du pas ou de la dimension d'un stock d'embeddings : 20 fonctions, dans `service.py` (8, dont `train_and_register`, `score_and_decide`, `evaluate_frozen`), `annotation/selection.py`, `combination/`, `evaluation/`, `exploration/explore.py`.
- Reçoit : l'identifiant du stock. Renvoie : les paramètres rangés par `embed`.
- Fait : `model_params` avec `kind="encoder"` ; si l'encodeur est inconnu, l'erreur dit quoi faire : « lancer `blanci embed` ».
- Détail : ces lignes ne sont pas écrites par `register_model` mais par `embedding/embed.py`, avec ses propres requêtes (`_register_identity`, `register_encoder`).

#### `register_model(con, model_id, kind, name, version, params, sha256=None)` — [db.py:207](db.py#L207)

- Quand : à la fin d'un entraînement ou d'une adoption ; appelée par [`service.train_and_register`](../service.py#L135) (sortes `head` puis `threshold`), [`service.train_fusion`](../service.py#L1428) (`fusion`) et [`service.adopt_head`](../service.py#L1655) (`adoption`).
- Reçoit : l'identifiant, la sorte, le nom, la version, les paramètres. Écrit : une ligne de `models`, puis valide (`commit`).
- Détail : si l'identifiant existe déjà, seuls `params_json` et `sha256` sont remplacés ; la sorte, le nom, la version et la date de création restent ceux de la première écriture.

#### `next_version(con, kind, name)` — [db.py:226](db.py#L226)

- Quand : juste avant d'enregistrer une nouvelle tête ; un seul appelant, [`service.train_and_register`](../service.py#L135), avec `kind="head"` et `name` = l'identifiant de l'encodeur. Les têtes sont donc numérotées par encodeur.
- Reçoit : la sorte et le nom. Renvoie : `v1`, `v2`…
- Fait : lit les versions déjà en base pour ce couple, garde celles de la forme `v<nombre>`, renvoie le plus grand numéro plus un. Rien n'est effacé : les versions précédentes restent en base (docstring).

#### `recording_id_for(path)` — [db.py:247](db.py#L247)

- Quand : à l'inventaire, pour chaque fichier ([`inputs/ingest.describe_recording`](../inputs/ingest.py#L233)) ; à l'import d'annotations faites sur une autre machine (`service._import_set`), quand le fichier reçu ne donne pas l'identifiant.
- Reçoit : un chemin. Renvoie : 16 caractères hexadécimaux.
- Fait : `recording_key` réduit le chemin au nom du fichier, sans dossier ni extension, en minuscules ; l'identifiant est le début du SHA-256 de ce nom.
- Pourquoi (docstring de `recording_key`) : `<série>_<AAAAMMJJ>_<HHMMSS>` est unique pour un enregistreur et survit à la copie sur un autre disque, à une réorganisation des dossiers et au passage WAV ↔ FLAC. Conséquence : deux fichiers de même nom dans deux dossiers ont le même identifiant ; c'est l'inventaire qui tranche entre déplacement, doublon et conflit.
- `recording_key(path)` ([db.py:238](db.py#L238)) : la partie « nom du fichier » ci-dessus ; accepte les `\` de Windows. Appelée seulement par `recording_id_for`.

#### `window_id_for(recording_id, offset_s, dur_s=3.0)` — [db.py:257](db.py#L257)

- Quand : chaque fois qu'une fenêtre est créée ou retrouvée : à l'encodage ([`embed.embed_recordings`](../embedding/embed.py#L198)), au poste d'annotation (`app.main`, `workbench._window_ids`, `workbench.ensure_window`), pour les baselines (`heads/baselines.evaluation_windows`), la calibration du contrôle qualité (`evaluation/qc_calibration._span_grid`) et l'exploration (`explore.py`).
- Reçoit : l'identifiant de l'enregistrement, le décalage et la durée en secondes. Renvoie : `<enregistrement>:<décalage>` pour une fenêtre de 3 s, `<enregistrement>:<décalage>/<durée>` pour toute autre durée (deux décimales).
- Pourquoi (DECISIONS n° 65) : sans la durée, une fenêtre de 5 s et une annotation de 3 s au même décalage partageaient une ligne de `windows`, et la seconde héritait de la durée de la première. Les fenêtres de 3 s gardent la forme d'origine pour que les identifiants déjà en base ne changent pas ; c'est le rôle de la constante `LEGACY_WINDOW_S` ([db.py:254](db.py#L254)).

#### `utc_now()` — [db.py:269](db.py#L269)

- Quand : pour dater tout ce qui s'écrit : labels et extraits (`service.append_label`, `append_span`, `append_labels_bulk`), modèles (`register_model`, `embed.register_encoder`), décisions (`service.score_and_decide`), débuts de notes (`signal_processing.store_onsets`), jeu gelé (`inputs/frozen.freeze`).
- Renvoie : l'heure UTC en texte, à la seconde, sous la forme `2026-11-10T14:03:00Z`.

## `locate.py` : retrouver un enregistrement sur les disques

La base garde le chemin de chaque enregistrement relatif à la racine du disque utilisée à l'inventaire. Ce fichier essaie ce chemin sous plusieurs racines jusqu'à trouver le fichier (DECISIONS n° 195). Lit : l'existence de fichiers sur les disques, la liste des disques, `racines_audio.txt`. Écrit : seulement `racines_audio.txt`, à côté de la base (`remember_root`) ; jamais sur les disques des enregistrements. Utilisé par le poste d'annotation, le paquet d'écoute, trois scripts et `documentation/tableau-de-bord/poste.py`. Tests : [`tests/core/test_locate.py`](../../tests/core/test_locate.py).

#### `locate(rel, raw, extra=(), renamed=None)` — [locate.py:119](locate.py#L119)

- Quand : au poste, pour le candidat affiché et pour le suivant, chargé d'avance ([`app.main`](../annotation/app.py#L897)) ; en fabriquant un paquet d'écoute ([`paquet.build_package`](../annotation/paquet.py#L96)) ; dans les scripts de contrôle qualité.
- Reçoit : le chemin relatif de la base, `paths.raw`, les racines collées à la main (`extra`), la table des dossiers renommés (`audio.renamed`). Renvoie : le chemin complet du fichier, ou `None` s'il n'est nulle part.
- Fait : 1. `variants` donne les noms possibles ; 2. pour chaque racine de `candidate_roots`, dans l'ordre, teste chaque nom ; 3. au premier fichier trouvé, sa racine passe en tête de `_FOUND` et le chemin est renvoyé.
- Détail : `_FOUND` ([locate.py:24](locate.py#L24)) est une liste gardée en mémoire le temps du processus. Après un premier succès, un seul test d'existence suffit pour les fichiers suivants du même disque. À l'inverse, un fichier absent de partout fait parcourir tous les disques à chaque appel.
- `candidate_roots(raw, extra)` ([locate.py:90](locate.py#L90)) : donne les racines une à une, sans doublon : celles qui ont déjà marché, `raw`, `extra`, puis les disques branchés, puis les dossiers de premier niveau de chaque disque. Les disques ne sont listés que si les racines connues n'ont pas suffi.
- `_children(root)` ([locate.py:83](locate.py#L83)) : les sous-dossiers d'un disque, triés, sans ceux qui commencent par un point ; une liste vide si le disque est illisible.

#### `variants(rel, renamed=None)` — [locate.py:107](locate.py#L107)

- Quand : chaque fois qu'un fichier audio est cherché ; appelée par `locate`, par [`app._missing_recording`](../annotation/app.py#L873), et par deux étapes qui cherchent sans `locate` : `prepare` dans [`inputs/qc.check_recordings`](../inputs/qc.py#L323) et `find` dans [`embed.embed_recordings`](../embedding/embed.py#L198).
- Reçoit : le chemin relatif et `audio.renamed` (ancien dossier → nouveau dossier). Renvoie : une liste, le chemin d'origine d'abord, puis le même chemin sous chaque nouveau nom dont l'ancien est le début du chemin.
- Pourquoi : les dossiers ont été renommés à la copie sur les disques du stage, et la base garde le chemin d'origine (commentaire de `config/default.yaml`, section `audio.renamed`). Les `\` sont remplacés par des `/` avant la comparaison.

#### `mounted_disks()` — [locate.py:67](locate.py#L67)

- Quand : appelée par `candidate_roots` et par [`embed._local_disks`](../embedding/embed.py#L188), qui dresse la liste des racines une fois au début de l'encodage.
- Renvoie : la liste des disques branchés. Sous Windows : `D:/` à `Z:/`, pour les seules lettres que garde `_local_drive_letters`. Ailleurs : les sous-dossiers de `/media/<utilisateur>`, `/run/media/<utilisateur>` et `/mnt`, précédés de ceux de `/Volumes` sur Mac ; un dossier absent est ignoré.
- `_local_drive_letters()` ([locate.py:54](locate.py#L54)) : demande à Windows les lettres attribuées (`GetLogicalDrives`) et ne garde, de D à Z, que les lecteurs amovibles, fixes ou virtuels en mémoire (`GetDriveTypeW` ∈ `_LOCAL_DRIVE_TYPES` = 2, 3, 6). Pourquoi (commentaire) : tester un fichier sur un partage réseau de l'ONF coûte une demi-seconde, et sonder ces lettres prenait 15 s.

#### `remembered_roots(db)` et `remember_root(db, root)` — [locate.py:32](locate.py#L32)

- Quand : au poste d'annotation seulement. `remembered_roots` à chaque rafraîchissement ([`app._recordings_root`](../annotation/app.py#L867)), pour donner `extra` à `locate`. `remember_root` quand l'annotateur colle un dossier dans le champ « Dossier des enregistrements » et que le fichier cherché s'y trouve ([`app._missing_recording`](../annotation/app.py#L873)).
- Reçoivent : le chemin de la base. `remembered_roots` renvoie la liste des dossiers retenus (vide si le fichier n'existe pas) ; `remember_root` ([locate.py:40](locate.py#L40)) réécrit le fichier avec le nouveau dossier en tête, sans doublon.
- `remembered_file(db)` ([locate.py:27](locate.py#L27)) : le fichier en question, `racines_audio.txt` dans le dossier de la base. Hors git et propre à chaque poste (docstring).

## `audio.py` : décoder, rééchantillonner, découper

Lit des fichiers audio avec `soundfile`. N'écrit rien et ne calcule jamais de spectrogramme : chaque encodeur calcule le sien (docstring). Utilisé par l'encodage, les débuts de notes, les jetons, les baselines et l'exploration. Tests : [`tests/core/test_audio.py`](../../tests/core/test_audio.py) et `tests/inputs/test_flags_and_channel.py`.

#### `load_audio(source, channel="mean")` — [audio.py:17](audio.py#L17)

- Quand : pour lire un enregistrement entier avant un calcul : `prepare` dans [`embed.embed_recordings`](../embedding/embed.py#L198) (encodage), `prepare` dans [`signal_processing.compute_onsets`](../heads/signal_processing.py#L281) (débuts de notes), [`service.compute_tokens`](../service.py#L1595), `exploration/explore.read_recording`, `anuraset/encodage/jetons.py`.
- Reçoit : un chemin ou un fichier déjà en mémoire, et le canal : `0`, `1`… ou `"mean"`. Renvoie : `(signal, fréquence d'échantillonnage)`, le signal en `float32` à une dimension, à la fréquence du fichier.
- Fait : lit tout le fichier ; `"mean"` renvoie la moyenne des canaux ; un fichier mono est renvoyé tel quel, quel que soit `channel` ; sinon le canal demandé, ou `ValueError` s'il n'existe pas.
- Pourquoi (docstring, DECISIONS n° 50) : les Song Meter du projet enregistrent deux micros, à 6 et 18 dB de gain ; les moyenner « mélange deux points d'écoute » et hérite des saturations du micro le plus amplifié. La configuration lit donc le canal 0 (`audio.channel: 0`). Le n° 54 corrige le n° 50 sur un point : l'écart entre les deux micros est inconnu.
- À savoir : la valeur par défaut de la fonction est `"mean"`, mais les commandes passent `cfg["audio"]["channel"]`, qui vaut `0`.

#### `resample(wav, sr, target_sr)` — [audio.py:34](audio.py#L34)

- Quand : pour amener le son à la fréquence qu'attend un calcul : [`embed.encoder_windows`](../embedding/embed.py#L140) (l'enregistrement entier, avant la découpe), [`BaseEncoder._prepare`](../embedding/encoders/base.py#L68) (un lot de fenêtres, si elles ne sont pas déjà à la fréquence de l'encodeur), `heads/baselines.window_features`, `heads/detectors/base.read_evaluation_audio`, `exploration/explore_plots.listen`.
- Reçoit : un signal ou un lot de fenêtres, sa fréquence, la fréquence voulue. Renvoie : le signal rééchantillonné, en `float32`.
- Fait : si les deux fréquences sont égales, rend le signal sans calcul ; sinon `scipy.signal.resample_poly` avec le rapport réduit par le PGCD (48 000 → 32 000 Hz : facteur 2/3), le long du dernier axe. Le filtre anti-repliement est celui de scipy.
- Détail : dans un lot, chaque fenêtre est traitée comme si elle était seule (docstring, vérifié par un test de `test_audio.py`).

#### `cut_windows(wav, sr, windows)` — [audio.py:43](audio.py#L43)

- Quand : juste après la lecture, pour donner à l'encodeur un tableau de fenêtres : [`embed.encoder_windows`](../embedding/embed.py#L140), `embed.embed_recordings` (fenêtres à la fréquence d'origine pour les portes du traitement du signal), `anuraset/encodage/jetons.py`.
- Reçoit : le signal, sa fréquence, une liste de `(décalage, durée)` en secondes. Renvoie : un tableau `(nombre de fenêtres, échantillons par fenêtre)`.
- Fait : vérifie que toutes les fenêtres ont la même durée (sinon `ValueError`), puis copie chaque morceau dans un tableau de zéros : une fenêtre qui dépasse la fin du fichier est complétée par du silence.
- Détail : une liste vide lève la même `ValueError`, avec le message « fenêtres de durées différentes : [] ».

## `ahead.py` : préparer en avance

Une seule fonction. Ne lit et n'écrit rien elle-même : elle fait tourner la fonction qu'on lui donne dans des fils à part. Utilisée par l'encodage, le contrôle qualité, les débuts de notes et `scripts/son_etouffe/qc_deux_pistes.py`. Test : `test_ahead_keeps_the_order_and_raises_in_turn`, rangé dans `tests/embedding/test_embed.py`.

#### `ahead(items, prepare, ahead=2, workers=1)` — [ahead.py:15](ahead.py#L15)

- Quand : dans les boucles qui lisent beaucoup de fichiers. [`embed.embed_recordings`](../embedding/embed.py#L198) l'importe sous le nom `read_ahead` et garde les valeurs par défaut : un fil, deux enregistrements d'avance. [`inputs/qc._read_tracks`](../inputs/qc.py#L436), [`signal_processing.compute_onsets`](../heads/signal_processing.py#L281) et `qc_deux_pistes.measure` passent `2 * workers` d'avance et `workers` fils (`blanci qc` : 4 fils par défaut, donc 8 d'avance).
- Reçoit : les éléments à traiter, la fonction `prepare` à appliquer à chacun, l'avance, le nombre de fils. Renvoie : un itérateur de `(élément, prepare(élément))`, dans l'ordre de départ.
- Fait : 1. soumet chaque élément à un groupe de fils (nommés `blanci-lecture`) ; 2. dès qu'il y a plus de `ahead` éléments en attente, rend le plus ancien, en attendant son résultat s'il n'est pas prêt ; 3. à la fin, rend ceux qui restent ; 4. si la boucle de l'appelant s'arrête avant la fin, annule ce qui n'a pas commencé et attend ce qui tourne.
- Détails : avec `ahead=0`, chaque élément est préparé à son tour. Une exception de `prepare` ne sort pas du fil : elle remonte dans la boucle de l'appelant au moment où son élément devait être rendu. Les `prepare` du projet attrapent eux-mêmes l'erreur de lecture et renvoient `None` ou un résultat marqué en échec.
- Pourquoi (docstring du fichier) : lire un fichier audio et calculer sur le signal avec numpy ou scipy relâche le verrou de Python ; ces étapes tournent donc vraiment pendant que le fil principal fait tourner l'encodeur ou écrit dans la base.

## `__init__.py`

Une docstring qui présente le dossier, aucun code, aucun import : on écrit toujours `from blanci.core.db import connect`, jamais `from blanci.core import connect`.

## Écarts relevés

Signalés, pas corrigés.

1. **`__init__.py` ne cite pas `locate`.** Sa docstring liste `config`, `db`, `audio`, `ahead` ; `locate.py` manque. À l'inverse, ni `README.md` ni `blanci/README.md` ne citent `ahead.py` (ils parlent de réglages, base, lecture de l'audio, recherche sur les disques).
2. **Sortes de modèles.** Le commentaire de la colonne `models.kind` ([db.py:58](db.py#L58)) dit « encoder, head, fusion, threshold ». `service.adopt_head` y range aussi la sorte `adoption`.
3. **Drapeaux de `recordings.qc_flags`.** Le commentaire du schéma ([db.py:25](db.py#L25)) dit « rain, saturation, muffled, silent, indices ». La docstring de `inputs/qc.py` dit que le contrôle ne signale ni la pluie ni la saturation (DECISIONS n° 174). Confirmé dans la [fiche de `inputs/`](../inputs/FONCTIONS.md) : le code n'écrit jamais de clé `saturation`, et `rain` n'existe que dans la liste `annotated` posée à l'écoute ; cette fiche donne le tableau des clés réellement écrites.
4. **`locate` n'est pas utilisée par toutes les étapes qui lisent l'audio.** La docstring de `locate.py` parle de retrouver un enregistrement « sans configuration ». C'est vrai du poste, du paquet d'écoute et des scripts. `blanci qc` essaie les noms de `variants` mais seulement sous `paths.raw`. `blanci embed` a sa propre recherche (`find`, racines listées une fois : son commentaire dit que `locate` reparcourt les disques à chaque fichier absent). `blanci onsets` (`compute_onsets`) et `blanci tokens` (`compute_tokens`) lisent `paths.raw/<chemin d'origine>` sans `variants` ni `locate` : ils ne voient ni les dossiers renommés ni un autre disque.
5. **Lecteurs réseau filtrés deux fois.** `mounted_disks` écarte déjà les lecteurs réseau sous Windows (`_LOCAL_DRIVE_TYPES`). `embed._local_disks` refiltre son résultat pour écarter le type 4 (réseau), ce qui ne retire plus rien.
6. **Docstring de `locate.py` et Linux.** Elle cite `/media/<moi>/*` et `/mnt/*` ; le code essaie aussi `/run/media/<moi>/*`, et essaie ces trois dossiers sur Mac également, après `/Volumes`.
7. **`db.py` renvoie à « §13.3 » et « §13.7 ».** Ces numéros sont ceux de `documentation/old/feuille-de-route-V4.md` (« 13.3 Schéma SQLite », « 13.7 Règles pour l'agent »), un document rangé dans `old/`.
8. **`load_audio` n'est pas le seul lecteur d'audio.** Malgré le rôle annoncé du fichier, le contrôle qualité (`qc.audio_qc`), l'inventaire, le poste (`workbench.py`), le paquet, les baselines, `evaluation/anuraset.py`, `evaluation/qc_calibration.py` et `explore.read_segment` appellent `soundfile` directement (lecture des deux pistes, ou lecture partielle d'un segment).
9. **Fonctions publiques sans appelant extérieur.** `migrate`, `recording_key`, `candidate_roots` et `remembered_file` ne sont appelées que depuis leur propre fichier. Aucune fonction de `core/` n'est sans appelant.
10. **Relevé en passant, à reprendre dans la fiche de `cli.py`.** `blanci/README.md` présente `service.py` comme le seul point d'entrée de la CLI ; `cli.py` appelle pourtant directement `connect`, `inputs/ingest.ingest`, `qc.check_recordings`, `embed.embed_recordings` et `signal_processing.compute_onsets`.
