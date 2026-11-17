# `blanci/embedding/encoders/` : le code fonction par fonction

Fiche vérifiée le 10/11/2026 sur le commit `9473508` du dépôt public (`git log -1 --format=%h`), arbre de travail sans modification dans ce dossier. Pour savoir si elle est périmée : `git diff --stat 9473508 -- blanci/embedding/encoders` ; une sortie vide veut dire que le code de ce dossier n'a pas bougé.

`encoders/` contient les encodeurs : 10 fichiers, 1 129 lignes. Un encodeur est un modèle pré-entraîné et gelé qui reçoit un lot de fenêtres de son et rend un vecteur par fenêtre. Ce dossier ne lit aucun enregistrement et n'écrit rien dans la base : il charge des modèles (bacpipe, AVEX, ONNX) et les fait tourner. Le reste de l'étape 2 est décrit dans la fiche du dossier parent : [`../FONCTIONS.md`](../FONCTIONS.md).

| Fichier | Lignes | Rôle |
|---|---|---|
| [`__init__.py`](__init__.py) | 76 | `get_encoder` : du nom de la configuration à l'encodeur prêt |
| [`base.py`](base.py) | 94 | le contrat d'un encodeur, le nom d'un stock, la base commune (lots, rééchantillonnage) |
| [`bacpipe_encoder.py`](bacpipe_encoder.py) | 337 | les encodeurs chargés par la bibliothèque bacpipe (perch_v2, birdmae, beats, birdnet…) |
| [`avex_encoder.py`](avex_encoder.py) | 128 | les encodeurs esp-aves2 chargés par la bibliothèque AVEX |
| [`onnx_encoder.py`](onnx_encoder.py) | 90 | l'encodeur du livrable : un paquet `manifest.json` + `model.onnx` |
| [`export.py`](export.py) | 75 | export d'un modèle torch en ONNX, quantification, test d'équivalence |
| [`pruned.py`](pruned.py) | 50 | graphe ONNX réduit aux sorties utiles |
| [`openvino_session.py`](openvino_session.py) | 194 | calcul par OpenVINO sur le processeur graphique intégré |
| [`lowpass.py`](lowpass.py) | 45 | enveloppe : filtre passe-bas avant l'encodeur |
| [`upstream.py`](upstream.py) | 40 | enveloppe : transformations du traitement du signal avant l'encodeur |

## La chaîne

Lecture : `a → b` veut dire « `a` appelle `b` » ; un point-virgule sépare les appels successifs d'une même fonction. Les noms sans préfixe sont dans `encoders/`. Chaque ligne a été remontée depuis `cli.py` et `service.py` en cherchant les appelants dans le code.

- **Qui demande un encodeur** : `cli.embed`, `cli.tokens`, `cli.anuraset_campaign`, `evaluation/throughput.measure_by_name` (commande `blanci throughput`), `anuraset/encodage/encoder.py`, `anuraset/encodage/jetons.main`. Tous passent par `get_encoder` → `_get_encoder` ; `_with_transforms`.
- **Chargement d'un encodeur bacpipe** : `_get_encoder` → `BacpipeEncoder.__init__` → `_default_device` ; `_load_bacpipe_model` → `_birdmae_with` (si un `checkpoint` est donné) ; `_light_session` ; `_resolve_classes` ; `_raw` (un essai sur une fenêtre de zéros, pour connaître la dimension). Puis `LowpassEncoder` si l'entrée a `lowpass_hz`.
- **`_light_session`, modèle ONNX (perch_v2)** : `pruned.pruned_model` ; avec le réglage `openvino` : `openvino_session.openvino_session` → `OpenVinoSession` ; puis `pruned.PrunedSession`.
- **`_light_session`, modèle torch avec le réglage `openvino`** : `_openvino_backbone` → `openvino_session.openvino_backbone` → `OpenVinoBackbone` → `weights_digest`, `_convert`.
- **Chargement d'un encodeur AVEX** : `_get_encoder` → `AvexEncoder.__init__` → `_load_fairseq_eat` → `remap_fairseq_eat` (famille EAT seulement) ; `_features` ; `_grid`.
- **Chargement d'un paquet ONNX** : `_get_encoder` → `OnnxEncoder.__init__` → `EncoderManifest.read` ; `file_sha256`.
- **Transformations en amont** : `_with_transforms` → `signal_processing.upstream_from_cfg` ; `UpstreamEncoder`.
- **Pendant `blanci embed`** : `embed.embed_recordings` → `stock_id` → `encoder_id` ; puis, par enregistrement, `encoder.embed` : `BaseEncoder.embed` → `_prepare` → `core/audio.resample` ; `_forward` par lot (`BacpipeEncoder._forward` → `_raw` ; `AvexEncoder._forward` → `_features`, `_grid` ; `OnnxEncoder._forward`). Pour un encodeur enveloppé : `UpstreamEncoder.embed` → `signal_processing.Upstream.transform`, ou `LowpassEncoder.embed` → `lowpass`, puis l'`embed` de l'encodeur enveloppé. Ensuite `embed._store_logits` → `pop_logits`.
- **Pendant `blanci tokens`** (et `anuraset/encodage/jetons.py`) : `service.compute_tokens` → `encoder.embed_tokens` : `BaseEncoder.embed_tokens` → `_prepare` ; `_forward_tokens` par lot (`BacpipeEncoder._forward_tokens` → `_model_tokens` ou `_raw` ; `AvexEncoder._forward_tokens` ; `OnnxEncoder._forward_tokens`).
- **`blanci throughput`** : `evaluation/throughput.measure_encoder` → `encoder.embed` sur du bruit.
- **`export.py`** : aucune commande n'y passe (voir les écarts).

## `__init__.py` : obtenir un encodeur

Lit la section `encoders` de la configuration et `paths.models`. Charge le modèle demandé (ce qui peut télécharger ses poids, voir `bacpipe_encoder.py`). N'écrit rien lui-même. Réexporte `BaseEncoder`, `Encoder`, `encoder_id`, `stock_id` de `base.py` (`__all__`, [\_\_init\_\_.py:10](__init__.py#L10)). Tests : `test_upstream.py` pour `_with_transforms` ; `get_encoder` elle-même n'est appelée par aucun test.

#### `get_encoder(name, cfg, upstream=None)` — [\_\_init\_\_.py:13](__init__.py#L13)

- Quand : au début de toute commande qui encode : [`cli.embed`](../../cli.py#L433), [`cli.tokens`](../../cli.py#L1529), [`cli.anuraset_campaign`](../../cli.py#L1657), `evaluation/throughput.measure_by_name`, `anuraset/encodage/encoder.py`, `anuraset/encodage/jetons.main`.
- Reçoit : le nom donné à `--encoder`, la configuration, et au besoin la chaîne du traitement du signal en amont (`signal_processing.Upstream`, option `--upstream`). Renvoie : un objet qui respecte le contrat `Encoder`.
- Fait : `_get_encoder` charge le modèle, puis `_with_transforms` l'enveloppe si des transformations sont demandées.
- `_get_encoder(name, cfg)` ([\_\_init\_\_.py:42](__init__.py#L42)) : cherche `name` dans `encoders.models`. `backend: bacpipe` → `BacpipeEncoder` (poids sous `paths.models/bacpipe`), enveloppé d'un `LowpassEncoder` si l'entrée a `lowpass_hz`. `backend: avex` → `AvexEncoder`, fenêtre de `window_s` (5 s par défaut). Sinon, si `paths.models/encoder/<name>/manifest.json` existe → `OnnxEncoder`. Sinon `ValueError` (« encodeur inconnu »). La taille de lot est celle de l'entrée (`batch_size`), à défaut `encoders.batch_size`. Les modules des encodeurs ne sont importés qu'ici, au moment de s'en servir.
- `_with_transforms(encoder, name, cfg, upstream)` ([\_\_init\_\_.py:25](__init__.py#L25)) : réunit les transformations de l'entrée (`transforms`) et celles de `upstream` ; sans aucune, rend l'encodeur tel quel ; sinon complète les réglages manquants par ceux de la configuration (`upstream_from_cfg`) et rend un `UpstreamEncoder`. Les portes de `upstream` ne passent pas par ici : c'est `embed_recordings` qui les applique.
- À savoir : pour un encodeur bacpipe, le nom de la configuration n'est transmis que si l'entrée a un `checkpoint` (`name=name if spec.get("checkpoint") else None`). Sans `checkpoint`, l'encodeur, et donc son stock, porte le nom du module bacpipe (`model`), pas celui de l'entrée. Dans `config/default.yaml`, les deux noms sont identiques pour toutes les entrées sans `checkpoint`, sauf `birdmae_lp8k`, dont le nom est refait par `LowpassEncoder`.
- Pourquoi : le lot propre au modèle existe parce que le meilleur lot dépend du moteur (commentaire, DECISIONS n° 172).

## `base.py` : le contrat et la base commune

Ne lit et n'écrit rien. Définit ce qu'est un encodeur pour le reste du projet, le nom d'un stock, et la classe dont héritent les trois encodeurs « réels ». Tests : [`tests/embedding/test_encoders_base.py`](../../../tests/embedding/test_encoders_base.py).

#### `Encoder` — [base.py:18](base.py#L18)

- Le contrat (un `Protocol` vérifiable à l'exécution) : six attributs, `name`, `version`, `sample_rate`, `window_s`, `dim`, `has_tokens`, et deux méthodes, `embed(wav, sr)` → tableau `(fenêtres, dim)` et `embed_tokens(wav, sr)` → tableau `(fenêtres, jetons, dim)` ou `None`.
- Sert d'annotation de type dans `embed.py`, `export.py`, `lowpass.py`, `upstream.py`. `LowpassEncoder` et `UpstreamEncoder` le respectent sans hériter de `BaseEncoder`.

#### `stock_id(encoder, overlap=0.5)` — [base.py:38](base.py#L38)

- Quand : pour nommer le stock d'embeddings : `embed.embed_recordings` et [`service.compute_tokens`](../../service.py#L1595).
- Renvoie : `<nom>-<version>` à 50 % de chevauchement ; sinon `<nom>-<version>@o<pourcentage>` (`birdmae-bacpipe1.3.5@o75`).
- Pourquoi (docstring) : deux chevauchements donnent deux grilles ; un stock par grille, jamais de mélange. Ce nom est aussi celui de tout ce qui dépend du stock : têtes, scores, décisions.
- `encoder_id(encoder)` ([base.py:34](base.py#L34)) : `<nom>-<version>`. Appelée seulement par `stock_id` (et un test).

#### `BaseEncoder` — [base.py:51](base.py#L51)

- Quand : classe mère de `BacpipeEncoder`, `AvexEncoder`, `OnnxEncoder`, et des faux encodeurs des tests (`ToyEncoder` de `tests/conftest.py`). Une sous-classe n'écrit que `_forward(lot)` (lot à la fréquence du modèle → embeddings) et, si elle a des jetons, `_forward_tokens(lot)`.
- `embed(wav, sr)` ([base.py:79](base.py#L79)) : prépare le lot (`_prepare`), le passe à `_forward` par paquets de `batch_size` (64 par défaut), met bout à bout, rend du `float32` ; un lot vide rend un tableau `(0, dim)`.
- `embed_tokens(wav, sr)` ([base.py:85](base.py#L85)) : `None` si `has_tokens` est faux ; sinon même découpe en paquets, avec `_forward_tokens`.
- `_prepare(wav, sr)` ([base.py:68](base.py#L68)) : met le son en `float32` à deux dimensions (une fenêtre seule devient un lot d'une ligne) ; rééchantillonne si `sr` n'est pas la fréquence du modèle ; puis force la longueur exacte `window_s × sample_rate` : une fenêtre trop longue est tronquée, une trop courte complétée par des zéros (écart d'arrondi du rééchantillonnage, commentaire).

## `bacpipe_encoder.py` : les encodeurs de recherche

Charge un modèle par la bibliothèque bacpipe et l'adapte au contrat. Lit : les poids sous `paths.models/bacpipe` (téléchargés par `bacpipe.ensure_models_exist` s'ils manquent ; birdmae passe par le cache Hugging Face, d'après la docstring). Écrit : ces poids, et un cache sous `paths.models/bacpipe/.cache/` (graphes ONNX élagués, graphes OpenVINO). bacpipe n'est installé qu'avec `uv sync --group research`. Tests : [`tests/embedding/test_bacpipe_encoder.py`](../../../tests/embedding/test_bacpipe_encoder.py), [`tests/embedding/test_openvino_session.py`](../../../tests/embedding/test_openvino_session.py).

#### `BacpipeEncoder(model_name, batch_size=64, device=None, model_base_path="data/models/bacpipe", logit_classes=None, checkpoint=None, name=None, openvino=None)` — [bacpipe_encoder.py:134](bacpipe_encoder.py#L134)

- Quand : construit par `_get_encoder` pour toute entrée `backend: bacpipe` de la configuration.
- Fait, à la construction (`__init__`, [bacpipe_encoder.py:140](bacpipe_encoder.py#L140)) : 1. vérifie que bacpipe est installé ; 2. `name` = nom donné, sinon nom du module ; `version` = `bacpipe<version installée>` ; 3. charge le modèle (`_load_bacpipe_model`) ; 4. allège ou déplace le calcul si possible (`_light_session`) ; 5. lit la fréquence et la durée de fenêtre dans les constantes du module bacpipe (`SAMPLE_RATE`, `LENGTH_IN_SAMPLES`) ; 6. retrouve les numéros des classes à garder (`_resolve_classes`) ; 7. fait tourner le modèle sur une fenêtre de zéros pour connaître `dim`, et décide de `has_tokens` : vrai si la sortie est une suite de jetons, si le modèle a laissé des jetons spatiaux (perch_v2), ou s'il est dans `TOKEN_FREQ_BANDS`.
- `_forward(batch)` ([bacpipe_encoder.py:299](bacpipe_encoder.py#L299)) : la sortie de `_raw` ; si c'est une suite de jetons, leur moyenne.
- `_forward_tokens(batch)` ([bacpipe_encoder.py:305](bacpipe_encoder.py#L305)) : pour un modèle de `TOKEN_FREQ_BANDS`, `_model_tokens` ; sinon la sortie de `_raw` si c'est une suite de jetons ; sinon les jetons spatiaux gardés au dernier appel (perch_v2), ou `None`.
- `_raw(batch)` ([bacpipe_encoder.py:275](bacpipe_encoder.py#L275)) : `model.preprocess` puis `model(...)`, sans gradient ; garde au passage les logits des classes demandées (perch_v2 : `results["logits"]` ; sinon `model.predictions`, les probabilités de birdnet_v3) et les jetons spatiaux de perch_v2 (`results["spatial_embeddings"]`, grille temps × fréquence gardée) ; rend `(lot, dim)` ou `(lot, jetons, dim)`.
- `_model_tokens(batch)` ([bacpipe_encoder.py:313](bacpipe_encoder.py#L313)) : jetons en grille `(lot, temps, fréquence, dim)`. birdmae : dernière couche cachée sans le jeton de classe ; beats et naturebeats : la moyenne que fait bacpipe est coupée le temps de l'appel (`avg_pooling`), puis rétablie ; audioprotopnet : la carte de la dernière couche, axes remis dans l'ordre temps, fréquence. Un nombre de jetons qui n'est pas un multiple du nombre de bandes lève `ValueError`.
- `_light_session(openvino, model_base_path, keep_logits, samples)` ([bacpipe_encoder.py:184](bacpipe_encoder.py#L184)) : si le modèle tourne sur une session ONNX Runtime (perch_v2), la remplace par celle d'un graphe sans les sorties inutiles (`spectrogram`, et `label` si aucun logit n'est gardé) ; avec le réglage `openvino` et le périphérique présent, ce graphe est calculé par OpenVINO et la taille de lot devient celle du réglage ; sinon par ONNX Runtime avec les mêmes fournisseurs. Si le modèle n'a pas de session ONNX et que `openvino` est demandé : `_openvino_backbone`. Sans effet sinon. Affiche les sorties gardées et le moteur.
- `_openvino_backbone(openvino, model_base_path, samples)` ([bacpipe_encoder.py:227](bacpipe_encoder.py#L227)) : pour les seuls modèles de `OPENVINO_TORCH` (sinon `ValueError`), remplace le réseau torch (`model.model`) par sa version OpenVINO ; le spectrogramme reste celui de bacpipe. Si OpenVINO ou le périphérique manque, garde torch et note la précision `f32` dans `self.openvino`.
- `_resolve_classes(names)` ([bacpipe_encoder.py:253](bacpipe_encoder.py#L253)) : la position de chaque classe de `logit_classes` dans `model.classes` ; une classe inconnue lève `ValueError`.
- Pourquoi : graphe élagué, DECISIONS n° 180 ; OpenVINO, n° 179 (perch_v2) et n° 208 (AudioProtoPNet : de ~0,7 à ~7 fenêtres/s sur l'Iris Xe en `float16`) ; si OpenVINO manque, la précision rangée avec le stock redevient `f32`, et un stock encodé en `f16` refuse alors la reprise au lieu de mélanger (docstring).

#### `BacpipeEncoder.pop_logits()` — [bacpipe_encoder.py:263](bacpipe_encoder.py#L263)

- Quand : après chaque `embed`, par `embed._store_logits` (directement, ou à travers `UpstreamEncoder.pop_logits`).
- Renvoie : `None` si aucune classe n'est gardée ; sinon un tableau `(fenêtres, classes)` des logits accumulés depuis le dernier appel, puis vide la réserve.
- Pourquoi (docstring du fichier) : les logits sortent du même calcul que l'embedding, sans seconde inférence.

#### `TOKEN_FREQ_BANDS` et `OPENVINO_TORCH` — [bacpipe_encoder.py:109](bacpipe_encoder.py#L109)

- `TOKEN_FREQ_BANDS` : les modèles dont le code sait lire les jetons, avec leur nombre de bandes de fréquence : `birdmae`, `beats`, `naturebeats` (8 chacun), `audioprotopnet` (`None` : il rend une carte, pas une suite de jetons).
- `OPENVINO_TORCH` ([bacpipe_encoder.py:113](bacpipe_encoder.py#L113)) : `audioprotopnet` et `convnext_birdset`, les modèles torch dont le réseau peut passer par OpenVINO.

#### Fonctions privées du fichier

- `_load_bacpipe_model(model_name, device, model_base_path, checkpoint=None)` ([bacpipe_encoder.py:68](bacpipe_encoder.py#L68)) : importe le module `bacpipe.model_pipelines.feature_extractors.<nom>` (absent : `RuntimeError` « bacpipe est-il installé ») ; télécharge les poids si le modèle est dans `NEEDS_CHECKPOINT` ; construit le modèle en lui passant tous les réglages de `bacpipe.settings`, avec le périphérique, le dossier des poids et `run_pretrained_classifier` à faux ; appelle `prepare_inference`. Un `checkpoint` pour un autre modèle que birdmae lève `ValueError`.
- `_birdmae_with(module, checkpoint)` ([bacpipe_encoder.py:42](bacpipe_encoder.py#L42)) : une classe `Model` de birdmae qui charge le `checkpoint` donné au lieu de Bird-MAE-Huge. L'extracteur de spectrogramme est toujours celui de `DBD-research-group/Bird-MAE-Base`, quel que soit le checkpoint. Pourquoi (docstring, DECISIONS n° 72) : bacpipe 1.3.5 impose la version Huge, 0,5 fenêtre/s sur l'i5.
- `_default_device()` ([bacpipe_encoder.py:116](bacpipe_encoder.py#L116)) : `mps` si torch le propose (Mac), sinon `cpu`.
- `_to_numpy(out)` ([bacpipe_encoder.py:125](bacpipe_encoder.py#L125)) : tenseur torch ou TensorFlow → numpy `float32`.

## `avex_encoder.py` : les encodeurs esp-aves2

Charge un modèle par la bibliothèque AVEX (`avex.load_model`). Lit : les poids, téléchargés par AVEX ou par `huggingface_hub`. N'écrit rien d'autre. Référence de benchmark seulement : les points de contrôle sont sous licence CC-BY-NC-SA-4.0, non déployables (docstring). Tests : [`tests/embedding/test_avex_encoder.py`](../../../tests/embedding/test_avex_encoder.py).

#### `AvexEncoder(model_name, batch_size=64, window_s=5.0, device="cpu", name=None)` — [avex_encoder.py:69](avex_encoder.py#L69)

- Quand : construit par `_get_encoder` pour une entrée `backend: avex` (dix entrées `esp_aves2_*` dans `config/default.yaml`).
- Fait, à la construction : lit la famille du modèle (`efficientnet`, `beats`, `eat…`) et sa fréquence dans `avex.get_model_spec` ; charge le modèle avec `return_features_only=True` ; pour la famille EAT, recharge les poids si le point de contrôle est au format fairseq (`_load_fairseq_eat`) ; fait tourner une fenêtre de zéros pour connaître `dim` ; `has_tokens` est toujours vrai ; `version` = `avex<version installée>`.
- `_grid(feats)` ([avex_encoder.py:109](avex_encoder.py#L109)) : rend `(embedding, jetons en grille)`. EfficientNet : moyenne de la carte, et la carte remise en temps × fréquence. EAT : l'embedding est le jeton de classe ; les jetons sont rangés en temps × 8 fréquences, et seules les lignes de temps de la fenêtre sont gardées (le modèle complète son spectrogramme à 10 s). BEATs : moyenne des jetons, et jetons en temps × 8 fréquences (sinon `ValueError`).
- `_forward(batch)` ([avex_encoder.py:124](avex_encoder.py#L124)) et `_forward_tokens(batch)` ([avex_encoder.py:127](avex_encoder.py#L127)) : la première et la seconde moitié de `_grid(_features(batch))`. `_features(batch)` ([avex_encoder.py:102](avex_encoder.py#L102)) : le modèle sans gradient, sortie en numpy `float32`. `_samples()` ([avex_encoder.py:99](avex_encoder.py#L99)) : le nombre d'échantillons d'une fenêtre.
- Pourquoi (docstring) : fenêtre de 5 s, la grille de perch_v2, pour comparer fenêtre à fenêtre (benchmark 07) ; encodeurs ajoutés par DECISIONS n° 151.

#### `remap_fairseq_eat(state, model_keys)` — [avex_encoder.py:32](avex_encoder.py#L32)

- Quand : dans `_load_fairseq_eat` ; testée seule.
- Reçoit : les tenseurs d'un point de contrôle et la liste des clés que le modèle attend. Renvoie : `None` si le point de contrôle n'est pas au format fairseq (aucune clé `modality_encoders.IMAGE.…`) ; sinon les mêmes tenseurs sous les noms du modèle AVEX. Une clé attendue qui manque lève `ValueError` ; les clés en trop (le décodeur de reconstruction) sont laissées de côté.
- Pourquoi (docstring) : `eat_bio` et `eat_all` sont publiés au format fairseq ; AVEX 1.3.0 n'en charge aucun tenseur (0 sur 150) et le modèle garderait alors les poids de l'EAT générique d'AudioSet.
- `_load_fairseq_eat(model, model_name)` ([avex_encoder.py:54](avex_encoder.py#L54)) : si le point de contrôle est sur Hugging Face (`hf://…`), le télécharge, le convertit, et le charge en mode strict ; sans effet sinon.

## `onnx_encoder.py` : l'encodeur du livrable

Fait tourner un paquet d'encodeur avec ONNX Runtime sur le processeur, sans torch ni TensorFlow. Lit : `manifest.json` et `model.onnx` d'un dossier `paths.models/encoder/<nom>`. Écrit : `manifest.json` (`EncoderManifest.write`, appelée par `export.py`). Tests : [`tests/embedding/test_onnx_export.py`](../../../tests/embedding/test_onnx_export.py).

#### `OnnxEncoder(directory, batch_size=16, threads=None)` — [onnx_encoder.py:57](onnx_encoder.py#L57)

- Quand : construit par `_get_encoder` quand le nom demandé n'est pas dans `encoders.models` et qu'un paquet de ce nom existe ; et par `export.check_equivalence`.
- Fait, à la construction : lit le manifeste ; recalcule le SHA-256 de `model.onnx` et refuse le paquet s'il diffère du manifeste (« paquet corrompu ») ; ouvre une session ONNX Runtime sur le processeur (`CPUExecutionProvider`), avec `threads` fils si demandé ; prend `name`, `version`, `sample_rate`, `window_s`, `dim` dans le manifeste ; `has_tokens` est vrai si le manifeste nomme une sortie de jetons.
- `_forward(batch)` ([onnx_encoder.py:82](onnx_encoder.py#L82)) et `_forward_tokens(batch)` ([onnx_encoder.py:86](onnx_encoder.py#L86)) : un appel de la session, sur la sortie d'embedding ou de jetons nommée par le manifeste.

#### `EncoderManifest` — [onnx_encoder.py:23](onnx_encoder.py#L23)

- La fiche d'identité d'un paquet : `name`, `version`, `sha256`, `license`, `sample_rate`, `window_s`, `dim`, et les noms d'entrée et de sortie du graphe (`waveform`, `embedding`, sortie de jetons facultative).
- `read(directory)` ([onnx_encoder.py:36](onnx_encoder.py#L36)) : lit `manifest.json` ; une des sept clés de `MANIFEST_KEYS` ([onnx_encoder.py:19](onnx_encoder.py#L19)) qui manque lève `ValueError`. `write(directory)` ([onnx_encoder.py:43](onnx_encoder.py#L43)) : l'écrit, indenté, en UTF-8.

#### `file_sha256(path)` — [onnx_encoder.py:49](onnx_encoder.py#L49)

- Quand : dans `OnnxEncoder.__init__`, `export.export_onnx`, `export.quantize_int8`.
- Renvoie : le SHA-256 du fichier, lu par blocs d'1 Mo.

## `export.py` : fabriquer un paquet ONNX

Lit et écrit des dossiers de paquet (`model.onnx`, `manifest.json`). torch n'est importé qu'à l'appel. Aucune commande ni aucun script n'appelle ce fichier : ses trois fonctions ne tournent que dans [`tests/embedding/test_onnx_export.py`](../../../tests/embedding/test_onnx_export.py).

#### `export_onnx(module, directory, manifest, opset=17)` — [export.py:17](export.py#L17)

- Reçoit : un `torch.nn.Module` qui prend la forme d'onde `(lot, échantillons)` et rend l'embedding `(lot, dim)`, le dossier du paquet, un manifeste. Écrit : `model.onnx` (taille de lot libre) et `manifest.json`, avec le SHA-256 du fichier écrit. Renvoie : le chemin de `model.onnx`.

#### `quantize_int8(directory, out_directory)` — [export.py:45](export.py#L45)

- Fait : quantification dynamique des poids en entiers 8 bits (`onnxruntime.quantization`) dans un nouveau dossier ; recopie le manifeste avec la version suivie de `-int8` et la nouvelle empreinte.

#### `check_equivalence(reference, directory, windows, sr, min_cosine=0.99)` — [export.py:66](export.py#L66)

- Fait : encode les mêmes fenêtres avec l'encodeur de référence et avec le paquet ONNX, et rend le cosinus minimal, le cosinus moyen et `passed` (minimum au-dessus de 0,99).
- Pourquoi (docstring du fichier) : c'est le critère d'acceptation du jalon M5, cosinus > 0,99 entre embeddings torch et ONNX, quantifié compris.
- `cosine_rows(a, b)` ([export.py:60](export.py#L60)) : cosinus ligne à ligne de deux matrices.

## `pruned.py` : graphe ONNX réduit

Lit un fichier ONNX ; écrit sa version réduite dans un dossier de cache. Utilisé par `BacpipeEncoder._light_session` seulement. Tests : dans [`tests/embedding/test_openvino_session.py`](../../../tests/embedding/test_openvino_session.py).

#### `pruned_model(source, outputs, cache_dir)` — [pruned.py:17](pruned.py#L17)

- Reçoit : le fichier ONNX d'origine, les sorties à garder, le dossier de cache. Renvoie : le chemin d'un fichier ONNX qui ne calcule que ces sorties ; `source` lui-même si le paquet `onnx` n'est pas installé.
- Fait : le fichier réduit s'appelle `<nom>-<sorties>-<taille du fichier d'origine>.onnx` ; il n'est fabriqué que s'il n'existe pas (extraction du sous-graphe, écriture dans un fichier temporaire puis renommage). Un modèle d'origine qui change de taille donne donc un nouveau fichier.
- Pourquoi (docstring, DECISIONS n° 180) : le graphe de perch_v2 calcule à chaque fenêtre les logits de ses 14 795 classes (364 Mo de prototypes sur 413) et son spectrogramme, et ONNX Runtime exécute tout le graphe quelles que soient les sorties demandées ; l'embedding et les jetons sont inchangés au bit près.

#### `PrunedSession(session, kept, names)` — [pruned.py:37](pruned.py#L37)

- Quand : posée par `_light_session` à la place de la session de bacpipe.
- `run(output_names, feeds)` ([pruned.py:47](pruned.py#L47)) : fait tourner la session réduite et rend les sorties dans l'ordre du graphe d'origine ; une sortie retirée est rendue vide, de forme `(lot, 0)`. L'argument `output_names` est ignoré. bacpipe ne voit donc pas la différence.

## `openvino_session.py` : calcul par OpenVINO

Fait calculer un modèle par OpenVINO sur le processeur graphique intégré d'Intel (Iris Xe). Lit : un fichier ONNX ou un réseau torch. Écrit : un cache (graphe compilé, graphe converti `torch_<empreinte>.xml`) dans le dossier donné. OpenVINO n'est pas une dépendance obligatoire : absent, ou sans le périphérique, l'appelant garde ONNX Runtime ou torch. Tests : [`tests/embedding/test_openvino_session.py`](../../../tests/embedding/test_openvino_session.py).

#### `openvino_session(model_path, settings, cache_dir=None)` — [openvino_session.py:179](openvino_session.py#L179)

- Quand : dans `BacpipeEncoder._light_session`, pour un modèle ONNX avec le réglage `openvino`.
- Reçoit : le fichier ONNX et le réglage (`device`, `batch_size`, `precision` ; par défaut `GPU`, 4, `f32`). Renvoie : une `OpenVinoSession`, ou `None` (avec un message) si OpenVINO n'est pas installé ou si le périphérique manque.

#### `OpenVinoSession(model_path, device="GPU", batch=4, precision="f32", cache_dir=None)` — [openvino_session.py:19](openvino_session.py#L19)

- Fait, à la construction : vérifie que le périphérique existe (sinon `RuntimeError`) ; fige la forme d'entrée à `(batch, échantillons)` ; compile avec la consigne `LATENCY` et la précision demandée ; avec `cache_dir`, garde le graphe compilé (23 s au premier lancement seulement, commentaire).
- `run(output_names, feeds)` ([openvino_session.py:63](openvino_session.py#L63)) : même appel qu'une session ONNX Runtime ; découpe le lot en paquets de `batch`, complète le dernier par des zéros (`_infer`, [openvino_session.py:56](openvino_session.py#L56)) et ne rend que les vraies lignes ; sorties dans l'ordre du modèle. `output_names` est ignoré.
- `get_providers()` ([openvino_session.py:53](openvino_session.py#L53)) : `OpenVINO (<nom du périphérique>)`, pour l'affichage de `_light_session`.
- Pourquoi (docstrings, DECISIONS n° 179) : un lot de taille fixe est plus rapide sur le processeur graphique ; mêmes sorties qu'ONNX Runtime à 5e-7 près en `float32`.

#### `openvino_backbone(backbone, example, settings, cache_dir=None)` — [openvino_session.py:160](openvino_session.py#L160)

- Quand : dans `BacpipeEncoder._openvino_backbone`, pour un modèle torch de `OPENVINO_TORCH`.
- Renvoie : un `OpenVinoBackbone`, ou `None` (avec un message) si OpenVINO ou le périphérique manque.

#### `OpenVinoBackbone(backbone, example, device="GPU", batch=4, precision="f32", cache_dir=None)` — [openvino_session.py:99](openvino_session.py#L99)

- Fait, à la construction : avec `cache_dir`, cherche `torch_<empreinte des poids>.xml` ; s'il manque, convertit le réseau (`_convert`) et l'écrit en `float32` (la précision se choisit à la compilation, commentaire) ; fige le lot ; compile.
- `__call__(x)` ([openvino_session.py:141](openvino_session.py#L141)) : s'appelle comme le réseau torch sur un lot de spectrogrammes ; découpe en paquets de `batch`, complète par des zéros ; rend un objet avec `last_hidden_state` et `pooler_output` en tenseurs torch, comme la sortie Hugging Face d'origine.
- `weights_digest(module)` ([openvino_session.py:87](openvino_session.py#L87)) : les 16 premiers caractères du SHA-256 des noms et des valeurs des poids ; un point de contrôle mis à jour donne donc un autre fichier de cache.
- `_convert(backbone, example)` ([openvino_session.py:69](openvino_session.py#L69)) : enveloppe le réseau pour qu'il rende le couple (carte, embedding), puis `ov.convert_model`.

## `lowpass.py` : le contrôle passe-bas

Ne lit et n'écrit rien. Sert à une seule entrée de la configuration, `birdmae_lp8k` (`lowpass_hz: 8000`). Tests : [`tests/embedding/test_lowpass.py`](../../../tests/embedding/test_lowpass.py).

#### `LowpassEncoder(inner, cutoff_hz)` — [lowpass.py:28](lowpass.py#L28)

- Quand : posé par `_get_encoder` autour d'un `BacpipeEncoder` dont l'entrée a `lowpass_hz`.
- Fait : reprend `version`, `sample_rate`, `window_s`, `dim`, `has_tokens` de l'encodeur enveloppé ; son nom devient `<nom>_lp<kHz>k`. `embed(wav, sr)` ([lowpass.py:41](lowpass.py#L41)) et `embed_tokens(wav, sr)` ([lowpass.py:44](lowpass.py#L44)) filtrent le son puis appellent l'encodeur enveloppé.
- Détail : l'attribut `inner` fait que `embed.resample_mode` garde le mode `window` : le filtre s'applique à la fréquence d'origine, avant le rééchantillonnage. Cette enveloppe n'a pas de `pop_logits`.
- Pourquoi (docstring) : le tuteur juge les modèles à 16 kHz « très limite » pour les paysages sonores ; la note d'A. blanci est à 4,4–5,5 kHz. Si birdmae sur l'audio filtré à 8 kHz égale birdmae sur l'audio natif, la fréquence de 16 kHz n'est pas un défaut pour ce chant.

#### `lowpass(wav, sr, cutoff_hz, order=8)` — [lowpass.py:20](lowpass.py#L20)

- Renvoie : le signal filtré par un Butterworth passe-bas d'ordre 8 appliqué dans les deux sens (phase nulle), en `float32` ; le signal inchangé si la coupure est à la moitié de la fréquence d'échantillonnage ou au-dessus.

## `upstream.py` : transformations en amont

Ne lit et n'écrit rien. Tests : [`tests/embedding/test_upstream.py`](../../../tests/embedding/test_upstream.py).

#### `UpstreamEncoder(inner, upstream)` — [upstream.py:17](upstream.py#L17)

- Quand : posé par `_with_transforms` quand l'entrée de la configuration a des `transforms` ou que `--upstream` en active (`bandpass`, `denoise`).
- Fait : refuse une chaîne sans transformation (`ValueError`) ; reprend les attributs de l'encodeur enveloppé ; son nom devient `<nom>+<étiquette>` (`birdmae+bp3-7k`), l'étiquette venant de `Upstream.transform_tag`. `embed(wav, sr)` ([upstream.py:31](upstream.py#L31)) et `embed_tokens(wav, sr)` ([upstream.py:34](upstream.py#L34)) transforment le son (`Upstream.transform`) puis appellent l'encodeur enveloppé.
- `pop_logits()` ([upstream.py:37](upstream.py#L37)) : ceux de l'encodeur enveloppé, calculés sur le son transformé ; `None` s'il n'en garde pas.
- Pourquoi (docstring) : le nom différent donne un autre stock, d'autres têtes et d'autres scores : c'est un autre encodeur, comparé par le benchmark. Comme pour le passe-bas, le son est transformé à la fréquence d'origine.

## Écarts relevés

Signalés, pas corrigés.

1. **`base.py` dit que le rééchantillonnage ne se fait jamais chez l'appelant.** Sa docstring : `embed(wav, sr)` reçoit les fenêtres « à la f_e native de l'enregistrement », et le rééchantillonnage « se fait ici, jamais chez l'appelant (§13.7) ». Depuis DECISIONS n° 176, `embed.encoder_windows` rééchantillonne l'enregistrement entier avant la découpe (mode `recording`, le défaut de `config/default.yaml`) ; `_prepare` ne rééchantillonne plus que dans le mode `window` et pour les encodeurs enveloppés.
2. **`export.py` n'a aucun appelant hors tests.** `export_onnx`, `quantize_int8` et `check_equivalence` ne sont appelées par aucune commande ni aucun script. `README.md` range l'export ONNX parmi les « emplacements réservés pour plus tard ». `OnnxEncoder` n'est atteint par `get_encoder` que si un paquet existe sous `paths.models/encoder/`.
3. **Nom du paquet ONNX.** La docstring de `onnx_encoder.py` range le paquet dans `data/models/encoder/<nom>-<version>/` ; `_get_encoder` cherche `paths.models/encoder/<name>`, où `name` est ce qui est passé à `--encoder`. Il faut donc donner `<nom>-<version>` en entier à la commande.
4. **Docstring de `get_encoder` incomplète.** Elle parle d'un « encodeur de recherche (bacpipe) ou paquet ONNX » ; le code a un troisième cas, `backend: avex`.
5. **« Jamais appelés en direct ».** La docstring de `__init__.py` dit que les encodeurs sont toujours obtenus par `get_encoder`. Hors tests, c'est vrai sauf `export.check_equivalence`, qui construit un `OnnxEncoder` lui-même.
6. **Message de `service.compute_tokens` périmé.** Il dit « seul perch_v2 aujourd'hui » expose des jetons ; `BacpipeEncoder` déclare aussi `has_tokens` pour birdmae, beats, naturebeats et audioprotopnet (`TOKEN_FREQ_BANDS`), et `AvexEncoder` pour tous ses modèles. L'aide de `cli.tokens` dit de même « (perch_v2) ». À reprendre dans la fiche de `service.py`.
7. **`anuraset/encodage/jetons.py` ne passe pas par `embed.encoder_windows`.** Il donne à `embed_tokens` des fenêtres découpées à la fréquence d'origine, donc rééchantillonnées fenêtre par fenêtre, quel que soit le mode rangé avec le stock ; `service.compute_tokens`, lui, relit ce mode. À reprendre dans la fiche d'`anuraset/encodage/`.
8. **Document cité à son ancien emplacement.** La docstring de `avex_encoder.py` renvoie à `documentation/encodeurs-bacpipe.md` ; le fichier existe à cet endroit dans l'arbre de travail (version complète, dépôt interne) et la version publique est `documentation/encodeurs/encodeurs-bacpipe.md`.
9. **Fonctions publiques à un seul usage.** `encoder_id` n'est appelée que par `stock_id` ; `cosine_rows` que par `check_equivalence` ; `weights_digest` que par `OpenVinoBackbone` ; `lowpass` que par `LowpassEncoder`. Aucune fonction de ce dossier n'est sans appelant si l'on compte les tests ; sans les tests, les trois fonctions de `export.py` le sont.
