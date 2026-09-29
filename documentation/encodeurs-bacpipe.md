# Encodeurs de bacpipe : ce qu'on peut récupérer

Relevé le 25/09/2026 dans le code de bacpipe 1.3.5 (`bacpipe/model_pipelines/feature_extractors/`)
et dans notre adaptateur (`blanci/encoders/bacpipe_encoder.py`). Aucun modèle n'a été exécuté
pour ce relevé : les tailles de grille marquées « à mesurer » se vérifient au premier chargement
(méthode dans `notes.md` : comparer l'embedding à la moyenne, au maximum et au jeton de classe).

**Convention** : ce fichier est la mémoire de tout ce qu'on apprend sur les encodeurs (articles,
cartes de modèles, mesures, explications échangées avec Léonard). Toute information nouvelle y
est consignée avec sa source et son niveau de vérification ; « Ce qui n'a pas été vérifié »
(fin de fichier) est tenu à jour.

**Mise à jour du 29/09/2026** (session « justification des exclusions ») : décision de
benchmarker douze encodeurs, corrections sur BEANS, licences, mémoire des jetons, familles à
variantes, Perch 2.0. Les sections ajoutées citent leurs sources en fin de fichier, et la
dernière liste ce qui n'a **pas** été vérifié.

## Décision du 29/09/2026 : douze encodeurs à benchmarker

Prise par Léonard après relecture des exclusions. Aucune n'avait été mesurée : c'étaient des
a priori de proximité de taxon (section « État de l'art », version du 29/09 matin). On
continue à benchmarker, sur AnuraSet puis sur les données ONF, en plus des cinq déjà faits
(perch_v2, perch_bird, birdnet, protoclr, Bird-MAE Base et Huge) :

`naturebeats`, `beats`, `convnext_birdset`, `birdnet_v3`, `rcl_fs_bsed`, `audioprotopnet`,
`avesecho_passt`, `biolingual`, `insect66`, `insect459`, `esp-aves2`, `MetaPerch`.

Précisions de Léonard le 29/09 après-midi : **`birdmae_large` s'ajoute à la liste** ;
`birdmae_large` et `esp-aves2` (variantes `-all` comprises) sont en **priorité 1 comme
références de benchmark**, pas comme livrables ; **gros plan sur `birdnet_v3`**, annoncé comme
meilleur partout : à vérifier sur nos données.

Inchangés : `mix2` (données ONF seulement : entraîné sur AnuraSet) ; `aves_especies` et
`birdaves_especies` (écartés : AVES est en bas du classement BirdSet de la revue de Schwinger
et al., voir « Écartés ») ; `surfperch`, `vggish`, `audiomae`, `bat`, `batdetect2_*`,
`google_whale`, `hbdet`.

- **Rien n'est encore dans `config/default.yaml`** parmi les sept encodeurs nouveaux
  (`birdnet_v3`, `rcl_fs_bsed`, `audioprotopnet`, `avesecho_passt`, `biolingual`, `insect66`,
  `insect459`), ni `esp-aves2` ni `MetaPerch` : à ajouter. `naturebeats`, `beats` et
  `convnext_birdset` y sont déjà ; ils n'ont simplement pas encore été lancés sur AnuraSet.
- **Le calcul n'est pas une contrainte du benchmark** (machines cloud), **le stockage des
  jetons si** (« Jetons et mémoire »), **et le livrable garde ses contraintes** (« Contraintes du
  livrable et licences ») : un encodeur lourd ou sous licence non commerciale reste une
  référence de benchmark ou un enseignant de la distillation.
- **Les quatre encodeurs « lointains » testent le critère de tri lui-même** (`insect66`,
  `insect459`, `avesecho_passt`, `biolingual`). Si leur AP suit l'ordre de proximité de taxon,
  le critère tient ; sinon il ne valait rien, et il faudra le dire.
- **Proposition d'ordre et de protocole (non décidée)** : (1) embeddings + sondage linéaire
  pour tous, protocole des benchmarks 01 à 07 ; (2) jetons et sondage attentif seulement pour
  les transformers (naturebeats, beats, les esp-aves2 EAT et BEATs), sur les fenêtres du
  benchmark seules ; (3) `rcl_fs_bsed` à part (fenêtres de 0,2 s, grille propre, candidats
  seulement) ; (4) `esp-aves2` : les six points de contrôle « architecture × bio/all » d'abord
  (effnetb0, eat, sl-beats), qui testent l'affirmation centrale de l'article (mélanger de
  l'audio général aide), un seul adaptateur AVEX pour tous.

## Règle : juger un encodeur avec la tête qui correspond à sa sortie (29/09/2026, n° 151)

Un test cassé fait écarter un bon modèle. Les transformers auto-supervisés (Bird-MAE, BEATs,
NatureBEATs, les EAT d'esp-aves2) ne donnent leur mesure qu'avec une tête sur leurs **jetons**
(sondage attentif, ou sondage par prototypes pour Bird-MAE). Un sondage linéaire sur leur
embedding moyenné les sous-estime : BEATs passe de 94,10 à 97,98 AUROC sur BEANS et de 72,70 à
82,28 sur BirdSet en changeant seulement de tête (revue de Schwinger et al.). Nos benchmarks 04
et 06 (Bird-MAE en sondage linéaire) ne sont donc **pas** un verdict sur Bird-MAE.

Avant d'écarter un encodeur, trois vérifications :

1. **La tête d'origine, ou son équivalent, a été essayée** (colonne « À essayer » ci-dessous).
   Sans elle, le statut est « en retrait en sondage linéaire », jamais « écarté ».
2. **On lit bien la sortie que ses auteurs évaluent** : couche, agrégation, normalisation
   (exemple : le `last_hidden_state` de Bird-MAE dans bacpipe est déjà une moyenne des patchs
   suivie de `fc_norm`, n° 111), f_e et durée de fenêtre du modèle.
3. **Un témoin passe** : BOAFAB, l'espèce facile d'AnuraSet, a une AP moyenne par site (minute)
   d'au moins 0,85 avec les six encodeurs déjà testés (benchmark 07). Un nouvel encodeur
   nettement en dessous signale d'abord un tuyau cassé (mauvaise f_e, fenêtre, couche), pas un
   mauvais modèle.

| Encodeur | Réseau et apprentissage | Tête de ses auteurs | À essayer chez nous avant verdict | État au 29/09 |
|---|---|---|---|---|
| perch_v2, MetaPerch | CNN (EfficientNet-B3), supervisé | linéaire sur l'embedding moyen, et prototypes sur la carte spatiale | logistique (fait) ; prototypes ou attentive sur les jetons spatiaux (déjà stockés pour perch_v2) | linéaire : référence |
| perch_bird, birdnet (v2.4) | CNN, supervisé | linéaire sur l'embedding | logistique | fait |
| birdnet_v3 | réseau non documenté (« improved architecture »), supervisé | son classifieur, 11 560 classes | logistique, et **son propre classifieur sans entraînement** sur 4 de nos 5 espèces | à faire |
| convnext_birdset, esp-aves2 effnetb0, insect66, insect459, mix2 | CNN, supervisé (réseau des insect* à lire) | linéaire | logistique ; l'attention n'apporte rien à ConvNeXt (revue) | à faire |
| audioprotopnet | CNN (ConvNeXt) + couche à prototypes (ProtoPNet) | prototypes sur les cases de la carte | logistique, puis prototypes sur les jetons (bacpipe les garde déjà) | à faire |
| birdmae_base, _large, _huge | ViT, auto-supervisé (MAE) | **sondage par prototypes sur les jetons** (proposé par ses auteurs) | jetons (32 × 8 patchs) + sonde à prototypes ou attentive | **en retrait en linéaire, verdict suspendu** (n° 147, 149) |
| beats, naturebeats, esp-aves2 eat-* | transformer, auto-supervisé (BEATs affiné sur AudioSet ; NatureBEATs dans NatureLM-audio) | sondage attentif (revue) | jetons + `attentive` | à faire |
| esp-aves2 sl-beats-*, sl-eat-* | transformer, auto-supervisé puis supervisé | linéaire et attentif (article) | les deux | à faire |
| avesecho_passt, biolingual | transformer, supervisé (PaSST) ; contrastif audio-texte (CLAP) | linéaire ; biolingual : similarité au texte, attentive dans la revue | logistique, puis attentive si en retrait | à faire |
| protoclr | CvT-13 (transformer à convolutions), contrastif supervisé | prototypes de classe, peu d'exemples (article, à relire) | `simple_prototype` (prototype simple, jamais lancé sur AnuraSet) ; prototype différentiel et kNN : faits, en retrait | en retrait |
| rcl_fs_bsed | CNN, contrastif régularisé, trames de 0,2 s | prototypes à peu d'exemples, détection trame par trame | grille de 0,2 s, prototype sur les candidats | à faire |

Conséquence pratique : l'adaptateur ne sait extraire les jetons que de perch_v2
(`spatial_embeddings`). Pour les transformers, l'extraction des jetons est un **prérequis** du
benchmark, pas une option. Les jetons ne tiennent pas dans une branche git (« Jetons et
mémoire ») : ils restent sur la machine qui encode, qui calcule aussi les têtes sur jetons et ne
pousse que les résultats.

## Vocabulaire

- **Embedding** : le vecteur unique que bacpipe rend par fenêtre (ce qu'on stocke aujourd'hui).
- **Jetons** : les vecteurs *avant* l'agrégation en embedding. Pour un transformer, un par
  morceau (patch) du spectrogramme ; pour un CNN, un par case de la dernière carte de
  caractéristiques (fréquence × temps). C'est l'entrée de l'attentive probing et des poolings.
- **Couches intermédiaires** : les sorties des couches plus basses du réseau, plus génériques.
- **Hook** : fonction PyTorch branchée sur une couche, qui en copie la sortie pendant le
  calcul, sans modifier le modèle ni ralentir l'inférence. Pour TensorFlow/Keras, l'équivalent
  est de reconstruire un modèle qui s'arrête à la couche voulue.
- **Sondage linéaire / attentif** : une tête entraînée sur un encodeur gelé. Linéaire : une
  couche sur l'embedding (C·d paramètres). Attentif : une attention multi-têtes sur les jetons,
  puis une couche linéaire (2d² + (C+1)d + C paramètres selon la revue de Schwinger et al. ;
  notre `blanci/attentive.py` : une requête, 2d + 1). Le coût du sondage attentif est le
  stockage des jetons, pas la tête (« Jetons et mémoire »).
- **SSL / SL** : apprentissage auto-supervisé (sans étiquettes d'espèces : reconstruire des
  morceaux masqués, par exemple) / supervisé (avec étiquettes). « Deux étages » : SSL puis SL.
- **bio / all / audioset** (esp-aves2) : données d'entraînement. `bio` : bioacoustique seule
  (Xeno-canto, iNaturalist, Animal Sound Archive, Watkins) ; `all` : bioacoustique + AudioSet
  (audio général) ; `audioset` : AudioSet seul.
- **Prédiction de source, auto-distillation** : deux ingrédients de l'entraînement de Perch 2.0,
  expliqués dans « Perch 2.0 ».

## Les encodeurs du projet (`encoders.models`)

Tableau en image : `documentation/tableaux/encodeurs.png` (avec le débit mesuré sur l'i5,
DECISIONS n° 72). Le même contenu en fiches, lisibles dans Xcode :

- **birdmae** (Huge ; `birdmae_base` : Base) — PyTorch, Hugging Face (code du modèle fourni
  par DBD-research-group). 32 kHz, 5 s, 1 280 dimensions (Base : 768).
  - Ce que bacpipe rend : l'embedding agrégé. Malgré son nom, `last_hidden_state` est ici la
    moyenne des patchs (jeton de classe exclu) puis `fc_norm`.
  - Jetons : **faciles**. `output_hidden_states=True` rend, par couche, 257 jetons = 1 jeton
    de classe + 256 patchs en grille **32 temps × 8 fréquences** (ordre temps d'abord :
    patch t·8 + f), avant `fc_norm`.
  - Couches intermédiaires : faciles, même option (13 sorties pour le modèle Base).
- **beats** — PyTorch. 16 kHz, 5 s, 768 dimensions.
  - Ce que bacpipe rend : la moyenne des jetons, faite *dans* bacpipe (`avg_pooling = True`).
  - Jetons : **faciles**. `avg_pooling = False` sur le modèle chargé, une ligne, sans hook.
    Grille ≈ 8 fréquences × 31 temps (patchs 16 × 16), à mesurer.
  - Couches intermédiaires : moyennes, hook sur les couches de l'encodeur transformer.
- **naturebeats** — PyTorch (BEATs, poids NatureLM-audio). 16 kHz, 5 s, 768 dimensions.
  Comme BEATs : jetons faciles (même réglage), couches moyennes.
- **perch_v2** — ONNX (onnxruntime), sans TensorFlow. 32 kHz, 5 s, 1 536 dimensions.
  - Ce que bacpipe rend : embedding, jetons spatiaux, spectrogramme, logits des 14 795 classes.
  - Jetons : **déjà utilisés**, 16 temps × 4 fréquences × 1 536 (`spatial_embedding`).
  - Couches intermédiaires : difficiles. Le graphe ONNX ne sort que ces quatre tenseurs ; il
    faudrait le modifier pour exposer des nœuds internes (faisable avec la bibliothèque
    `onnx`, à valider).
- **perch_bird** (Perch v1) — TensorFlow SavedModel (via perch-hoplite). 32 kHz, 5 s,
  1 280 dimensions.
  - Ce que bacpipe rend : embedding, logits, spectrogramme.
  - Jetons et couches : **difficiles**, la signature du SavedModel n'expose pas la carte
    avant agrégation.
- **birdnet** (v2.4) — TensorFlow / Keras. 48 kHz, 3 s, 1 024 dimensions.
  - Ce que bacpipe rend : la sortie de l'avant-dernière couche (`layers[-3]`).
  - Jetons : **faciles**. La carte de caractéristiques avant l'agrégation globale s'obtient
    comme bacpipe obtient l'embedding (`tf.keras.Model(entrée, couche.output)`). Taille à
    mesurer.
  - Couches intermédiaires : faciles, n'importe quelle couche, même méthode. Licence non
    commerciale : référence seulement, non déployable.
- **protoclr** — PyTorch (CvT-13). 16 kHz, 6 s, 384 dimensions.
  - Ce que bacpipe rend : moyenne des jetons du dernier étage (ou jeton de classe selon la
    config).
  - Jetons : **faciles**, hook sur le dernier étage. Couches : faciles, 3 étages, un hook
    chacun.
- **convnext_birdset** — PyTorch, Hugging Face (ConvNeXt). 32 kHz, 5 s, 1 024 dimensions.
  - Ce que bacpipe rend : `pooler_output` (moyenne de la dernière carte).
  - Jetons : **faciles**, `output_hidden_states=True` rend la dernière carte
    fréquence × temps. Couches : faciles, les 4 étages, même option.

« Facile » : une option ou un hook, puis la vérification du `notes.md`. « Moyen » : lire le code
du modèle pour choisir la couche. « Difficile » : il faut modifier ou reconstruire le modèle
exporté.

**Correction** (n° 111) : une première version de ce tableau disait que bacpipe rendait déjà
les jetons de Bird-MAE. Faux : le code du modèle (`modeling_bird_mae.py` de Bird-MAE-Base, lu le 25/09 ; Huge à confirmer) nomme
`last_hidden_state` la sortie agrégée. Le n° 77 avait raison : seul perch_v2 rend ses jetons
tels quels aujourd'hui.

## Les autres modèles de bacpipe (hors projet, sauf décision du 29/09)

Même image, second tableau. En bref (nom : framework, f_e, fenêtre, domaine ; jetons ou
couches ; intérêt ici). Les modèles marqués **[29/09]** sont à benchmarker (voir plus haut) :

- `audioprotopnet` **[29/09]** : PyTorch, Hugging Face, 32 kHz, 5 s, oiseaux (BirdSet) ; jetons
  **déjà gardés** par bacpipe (`results.last_hidden_state`). Aucune décision écrite ne
  l'écartait : il n'avait pas été trié. Quatre variantes existent (1, 5, 10, 20 prototypes par
  classe) : voir « Familles à plusieurs variantes ».
- `birdnet_v3` **[29/09]** : ONNX (dans une enveloppe PyTorch), 32 kHz, 3 s dans bacpipe (le
  modèle accepte une durée variable) ; BirdNET+ V3.0, **préversion** (« preview 3.1 », 11 000
  espèces dont des non-oiseaux) ; modèles sous CC BY-SA 4.0, Terms of Use **lus le 29/09**
  (`TERMS_OF_USE.txt` du dépôt Zenodo 20703646) : usage commercial permis, dérivés sous la même
  licence, attribution obligatoire (« Powered by BirdNET »), interdits : braconnage et tout
  usage militaire ; la description dit « fourni pour la recherche et l'évaluation ».
  **Déployable**, contrairement à la v2.4 (non commerciale). Étiquettes (fichier
  `…_Global_11K_Labels.csv`, lu) : 11 560 classes, dont 9 834 oiseaux, 699 insectes,
  **647 amphibiens** et 350 mammifères. 4 de nos 5 espèces AnuraSet y sont (DENMIN, LEPLAT,
  PHYCUV, BOAFAB sous « Hypsiboas faber » ; pas PITAZU, *Pithecopus azureus*) : son classifieur
  se juge **sans entraînement** sur ces quatre-là. A. blanci n'y est pas ; son congénère
  *Anomaloglossus baeobatrachus* (Guyane) y est, générateur de candidats possible comme les
  congénères de Perch (n° 70). Sources d'entraînement non publiées : AnuraSet y est peut-être
  (fuite possible, comme mix2) ; un score spectaculaire sur AnuraSet se revérifie sur les
  données ONF.
- `aves_especies`, `birdaves_especies` : PyTorch (wav2vec2), 16 kHz, 1 s, animaux ; **toutes
  les couches** rendues par `extract_features` ; moyen (fenêtre courte, adaptée à une note).
  Écartés (voir « Écartés »).
- `avesecho_passt` **[29/09]** : PyTorch (PaSST), 32 kHz, 3 s, oiseaux ; hook ; moyen.
- `mix2` : PyTorch (MobileNetV3-Large), 16 kHz, 3 s ; **entraîné sur AnuraSet** (Moummad et
  al., EUSIPCO 2024, 42 espèces d'anoures) : exclu de nos benchmarks AnuraSet (il en a vu les
  labels) ; seul encodeur spécialisé anoures, très léger ; candidat pour les données ONF.
- `rcl_fs_bsed` **[29/09]** : PyTorch, 22,05 kHz, 0,2 s ; contrastif régularisé pour la
  détection à peu d'exemples (Moummad et al., DCASE few-shot) ; fenêtre de la taille d'une note
  d'A. blanci, mais 300 embeddings par minute : sur les candidats seulement, grille propre.
- `surfperch` : TensorFlow (Perch), 32 kHz, 5 s, récifs coralliens ; comme Perch v1 ; faible.
- `biolingual` **[29/09]** : PyTorch (CLAP), 48 kHz, 10 s, bioacoustique + texte ; couches du
  modèle audio via Hugging Face ; fenêtre de 10 s (notes de 0,1 s diluées) et conçu pour
  l'interrogation par texte, mais la revue de Schwinger et al. lui donne 97,76 AUROC sur BEANS
  (classification, sondage attentif) et 82,51 sur BirdSet : correct, pas faible.
- `audiomae` : PyTorch (ViT), 16 kHz, 10 s, audio général ; hook ; faible.
- `vggish` : TensorFlow, 16 kHz, 1 s, audio général ; faible (référence ancienne).
- `insect66`, `insect459` **[29/09]** : PyTorch, 44,1 kHz, 5,5 s, insectes ; `embeddings` du
  modèle ; deux jeux d'étiquettes, apparemment 66 et 459 espèces (d'après les noms, non vérifié).
  A priori faibles (orthoptères =
  faux amis), mais l'argument se retourne : un encodeur qui sépare bien les insectes peut aider
  la tête à séparer les faux amis des anoures. Jamais testé.
- `bat`, `batdetect2_clip_avg`, `batdetect2_dets_avg` : PyTorch, 256 kHz, 1 s,
  chauves-souris ; aucun intérêt (ultrasons).
- `google_whale`, `hbdet` : TensorFlow, 24 kHz / 2 kHz, 2–4 s, cétacés ; aucun intérêt.

## Ce que ça change pour la suite

- **Bird-MAE, BEATs, NatureBEATs, ProtoCLR et ConvNeXt-BirdSet** : une option ou un hook par
  modèle pour les jetons, à ajouter à l'adaptateur (non fait). Bird-MAE est le plus simple
  (option Hugging Face, grille connue : 32 × 8).
- **Couches intermédiaires** (R23) : faciles pour les modèles PyTorch et BirdNET, difficiles
  pour Perch v1 et v2. Le coût est ailleurs : stocker plusieurs couches multiplie le volume des
  embeddings, sur des centaines de milliers de fenêtres.
- **Encodeurs de la décision du 29/09** : `birdnet_v3`, `rcl_fs_bsed`, `audioprotopnet`,
  `avesecho_passt`, `biolingual`, `insect66`, `insect459` sont dans bacpipe (une entrée de
  `encoders.models` chacun, sur le modèle de `birdnet` : `{backend: bacpipe, model: <nom>}`) ;
  `esp-aves2` et `MetaPerch` n'y sont pas : un adaptateur chacun (AVEX pour esp-aves2 ;
  MetaPerch : voir « MetaPerch »). Vérifier la f_e et la fenêtre lues dans chaque modèle : la
  grille en découle.

## État de l'art hors bacpipe et encodeurs à tester ensuite (29/09/2026)

Relevé en ligne le 29/09 ; utilisés sur AnuraSet à cette date (benchmarks 01 à 07) : perch_v2,
perch_bird, birdnet (v2.4), protoclr, birdmae_base, birdmae_huge, soit 5 des 26 modèles de
bacpipe 1.3.5 (Bird-MAE en deux tailles). Critère de tri : l'encodeur a-t-il appris sur des sons proches des nôtres
(notes brèves et tonales de 4 à 6 kHz, dans un fond de forêt tropicale : insectes, autres
anoures, oiseaux) ? Les modèles appris sur des taxons lointains (cétacés, chauves-souris,
éléphants) apprennent d'autres échelles de fréquence et de temps : aucun intérêt ici. En
revanche, ajouter des taxons variés à un modèle généraliste l'améliore (Perch 2.0, « The Bittern
Lesson ») : un généraliste qui a entendu des anoures vaut mieux qu'un spécialiste d'autre chose.
Ce critère est un a priori : aucune de ses conséquences n'avait été mesurée le 29/09 matin.
Le même jour, la décision de benchmarker douze encodeurs (plus haut) le met à l'épreuve.

| Encodeur | Origine | Dans bacpipe | Pourquoi le tester | Ordre proposé |
|---|---|---|---|---|
| naturebeats | ESP, encodeur BEATs de NatureLM-audio (2024) | oui (déjà dans `encoders.models`) | 1er de la revue de Schwinger et al. sur les 5 tâches de **classification** de BEANS en sondage attentif (98,57 AUROC), mais **sans anoure** et à moins d'un point de Bird-MAE et de Perch 2.0 (voir « BEANS ») ; 16 kHz, 5 s : même grille que perch_v2 ; la carte ESP de ses poids (probablement les mêmes) est **non commerciale** (voir « Contraintes du livrable ») | 1 |
| birdnet_v3 | BirdNET+ V3.0, préversion 3.1 (2026) | oui | 11 000 espèces dont des non-oiseaux ; CC BY-SA 4.0 mais préversion (« developer preview », Terms of Use à lire) : à refaire à la version finale | 1 |
| esp-aves2 (11 points de contrôle, voir « Familles à plusieurs variantes ») | ESP, « What Matters for Bioacoustic Encoding » (ICLR 2026) | non (Hugging Face, bibliothèque AVEX) | encodeurs récents ; l'article montre que mélanger audio général et bioacoustique aide ; effnetb0 est un CNN léger ; **CC-BY-NC-SA-4.0** : référence de benchmark, non déployable ; 16 kHz ; adaptateur à écrire | 1 |
| audioprotopnet | DBD (Heinrich et al.), BirdSet | oui | oiseaux, 32 kHz, 5 s ; avec sa tête d'origine (5 prototypes par classe), ROC-AUC BirdSet 0,896, contre 0,908 pour Perch 2.0 et 0,886 pour Bird-MAE-L (tableaux de MetaPerch) ; même laboratoire que Bird-MAE et convnext_birdset | 2 |
| convnext_birdset | DBD, BirdSet (2024) | oui (déjà dans `encoders.models`) | CNN appris sur BirdSet ; 85,75 AUROC sur BirdSet en sondage linéaire dans la revue (Bird-MAE : 86,54 en attentif) ; ne profite pas du sondage attentif : pas de jetons à stocker | 2 |
| beats | Microsoft, audio général (2022) | oui | point de comparaison : la revue le classe 3e de BEANS (97,98 en attentif, 94,10 en linéaire) : un modèle général fait-il aussi bien ? | 2 |
| MetaPerch | Google, Perch + métadonnées (arXiv 2607.14072, ICML 2026) | non | poids annoncés publics par le papier (dépôt non vérifié) ; gain modeste sur Perch 2.0, plus net en Amérique du Sud (voir « MetaPerch ») ; export ONNX à prévoir | 2 |
| mix2 | Moummad et al., AnuraSet (2024) | oui | seul encodeur spécialisé anoures ; **pas sur AnuraSet** (vu à l'entraînement) : sur les données ONF | 2 (ONF) |
| avesecho_passt | PaSST, oiseaux (bacpipe) | oui | « oiseaux d'Europe » : a priori non testé ; teste le critère de proximité de taxon | 3 |
| biolingual | CLAP bioacoustique + texte | oui | 97,76 sur BEANS et 82,51 sur BirdSet dans la revue ; fenêtre de 10 s | 3 |
| insect66, insect459 | insectes | oui | test des faux amis (voir plus haut) ; 44,1 kHz, 5,5 s | 3 |
| rcl_fs_bsed | Moummad et al. (2024) | oui | fenêtre de 0,2 s, à peu d'exemples ; sur les candidats seulement | 3 |

Écartés (maintenus) : `surfperch` (récifs), `google_whale`, `hbdet` (cétacés), `bat`,
`batdetect2_*` (ultrasons), `vggish` et `audiomae` (audio général, supplantés par BEATs ; dans
la revue, AudioMAE est le dernier de BEANS en linéaire, 84,47 AUROC), `aves_especies` et
`birdaves_especies`. Pour ces deux derniers, l'argument du 29/09 matin (« supplantés par
esp-aves2 ») n'était pas mesuré, mais la revue classe AVES et BirdAVES parmi les derniers de
BirdSet (AVES 63,80 en linéaire, 74,48 en attentif ; BirdAVES 65,58 et 78,87) : l'exclusion tient
sur des données, pas sur un a priori.

Écartés le 29/09 matin puis **réintégrés** le 29/09 : `insect66` et `insect459` (« nos faux amis,
pas nos cibles » : a priori, non testé), `biolingual` (fenêtre de 10 s, fait pour le texte :
mais 97,76 sur BEANS), `avesecho_passt` (« oiseaux d'Europe » : a priori, non testé),
`audioprotopnet` (jamais trié : absent à la fois du tableau et de la liste des écartés) et
`rcl_fs_bsed` (0,2 s, « pour plus tard »).

À retenir de la revue comparative (Foundation Models for Bioacoustics, arXiv 2508.01277, 2025 ;
Ecological Informatics 2026) : Perch 2.0 mène en sondage linéaire (97,78 AUROC sur les
tâches de classification de BEANS ; meilleur score BirdSet, 90,78, avec sa tête d'origine
restreinte aux classes du test). Les transformers (BEATs, Bird-MAE) ne donnent leur mesure
qu'en **sondage attentif sur les jetons** (BEATs : 94,10 → 97,98 sur BEANS, 72,70 → 82,28 sur
BirdSet). Cohérent avec nos benchmarks 04 et 06 (Bird-MAE loin derrière en sondage linéaire) :
un encodeur transformer ne s'écarte qu'après l'essai d'une tête sur jetons. Autre résultat :
« une taille de modèle plus grande ne donne pas d'avantage net » (Perch ≈ 8 M de paramètres :
85,63 sur BirdSet ; ConvNeXt-BirdSet 88 M : 85,75 ; Bird-MAE 300 M : 86,54), ce qui vaut la peine
d'être gardé en tête quand le calcul est gratuit : plus gros n'est pas plus performant.

## BEANS : ce que le classement de NatureBEATs mesure (29/09/2026)

BEANS (BEnchmark of ANimal Sounds ; Hagiwara et al., ESP, ICASSP 2023, arXiv 2210.12300) :
12 jeux de données.

- **5 en classification** (exactitude) : Watkins (mammifères marins, 31 classes), Bats
  (chauves-souris, 10 individus, 250 kHz), CBI (oiseaux, 264 espèces), Dogs (chiens, 10
  individus), HumBugDB (moustiques, 14 espèces) ;
- **5 en détection** (cmAP, enregistrements continus) : DCASE 2021 tâche 5 (oiseaux et
  mammifères), ENABirds, Hiceas (mammifères marins), **RFCX (oiseaux et grenouilles, 24
  espèces, 48 kHz)**, Hainan Gibbons ;
- 2 hors bioacoustique : ESC-50, Speech Commands.

Le seul jeu de BEANS qui contient des anoures est **RFCX, en détection**.

**La revue de Schwinger et al. écrit « we omit the detection tasks from BEANS »** : son
« BEANS » se réduit aux 5 tâches de classification, **sans aucun anoure**. Le classement où
NatureBEATs (« BEATs_NLM » dans la revue) est premier ne dit donc rien sur les anoures. Ses
chiffres (AUROC moyen sur les 5 tâches, sondage attentif) : BEATs_NLM 98,57 ; Bird-MAE 98,18 ;
BEATs 97,98 ; BioLingual 97,76 ; EAT 97,51 ; AudioMAE 97,19. Perch 2.0, sans jetons
accessibles, n'est jugé qu'en linéaire : 97,78. Trois réserves :

- **Classement saturé** : moins d'un point d'AUROC entre les quatre premiers.
- **Sur BirdSet** (enregistrements continus d'oiseaux, plus proche de notre usage), BEATs_NLM
  vaut 84,55, contre 86,54 pour Bird-MAE et 90,78 pour Perch 2.0 avec sa tête d'origine (84,49
  en sondage linéaire, à égalité avec NatureBEATs en attentif).
- **Fuite possible** : la revue note que BEATs_NLM a vu le jeu SSW de BirdSet à l'entraînement.
  Elle avance aussi que le 16 kHz peut expliquer une part de son retard sur les oiseaux
  (les modèles à 32 kHz dominent BirdSet) ; notre note à 4,4–5,5 kHz n'est pas concernée
  (contrôle passe-bas, DECISIONS n° 67).

Sur la suite complète de BEANS, avec RFCX, le papier MetaPerch (tableau 3) donne, en cmAP sur
RFCX : Perch 1.0 0,232 ; MetaPerch 0,209 ; Perch 2.0 0,200 ; BioLingual affiné 0,178 ; BirdNET
0,148 ; AVES-Bio 0,130. Tâche difficile pour tous, et Perch 1.0 devant Perch 2.0 : de quoi ne
pas abandonner `perch_bird`. NatureBEATs n'y figure pas (NatureLM-audio, le modèle de langage
complet, y est mesuré sans entraînement : 0,025, non comparable). **C'est AnuraSet qui
tranchera pour les anoures**, pas BEANS.

## Jetons et mémoire : ce que coûte le sondage attentif (29/09/2026)

La tête attentive est petite : 2d² + (C+1)d + C paramètres selon la revue, soit environ 1,2 M
pour d = 768 ; notre `blanci/attentive.py` : 2d + 1. Le coût est le **stockage des jetons de
chaque fenêtre**. La revue le dit elle-même : les modèles qui marchent avec un embedding moyenné
sont plus faciles à stocker et à traiter que ceux qui demandent les jetons.

Ordre de grandeur pour une campagne d'une semaine (575 h), fenêtres de 5 s au pas de 2,5 s
(828 000 fenêtres), jetons en fp16 (fp32 : le double) :

| Encodeur | Jetons par fenêtre | Par fenêtre | Campagne | Embedding moyen, fp32 |
|---|---|---|---|---|
| naturebeats, beats | 8 × 31 = 248 jetons × 768 (grille à mesurer) | 381 Ko | ≈ 315 Go | 2,5 Go |
| birdmae_base | 256 × 768 | 393 Ko | ≈ 326 Go | 2,5 Go |
| birdmae_huge | 256 × 1 280 | 655 Ko | ≈ 543 Go | 4,2 Go |
| perch_v2 | 16 × 4 = 64 × 1 536 (mesuré, n° 77) | 197 Ko | ≈ 163 Go | 5,1 Go |

Le papier de Perch 2.0 annonce un embedding spatial de forme (5, 3, 1536) ; l'ONNX de bacpipe
rend 16 × 4 × 1536 (mesuré, n° 77) : on garde la mesure.

Parades :

1. **Jetons sur les seules fenêtres du benchmark** (~1 500, n° 77) : c'est déjà la pratique.
2. **Moyenne sur l'axe fréquence** (8 fois moins pour BEATs), comme pour perch_v2 (n° 77).
3. **Cascade** : embedding moyen pour tout le corpus, jetons recalculés sur la fraction la
   mieux classée (`head.cascade_fraction` : 0,2 dans la config).
4. **CNN** (convnext_birdset, esp-aves2-effnetb0, audioprotopnet, MetaPerch) : la revue n'a
   testé que ConvNeXt-BirdSet, et moyenner la dernière carte y suffit (l'attention n'apporte
   rien). À vérifier pour les autres CNN.

Les stocks d'embeddings d'AnuraSet sont versionnés sur des branches git (66 à 144 Mo par
encodeur) : des jetons n'y tiennent pas.

## Contraintes du livrable et licences (29/09/2026)

- **Benchmark** : ni la taille ni la durée de calcul ne comptent (machines cloud ; le
  benchmark global du 29/09 a tourné sur des machines de 4 cœurs). La limite est le stockage
  des jetons.
- **Livrable** : portable Windows, i5-1145G7, 16 Go, CPU seul ; ONNX Runtime, NumPy,
  scikit-learn ; ni PyTorch ni TensorFlow à l'exécution (V4 §7) ; l'ONF réentraîne la tête et
  ré-encode par lots. Un encodeur qui ne s'exporte pas en ONNX, ou trop lent (débits :
  DECISIONS n° 72), reste une référence de benchmark ou un enseignant de la distillation
  (`feuille-de-route-V4.md`, option « Distillation (modèle maison) »).
- **Licences** (règle du projet : non commerciale = « non déployable », appliquée à BirdNET
  v2.4) :
  - **esp-aves2** : CC-BY-NC-SA-4.0 sur les **dix** points de contrôle publiés, variantes `-all`
    comprises (métadonnées Hugging Face lues le 29/09 : effnetb0-bio, -all, -audioset ; eat-bio,
    -all ; sl-eat-bio-ssl-all, sl-eat-all-ssl-all ; sl-beats-bio, -all ;
    naturelm-audio-v1-beats) → non déployable ; référence de benchmark, priorité 1. Le tableau
    du 29/09 matin disait effnetb0 « déployable sur l'i5 » : vrai pour la taille, faux pour la
    licence.
  - **naturebeats** : `esp-aves2-naturelm-audio-v1-beats` (l'encodeur BEATs de NatureLM-audio
    v1, probablement les mêmes poids) est CC-BY-NC-SA-4.0. **À vérifier avant de lui faire une
    place dans le livrable.**
  - **birdnet** (v2.4) : non commerciale. **birdnet_v3** : CC BY-SA 4.0, mais préversion et
    Terms of Use non lus.
  - **MetaPerch** : licence des poids non lue. Bird-MAE, ConvNeXt-BirdSet, AudioProtoPNet,
    BEATs : non relevée.
  - **Relevé du 29/09 au soir** (fiches Hugging Face ; feuille de route V5 §2, DECISIONS
    n° 156) : AudioProtoPNet CC BY-NC 4.0 → non libre ; NatureLM-audio CC BY-NC-SA 4.0
    (confirme la réserve sur naturebeats) ; Bird-MAE Base, Large et Huge, ConvNeXt-BirdSet et
    BioLingual : **aucune licence déclarée**, donc non libres tant que les auteurs n'en publient
    pas ; BEATs : dépôt `microsoft/unilm` sous MIT.
  - **Règle V5** : encodeur libre impératif pour le livrable ; au benchmark ONF, au plus deux
    non libres, seulement s'ils font mieux que le meilleur libre sur AnuraSet.

## Familles à plusieurs variantes (29/09/2026)

Oui, plusieurs encodeurs sont en fait des familles. Relevé (les variantes non testées par le
projet sont des candidats à ajouter) :

| Famille | Variantes | Dans le projet ou bacpipe |
|---|---|---|
| Bird-MAE (DBD) | Base, Large, Huge (`DBD-research-group/Bird-MAE-*`, les trois dépôts existent, vérifié le 29/09) | `birdmae_base`, `birdmae_huge` testés en linéaire ; la **Large** (≈ 300 M de paramètres, celle qu'évaluent la revue et MetaPerch, « BirdMAE-L ») : priorité 1 comme référence de benchmark (décision du 29/09) |
| AudioProtoPNet (DBD) | 1, 5, 10, 20 prototypes par classe (`AudioProtoPNet-{1,5,10,20}-BirdSet-XCL`) ; 5, 10 et 20 se valent, 1 est un peu moins bon | `audioprotopnet` : checkpoint chargé par bacpipe à lire dans son code |
| ESP-AVES2 | 11 points de contrôle (tableau ci-dessous) | aucun : adaptateur AVEX à écrire |
| AVES, BirdAVES (ESP, génération précédente) | plusieurs points de contrôle (non relevés) | bacpipe en expose un chacun (`aves_especies`, `birdaves_especies`), écartés |
| Perch | 1.0 (EfficientNet-B1, ≈ 8 M), 2.0 (EfficientNet-B3, ≈ 12 M, **une seule taille**), SurfPerch (récifs), MetaPerch ; ONNX `perch_v2_no_dft` fourni par bacpipe | `perch_bird`, `perch_v2` testés |
| BirdNET | v2.4 (TensorFlow, 48 kHz, 3 s) ; v3.0 préversion 3.1 (32 kHz, durée variable ; ONNX, PyTorch, TFLite, TensorFlow ; FP32 ≈ 540 Mo, FP16 ≈ 270 Mo, FP16 élagué ≈ 70 Mo ; un modèle global de 11 000 espèces et des modèles régionaux) | `birdnet`, `birdnet_v3` |
| BEATs (Microsoft) | points de contrôle préentraînés (SSL) et affinés sur AudioSet (la revue prend l'affiné) ; NatureBEATs = BEATs affiné dans NatureLM-audio | `beats`, `naturebeats` ; point de contrôle chargé par bacpipe pour `beats` : à lire |
| MetaPerch (Google) | BioBaseline (même réseau, sans métadonnées) et MetaPerch ; 9 sources de métadonnées étudiées dans l'article | à ajouter |
| insect (bacpipe) | `insect66`, `insect459` : deux jeux d'étiquettes du même type de modèle | à benchmarker |
| BatDetect2 (bacpipe) | `_clip_avg`, `_dets_avg` : deux façons d'agréger la même sortie | écartés |

### ESP-AVES2 : 11 points de contrôle, trois architectures

Article « What Matters for Bioacoustic Encoding » (arXiv 2508.11845, ICLR 2026 : 19 modèles sur
26 jeux de données). Collection Hugging Face `EarthSpeciesProject/esp-aves2`, bibliothèque AVEX
(`pip install avex`, `load_model("esp_aves2_…")`, `return_features_only=True` pour l'embedding).
16 kHz, mel-spectrogramme à 128 bandes, licence CC-BY-NC-SA-4.0. Données : Xeno-canto 10 416 h,
iNaturalist 1 539 h, Watkins 27 h, Animal Sound Archive 78 h (« bio », environ 12 000 h) ;
AudioSet 5 700 h en plus pour « all ». Les cartes ne listent pas d'amphibiens parmi les sources
(iNaturalist en contient peut-être quelques-uns) : c'est surtout des oiseaux.

Conclusion de l'article : SSL puis SL sur un mélange bio + audio général donne les meilleurs
résultats ; les données générales aident (EffNetB0-all ≥ EffNetB0-bio sur presque toutes les
métriques). Nom : `esp-aves2-<architecture>-<données>[-ssl-<données>]`, `sl` = supervisé.

| Point de contrôle | Architecture | Recette | Données |
|---|---|---|---|
| effnetb0-bio | EfficientNet-B0 (CNN) | supervisé, depuis EfficientNet-B0 préentraîné sur ImageNet | bio |
| effnetb0-all | idem | idem | bio + AudioSet |
| effnetb0-audioset | idem | idem | AudioSet seul |
| eat-bio | EAT (transformer) | auto-supervisé seul (distillation d'un enseignant et reconstruction de patchs masqués), 768 d | bio |
| eat-all | EAT | auto-supervisé seul | bio + AudioSet |
| sl-eat-bio-ssl-all | EAT | SSL sur all, puis SL sur bio (d'après le nom et la page AVEX) | bio |
| sl-eat-all-ssl-all | EAT | SSL sur all, puis SL sur all | all |
| sl-beats-bio | BEATs | BEATs SSL sur AudioSet, puis SL sur bio, 768 d | bio |
| sl-beats-all | BEATs | BEATs SSL sur AudioSet, puis SL sur all, 768 d | all |
| naturelm-audio-v1-beats | BEATs | l'encodeur BEATs de NatureLM-audio v1, dégelé pendant l'entraînement multimodal ; 768 d ; probablement les mêmes poids que `naturebeats` de bacpipe (à vérifier avant de le compter deux fois) | NatureLM-audio |
| sed-birdcode-ablation-ssl-beats-clip-pseudo-encoder | non relevée (ablation, d'après le nom) | non relevé | non relevé |

La page « Supported Models » d'AVEX étiquette `eat_bio` et `eat_all` « supervised learning » ;
les cartes Hugging Face disent auto-supervisé : à trancher au chargement.

## Perch 2.0 : prédiction de source et auto-distillation (arXiv 2508.04665)

Perch 2.0 : EfficientNet-B3 (≈ 12 M de paramètres), 32 kHz, fenêtres de 5 s, embedding moyen
de 1 536 dimensions, 14 795 classes (14 597 espèces), 1 542 778 enregistrements (Xeno-canto
896 255, iNaturalist 571 698, Tierstimmenarchiv 33 859, FSD50K 40 966). Trois têtes pendant
l'entraînement :

1. un **classifieur linéaire** sur l'embedding moyen (les espèces) ;
2. un **classifieur à prototypes**, sur les jetons spatiaux : 4 prototypes par classe,
   prédiction = maximum des activations, perte d'orthogonalité entre prototypes ;
3. une **tête de prédiction de source**.

**Prédiction de source** (idée de DIET, Balestriero 2023) : chaque *enregistrement* du jeu
d'entraînement est sa propre classe (plus de 1,5 million de classes, d'où une tête de rang 512),
et le réseau doit retrouver l'enregistrement d'origine à partir d'une fenêtre de 5 s, depuis
l'embedding moyen. L'augmentation de données est le fenêtrage : plusieurs fenêtres disjointes du
même enregistrement doivent recevoir la même classe. Effet recherché : forcer l'embedding à
garder des détails fins, pas seulement l'espèce. Les auteurs y voient un problème de
classification supervisée extrêmement fin. Poids de la perte : 0,1–0,9 en phase 1, 0–0,4 en
phase 2. Le papier ne donne pas d'ablation isolée. MetaPerch l'a retirée en supposant qu'elle
apprend implicitement de l'information liée aux métadonnées de l'enregistrement (lieu,
conditions d'enregistrement) : hypothèse des auteurs de MetaPerch. **Piste pour nous, non
testée** : un embedding entraîné à reconnaître l'enregistrement d'origine peut aussi encoder le
micro et le fond sonore, ce qui compterait devant les échecs de transfert entre sites.

**Auto-distillation** : deux phases. *Phase 1* (jusqu'à 300 000 pas) : on entraîne le
classifieur linéaire, la tête de source et, séparément, le classifieur à prototypes ; un
stop-gradient l'empêche d'influencer l'embedding. *Phase 2* (jusqu'à 400 000 pas), reprise du
meilleur modèle de la phase 1 : les prédictions (probabilités) du classifieur à prototypes
servent de **cibles douces** au classifieur linéaire. Les prototypes sont l'enseignant, le
linéaire l'élève, tous deux sur le même réseau d'embedding : d'où « auto ». Poids de la perte de
distillation 1,5–4,5 (4,22 au final). Le gain est modeste : ROC-AUC BirdSet 0,902 (fin de phase
1) → 0,907 (final), exactitude BEANS en sondage linéaire 0,835 → 0,839. Les auteurs supposent
que la distillation atténue le bruit d'étiquettes (Xeno-canto : les espèces de fond ne sont pas
annotées) au point que des fenêtres aléatoires valent des fenêtres choisies au pic d'énergie.

### Lire ces têtes par rapport aux nôtres (précisions du 29/09)

Schéma (la carte spatiale E_S vaut 5 × 3 × 1 536 dans le papier, 16 × 4 × 1 536 dans l'ONNX de
bacpipe) :

```
fenêtre 5 s → réseau (EfficientNet-B3) → carte spatiale E_S
   moyenne de E_S → E_A (1 536) → tête linéaire → 14 795 scores           (élève)
                                → tête de source
   E_S (cases locales) → prototypes, 4 par classe → 14 795 scores         (enseignant, stop-gradient)
phase 2 : perte = CE(linéaire, étiquettes) + λ · CE(linéaire, probabilités des prototypes)
```

- **Classifieur à prototypes (ProtoPNet, repris par AudioProtoPNet puis Perch 2.0)** : pour
  chaque classe, **4 vecteurs appris** (« prototypes », de la taille d'une case de la carte
  spatiale, 1 536), appris par descente de gradient. Une case de la carte est comparée aux
  prototypes ; le score d'une classe est l'activation maximale sur ses 4 prototypes (papier).
  « 4 par classe » : quatre motifs typiques par espèce (variantes de chant, par exemple) ; la
  perte d'orthogonalité les empêche de se recopier. Le maximum sur les cases de la carte est la
  définition classique de ProtoPNet (non relu dans le papier de Perch).
- **Ce n'est pas notre « prototype »** (`blanci/head.py`, `differential_prototype`) : le nôtre est
  **un seul** vecteur par espèce, w = moyenne des positifs − moyenne des négatifs, calculé en
  forme fermée (pas de gradient), sur l'embedding **moyenné** de la fenêtre ; le score est
  linéaire (w·x + b). Celui de Perch 2.0 est appris, multiple, **local** (une note brève dans une
  seule case n'est pas diluée par la moyenne) et non linéaire (maximum). Même idée de
  ressemblance à un exemple type, mécanisme différent. Il se rapproche plutôt de notre sonde
  attentive (`blanci/attentive.py`), qui vise elle aussi la dilution d'une note dans la fenêtre.
- **Classifieur linéaire de Perch 2.0 ≠ notre linear probe** : même forme (une couche linéaire
  sur l'embedding moyen), mais il est appris **pendant le préentraînement**, avec tout le
  réseau, sur 14 795 classes, et façonne l'embedding. Notre linear probe est une logistique
  apprise **après**, sur embeddings gelés, avec nos quelques annotations. Ses 14 795 logits sont
  ceux que le projet lit pour les congénères d'A. blanci (`logit_classes`, n° 70).
- **Pourquoi deux phases** : l'enseignant doit exister avant d'enseigner. En phase 1, la tête à
  prototypes apprend en « observant » l'embedding (stop-gradient : elle ne le modifie pas) ;
  quand elle est bonne, la phase 2 reprend le meilleur modèle de la phase 1 et allume la perte
  de distillation. La phase 2 est un raffinement : pas de mixup (N = 1 le plus souvent), taux
  d'apprentissage plus petit, moins de dropout, poids de la source réduit, poids de la
  distillation élevé (1,5–4,5). Deux recherches d'hyperparamètres (Vizier), une par phase.
- **Ce que la distillation apporte** : les probabilités de l'enseignant sont plus nuancées que
  l'étiquette (par exemple 0,7 / 0,2 / 0,1 au lieu de 1 / 0 / 0) : elles disent qu'une fenêtre
  contient sans doute aussi une seconde espèce que l'étiquette Xeno-canto ignore. L'article
  invoque seulement Allen-Zhu et Li (2022) : « la self-distillation améliore la performance ».
  L'explication par le bruit d'étiquettes est l'hypothèse des auteurs.
- **Transposition à notre projet (piste, non testée)** : la distillation de Perch 2.0 change
  l'**embedding** parce que l'encodeur s'entraîne ; avec un encodeur gelé, seule la
  distillation **entre têtes** est possible (enseignant : sonde attentive ou prototype ;
  élève : logistique sur l'embedding moyen, entraînée sur les probabilités de l'enseignant
  sur des fenêtres non annotées du site cible). Elle diffère de R30 (`logistic_to_prototype`),
  qui tire les **poids** vers le prototype, et de R47, qui tire vers la logistique : ici on tire
  les **sorties**. Le gain observé dans Perch 2.0 est petit (+0,005 de ROC-AUC).
  L'hypothèse « la prédiction de source encode le micro » (plus haut) est une autre affaire :
  elle concerne la prédiction de source, pas l'auto-distillation.

## MetaPerch (arXiv 2607.14072, ICML 2026)

Chasmai, Dumoulin, Hamer (Google), juillet 2026. Même réseau que « BioBaseline » : EfficientNet-B3
(≈ 12 M de paramètres), fenêtres de 5 s, embedding spatial avant agrégation (donc des jetons),
tête à prototypes et tête linéaire, mixup. **Sans** auto-distillation ni prédiction de source
(≠ Perch 2.0). Entraîné sur Xeno-canto, iNaturalist, Tierstimmenarchiv et d'autres sources (non
relevées). Nouveauté : la localisation, la date et d'autres métadonnées de l'enregistrement (9
sources étudiées) servent de **pertes auxiliaires pendant l'entraînement seulement** : rien à
fournir à l'inférence.

- **Poids** : note 1 du papier, « Model released at github:google-research/perch/metaperch ».
  Dépôt non vérifié (hors du périmètre de la session), licence des poids non lue, format non
  relevé (Perch est en TensorFlow/JAX : un export ONNX est nécessaire pour le livrable).
- **Face à Perch 2.0** (tableaux 2 et 3 du papier, MetaPerch en moyenne de 5 exécutions) :
  ROC-AUC BirdSet 0,906 contre 0,908 (cmAP 0,438 contre 0,431) ; sur PER (Amazonie péruvienne)
  0,801 contre 0,786 ; BEANS sans CBI, exactitude moyenne 0,870 contre 0,847 et cmAP moyen 0,512
  contre 0,502 ; RFCX (oiseaux et grenouilles) 0,209 contre 0,200. Les auteurs jugent le gain
  « limité » quand l'espèce **et** l'acoustique changent en même temps.
- **Face à leur base sans métadonnées**, les gains sont nets, surtout dans les régions
  sous-représentées du corpus d'entraînement (Amérique du Sud, Hawaï). Pertinent pour la
  Guyane : à tester sur les données ONF. Ce n'est pas un changement d'encodeur : c'est un
  successeur possible de perch_v2.

## Ce qui n'a pas été vérifié

- La grille de jetons de BEATs (8 × 31 × 768) : à mesurer au premier chargement (méthode du
  `notes.md`).
- Les poids et la licence de MetaPerch : le dépôt `google-research/perch` n'a pas été ouvert.
- L'identité des poids `naturelm-audio-v1-beats` et `naturebeats` (à comparer sur quelques
  fenêtres) ; le caractère auto-supervisé ou non de `eat-*` (cartes contre page AVEX).
- Les sources d'entraînement de BirdNET+ V3.0 (AnuraSet y est-il ?) et son architecture.
- La licence de Bird-MAE (absente des métadonnées Hugging Face).
- Perch 2.0 : la forme exacte de la perte de distillation (température, mélange avec la perte
  sur les étiquettes) et le maximum spatial du classifieur à prototypes ne sont pas relus dans le
  texte du papier ; le schéma ci-dessus est une lecture, pas une citation.
- Le point de contrôle que bacpipe charge pour `audioprotopnet` et `beats` ; que 66 et 459 sont
  bien des nombres d'espèces pour `insect66` et `insect459`.
- Les fiches ESP et BirdNET ont été lues par un outil de synthèse de pages web (pas en texte
  intégral) ; les chiffres des revues (Schwinger et al., Perch 2.0, MetaPerch) ont été lus dans
  le texte des PDF.

## Sources (consultées le 29/09/2026)

- BEANS : <https://arxiv.org/abs/2210.12300>
- Foundation Models for Bioacoustics, a Comparative Review : <https://arxiv.org/abs/2508.01277>
- Perch 2.0, The Bittern Lesson : <https://arxiv.org/abs/2508.04665>
- MetaPerch : <https://arxiv.org/abs/2607.14072>
- What Matters for Bioacoustic Encoding (ESP-AVES2) : <https://arxiv.org/abs/2508.11845> ;
  collection <https://huggingface.co/collections/EarthSpeciesProject/esp-aves2> ; page des modèles
  AVEX <https://projects.earthspecies.org/avex/supported_models.html>
- BirdNET+ V3.0 préversion : <https://zenodo.org/records/20703646>
- AudioProtoPNet : <https://huggingface.co/collections/DBD-research-group/audioprotopnet> ;
  Bird-MAE : <https://huggingface.co/collections/DBD-research-group/bird-mae>
