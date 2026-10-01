# Décisions d'implémentation

Ce fichier tient lieu de feuille de route. Il se lit en deux temps :

1. le **cadre du projet** (objectifs, protocole, plan d'annotation, livrable, planning, risques),
   repris de la feuille de route V5 ;
2. le **journal des décisions**, numérotées et datées : écarts au cadre, précisions qu'il ne
   tranche pas, résultats qui changent la suite. Une entrée par décision ; une décision remise en
   cause reçoit une nouvelle entrée, l'ancienne reste. Une entrée l'emporte sur le cadre.

Les renvois « §n » désignent les sections du cadre. La table des matières du journal : les titres
datés `## 2026-…` ci-dessous (`grep '^## ' DECISIONS.md`).

## Cadre du projet (repris de la feuille de route V5)

Stage ONF Guyane, 15/09/2026 → 14/03/2027 : détection acoustique automatique d'*Anomaloglossus
blanci*. Cette section est ce qui reste utile de la feuille de route V5 (29/09/2026, après
l'entretien de Léonard avec Élodie), reprise ici le 01/10/2026 quand le fichier a été supprimé.
**Les numéros de section (§0 à §14) sont conservés** : les renvois « §5 », « §6 », « V5 §7 »… de
ce journal, du code et de la configuration désignent les sections ci-dessous. En cas de
désaccord, **une décision numérotée plus bas l'emporte sur le cadre**.

Légende : `[HYPOTHÈSE Hn]` = information manquante (§12) ; `[À VÉRIFIER]` = à contrôler sur les
données ou les sources ; « établi » = bibliographie (dont Courtois et al. 2025, voir
`documentation/biblio/biblio.md`) ou source vérifiée ; « jugement » = arbitrage d'ingénieur ;
« tuteur » = confirmé en réunion ; « Élodie » = décidé à l'entretien du 29/09.

Archives : les feuilles de route V1 à V4 sont dans `documentation/old/` ; la V5 complète
(≈ 83 Ko, avec le journal v2 → v5, la montée en compétence et les interfaces détaillées) reste
dans l'historique git : `git show ad43369:documentation/feuille-de-route-V5.md`.

### Positions des encadrants (établi, extraits reçus)

- **Tuteur (B)** : idée initiale = embeddings BirdNET avec entraînement progressif *human in the
  loop* pour ajouter des sites ; BirdNET et Perch + linear probing lui ont donné satisfaction ;
  ouvert à d'autres modèles ; suggère bacpipe ; NatureLM « limité à 16 kHz, très limite pour les
  paysages sonores ».
- **Encadrant A (auteur de ProtoCLR)** : éviter BirdNET (TensorFlow, dépendance à la base de code
  de Cornell, extraction d'embeddings peu pratique ; seule l'interface graphique vaut, pour les
  non-codeurs) ; modèles Hugging Face ; d'après la revue d'*Ecological Informatics* (Schwinger et
  al. 2026), BEATs_NLM et Bird-MAE au-dessus du lot ; ProtoCLR pas loin mais pas facile à
  réutiliser, non recommandé ; Streamlit ou Gradio pour la réannotation.
- **Arbitrage** : aucun TensorFlow à l'exécution dans le livrable ; benchmark bacpipe sur
  `birdmae`, `beats`, `naturebeats`, `perch_v2`, `birdnet` (référence du tuteur, non déployable :
  licence, TensorFlow) ; `protoclr` seulement s'il tourne sans effort ; 16 kHz tranché par le
  contrôle passe-bas (§2).
- **Élodie (entretien du 29/09/2026)** : encodeur libre impératif ; annotations reprises de zéro,
  vérifiées par elle et Benoît (experts naturalistes) avant tout benchmark ONF ; moins de temps
  sur les benchmarks, plus sur l'interface finale ; critères : performance d'abord, puis facilité
  d'utilisation (et durée d'encodage) ; pas plus d'une ou deux lignes de code pour un
  naturaliste. Elle propose, sans certitude et en demandant l'avis de Sylvain, d'écouter en
  priorité février–mars (pic d'activité de son rapport de phénologie) et les pics journaliers,
  avec aussi du milieu de journée pour trouver des chants manqués (avis en §5.8).

### §0. Objectifs et critères

**Objectif** : un détecteur d'*A. blanci* qui fonctionne sur les **points d'écoute des futures
campagnes** (couples micro + site jamais vus), livré à l'ONF sous une forme qu'un naturaliste
utilise sans écrire de code et que l'ONF réentraîne seul après le stage.

| Rang | Critère | Mesure (§6) | Rôle |
|---|---|---|---|
| 0 | Encodeur libre d'accès | poids publics, licence qui permet l'usage par l'ONF | **filtre** : un encodeur non libre n'entre jamais dans le livrable |
| 1 | Performance du modèle | AP et rappel par enregistrement sur des points tenus à l'écart | départage d'abord |
| 2 | Facilité d'utilisation | installation et usage sans code ; tâches réussies seul | départage ensuite |
| 3 | Durée d'encodage | heures pour une campagne d'une semaine sur l'i5 de l'ONF | seulement à performance égale |

« Performance égale » : l'intervalle apparié contient zéro (§6). La licence n'est plus un critère
de départage : c'est un filtre posé avant toute comparaison.

**« Libre d'accès »** (jugement, à confirmer par Élodie, Q1 du §10) : poids téléchargeables sans
demande **et** licence qui autorise l'usage par l'ONF sans accord particulier. L'ONF est un
établissement public à caractère industriel et commercial : une clause « non commerciale » ne le
couvre pas à coup sûr. Libres : Apache 2.0, MIT, BSD, CC BY, CC BY-SA (CC BY-SA oblige à
redistribuer les dérivés sous la même licence ; une tête entraînée sur les embeddings n'est pas
un dérivé des poids `[À VÉRIFIER]`). Non libres : clause NC ; **aucune licence déclarée** (par
défaut « tous droits réservés ») ; accès sur demande.

**Livrables, par priorité** : (1) jeu annoté v1, vérifié par Élodie et Benoît (§5) — sans lui,
aucun chiffre ne vaut ; (2) détecteur (encodeur libre + tête) évalué sur des points tenus à
l'écart (§6) ; (3) application Windows installable sans code et guide de réentraînement pour
l'ONF (§7) ; (4) rapport de stage et soutenance (mars 2027) ; (5) si 1 à 4 sont finis en avance :
article de phénologie et gabarit pour d'autres espèces (§14).

### §1. Le problème

**Cadre (tuteur)** : détecteur robuste d'*A. blanci* sur de nouveaux sites ; métriques AP et
rappel ; six mois, MacBook M4 16 Go ; logiciel prenable en main par un naturaliste ; comparaison
des approches par score, vitesse, prise en main.

**Données**
- Enregistreurs : Wildlife Acoustics Song Meter Mini 2 partout ; fréquence d'échantillonnage et
  format lus dans les en-têtes (96 176 des 96 292 enregistrements sont en 48 kHz stéréo, WAV et
  FLAC, n° 153).
- **Jeu 2023** (établi, Courtois et al. 2025) : 6 enregistreurs, 2 par site (Kaw A/B, Molokoï
  E/F, Trésor C/D), 25/11/2023 → 24/11/2024, 2 min toutes les 30 min de 5 h à 20 h, ≈ 2 200 h ;
  capteurs de température et d'humidité associés.
- **Jeu 2026** (inventaire du 29/09, n° 153) : 5 sites (CDR, Mataroni, Patawa Est, Patawa Ouest,
  RNRT), **un relevé d'environ une semaine par site**, décembre 2025 → février 2026 (mois de
  chaque relevé `[À VÉRIFIER]` par `blanci status`) ; 2 min toutes les 30 min de 5 h à 19 h 30 ;
  29 513 enregistrements, 979 h. **En 2026, le mois est confondu avec le site.**
- Total : 96 292 enregistrements, 3 204 h de 120 s, 2 216 Go ; 94 588 encodables après les
  drapeaux ; 4,5 millions de fenêtres de 5 s au pas de 2,5 s.
- **Annotations : reprises de zéro** (§5). Les 345 positifs et 150 négatifs Blancinet (51
  enregistrements, 13 micros, tous à Mataroni, n° 34–35) restent dans la base (ajout seul) mais
  sortent de l'entraînement et de l'évaluation : ils sont conditionnés par un détecteur, ne
  couvrent qu'un site, et leurs négatifs sont des faux amis choisis, pas un échantillon du stock.
  Un jeu étiqueté conditionné par un détecteur ne contient pas les chants que ce détecteur n'a
  pas vus : le rappel mesuré dessus est relatif. D'où un premier lot tiré sans aucun détecteur.

**Phénologie (établi, Courtois et al. 2025)**
- Journalier : pics 7–9 h et 15–17 h à Kaw et Trésor ; à Molokoï, activité haute et constante de
  7 h à 17 h en saison.
- Annuel : activité forte de janvier à avril (jusqu'en juin à Trésor et Molokoï) ; quasi nulle de
  juillet à octobre ; reprise en novembre–décembre avec les premières pluies. Septembre–novembre
  est donc la saison basse du stage.
- Détection : à Molokoï, probabilité journalière ≈ 1 de fin novembre à mars ; à Trésor, maximale
  en décembre puis ≈ 0,5 jusqu'en juin ; Kaw_B ne dépasse jamais 0,5.
- Chant continu (tuteur) : elle chante du début à la fin des 2 min ; l'indice horaire du rapport
  atteint rarement 1, mais ce chiffre mélange silences réels et rappel du détecteur (0,69) : H20
  reste ouverte.

**Problème d'apprentissage** : espèce à note brève mais à chant continu, dans une bande saturée,
avec ≈ 50 sources de fausses alarmes recensées (oiseaux, amphibiens, orthoptères, artefacts).
Chant d'*A. blanci* : note simple de 0,090–0,103 s (moyenne 0,094 s), légère modulation
ascendante (≈ 0,1 kHz), fréquence dominante 4,48–5,41 kHz (moyenne 4,75 kHz), intervalle entre
notes 1,414 s en moyenne (1,200–1,906 s), structure harmonique développée ; mâles chantant au
bord des cours d'eau le jour, pic à l'aube (6–7 h) et en fin d'après-midi (16–17 h) en saison des
pluies, aussi les jours humides de saison sèche. Congénères : *A. surinamensis* (note 0,028–0,037
s, intervalle 0,37–0,83 s, 4,55–5,35 kHz) ; *A. degranvillei* (3,60–3,62 kHz, notes plus longues
de 0,157–0,160 s). Ces valeurs sont dans `config/default.yaml` (section `signal`).

**Unité de décision (jugement)**

| Niveau | Rôle | Règle |
|---|---|---|
| Fenêtre (3 s ou 5 s selon l'encodeur, pas ≤ moitié) | apprentissage, score, vérification | métriques de développement |
| Enregistrement (2 min) | **unité principale** ; score = proportion de fenêtres où *A. blanci* est détectée | scores faibles à vérifier ; détection sur des fenêtres isolées → file « suspect » |
| Point × période | décision gestionnaire `[H5]` | « à vérifier » dès qu'un enregistrement est positif ; points classés par force (fraction, nombre d'enregistrements, jours) ; « présence confirmée » après validation humaine ; sinon « non détecté », jamais « absent » |

**Écart métrique / décision** : fausse absence irréversible, fausse présence réversible → rappel
prioritaire ; chant continu → un créneau en saison suffit, la contrainte est la **saison**.

**Échelle** (note de 0,09 s dans une fenêtre de 3–5 s) : fenêtres pleines par défaut. Contre le
détecteur amont : bande saturée, rappel plafonné. Le seuillage spectral ne sert qu'aux onsets du
module séquentiel. Variantes à comparer (plis par micro) : A grille standard ; B pas resserré ;
C fenêtres centrées sur onsets ; E agrégation des jetons (moyenne / max / attention) ; C adopté
seulement s'il dépasse A et B de plus que l'incertitude.

### §2. Benchmark des encodeurs

- **AnuraSet clôt le choix des candidats.** La vague 2 (`anuraset/VAGUE_ENCODAGE_2.md`)
  se termine le **09/10/2026** ; les sessions non finies ce jour-là sont abandonnées. Pas de
  nouvelle tête ni de nouvelle régularisation sur AnuraSet, fiches courtes (≤ 30 lignes).
- **Filtre de licence (§0) avant le benchmark ONF.** Tous les encodeurs libres qui passent le
  témoin BOAFAB (n° 151) y vont, plus **au plus deux non libres**, choisis parmi ceux qui font
  mieux que le meilleur libre sur AnuraSet : objectifs à battre, jamais livrés. BirdNET 2.4
  resterait le seul non libre comme repère connu des naturalistes (à valider avec Élodie).
- **Pas de benchmark sur les données ONF avant le go d'Élodie et Benoît** (§5.7). Ensuite : les
  encodeurs retenus × logistique (plus une tête sur jetons pour un transformer libre, n° 151),
  courbe d'amorçage par point ; **choix de l'encodeur le 20/11/2026**, puis plus aucun
  benchmark d'encodeur.

**Licences relevées (29/09/2026 ; fiches Hugging Face, dépôts, `documentation/encodeurs-bacpipe.md`)**

| Encodeur | Licence des poids | Statut |
|---|---|---|
| perch_v2 (Perch 2.0) | Apache 2.0 | **libre** |
| perch_bird (Perch 1) | Apache 2.0 `[À VÉRIFIER]` (page Kaggle) | libre sous réserve |
| birdnet_v3 (BirdNET+ 3.0, préversion) | CC BY-SA 4.0 ; mention « Powered by BirdNET », braconnage et usage militaire interdits (n° 151) | **libre** ; ONNX officiel |
| beats | dépôt `microsoft/unilm` sous MIT | libre sous réserve |
| birdnet (2.4) | CC BY-NC-SA 4.0 | non libre : repère des naturalistes |
| esp-aves2 (dix points de contrôle) | CC BY-NC-SA 4.0 (n° 151) | non libre |
| naturebeats | NatureLM-audio : CC BY-NC-SA 4.0 ; mêmes poids `[À VÉRIFIER]` | non libre |
| audioprotopnet | CC BY-NC 4.0 | non libre |
| Bird-MAE (Base, Large, Huge), convnext_birdset, biolingual | aucune licence sur la fiche Hugging Face ; dépôt de code non lu | non libre tant qu'aucune licence n'est publiée ; demander aux auteurs si l'un gagne |
| protoclr, rcl_fs_bsed, mix2, avesecho_passt, insect66, insect459, MetaPerch (poids pas publiés) | non relevée | à relever avant le benchmark ONF |

**Méthode**
- Colonnes : licence ; exécution sur M4 ; vitesse sur M4 et projection sur i5-1145G7 ; prise en
  main ; jetons accessibles ; AP ; rappel à précision 0,5 et 0,1 ; kNN top-1.
- Pré-benchmark AnuraSet (Cañas et al. 2023) : deux ou trois anoures à note brève en 3–6 kHz,
  plis par site, milliers de positifs. Indicateur, pas garantie.
- Sondes : kNN cosinus ; prototype différentiel ; régression logistique L2. Validation **groupée
  par micro**. Comparaisons appariées sur les mêmes plis, bootstrap.
- Générateur de candidats : Perch 2.0 contient *A. baeobatrachus*, *A. stepheni*,
  *A. surinamensis* ; leurs logits (non calibrés) donnent une première liste de candidats et des
  descripteurs optionnels.
- Mesures trompeuses : exactitude, AUROC, F1 au seuil 0,5, AMI/ARI.
- `perch_v2` (TensorFlow, GPU) : wrapper CPU `[À VÉRIFIER]` ; `perch_v2_no_dft.onnx` repéré dans
  un notebook BirdCLEF+ 2026, à valider contre des embeddings de référence ; sinon extraction
  déportée pour le benchmark seul.
- Désaccords → mesures : AP(birdnet) contre les autres ; BEATs/NatureBEATs à 16 kHz contre
  Bird-MAE à 32 kHz **avec contrôle** (Bird-MAE sur audio filtré à 8 kHz : si le contrôle égale
  l'encodeur natif, l'objection du tuteur ne s'applique pas à ce signal).

### §3. Cartographie des approches

| Approche | Principe | Annotations | Verdict |
|---|---|---|---|
| Template matching | corrélation croisée de spectrogrammes | 1–10 | baseline obligatoire, indépendante de l'encodeur |
| Seuillage spectral | passe-bande 4,4–5,5 kHz, enveloppe, seuil, durée 0,08–0,11 s | 0 | onsets du module séquentiel ; pas de filtre amont |
| Indices acoustiques | statistiques par enregistrement | 0 | contrôle qualité : pluie, saturation, « micro dans sac » |
| Prototype simple | cosinus au centroïde des positifs | 5–30 | baseline de similarité, sensible au fond partagé |
| Prototype différentiel | ⟨x, μ₊ − μ₋⟩ + b, négatifs appariés | 10–50 | baseline permanente ; retire le fond partagé |
| Linear probing | régression logistique L2 sur embeddings gelés | dizaines → centaines | **tête principale** |
| Attentive probing | tête d'attention sur jetons pris avant agrégation | ≥ 150–200 | si jetons accessibles (nécessaire pour les transformers) |
| Logits de congénères (Perch 2.0) | scores des 3 *Anomaloglossus* connus | 0 | générateur de candidats ; descripteur optionnel |
| Clustering | HDBSCAN sur ACP ; UMAP pour voir | 0 | §5 bis |
| LoRA / fine-tuning | adaptation partielle ou totale (`blanci/heads/finetune.py`, réservé) | centaines à milliers | conditionné ; hors chemin critique |
| Distillation / modèle maison (`blanci/heads/detectors/`, réservés) | petit CNN bande 3–7 kHz imitant la chaîne gelée | 0 | livrable léger pour l'i5 ; après le choix de l'encodeur |

**Prototype différentiel** : x ≈ c_site + c_espèce + ε ; w = μ₊ − μ₋ ≈ c_espèce avec des négatifs
**appariés** (mêmes micros, heures, jours) ; centroïde le plus proche sous variance commune =
analyse discriminante linéaire à covariance identité = solution fermée de la régression
logistique ; première instance de l'arithmétique d'embedding (tuteur).

**Module séquentiel** (descripteurs calculés depuis l'audio, hors encodeur) : rythme intra-fenêtre
(débuts de notes dans la bande) ; persistance (fraction de fenêtres positives dans
l'enregistrement, continuité entre fenêtres voisines, créneaux voisins du même micro : sépare
chant continu et chant ponctuel, isole les détections uniques) ; solo contre chœur `[H21]`
(densité d'onsets, chevauchements, étalement spectral). **Heure et saison restent hors du
classifieur par défaut** (risque d'apprendre la phénologie de Mataroni) : elles servent à
l'échantillonnage, au classement des points et à un drapeau de plausibilité issu des courbes de
Courtois et al. ; entrée du classifieur seulement si validée sur Trésor et Kaw.

**Tête de fusion (stacking à deux niveaux)** : `head` et `sequential` au niveau 1 ; régression
logistique au niveau 2 sur (score de `head` **hors-pli**, 2–4 descripteurs, logits de congénères
en option). Hors-pli = score produit par une version de `head` entraînée sans l'exemple.
≈ 10 enregistrements positifs indépendants par coefficient. Le score séquentiel module, jamais
de veto.

### §4. Architecture

```
audio brut (Song Meter Mini 2)
  └─ ingest      inventaire, métadonnées (jeu, site, micro, horodatage), drapeaux QC → SQLite
  └─ decode      forme d'onde float32 mono, f_e native (soundfile)
  └─ grid        fenêtres (recording_id, offset_s, dur_s), indépendantes de l'encodeur
  └─ encoder[k]  rééchantillonnage vers f_e(k) ; spectrogramme interne ; E_k ∈ R^{N×d_k}
  └─ store       Parquet partitionné encodeur / jeu / site / mois, float16 (audio intact)
  └─ index       cosinus exhaustif par fragments ; requêtes positives et négatives empilées
  └─ head[v]     régression logistique → score par fenêtre
  └─ sequential  rythme, persistance, solo/chœur → fusion
  └─ aggregate   fenêtre → enregistrement (fraction) → point (classement)
  └─ queue       file de vérification → labels en ajout seul → réentraînement de head
```

- `decode` s'arrête à la forme d'onde ; le log-mel est calculé dans chaque encodeur.
- Labels attachés à (enregistrement, décalage) ; encodeur derrière un `Protocol` ; GUI → couche
  de service (`blanci/service.py`).
- Volume : ≈ 4 millions de fenêtres → 6 Go en 768-d float16, 12 Go en 1 536-d.
- Changement d'encodeur : ré-encodage en tâche de fond ; tête réentraînée sur les mêmes labels ;
  rapport de migration sur le jeu gelé ; décisions estampillées (`encoder_id`, `head_version`,
  `threshold_id`).
- Croissance : encodeur figé ; index en ajout ; tête réentraînée de zéro sur tous les labels
  (réentraînement périodique) ; seuils recalibrés.

### §5. Annotation et apprentissage actif (reprise de zéro)

**Décision (Élodie)** : on repart de zéro. Léonard tire et annote seul, envoie un sous-échantillon
à Élodie et Benoît qui le vérifient. Aucun benchmark sur ces données avant leur go. But : un jeu
aussi équilibré que possible pour **généraliser à de nouveaux points**.

#### 5.1 Principes (jugement)

1. **Le point d'écoute (site + micro, n° 38) est l'unité qui compte**, ni la fenêtre ni
   l'enregistrement : 40 fenêtres d'un enregistrement valent à peine plus qu'une. Beaucoup de
   points, peu d'enregistrements par point. Benchmark 07 : sur un site neuf, 5 à 10
   enregistrements positifs annotés relèvent l'AP de 0,81 à 0,89 ; au-delà, c'est de la
   diversité qu'il faut.
2. **Deux jeux, deux règles.** L'**entraînement** peut être enrichi là où l'espèce chante (un
   tirage biaisé ne fausse pas une tête si les négatifs couvrent tous les fonds). L'**évaluation**
   est tirée au hasard dans des strates, sur des **points tenus à l'écart**, avec la probabilité
   de tirage de chaque enregistrement : sans elle, aucune métrique ne se ramène au stock réel.
3. **Aucun détecteur dans le tirage du premier lot** (ni Blancinet, ni nos têtes). Les modèles
   reviennent après le go, pour les lots d'entraînement seulement (§5.9).
4. **Chaque strate qui donne des positifs donne aussi des négatifs**, des mêmes points aux mêmes
   heures. Sinon la tête apprend l'heure, la saison ou le site à la place du chant.
5. **Tout tirage est fait par un script à graine fixée, avant écoute**, et sa liste est
   versionnée.

#### 5.2 Partition des points, avant toute écoute

| Jeu | Points | Pourquoi |
|---|---|---|
| Évaluation, niveau 2 | **un site 2026 entier**, choisi avec Élodie : présence connue, ≥ 10 points | mesure « nouveau site » (futures campagnes) |
| Évaluation, niveau 1 | **20 % des points** de chacun des autres sites 2026, au hasard | mesure « nouveau point d'un site connu » |
| Évaluation, niveau 3 | 2023 : **une station sur deux** par site (A ou B, au hasard) | toutes les saisons, sur un enregistreur jamais vu |
| Entraînement | tout le reste | — |

Un point tenu à l'écart ne donne jamais rien à l'entraînement, à la validation croisée ni au
réglage du seuil. Les enregistrements qui portent un label Blancinet sont exclus du tirage
d'évaluation. Pour 2023, le point est la station (Kaw_A, Kaw_B…) `[À VÉRIFIER]`.

#### 5.3 Strates

- **Tranche horaire** (heure locale) : aube 5–7 h ; pic du matin 7–9 h ; journée 9–15 h ; pic du
  soir 15–17 h ; soir 17–20 h.
- **Période** (2023 seulement) : haute (décembre–avril, dont **février–mars** en sous-strate) ;
  transition (mai–juin, novembre) ; basse (juillet–octobre). En 2026, la période est fixée par le
  site : on stratifie par point et par tranche horaire.
- Dans une strate : jour tiré au hasard, puis enregistrement tiré au hasard ; au plus un
  enregistrement par point, jour et tranche. Les enregistrements écartés par un drapeau (n° 79,
  153, 154) sont exclus ; pluie et saturation restent (conditions réelles).

#### 5.4 Lot d'entraînement 1 : environ 450 extraits de 30 s

| Jeu | Part | Répartition |
|---|---|---|
| 2026, points d'entraînement | ≈ 300 extraits | **3 par point** : un au pic du matin, un au pic du soir, un dans une autre tranche tirée au hasard |
| 2023, stations d'entraînement (3) | ≈ 150 extraits (50 par station) | période haute 50 % (dont deux tiers en février–mars), transition 20 %, basse 30 % ; pics 60 %, autres tranches 40 % |

Au total ≈ 60 % des extraits aux heures de pic, 40 % hors pic. Le quota par point se cale sur le
décompte réel (`blanci status`). Ce lot mesure aussi la **prévalence** par strate
(`decision.prevalence`, R73, n° 128). Temps : ≈ 1 min par extrait, soit 8 à 10 h (à chronométrer
sur les 50 premiers).

#### 5.5 Jeu d'évaluation v1 : environ 250 enregistrements entiers

- Points tenus à l'écart (§5.2) ; enregistrements de 2 min **écoutés en entier** (l'unité de
  décision est l'enregistrement ; l'écoute entière teste H20).
- Tirage stratifié (tranche horaire, et période en 2023). Les strates de pic, et la période haute
  en 2023, sont **surreprésentées deux fois** ; la probabilité de tirage est notée et les
  métriques repondérées (§6).
- Cible : **au moins 60 enregistrements positifs** (rappel 0,9 → Wilson [0,80 ; 0,95]) et au
  moins 150 négatifs ; sinon on complète dans les strates de pic, probabilités notées.
- Après le go (§5.7), c'est le **jeu gelé v1** (`blanci freeze`), jamais entraîné. Il remplace le
  jeu gelé de 60 enregistrements et l'audit aléatoire de 300 enregistrements de Mataroni.
  Temps : ≈ 3 min par enregistrement, ≈ 12 h.

#### 5.6 Que noter, et combien de fenêtres par enregistrement

- **Unité d'écoute** : un extrait de **30 s** (position tirée au hasard) à l'entraînement ;
  l'enregistrement entier à l'évaluation.
- **On note les intervalles** où *A. blanci* chante (début, fin, à 0,5 s près, sûr ou incertain)
  et, pour l'extrait : familles présentes (schéma ci-dessous), qualité A/B/C, pluie, chœur ou
  solo si on l'entend, canal écouté.
- **Les labels de fenêtres se déduisent des intervalles, pour n'importe quelle grille** (3, 5 ou
  6 s) : positive si dans un intervalle, négative si aucun n'est touché, « bord » sinon (exclue de
  l'entraînement). Les labels ne dépendent plus de l'encodeur.
- À l'entraînement, **chaque enregistrement pèse autant** (poids 1 / nombre de fenêtres).
- Pourquoi 30 s : ≈ 20 notes, de quoi reconnaître le rythme ; 1 min par extrait contre 3 min par
  enregistrement entier : trois fois plus de points et de jours pour le même temps. À vérifier sur
  le jeu d'évaluation : un chant absent des 30 s tirés mais présent ailleurs. Au-delà de 10 %,
  passer à 60 s (risque 5, §9).

#### 5.7 Vérification par Élodie et Benoît, et règle du go

Paquet **à l'aveugle** (label de Léonard caché, ordre mélangé), envoyé fin S6 :

| Contenu | Part vérifiée | Nombre attendu |
|---|---|---|
| Tout ce que Léonard a noté « incertain » | 100 % | ≈ 40 |
| Positifs du jeu d'évaluation | 100 % | 60–80 enregistrements |
| Négatifs du jeu d'évaluation | 20 %, au hasard | ≈ 35–40 enregistrements |
| Extraits d'entraînement | au hasard, stratifiés par site | 60 positifs + 100 négatifs |
| Recouvrement (deux experts) | 30 éléments pris dans ce qui précède | accord entre experts (`blanci agreement`) |

≈ 15 à 20 % de ce que Léonard a annoté ; ≈ 2 h 30 d'écoute par expert. Format sans installation :
dossier d'extraits WAV nommés par identifiant et feuille Excel (identifiant, *A. blanci* oui / non
/ incertain, commentaire) ; outil au choix (Raven, Audacity, Kaleidoscope, poste d'annotation).

**Règle du go** (jugement ; bornes unilatérales à 95 %) : 0 erreur sur 60 positifs de Léonard
(taux d'erreur ≤ 5 %) ; au plus 1 chant manqué sur 100 négatifs (≤ 4,7 %) ; au-delà, la strate
fautive (site, tranche, type de fond) est réécoutée puis contrôlée par un nouveau tirage. Toute
réponse d'expert s'ajoute comme label (annotateur = l'expert) et prime ; la base reste en ajout
seul. Le go s'écrit dans ce journal, avec ses chiffres ; avant, aucune tête n'est jugée sur ces
données.

#### 5.8 Avis sur les propositions d'Élodie (à discuter avec Sylvain)

**Écouter en priorité février–mars ? Oui pour récolter des positifs, non comme seule période.**
En 2026 chaque site n'a été enregistré qu'une semaine et le mois y est confondu avec le site :
« février–mars » reviendrait à choisir des sites, donc moins de points. Une tête qui n'a vu que
des fonds de saison des pluies rencontrera en saison sèche d'autres insectes et oiseaux, source de
fausses alarmes qui dessineraient une fausse activité de juillet à octobre. Les chants de début
et de fin de saison, plus rares et peut-être plus faibles `[À VÉRIFIER]`, datent la saison : la
tête doit en avoir vu. Un jeu d'évaluation tiré en février–mars ne dit rien des autres mois.
Proposition : en 2023, période haute 50 % du tirage dont deux tiers en février–mars (facteur 1,7
par rapport au calendrier), transition 20 %, basse 30 %.

**Pics journaliers en priorité, et milieu de journée pour les faux négatifs ? D'accord**, en
tirant aussi des négatifs aux mêmes points aux heures de pic (sinon la tête apprend
« matin = *A. blanci* »). Les heures creuses servent autant aux négatifs qu'aux chants manqués ;
le chœur d'oiseaux de l'aube (5–7 h) et les amphibiens du soir (17–20 h) sont des sources
probables de faux amis. Proposition : 60 % aux heures de pic, 40 % ailleurs.

**Questions pour Sylvain** : (1) enrichir l'évaluation (×2 aux pics) et repondérer, ou tirer
proportionnellement ? (2) 30 s à l'entraînement et enregistrement entier à l'évaluation ?
(3) la partition des points est-elle trop ou pas assez ? (4) poids 1 / nombre de fenêtres par
enregistrement, ou sous-échantillonnage ? (5) après le go, mélanger des lots d'apprentissage actif
à l'entraînement si l'évaluation reste purement probabiliste ?

#### 5.9 Après le go : lots suivants (entraînement seulement)

- Apprentissage actif : file de 60 % d'incertains, 20 % de scores maximaux, 20 % tirés au hasard
  et stratifiés. Récolte par similarité sur les points peu servis ; negative mining parmi les
  mieux classés. Ces lots ne vont **jamais** dans l'évaluation ; leur source est notée
  (`active`, `similarity`).
- Vérification experte : 10 % tirés au hasard par lot, plus les incertains.
- Seuil d'un site : 5 à 10 enregistrements positifs vérifiés sur place (le seuil ne voyage pas
  d'un site à l'autre, benchmark 07) ; c'est aussi la procédure de l'ONF pour chaque nouvelle
  campagne (§7).
- Arrêt : gain d'AP sur le jeu gelé v1 inférieur à 0,02 sur deux tours.

#### 5.10 Poste d'annotation

Poste Streamlit existant (`blanci annotate`, écoute des deux canaux) avec un nouveau mode
**« extrait + intervalles »** : curseur double sous le spectrogramme de 2 à 8 kHz, raccourcis
clavier, labels du schéma. Export du paquet d'experts (WAV + Excel, à l'aveugle) et réimport de
leurs réponses comme labels. YAPAT et Whombat restent possibles : outils de travail, pas
livrable.

#### 5.11 Classification : à trancher avant la première écoute

Ce qu'on note à l'écoute fixe ce que la tête pourra apprendre : la décision précède le lot 1
(n° 160).

| Option | Classes | Pour | Contre |
|---|---|---|---|
| A. Une seule classe | *A. blanci* / non | décision gestionnaire binaire ; toutes les données servent une frontière | ne sait pas qu'un chœur proche et un mâle lointain sont le même animal |
| B. Trois niveaux | clair / chevauchement / lointain (qualité A/B/C) | rappel par niveau ; C est celui qu'on rate | un **degré** de la même classe, pas une classe |
| C. Chœur contre solo | blanci chœur / solo / non | pèse sur la phénologie et le module séquentiel (H21) | H21 non levée ; peu de solos |
| D. Hiérarchie | niveau 1 : blanci / non ; niveau 2 : solo / chœur ; attribut : qualité | A pour décider, B et C pour apprendre et mesurer | annotation plus longue ; sous-classes rares |

Classification hiérarchique et apprentissage multitâche existent (BirdNET, Perch rangent leurs
espèces par genre et famille) ; la tête R67 (blanci / congénères / faux amis / bruit, n° 124) en
est une version à un niveau côté négatifs. **C'est une régularisation** au sens large : une tâche
auxiliaire contraint la représentation sans changer la décision ; effet non garanti avec peu
d'exemples par sous-classe.

**Recommandation (jugement) : annoter fin, décider gros (option D).** (1) À l'écoute, chaque
intervalle positif reçoit sa sous-classe (solo, chœur, indécis) et sa qualité (A, B, C) ;
« incertain » est un doute de l'annotateur, exclu de l'entraînement (n° 3). (2) Décision et
livrable : une seule sortie, *A. blanci* présente ou non. (3) Qualité A/B/C : attribut (rappel par
qualité, éventuellement poids d'entraînement). (4) Sous-classes en tâches auxiliaires seulement si
chacune compte ≥ 30 enregistrements indépendants après le lot 1, adoptées si l'AP binaire sur les
points tenus à l'écart progresse (intervalle apparié qui exclut zéro). (5) Toute sous-classe se
replie sur *A. blanci*.

**Schéma de labels** : positifs {blanci-solo, blanci-chœur, blanci-incertain} ; négatifs par
famille {oiseau:<espèce>, amphibien:<espèce>, orthoptère, cri-de-contact-amphibien, pluie,
artefact:micro-dans-sac, fond, autre} ; espèces recensées : fourmilier tacheté (le plus
fréquent), moucherolle, manakin, tangara mordoré, pigeon plombé, évêque de Rothschild, sclérures,
myrmidon, psittacidés, pic à cou rouge, martinet ; *Adenomera andreae*, *Allobates femoralis*,
*A. hahneli* (n° 10), *Hyalinobatrachium cappellei / mondolfii / iaspidiense*, *Amazophrynella
teko*, *Otophryne* ; grillons. Noms scientifiques des oiseaux `[À VÉRIFIER]` avant publication.

**Budget** : Léonard ≈ 25 h avec la manipulation (lot 1 : 8–10 h ; évaluation ≈ 12 h), de S4 à
S7 ; experts ≈ 2 h 30 chacun. L'annotation est le chemin critique. **Après le stage**
(H13) : l'ONF réentraîne sur ses portables à chaque nouvelle campagne, sur plusieurs années :
procédure documentée, tête réentraînable sans GPU (une nuit de calcul acceptable), ré-encodage
par lots reprenable, comparaison automatique au jeu gelé avant bascule.

### §5 bis. Voie non supervisée : clustering

Après le go seulement, sur le jeu v1 ; C1 se refait sur les points d'entraînement. Fonctions :
négatifs en volume, exploration, détecteur seulement si C1 réussit. HDBSCAN sur ACP (≈ 50
composantes), UMAP pour visualiser. C0 : clustering global d'un échantillon, AMI entre groupes et
micros, anomalies. C1 : positifs connus + 5 000 fenêtres aux mêmes heures, fenêtres pleines contre
recadrées ; seuils (jugement) : rappel du meilleur groupe ≥ 0,5, enrichissement ≥ 20, AMI micro
faible. C2 : étiquetage en bloc des groupes homogènes après dix écoutes. C3 : sub-clustering par
micro sur les heures de pic. Limite : 2–3 % de fenêtre pour la note → groupes = paysages
sonores.

### §6. Évaluation

Rien n'est jugé sur les données ONF avant le go (§5.7).

- **Niveau 2, nouveau site** : le site 2026 tenu entier à l'écart. **La** mesure principale.
- **Niveau 1, nouveau point d'un site connu** : les 20 % de points tenus à l'écart.
- **Niveau 3, temporel** : les stations 2023 tenues à l'écart, précision et rappel **par période**
  (haute, transition, basse) ; et **reproduction des patrons publiés** : le détecteur appliqué aux
  2 225 h de 2023 doit retrouver les courbes de Courtois et al. (pics 7–9 h et 15–17 h, creux de
  juillet à octobre, Kaw_B faible). Un désaccord signale un problème de généralisation ou un
  artefact.
- **Développement** (tête, C, pooling) : plis par point sur les seuls points d'entraînement.
- **Repondération** (Horvitz–Thompson) : sur le jeu d'évaluation chaque enregistrement pèse
  1 / sa probabilité de tirage ; l'AP, la précision, les fausses alarmes par heure et la
  prévalence sont celles du stock réel. Le rappel est rapporté aussi strate par strate.
- **Jeu gelé** : le jeu d'évaluation v1 après le go, versionné, jamais entraîné ; qualité A/B/C,
  solo ou chœur, SNR et probabilité de tirage renseignés.
- **Métriques** : AP et rappel par enregistrement et par fenêtre ; rappel à précision ≥ 0,1 et
  ≥ 0,5 ; fausses alarmes par heure ; rappel par qualité, SNR, période, tranche horaire ; accord
  du classement des points avec l'expert. Rejetées : exactitude, AUROC, F1 au seuil 0,5, kappa
  (pour les modèles).
- **Baseline = écoute humaine** : protocole homme–machine sur 60 enregistrements communs (30
  positifs, 30 négatifs difficiles) : rappel, précision et temps d'un naturaliste à l'oreille
  contre l'outil suivi d'une vérification des seuls candidats. Comparaison principale du rapport,
  exécutée en P4 avec deux naturalistes.
- **Repères** : au plus deux encodeurs non libres comme objectifs à battre ; détections Blancinet
  v0.1.0 (`import-detections`, score ≥ 0,10, jeu 2026) jugées au niveau de l'enregistrement sur
  les mêmes enregistrements (réserve : on ignore sur quoi il a été entraîné) ; détecteur
  Biophonia 2024 (Courtois et al. : ≈ 1 000 extraits annotés, test sur 450 : 0 faux positif, 24
  faux négatifs, 53 vrais positifs, rappel ≈ 0,69 à précision 1) : indicatif seulement.
- **Cibles** : rappel ≥ 0,9 au seuil le plus élevé qui garde une précision ≥ 0,1 sur le site tenu
  à l'écart ; file hebdomadaire ≤ 1 h à 10 s par candidat. Intervalles : Wilson en
  enregistrements ; bootstrap par point pour l'AP (n° 139) ; comparaisons appariées, Holm ;
  « A meilleur que B » seulement si l'intervalle apparié exclut zéro.
- **Calibration** : seuils sur scores hors-pli ; **seuil propre à chaque site** (§5.9).
- Combinaisons : concaténation d'embeddings et fusion séquentielle d'abord ; le reste si le jeu
  gelé compte ≥ 200 positifs indépendants (chaque encodeur ajouté double l'inférence sur l'i5).

| Rang | Critère | Mesure | Comment |
|---|---|---|---|
| 1 | Performance | AP repondérée et rappel par enregistrement : niveau 2, puis 1 et 3 | bootstrap par point, comparaisons appariées, Holm |
| 2 | Facilité d'utilisation | installation (clics, minutes) ; lignes de code (cible 0, plafond 2) ; tâches réussies seul (analyser un dossier, vérifier une file, régler le seuil d'un site, exporter) ; grille à 5 points | Élodie et Benoît en P4, sur une machine ONF |
| 3 | Durée d'encodage | heures pour une campagne d'une semaine (575 h) sur l'i5 | même fichier ; encodage + tête ; CPU seul |

### §7. Outil livrable

**Cible** : portables Windows 10/11 x64, Intel Core i5-1145G7 (4 cœurs, GPU intégré Iris Xe),
16 Go, pas de CUDA ; l'ONF réentraîne après le stage.

**Retenu (jugement)** : une **application de bureau Windows** installée par un `.exe`
(double-clic), sans aucune ligne de code, sur le modèle de BirdNET-Analyzer (interface Gradio
dans une fenêtre pywebview).

| Brique | Choix |
|---|---|
| Moteur | `blanci/service.py` réduit au livrable : ONNX Runtime, NumPy, scikit-learn ; ni PyTorch ni TensorFlow |
| Interface | pages locales dans une fenêtre (pywebview) ou le navigateur ; l'écran « Vérifier » reprend le poste Streamlit ; Gradio si l'empaquetage de Streamlit résiste |
| Empaquetage | PyInstaller puis Inno Setup (installateur, raccourci, désinstallation) |
| Construction | GitHub Actions sur une machine Windows, à chaque version |

Étape intermédiaire et repli : avec `uv`, une ligne PowerShell installe l'outil, une autre le
lance. Écartés : serveur web hébergé, PAMGuard comme hôte (pas de réentraînement sur place),
notebooks ou ligne de commande seule, application Qt. **Essai d'empaquetage en S4** : le poste
d'annotation actuel en `.exe` sur une machine Windows propre (lève le risque 6).

**Écrans** : Analyser (dossier de campagne, avancement, reprise après coupure) ; Vérifier (file,
spectrogramme 2–8 kHz, boutons du schéma, raccourcis) ; Seuil du site (5 à 10 positifs vérifiés) ;
Résultats (points classés, export CSV et Excel) ; Modèle (versions, réentraînement, comparaison au
jeu gelé avant bascule).

**Budget de calcul sur la cible** (bruit synthétique, i5-1145G7, n° 72) : perch_v2, fenêtres de
5 s au pas de 2,5 s : 7 fenêtres/s, **33 h pour une campagne d'une semaine** (575 h), 9 h pour les
seules heures de pic ; BirdNET 2.4 : 9 h ; Bird-MAE-Base et ConvNeXt : 62 h ; BEATs : 81 h.
Leviers : fenêtres jointives (÷ 2) ; quantification int8 (× 2 environ `[À MESURER]`) ; heures de
pic d'abord (÷ 3,6) ; sous-échantillonnage sous H20 (÷ 5) ; OpenVINO pour le GPU intégré
`[À VÉRIFIER]`. Réentraînement de la tête : secondes ; ré-encodage complet : plusieurs nuits, par
lots reprenables.

Options : A application installable (3–4 semaines réparties de S4 à S16) **retenue** ;
B `uv`, deux lignes PowerShell (étape intermédiaire et repli) ; C PAMGuard (écartée) ; D dossier
Python portable + scripts (repli ultime).

Mise à jour : tête dans l'application ; encodeur par paquet (`manifest.json` : nom, version,
SHA-256, **licence**, f_e, fenêtre) ; un paquet dont la licence n'est pas libre est refusé.
Documentation : README ; « Interpréter un score » ; « Vérifier une file » ; « Régler le seuil
d'un nouveau site » ; guide de réentraînement pour l'ONF (pas à pas, une nuit) ; `LICENSES.md`.
Licence d'AnuraSet : CC BY (n° 78 ; la V5 disait CC0 à tort). Confirmation ONF des licences
avant S12.

### §8. Planning (S1 = 14–18/09/2026, aucune semaine réduite)

| Phase | Semaines | Contenu | Go / no-go |
|---|---|---|---|
| P0 Cadrage (fait) | S1–S3 | lecture ; inventaire complet ; chaîne CLI (M0–M5 écrits) ; benchmarks AnuraSet 01–07 | — |
| P0 bis Clôture et plan | S3–S4 (29/09–09/10) | vague 2 AnuraSet close le 09/10 ; classification tranchée (§5.11) ; licences relevées ; partition des points et plan de tirage figés et versionnés ; mode « extrait + intervalles » ; essai chronométré sur 50 extraits ; avis de Sylvain ; essai d'empaquetage `.exe` | plan validé par Sylvain et Élodie ; au plus deux non libres retenus |
| P1 Annotation v1 | S4–S7 (05/10–30/10) | lot d'entraînement 1 (≈ 450 extraits) ; jeu d'évaluation (≈ 250 enregistrements entiers) ; paquet de vérification fin S6 ; **en parallèle** : application v0 | ≥ 60 enregistrements positifs à l'évaluation, sinon tirage complémentaire |
| Go des experts | S7–S8 (≈ 06/11) | retour d'Élodie et Benoît ; corrections ; règle du go (§5.7) | sans go : pas de benchmark ONF, l'application continue |
| P2 Benchmark ONF borné | S8–S10 (02/11–20/11) | encodeurs libres + au plus deux non libres × logistique ; courbe d'amorçage par point ; seuil par site ; lot 2 par apprentissage actif | **choix de l'encodeur le 20/11** ; rappel ≥ 0,85 à précision ≥ 0,1 sur le site tenu à l'écart, sinon pivot (limite de pivot : fin S12) |
| P3 Outil v1 | S5–S16 (effort S11–S16) | export ONNX + int8 de l'encodeur retenu ; écrans ; seuil par site ; installateur ; test sur une machine ONF ; guide de réentraînement v0 | 1 h d'audio en < 10 min sur l'i5 ; installation et analyse sans code par Élodie ; sinon option B |
| P4 Transfert | S17–S21 (04/01–05/02) | test homme–machine ; test d'usage par Élodie et Benoît ; répétition du réentraînement par un agent ONF ; documentation | gel fin S21 |
| P5 Rapport | S22–S26 (08/02–12/03) | rédaction, soutenance en mars 2027, transfert | — |
| Pour aller plus loin (§14) | dès S17, si P1–P3 tiennent | 2 225 h de 2023, courbes de phénologie, gabarit multi-espèces | seulement si go et application v1 acquis |

Rédaction continue dès S12 (½ j/sem.). Réunion hebdomadaire : avancement, chiffre clé (extraits
annotés, positifs, points couverts ; puis AP repondérée sur le jeu gelé), décision demandée.
**Chemin critique** : plan de tirage (S4) → annotation v1 (S7) → go des experts (S8) → choix de
l'encodeur (S10) → ONNX + application v1 (S16) → tests d'usage (S19–S20) → gel (S21). Après le
20/11 : plus de benchmark d'encodeur ; après S16 : plus de changement d'encodeur. Ce qui n'attend
pas le go : l'application, l'export ONNX de perch_v2 et sa validation sur l'i5, le guide, la
lecture.

### §9. Risques

| Risque | Seuil | Repli |
|---|---|---|
| 1. Go des experts tardif | paquet non rendu > 2 semaines après l'envoi | l'application avance ; benchmark décalé ; paquet réduit aux positifs et incertains |
| 2. Trop peu de positifs | < 40 à l'entraînement ; < 60 à l'évaluation | tirage complémentaire dans les strates de pic, probabilités notées ; récolte par similarité (entraînement seulement) |
| 3. Erreurs d'annotation de Léonard | au-delà de la règle du go | réécoute de la strate ; séance des faux amis avec Élodie ; nouveau contrôle |
| 4. Le meilleur encodeur n'est pas libre | écart apparié significatif avec le meilleur libre | le libre est livré, l'écart rapporté ; distillation seulement si la licence de l'enseignant le permet |
| 5. Chant ponctuel avéré (H20 fausse) | > 10 % des positifs chantent < 50 % du temps, ou > 10 % absents des 30 s tirés | extraits de 60 s ; score par enregistrement = max ou top-k ; file « suspect » réintégrée |
| 6. Empaquetage Windows | `.exe` qui ne tourne pas sur une machine propre | Gradio à la place de Streamlit ; sinon option B |
| 7. Cible trop lente | > 15 h par campagne d'une semaine après leviers | fenêtres jointives, int8, heures de pic ; BirdNET 3 si ses performances suivent |
| 8. Pas de Perch 2.0 en ONNX valide | cosinus < 0,99 à la fin de S10 | BirdNET 3 (ONNX officiel, libre) ; sinon l'encodeur libre suivant |
| 9. Confusion avec les faux amis | > 30 % des 100 meilleurs candidats après le tour 3 ; précision < 0,1 à rappel 0,85 fin S11 | poids accru de persistance et rythme ; négatifs par espèce ; sortie hiérarchique |
| 10. Décalage 2023 (capteurs, saison) | corrélation des courbes journalières < 0,5 | recalibration par jeu ; négatifs 2023 appariés ; résultat rapporté tel quel |
| 11. Les benchmarks débordent | une semaine sans extrait annoté ni écran écrit | dates fermes : AnuraSet le 09/10, ONF le 20/11 |
| 12. Bande saturée | > 500 candidats/h aux heures de pic | fenêtres pleines ; seuillage limité aux onsets |
| 13. Perte machine ou données | machine unique | SSD externe quotidien, copie ONF hebdomadaire, labels sous git |

### §10. Questions restantes

1. **Élodie** : « libre d'accès » = licence qui permet l'usage par l'ONF sans clause non
   commerciale (définition du §0), ou seulement « téléchargeable gratuitement » ? Dans le second
   cas, BirdNET 2.4 et esp-aves2 redeviennent candidats.
2. **Élodie, Benoît** : temps et délai pour la vérification (≈ 2 h 30 chacun) ; outil d'écoute
   habituel (Raven, Audacity, Kaleidoscope, notre poste) ?
3. **Sylvain** : les cinq questions du §5.8.
4. **Élodie** : quel site 2026 tenir entier à l'écart ? Existe-t-il une carte des points
   (distance à la crique, type de crique) ?
5. **Élodie** : à quels mois poseront les futures campagnes (poids de la saison basse dans le
   tirage) ?
6. **Élodie** : qui, à l'ONF, lancera l'application et réentraînera la tête, sur quelle machine ?
7. *A. blanci* chante-t-elle parfois de façon ponctuelle ? Chœur et mâle seul sont-ils
   distinguables à l'oreille ? (H20, H21)
8. Les sorties du détecteur Biophonia sur 2023 et ses ≈ 1 000 extraits annotés sont-ils
   récupérables auprès d'ENIA ou de Trésor ?
9. Quand un point est remonté « à vérifier », que se passe-t-il (visite de terrain, contrainte
   d'exploitation) et quel volume l'ONF peut-il vérifier par semaine ? (H5)
10. Taux d'émission et intervalles entre notes d'*A. blanci* (H9) ; réponse et réglages des
    enregistreurs 2023 et 2026 (H23).
11. Qui tranche les licences à l'ONF, et à quelle échéance (avant S12) ? Une machine ONF est-elle
    disponible pour mesurer le débit et tester l'installateur ? Date du prochain rapatriement de
    terrain (H13) ?

### §12. Hypothèses restantes

| Id | Hypothèse | Levée par |
|---|---|---|
| H5 | Décision gestionnaire = intégration dans la planification forestière, à l'échelle du point ou du bassin de crique | Q9 |
| H9 | IOI de l'ordre de la seconde ; plusieurs notes par fenêtre | Q10 |
| H13 | Un rapatriement de terrain avant la fin du stage | Q11 |
| H16 | Relevés 2026 de décembre 2025 à février 2026, un par site | `blanci status` |
| H20 | *A. blanci* chante rarement de façon ponctuelle ; détection isolée = suspicion de faux positif | Q7 ; jeu d'évaluation |
| H21 | Solo et chœur séparables par densité d'onsets et chevauchements | Q7 |
| H23 | Les enregistreurs 2023 et 2026 ont une réponse comparable | Q10 |
| H24 | « Libre d'accès » = licence qui permet l'usage par l'ONF (§0) | Q1 |
| H25 | L'annotation de Léonard seul atteint la règle du go (≤ 5 % d'erreur par classe) | vérification (§5.7) |
| H26 | 30 s suffisent à dire si un enregistrement contient le chant | jeu d'évaluation |
| H27 | Les futures campagnes se posent en saison d'activité, comme en 2026 | Q5 |

### §13. Règles d'implémentation

**13.1 Contraintes.** Développement : MacBook Air M4, 16 Go, PyTorch avec backend MPS, pas de
CUDA. Cible : Windows x64, i5-1145G7, 16 Go, CPU seul. **Interdit à l'exécution du livrable** :
TensorFlow, TFLite, PyTorch ; autorisé : ONNX Runtime, NumPy, scikit-learn. En recherche :
bacpipe, torch. Python 3.11, `uv`, `pyproject.toml`, `pytest`, `ruff`.

**13.2 Arborescence.** Celle de la V5 (`blanci/` : `ingest`, `audio`, `grid`, `qc`, `encoders/`,
`store`, `index`, `head`, `sequential`, `fusion`, `aggregate`, `evaluate`, `active`, `labels`, `db`,
`cli` ; `app/streamlit_app.py` ; `data/` hors git) a été dépassée par le code (n° 11 et suivants) :
la structure réelle est décrite dans le README. `data/` : `raw/{2023,2026}/<site>/<micro>/*.wav`
(jamais écrit), `db/blanci.sqlite`, `embeddings/<encoder_id>/<dataset>/<site>/<yyyymm>.parquet`,
`labels/imports/` (fichiers reçus, jamais modifiés), `models/<kind>/<name>-<version>/`
(`manifest.json` + poids), `frozen_test/` (jeu gelé, lecture seule).

**13.3 Schéma SQLite** (implémenté dans `blanci/core/db.py`, qui fait foi ; labels en ajout seul).

```
recordings(recording_id TEXT PK, path TEXT UNIQUE, dataset TEXT, site TEXT, mic_id TEXT,
           start_utc TEXT, duration_s REAL, sample_rate INT, channels INT, sha256 TEXT,
           qc_flags TEXT)                       -- JSON : rain, saturation, in_bag, silent
windows(window_id TEXT PK, recording_id TEXT FK, offset_s REAL, dur_s REAL)
labels(label_id INTEGER PK, window_id TEXT FK, label TEXT, quality TEXT, species TEXT,
       conditions TEXT, annotator TEXT, source TEXT, created_at TEXT)
       -- label ∈ {blanci (n° 3), blanci_solo, blanci_chorus, blanci_uncertain, bird, amphibian,
       --          orthoptera,
       --          amphibian_contact_call, rain, artefact_in_bag, background, other, uncertain}
       -- quality ∈ {A, B, C, NULL} ; source ∈ {import, similarity, active, random, audit, flag}
models(model_id TEXT PK, kind TEXT, name TEXT, version TEXT, sha256 TEXT, params_json TEXT,
       created_at TEXT)                        -- kind ∈ {encoder, head, fusion, threshold}
scores(window_id TEXT, model_id TEXT, score REAL, PRIMARY KEY(window_id, model_id))
decisions(recording_id TEXT, encoder_id TEXT, head_version TEXT, threshold_id TEXT,
          fraction REAL, status TEXT, created_at TEXT)
          -- status ∈ {positive, suspect, negative, verified_positive, verified_negative}
```

`window_id = f"{recording_id}:{offset_s:.2f}"` ; `recording_id` : sha256 du chemin relatif (n° 2).
Parquet : colonnes `window_id, recording_id, offset_s, emb` (`emb` en liste de taille fixe
`float16[dim]`), un fichier par (encodeur, jeu, site, mois).

**13.4 Interfaces** (signatures d'origine ; le code fait foi).

```
class Encoder(Protocol):
    name: str; version: str; sample_rate: int; window_s: float; dim: int; has_tokens: bool
    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray            # (n_windows, dim)
    def embed_tokens(self, wav: np.ndarray, sr: int) -> np.ndarray | None  # (n_windows, n_tokens, dim)

def window_grid(duration_s: float, window_s: float, hop_s: float) -> list[tuple[float, float]]
def differential_prototype(E_pos, E_neg_paired) -> tuple[np.ndarray, float]
def oof_scores(X, y, groups, n_splits: int = 5) -> np.ndarray
def detect_onsets(wav, sr, band: tuple[int, int] = (4400, 5500)) -> np.ndarray
def aggregate_recording(window_scores, threshold) -> tuple[float, str]
def build_queue(scores, labels, n, mix: tuple[float, float, float] = (0.6, 0.2, 0.2))
def evaluate(scores, labels, groups, level: Literal["window", "recording"]) -> dict
```

**13.5 Commandes.** `ingest`, `import-labels`, `embed`, `benchmark`, `search`, `train`, `score`,
`queue`, `evaluate`, `export-onnx` à l'origine ; la liste réelle (une cinquantaine de commandes)
est dans `documentation/commandes.md`.

**13.6 Jalons et critères d'acceptation.** M0 dépôt, `ingest`, `import-labels` (accepté, n° 49) ;
M1 `embed`, `store`, `benchmark` en plis par micro ; M2 `head`, `search`, `queue`, prototype
Streamlit ; M3 `sequential`, `fusion`, `aggregate`, audit aléatoire ; M4 jeu gelé,
`evaluate --holdout`, reproduction des patrons 2023 ; M5 `export-onnx`, quantification, test
d'équivalence (cosinus > 0,99 avec torch), bundle Windows : 1 h d'audio en < 10 min sur l'i5,
réentraînement de la tête sans intervention ; M6 guide de réentraînement, test homme–machine,
transfert : un agent ONF réentraîne seul sur une nouvelle campagne. M0 à M5 sont écrits (état :
README).

**À coder (liste v5, dans l'ordre).** (1) Plan de tirage : partition des points versionnée,
`candidates --plan` (strates, quotas, graine), probabilité de tirage enregistrée. (2) Annotation
par intervalles : mode « extrait + intervalles », labels de fenêtres déduits pour chaque grille,
poids 1 / nombre de fenêtres. (3) Sources exclues par défaut : `import` (Blancinet) hors
entraînement et évaluation ; `active` et `similarity` hors évaluation. (4) Vérification à l'aveugle :
export WAV + Excel sans le label de Léonard, réimport des réponses comme labels, `agreement` par
classe avec bornes. (5) Métriques repondérées (AP, précision, fausses alarmes par heure, rappel
par strate). (6) Licence dans le manifeste : paquet refusé si la licence n'est pas libre. (7)
Application installable : essai PyInstaller + Inno Setup, construction Windows par GitHub Actions.

**13.7 Règles pour l'agent de code.**
- Ne jamais écrire dans `data/raw` ni `data/frozen_test`. Les labels ne se modifient pas, ils
  s'ajoutent.
- Toute évaluation passe par `evaluate.py` avec groupes explicites ; un découpage aléatoire est
  une erreur. Aucun score n'entre dans `fusion` s'il n'est pas hors-pli.
- Un encodeur n'est jamais appelé sans passer par `Encoder` ; le rééchantillonnage se fait dans le
  wrapper.
- Tout résultat chiffré vient d'une commande reproductible avec graine, jamais d'un notebook non
  versionné.
- Aucun résultat sur les données ONF avant le go d'Élodie et Benoît, écrit dans ce journal (§5.7).
- Un encodeur non libre (§0) n'entre jamais dans le livrable ; au benchmark ONF, au plus deux,
  comme objectifs à battre.
- Aucun label choisi par un modèle (sources `active`, `similarity`, `import`) dans le jeu
  d'évaluation.
- L'application avance en parallèle de l'annotation ; PAMGuard est écarté ; la distillation reste
  après le choix de l'encodeur.
- En cas de doute sur une décision de conception, se référer au numéro de section du cadre et ne
  pas rediscuter les choix « jugement » : les remettre en question par une entrée datée de ce
  journal.

### §14. Pour aller plus loin (si le cœur du stage est fini en avance)

**Article de phénologie d'*A. blanci*** : le détecteur validé appliqué aux 2 225 h de 2023 (trois
sites, une année entière) plus les points 2026. Questions possibles (à choisir avec Élodie) :
courbes journalières et saisonnières par site ; lien avec la pluie, la température et l'humidité ;
différences entre criques rocheuses (Kaw, Molokoï) et lit évasé et marécageux (Mataroni) ;
probabilité de détection et occupation. Prérequis : un rappel et une précision **connus par
période**, y compris en saison basse (une fausse alarme constante dessine une fausse activité hors
saison). Auteurs, revue, calendrier : à décider avec Élodie, Benoît et Sylvain ; rien n'est écrit
avant l'application v1 (S16).

**Gabarit du projet pour d'autres espèces** : propre à *A. blanci* : bande de fréquence
(4,4–5,5 kHz), durée de note et intervalle, heures et mois d'activité, faux amis et schéma de
labels, congénères dans les classes de Perch et de BirdNET 3. Le reste est générique (inventaire,
drapeaux, plan de tirage, poste d'annotation, encodeur, tête, évaluation, application). Travail :
regrouper ce qui est propre à l'espèce dans une section `species` de la configuration, rien de
codé en dur ; un guide « adapter à une nouvelle espèce » ; un essai sur une deuxième espèce. À
faire au fil de l'eau : chaque nouvelle fonction évite de coder *A. blanci* en dur.

### Angles morts

- **Un seul annotateur** : la vérification à l'aveugle mesure son erreur sur un échantillon, pas
  sur tout ; une confusion systématique peut passer entre les mailles si la strate n'est pas
  tirée.
- **Le go des experts est au chemin critique** : paquet court, clair, sans installation, envoyé
  tôt (fin S6).
- **Sans probabilités de tirage, pas de métrique honnête** : un enregistrement ajouté « à la
  main » au jeu d'évaluation le casse.
- **En 2026, mois et site sont confondus** : l'effet saison ne se mesure que sur les six stations
  de 2023.
- **Le modèle précédent reste mal comparable** (Blancinet) : on ignore sur quoi il a été entraîné.
- **H20 porte trois économies** (règle d'agrégation, sous-échantillonnage des fenêtres, extraits
  de 30 s) : si elle est fausse, elles disparaissent ensemble.
- **Phénologie comme piège** : très prédictive d'un site à l'autre ; hors classifieur par défaut,
  mais la tentation reviendra.
- **La machine cible est lente** (33 h par campagne sans levier, n° 72) ; export ONNX et int8 non
  encore validés sur l'i5.
- **« Libre » est une question juridique** : la définition du §0 est un jugement d'ingénieur ;
  l'ONF tranche (Q11).
- **Transfert à l'ONF** : réentraînement par des non-développeurs sur plusieurs années : le guide
  et le réglage du seuil d'un nouveau site sont des livrables à part entière.
- **Sur-ingénierie** : chaque heure passée sur un benchmark est retirée à l'annotation et à
  l'application, désormais au chemin critique.
- **Occupation** : la non-détection devrait passer par un modèle d'occupation `[À VÉRIFIER]` ;
  hors périmètre du stage, dans celui de l'article (§14).

---

# Journal des décisions

## 2026-09-21 — Jalon M0

1. **Python 3.11, numpy < 2.** Conforme au §13.1 (le projet `uv init` était en 3.13).
   bacpipe 1.3.5 exige Python ≥ 3.11 et < 3.13 et fige `numpy==1.26.4`, `soundfile==0.12.1`,
   `pandas<3`, `pyarrow<26` : les dépendances de base sont bornées en conséquence pour que
   l'ajout de bacpipe en M1 ne casse rien. bacpipe n'est pas encore installé (torch +
   TensorFlow, plusieurs Go).

2. **`recording_id` = sha256 du chemin *relatif* à `paths.raw`** (POSIX), et non du chemin
   absolu. Les identifiants survivent au passage sur SSD externe ou sur un portable Windows de
   l'ONF. `recordings.path` stocke ce même chemin relatif.

3. **Label `blanci` ajouté** au vocabulaire du §13.3 : positif dont le solo/chœur n'est pas
   renseigné. Les 345 positifs importés n'ont pas cette information (H21 ouverte) ; les
   ranger en `blanci_solo` serait faux, en `blanci_uncertain` aussi (l'identification est
   sûre, H19). Positifs = {`blanci`, `blanci_solo`, `blanci_chorus`} ; `blanci_uncertain` et
   `uncertain` sont exclus de l'entraînement par défaut.

4. **Fenêtres annotées stockées telles quelles** dans `windows` (décalage et durée de
   l'annotation, 3 s Biophonia), indépendamment des grilles d'encodeur. Le transfert des
   labels vers une grille (3 s / 1,5 s ou 5 s / 2,5 s) se fera par inclusion au moment de
   l'entraînement (M1). `blanci check-grid` vérifie que chaque annotation positive tient
   entière dans une fenêtre de chaque grille.

5. **Qualité A/B/C déduite du commentaire** quand le fichier reçu n'a pas de colonne
   qualité : C si « lointain » ou « second plan », B si pluie, bruit ou autre espèce citée,
   A sinon. Marquée `quality_inferred: true` dans `labels.conditions` ; à relire avec le
   tuteur. Une colonne qualité explicite l'emporte toujours.

6. **Import d'un fichier d'annotation en tout ou rien.** Une ligne dont l'enregistrement est
   introuvable ou ambigu bloque le fichier (sauf `--allow-partial`). Table `imports` ajoutée
   au schéma : un fichier déjà importé (même SHA-256) n'est pas réimporté.

7. **Labels en ajout seul imposés par la base** : triggers SQLite refusant UPDATE et DELETE
   sur `labels` (§13.7).

8. **QC en numpy/scipy, pas scikit-maad, pour M0.** Quatre indices suffisent aux drapeaux
   demandés (RMS, fraction saturée, part d'énergie > 2 kHz, platitude spectrale 1–10 kHz).
   Seuils provisoires dans `config/default.yaml`, à calibrer sur les enregistrements étiquetés
   `rain` et `artefact_in_bag`. scikit-maad reste possible si des indices écoacoustiques
   deviennent utiles.

9. **Horodatage** : bloc GUANO du WAV en priorité (fuseau inclus), sinon nom de fichier Song
   Meter + `recorder.filename_utc_offset_h` (UTC−3). [À VÉRIFIER] sur les premiers fichiers
   réels : réglage horaire des enregistreurs 2023 et 2026.

10. **« A. hahneli » interprété comme *Ameerega hahneli*** (Dendrobatidae, présente en
    Guyane). La feuille de route écrit *A. hahneli* à la suite d'*Allobates femoralis*, ce qui
    suggère *Allobates*. [À VÉRIFIER] avec le tuteur. « piau » interprété comme Piauhau hurleur
    (*Lipaugus vociferans*), [À VÉRIFIER].

## 2026-09-21 — Environnement, tests des modules M1

11. **Modules hors de l'arborescence du §13.2** : `dataset.py` (transfert des annotations vers
    une grille d'encodeur, négatifs appariés, assemblage du jeu d'apprentissage) et `embed.py`
    (extraction reprenable, remplissage de `windows`, mesure du débit). Le §13.2 les laisse
    implicites entre `store.py` et `head.py` ; les isoler évite de gonfler `cli.py` et rend
    l'apprentissage testable sans audio. `benchmark.py` et `service.py` suivront de même.

12. **`Encoder.embed(wav, sr)` reçoit un lot de fenêtres**, de forme (n_fenêtres, n_échantillons)
    à la fréquence d'échantillonnage native de l'enregistrement, et non un signal continu. Le
    §13.4 écrit `embed(self, wav, sr) -> (n_windows, dim)` sans dire qui découpe. Découper dans
    `grid.py` puis passer le lot garde une grille unique, indépendante de l'encodeur (§4), et
    permet le traitement par lots. Le rééchantillonnage et le complément de zéros sont faits
    dans `BaseEncoder._prepare`, jamais chez l'appelant (§13.7).

13. **Grille d'un encodeur = sa fenêtre × `grid_hop_ratio` (0,5)**. Un encodeur à fenêtre de 3 s
    est donc échantillonné tous les 1,5 s, un encodeur à 5 s tous les 2,5 s. Le §1 impose un pas
    ≤ la moitié de la fenêtre pour qu'aucune note de 0,09 s ne soit coupée ; un ratio unique
    évite d'avoir à régler le pas encodeur par encodeur.

14. **Score d'enregistrement pour l'AP = maximum des fenêtres** (`to_recordings(how="max")`),
    avec `top3` en variante. Le §1 définit le score d'un enregistrement comme la *proportion* de
    fenêtres positives, mais une proportion suppose un seuil déjà fixé : circulaire pour une
    métrique de classement. Le maximum s'en passe. La proportion reste la sortie de
    `aggregate_recording`, pour la décision, pas pour l'AP.

15. **Négatifs appariés présumés, jamais écrits dans `labels`.** `paired_negatives` marque ses
    fenêtres `background_presumed` avec une colonne `presumed`. Personne ne les a écoutées : en
    saison et à l'heure de pic, une partie contient sans doute *A. blanci*. Les inscrire dans
    `labels` polluerait le jeu annoté et fausserait tout comptage de positifs. Le bruit
    d'étiquette est identique pour tous les encodeurs, donc sans effet sur leur comparaison (§2).

16. **`recall_at_precision` renvoie le seuil le plus élevé à rappel égal.** Il renvoyait le plus
    bas : même rappel, mais davantage de faux positifs dans la file de vérification. Le §6
    demande « le seuil le plus élevé gardant précision ≥ 0,1 ». Corrigé, avec test.

17. **Ruff ne formate pas le Markdown** (`extend-exclude = ["*.md"]`). `ruff format` reformatait
    les blocs de code Python du glossaire, qui est un document manuscrit.

## 2026-09-21 — Benchmark (§2)

18. **`knn_top1` exclut les fenêtres du même enregistrement** du voisinage. Deux fenêtres
    voisines d'un même fichier de 2 min se chevauchent (pas = moitié de la fenêtre) et
    partagent le fond sonore : le plus proche voisin d'un positif serait presque toujours la
    fenêtre d'à côté, et la mesure vaudrait 1 pour tous les encodeurs. Le §2 demande un
    « kNN top-1 » sans préciser le voisinage.

19. **Comparaisons appariées au niveau enregistrement seulement.** Le §2 demande des
    comparaisons appariées « sur les mêmes plis ». Deux encodeurs de fenêtres différentes
    (3 s et 5 s) n'ont pas la même grille, donc pas les mêmes fenêtres ; ils voient en
    revanche les mêmes enregistrements. `compare_encoders` apparie donc sur l'intersection
    des enregistrements, avec la sonde logistique.

20. **`paired_bootstrap` renvoie NaN plutôt que de planter** quand la métrique est indéfinie
    sur tous les rééchantillonnages (une seule classe). Il appelait `np.quantile` sur un
    tableau vide ; `bootstrap_ci` se protégeait déjà de ce cas.

## 2026-09-21 — Couche de service et CLI (§13.5)

21. **Format des annotations confirmé** (Léonard) : ce sont des **fenêtres de 3 s**, ce que
    `labels.import.window_s` suppose déjà. Reste inconnu : si ce sont des lignes de tableau
    (fichier + début) ou des extraits WAV découpés. Dans le second cas, il faudra un
    importeur qui retrouve l'enregistrement d'origine et le décalage.

22. **Registre des modèles dans `db.py`** (`model_params`, `encoder_params`, `register_model`,
    `next_version`), puisque c'est lui qui possède la table `models`. Les versions de tête
    s'incrémentent par encodeur : `v1`, `v2`… et les précédentes restent en base, pour
    comparer une décision ancienne au modèle qui l'a produite (§4).

23. **Identifiants estampillés** : `<encodeur>:head:<version>` pour une tête,
    `<encodeur>:head:<version>:p<précision>` pour un seuil. Le §13.3 impose les colonnes
    `encoder_id`, `head_version`, `threshold_id` dans `decisions` sans fixer leur forme.

24. **Le seuil est calibré sur les scores hors-pli**, pas sur les scores d'entraînement. Un
    seuil calibré en-pli serait optimiste, la tête ayant vu chacun de ses exemples (§6).

25. **Les décisions se remplacent, les labels s'ajoutent.** `score_and_decide` efface les
    décisions du même triplet (encodeur, tête, seuil) avant d'écrire : elles sont
    reproductibles à partir du modèle. Les labels, eux, sont en ajout seul (§13.7).

26. **`evaluate --holdout` sans site** retombe sur la validation groupée par micro (§6,
    niveau 1) au lieu d'échouer : c'est le protocole par défaut tant qu'il n'y a pas de
    positifs hors Mataroni.

27. **`embedded_training_set` dans `dataset.py`** : point d'entrée commun du benchmark et de
    l'entraînement. Les deux doivent voir exactement les mêmes labels et les mêmes négatifs
    appariés, sinon le benchmark ne prédit pas ce que fera la tête.

## 2026-09-21 — Format réel des données (Léonard)

28. **Format confirmé.** Enregistrements de 2 min sur disque externe, nommés
    `2la04530_20260106_103000` : micro `2LA04530`, 6 janvier 2026 à 10 h 30 locales. Les
    fichiers sont des **`.wav`**, mais l'Excel les cite en **`.flac`**. L'inventaire accepte
    désormais les deux (`audio.suffixes`), et l'appariement Excel ↔ disque se fait sur le nom
    **sans extension** : une citation `.flac` retrouve un `.wav`. Si les deux formats
    existent pour un même nom, l'extension citée départage ; sans extension citée, la ligne
    est signalée comme ambiguë plutôt que devinée.

29. **Le micro vient du préfixe du nom de fichier** quand l'arborescence n'a pas de dossier
    micro. Il est conservé tel quel (`2la04530`, minuscules) ; les comparaisons avec l'Excel
    passent par `normalize`, la casse n'a donc pas d'importance.

30. **Colonne « vérif manuelle » = verdict.** `parse_verdict` lit les oui/non (et variantes
    `o`, `x`, `ok`, `vrai`, `1`, `True`, `confirmé`…) et les réponses rédigées mentionnant
    *blanci*. Si la cellule nomme un faux ami reconnu (« fourmilier tacheté »), l'import en
    tire un négatif **et** l'espèce. Si elle ne dit rien de reconnaissable (« à revoir »,
    « ? »), la ligne est signalée : l'ancien code l'aurait rangée en négatif sans un mot, ce
    qui aurait transformé 345 positifs en 345 négatifs silencieusement.

31. **Score de l'ancien prestataire conservé** dans `labels.conditions.previous_model_score`,
    et résumé (min / médiane / max) par `--dry-run`. C'est le repère chiffré du §6 : il
    permettra plus tard de mesurer le rappel relatif au détecteur Biophonia, et de voir quels
    positifs il ne remontait qu'à score médiocre (§5, conséquence 2).

32. **Détection des colonnes en deux passes** : égalité exacte d'abord, puis libellé composé
    par mots entiers (« Nom de l'enregistrement » → `fichier`). Une colonne ne sert qu'à un
    seul champ. La comparaison par mots évite qu'un candidat `time` capture `timecode`.

33. **Timecode robuste aux formats Excel** : nombre, « mm:ss », « hh:mm:ss », `datetime.time`,
    `timedelta` et `Timestamp`. Excel rend une cellule horaire différemment selon son format
    d'affichage, et l'erreur serait silencieuse.

## 2026-09-22 — Acceptation M0 sur le disque réel

34. **Le fichier d'annotations est l'export complet de Blancinet v0.1.0**
    (`All_detections_blancinet_v0.1.0_dataset1BV.xlsx`) : 74 785 détections (score ≥ 0,10),
    dont **345 vérifiées `True`, 150 `False`**, 4 en attente (« à vérif », « à conf ») et
    74 286 jamais écoutées. Seules les vérifiées deviennent des labels. Une cellule de
    vérification vide est comptée « non vérifiée » et ignorée ; « à vérif » / « à conf » sont
    listées et mises de côté, sans label inventé et sans bloquer l'import. Un second fichier
    vérifié existe sur le disque (`..._dataset2_verifBV.xlsx`), non importé à ce jour.

35. **Les 345 positifs viennent de 51 enregistrements, pas de 345.** 13 micros, tous à
    Mataroni ; le point RB04 porte à lui seul 146 fenêtres (42 %). Le §1 (« 345 positifs =
    345 enregistrements distincts ») et H2 sont à corriger : l'effectif indépendant pour les
    intervalles de confiance est de l'ordre de 51 enregistrements, et bien moins de points.
    Conséquence directe sur le §6 : l'intervalle de Wilson « n = 345 → [0,86 ; 0,93] » ne
    s'applique pas ; avec n = 51, un rappel de 0,9 donne environ [0,79 ; 0,96].

36. **Clé S3 de l'ancien prestataire.** Les noms sont préfixés d'un identifiant numérique
    (`2353462-2la03550_20260108_143000.flac`). `file_key` retire ce préfixe, mais seulement
    devant un nom Song Meter, pour ne jamais tronquer un vrai nom de fichier.

37. **Commentaires dans des colonnes sans nom.** Le fichier range ses commentaires dans
    `Colonne1` (nom par défaut d'un tableau Excel) et `Unnamed: 12` (cellule sans en-tête).
    Toute colonne anonyme non attribuée alimente le commentaire, ainsi que le texte de la
    vérification. Une cellule qui recopie un en-tête (« Colonne1 ») est écartée.

38. **Micro = numéro de série, plus le dossier.** Remplace l'ordre de DECISIONS n° 2 et du
    jalon M0 (dossier d'abord). Sur le disque, le même enregistreur s'appelle « SM4 A »,
    « SM A_SMA13417 », « SMA13417 A » ou « SMA13417 » selon le relevé, et les fichiers sont
    souvent dans un sous-dossier `Data`. Priorité : `Serial` GUANO, préfixe du nom de fichier,
    puis dossier en dernier recours. Un même enregistreur change de site d'un relevé à
    l'autre (2LA03550 : Mataroni en janvier, RNRT en février) : le groupe des plis reste le
    couple site/micro (`point`), jamais le micro seul.

39. **Site donné à l'inventaire** (`blanci ingest --site`), et dossier scanné découplé du
    jeu. L'arborescence réelle (`<projet>/RELEVE n Site - dates/<SERIE>[_POINT]/Data/`) ne
    suit pas `<jeu>/<site>/<micro>/`. Les chemins restent stockés relatifs à `paths.raw`
    (racine du disque), dans `config/local.yaml`, ignoré par git.

40. **Doublons de relevés.** 3 668 noms existent deux fois sur le disque : les cartes SD
    emportées à Mataroni contenaient encore les fichiers de CDR, copies identiques octet pour
    octet (12/12 sur échantillon). Sans précaution, un même audio compterait deux fois, sous
    deux sites — fuite entre plis et entre sites tenus à l'écart. L'inventaire écarte un nom
    déjà vu sous un autre chemin et le signale (`ingest_duplicates_<jeu>_<site>.csv`).
    Le premier inventorié l'emporte : **inventorier les relevés dans l'ordre chronologique**.
    Vérifié : les 34 lignes vérifiées tombant sur un nom en double sont toutes des stations
    CDR, datées de la période du relevé CDR.

41. **Horodatage GUANO : 3 h d'erreur corrigées.** Les Song Meter écrivent le fuseau sans
    zéro (`2025-12-27 15:30:00-3:00`), que `datetime.fromisoformat` refuse. Le code se
    rabattait sans le dire sur « nom de fichier + UTC−3 » : juste par chance pour un
    enregistreur en heure locale, **faux de 3 h pour un enregistreur réglé en UTC** (`+0:00`,
    observé sur le disque). Le décalage est maintenant normalisé avant lecture. Répond à la
    question de DECISIONS n° 9 : le fuseau varie d'un enregistreur à l'autre, seul le GUANO
    fait foi.

42. **Inventaire rapide.** Sans contrôle qualité ni empreinte, seuls les en-têtes sont lus :
    16 ms par fichier au lieu de 0,8 s, soit le disque entier (33 210 fichiers) en quelques
    minutes au lieu de 7 h. Le contrôle qualité se lance ensuite sur ce qui sera encodé.

43. **Fichiers vides.** 8 fichiers de 0 octet (enregistreur SMA14826, relevé Patawa, heures
    irrégulières) : signalés en erreur, sans bloquer l'inventaire.

44. **À trancher avec le tuteur — audio stéréo à deux gains.** Les fichiers sont en 48 kHz,
    2 canaux, gains 6 et 18 dB (`WA|Song Meter|Audio settings`). `load_audio` fait la
    moyenne des deux, dominée par le canal à 18 dB, qui sature parfois (crête 0,987 observée).
    Garder un seul canal (lequel ?) ou la moyenne change les embeddings : à décider avant M1.

45. **Identifiant d'enregistrement = nom de fichier.** Remplace DECISIONS n° 2 (et le
    `sha256(path)` du §13.3). `recording_id = sha256(<série>_<date>_<heure> en minuscules)` :
    il ne dépend plus ni du dossier ni de l'extension. Motif : les enregistrements seront
    copiés sur un disque propre, dans une arborescence réorganisée ; avec l'ancien identifiant,
    chaque label et chaque embedding aurait perdu son enregistrement. Le nom est unique par
    construction (numéro de série + horodatage), et l'inventaire écarte déjà les doublons.

46. **Déplacement plutôt que doublon.** Un nom déjà inventorié dont l'ancien chemin n'existe
    plus sous la racine est un fichier déplacé : même identifiant, ligne mise à jour, labels et
    embeddings conservés. Si l'ancien chemin existe encore, c'est un doublon (écarté). Un
    réinventaire sans contrôle qualité n'efface pas le contrôle qualité déjà calculé.

47. **Qualité inconnue sans commentaire.** Remplace la règle « A sinon » de DECISIONS n° 5 :
    sans commentaire, la qualité reste vide. Sur le fichier réel, 339 positifs sur 345
    auraient reçu A par défaut, ce qui aurait vidé de sens le rappel par qualité (§6). Avec la
    nouvelle règle : A 37, B 4, C 2, inconnue 302.

48. **Ne rien écrire sur les disques externes.** Ils n'appartiennent pas au projet. L'inventaire
    et le contrôle qualité n'y font que des lectures ; la base, les rapports et les modèles
    vont dans `data/`, sur la machine. Vérifié le 22/09 : aucun fichier des deux projets n'a
    été modifié par l'inventaire (seuls l'Excel et son verrou `~$` portent une date du jour,
    08:56, antérieure à l'inventaire). L'export Excel du prestataire est ignoré par git
    (`documentation/*.xlsx`) : ce ne sont pas nos données.

49. **Résultat de l'acceptation M0** (22/09, disque « Projet blanci 2025 ») :
    - inventaire de 33 210 fichiers en 8 min (en-têtes seuls) : 29 513 enregistrements,
      980 h, 3 673 doublons de relevés écartés, 24 fichiers vides en erreur ;
    - 148 fichiers d'une durée autre que 2 min (0,1 s à 14 min : tests, déclenchements
      manuels) et 69 datés à plus de 15 jours de leur relevé (juillet 2024 à décembre 2025) :
      aucun n'est vérifié, mais **à écarter avant tout tirage de négatifs** (à faire) ;
    - quelques fichiers à 24 kHz (enregistrements de test) parmi les 48 kHz ;
    - import : 345 positifs, 150 négatifs, 74 286 détections non vérifiées ignorées,
      4 en attente ; les 495 lignes retrouvent leur enregistrement, et le site de
      l'inventaire concorde avec la station de l'Excel sur les 495 (CDR 41, Mataroni 445,
      PatawaOuest 9) — ce qui confirme au passage que « RELEVE 2 Patawa » est PatawaOuest ;
    - `check-grid` : 0 annotation positive coupée sur 345, grilles 3 s et 5 s ;
    - trois enregistreurs ont un préfixe de nom personnalisé (ALOUATTA, LAGOTHRIX, SAIMIRI) :
      le micro retenu est leur numéro de série GUANO (SMA11407, 2MA02726, SMA11393).

## 2026-09-22 — Canal audio, drapeaux d'inventaire, deux disques

50. **Canal audio : premier micro (gain 6 dB) par défaut** (`audio.channel: 0`), plus la
    moyenne des deux canaux. Mesuré sur 91 enregistrements (les 51 à positifs + 40 d'autres
    sites) :
    - les deux canaux sont **deux micros distincts** qui écoutent la même scène : enveloppes
      d'énergie très corrélées (0,75 à 0,94 le plus souvent), formes d'onde presque pas
      (0,05 à 0,40, sans décalage constant) ; un micro unique enregistré deux fois donnerait
      une corrélation proche de 1 au décalage 0 ;
    - l'écart de niveau vaut bien le réglage : +10 à +12 dB pour le second canal ;
    - **les 12 dB de plus n'améliorent pas la lisibilité des notes d'A. blanci** : rapport
      signal/bruit dans la bande 4,4–5,5 kHz de 8,5 dB (6 dB de gain) contre 9,8 dB (18 dB),
      écart médian par enregistrement +0,2 dB. Le fond de la forêt est 46 dB au-dessus du
      bruit de quantification 16 bits : le gain amplifie le fond autant que les notes ;
    - le micro à 18 dB sature dans 12 % des fichiers, celui à 6 dB dans 4 %.
    Moyenner deux micros ne gagne rien, hérite des saturations du plus amplifié (4 fois plus
    fort, il domine la moyenne) et additionne deux points d'écoute décalés de quelques
    centimètres, ce qui creuse des annulations de phase à des fréquences qui dépendent de la
    direction du son — de l'ordre de la bande d'A. blanci pour un écart de ~3,6 cm. Le canal
    est inscrit dans les paramètres de l'encodeur (`models.params_json`) : des embeddings de
    micros différents ne se comparent pas. Piste pour plus tard : le second micro comme
    augmentation de données (deux « vues » du même instant).

51. **Drapeaux d'inventaire, sans lire l'audio** (`duration_off`, `off_campaign`), recalculés
    à chaque inventaire et par `blanci flag`. Un enregistrement signalé reste dans la base et
    sur le disque ; il n'est jamais encodé, donc jamais tiré comme négatif apparié, proposé à
    la vérification ni scoré.
    - `duration_off` : durée hors de 120 s ± 1 s (tests, déclenchements manuels).
    - `off_campaign` : pour chaque (jeu, site, micro), la série de dates est coupée à chaque
      trou de plus de 7 jours ; le plus gros bloc est la campagne, le reste en sort. Un micro
      posé en continu sur plusieurs relevés d'un même site (phénologie 2023-2024) reste un bloc.
    Résultat sur l'inventaire 2026 : 148 durées anormales, 70 hors relevé, 149 enregistrements
    signalés au total (69 cumulent les deux), **aucun enregistrement annoté**. Un seul hors
    relevé a une durée normale (2LA04429, 17 décembre, carte posée à Mataroni le 6 janvier).

52. **Deux disques, un seul utile en plus.** D: (Samsung T7 Shield, 4 To) et E: (Seagate
    Basic, 1 To). Comparaison des noms de fichiers :
    - E: `enregistrements_blanci_mataroni_01-26` = copie exacte du relevé Mataroni du D:
      (les 9 noms « en plus » sont les fichiers vides) ;
    - E: `Pheno_blanci_Molokoi` = copie de la partie Molokoï de « Projet Phénologie blanci »
      du D: (21 009 noms sur 21 012) ;
    - **E: `Enregistrements blanci Mataroni mai 2026` : 18 fichiers absents du D:** (un micro,
      2LA03595 au point GI07, 19 mai 2026) ;
    - D: « Projet Phénologie blanci » (2023-2024, Trésor, Kaw, Molokoï, 66 824 fichiers, jeu
      de test temporel du §6) **n'est pas encore inventorié** ; « Mares » (16 568 fichiers,
      nov. 2024 – fév. 2025) et « SM_MaraisKaw_sd1/sd2 » (808 fichiers, 2026) : rôle à préciser.
    L'inventaire ne suit qu'une racine (`paths.raw`) : les deux disques ne s'inventorient pas
    ensemble. Inutile de le changer, les enregistrements seront regroupés sur le disque du
    stage (DECISIONS n° 45-46).

53. **Feuille de route corrigée par Léonard** : 345 fenêtres positives pour 51 enregistrements
    (DECISIONS n° 35 confirmé). D'autres annotations suivront, sur plus de sites. Le second
    export (`..._dataset2_verifBV.xlsx`) n'est pas importé : même format, aucune fenêtre
    vérifiée.

## 2026-09-23 — Corrections et précisions

54. **Correction de DECISIONS n° 50 : l'écart entre les deux micros est inconnu.** « Décalés de
    quelques centimètres » était une supposition, pas une mesure. Ce qui est établi : ce sont
    deux entrées sonores distinctes (aucune superposition des canaux, même en cherchant un
    décalage jusqu'à ±250 ms : corrélation maximale 0,07 à 0,42) ; le GUANO ne décrit pas les
    micros. L'argument contre la moyenne tient sans connaître l'écart : additionner deux micros
    distants crée des annulations à des fréquences qui dépendent de l'écart et de la direction
    du son. Écart à vérifier sur un enregistreur (deux ouvertures de micro sur le boîtier ?).
    Extraits d'écoute comparée : `data/ecoute/` (hors git), produits le 23/09.

55. **Le gain est fixé à l'enregistrement.** Il s'applique au signal analogique avant la
    numérisation : impossible de le changer après coup. Multiplier un canal sur ordinateur
    change son volume, pas son rapport signal/bruit, et ne rend pas ce qu'une saturation a
    coupé. Les deux canaux étant enregistrés, le choix se fait à la lecture (`audio.channel`).

56. **« Mares » et « SM_MaraisKaw_sd1/sd2 » (disque D:) n'appartiennent pas au projet** :
    autre étude sur le même disque. À ne jamais inventorier.

57. **La machine de travail actuelle est la machine cible de l'ONF** (§7) : Intel Core
    i5-1145G7, 4 cœurs / 8 fils, 16 Go, Windows. Le débit des encodeurs (risque 5, question 8
    du §10) se mesure donc ici, sans attendre le Mac.

## 2026-09-23 — Les deux micros, baselines, sous-ensemble du benchmark, poste d'annotation

58. **Les deux canaux sont conservés** (décision de Léonard après écoute, complète n° 50 et 55).
    Le micro 2 (18 dB) fait ressortir le chant à l'oreille ; sur un positif faible (chant
    lointain, chevauchement), il peut aider l'annotateur. Donc :
    - l'encodage reste sur `audio.channel` (0 par défaut), en attendant que le benchmark dise
      si le canal 1 change quelque chose pour les modèles ;
    - le poste d'annotation fait écouter **les deux micros côte à côte** et note dans
      `conditions.channel_listened` le canal du spectrogramme affiché ;
    - les baselines comparent déjà les deux canaux (`blanci baselines --channels 0,1`).
    L'écart de quelques centimètres entre les deux micros est confirmé sur un enregistreur
    (corrige la réserve du n° 54). Extrait d'écoute d'un positif faible (seul positif noté C,
    RB04) ajouté dans `data/ecoute/3_faible_*`.

59. **bacpipe sous Windows** : TensorFlow tire `tensorflow-io-gcs-filesystem`, sans roue Windows
    depuis la 0.32. Contrainte `<0.32` sous Windows dans `[tool.uv]` (lecture de fichiers sur
    Google Cloud, jamais utilisée ici). Groupes `research` et `app` installés sur la machine ONF.

60. **Sous-ensemble du benchmark** (`dataset.benchmark_recordings`, `blanci embed --subset
    benchmark`) : les enregistrements annotés + tous les candidats aux négatifs appariés (même
    micro, créneau à ± 30 min, non signalés). Sur la base actuelle : 131 annotés + 716 candidats
    = 847 enregistrements, 28 h d'audio au lieu de 980 h. Aucun encodage lancé : il attend
    l'accord de Léonard, une fois le squelette complet.

61. **Prototype simple ajouté aux sondes du benchmark** (§3 le liste comme baseline de
    similarité ; §6/§13 l'avaient omis). L'écart prototype simple / différentiel mesure le fond
    sonore capté par l'embedding (note de Léonard).

62. **Baselines sans encodeur** (`blanci/heads/baselines.py`, `blanci baselines`) : énergie en bande,
    contraste bande / bandes voisines, onsets, rythme, template matching (gabarit moyen et
    meilleur de 30 exemplaires, appris dans chaque pli sans le micro testé). Même jeu que le
    benchmark : 345 positifs, 150 négatifs vérifiés, 1 020 négatifs appariés présumés (20 par
    enregistrement positif), plis par micro. Premier passage le 23/09 (2 min 48 s, lecture seule
    du disque D:) :
    - **AP fenêtre 0,22 à 0,37** (hasard 0,23) ; niveau enregistrement 0,10 à 0,15 (hasard 0,08).
      Aucune baseline n'approche le seuil du go/no-go ;
    - contre les seuls faux amis vérifiés, le template matching sépare bien (AP 0,93 au canal 1,
      hasard 0,70) ; contre les négatifs appariés, presque pas (0,33, hasard 0,25) ;
    - 28 % des négatifs présumés dépassent le score médian des positifs (gabarit moyen). Soit
      ils ressemblent au chant, soit **ils contiennent A. blanci** : à Mataroni, en saison, au
      même créneau, c'est plausible. Le bruit d'étiquette des négatifs présumés (§2) est donc à
      mesurer avant de juger les encodeurs : file d'audit `candidats_audit_negatifs.csv`
      (20 présumés les mieux notés + 20 au hasard) ;
    - canal 1 légèrement devant le canal 0 sur presque toutes les baselines (écarts dans les
      intervalles de confiance) ;
    - **le critère « rappel ≥ 0,8 à précision ≥ 0,1 » est vide au niveau fenêtre** avec ce jeu :
      23 % de positifs, tout accepter donne déjà une précision de 0,23. Il ne départage qu'au
      niveau enregistrement (8 % de positifs). À trancher avec le protocole figé (S3).

63. **Poste d'annotation** (`blanci/annotation/workbench.py`, `blanci/annotation/app.py` Streamlit, `blanci annotate`,
    `blanci candidates`). Files CSV dans `data/reports/candidats_*.csv` (aussi `queue_*` et
    `search_*`). `candidates --from <export Blancinet>` tire les détections jamais écoutées,
    `per_site` par site, à parts égales entre tranches de score (0–0,3 ; 0,3–0,7 ; 0,7–1) puis
    entre micros, une fenêtre par enregistrement ; `--random` ajoute des fenêtres au hasard aux
    heures de pic, à parts égales entre sites. Chaque réponse est un label en ajout seul
    (source `active`, `random` ou `audit`) ; la fenêtre est créée si besoin. Lot 1 : 100
    candidats (CDR 31, Mataroni 33, PatawaOuest 32, PatawaEst 2, RNRT 2). YAPAT n'est pas
    essayé : le poste suffit pour les premiers lots, l'essai reste possible (§5).

64. **Adaptateur bacpipe validé contre bacpipe 1.3.5** (torch 2.6, TensorFlow 2.15, Windows,
    i5-1145G7) le 23/09. Trois corrections :
    - les modèles sont dans `bacpipe.model_pipelines.feature_extractors.<nom>` (l'adaptateur
      cherchait `embedding_generation_pipelines`) ;
    - `Model(...)` ne lit ses réglages dans `bacpipe.settings` que si `device` est absent : on
      les passe tous, classifieur désactivé, et `prepare_inference()` est appelé ;
    - certains modèles rendent une séquence de jetons (birdmae) : moyennée pour l'embedding,
      exposée par `embed_tokens`, `has_tokens` mesuré au chargement.
    Poids dans `paths.models/bacpipe` (hors git), pas dans le dossier courant. Sous Windows,
    `tensorflow-intel==2.15.1` doit être déclaré (uv oubliait cette dépendance de tensorflow).
    Mesures sur bruit synthétique, CPU de la machine ONF (risque 5) :

    | encodeur | fenêtre | f_e | dim | fenêtres/s | temps réel (grille pas = ½ fenêtre) |
    |---|---|---|---|---|---|
    | birdnet | 3 s | 48 kHz | 1 024 | 26,0 | ×39 |
    | beats | 5 s | 16 kHz | 768 | 3,4 | ×8,5 |

    Ordre de grandeur : le sous-ensemble du benchmark (28 h) prend ~45 min avec birdnet et
    ~3 h 20 avec beats ; le disque entier (980 h), ~25 h et ~115 h. birdmae (version « Huge »,
    plusieurs Go) et perch_v2 ne sont pas encore téléchargés : à mesurer avant de les retenir.

## 2026-09-23 (après-midi) — Encodeurs, clustering, audit, congénères

65. **L'identifiant d'une fenêtre contient sa durée** (sauf 3 s, forme historique inchangée :
    les 495 fenêtres de la base gardent leur identifiant). Sans la durée, une fenêtre de 5 s
    (grille de beats, naturebeats, perch_v2) et une annotation de 3 s au même décalage, ou un
    label d'enregistrement entier (0 s, 120 s) et la fenêtre de 3 s à 0 s, partageaient une
    seule ligne de `windows` : la seconde héritait de la durée de la première, et le transfert
    des labels vers la grille devenait faux. Corrigé avant tout encodage : `<enr>:<décalage>`
    pour 3 s, `<enr>:<décalage>/<durée>` sinon (`db.window_id_for`).

66. **Relevé des encodeurs** (`blanci throughput`, §2 et §7) : débit, mémoire, dimension,
    jetons, projection sur une campagne (575 h) et sur les heures de pic seules. Bruit
    synthétique à 48 kHz, un processus par encodeur, rien n'est lu sur les disques. Premier
    relevé, i5-1145G7, **provisoire** (téléchargement de birdmae en parallèle : beats y tombe à
    2,1 fenêtres/s contre 3,4 mesurées seul, DECISIONS n° 64) :

    | encodeur | f_e | fenêtre | dim | temps réel | campagne 575 h | mémoire |
    |---|---|---|---|---|---|---|
    | birdnet | 48 kHz | 3 s | 1 024 | ×60 | 10 h | 2,0 Go |
    | perch_v2 | 32 kHz | 5 s | 1 536 | ×14 | 41 h | 4,1 Go |
    | beats | 16 kHz | 5 s | 768 | ×5 | 109 h | 2,2 Go |
    | naturebeats | 16 kHz | 5 s | 768 | ×5 | 106 h | 2,2 Go |

    **perch_v2 tourne en ONNX Runtime, sans TensorFlow** (`perch_v2_no_dft.onnx` fourni par
    bacpipe) : le repli (2) du §2 fonctionne sur la cible ; risque 7 levé pour l'exécution,
    la conformité aux embeddings de référence reste à vérifier. Il expose aussi des jetons
    spatiaux (16 × 4 × 1 536) que l'adaptateur n'utilise pas encore (attentive probing, P2).
    birdmae (Bird-MAE-**Huge** dans bacpipe, pas la version Base du §7) : téléchargement en
    cours ; protoclr, perch_bird, convnext_birdset ensuite.

67. **Contrôle passe-bas du §2** : `birdmae_lp8k` = birdmae sur l'audio filtré à 8 kHz
    (Butterworth d'ordre 8, phase nulle, à la f_e d'origine). Si son AP égale celle de
    birdmae, l'objection « 16 kHz, très limite » ne vaut pas pour la note de 4,4–5,5 kHz.
    Toute entrée de `encoders.models` accepte `lowpass_hz`.

68. **Clustering C0/C1** (`blanci cluster --mode c0|c1`, §5 bis) : normalisation L2, ACP
    (50 composantes), HDBSCAN (scikit-learn). C0 : échantillon du stock (tirage au prorata des
    partitions, une à la fois en mémoire), AMI groupes/micros, part de fenêtres signalées par
    groupe. C1 : positifs de Mataroni + 5 000 fenêtres des mêmes micros aux mêmes heures ;
    verdict sur le meilleur groupe (rappel ≥ 0,5, enrichissement ≥ 20, AMI micro ≤ 0,3 :
    seuils de jugement, dans `cluster` de la config). Les affectations par fenêtre sont
    écrites pour C2 (étiquetage en bloc). Fenêtres recadrées sur les onsets : pas encore.

69. **Écoute d'enregistrements entiers** (`blanci candidates --entiers N --reason …`) : audit
    aléatoire (§6, 300 enregistrements de Mataroni) et jeu gelé (60, stratifiés). Parts
    égales entre micros, puis heures locales les moins servies d'abord ; source `audit` ;
    le label vaut pour toute la grille (annotation par enregistrement, §5). Le jeu gelé n'est
    pas encore **exclu de l'entraînement** : à faire avant d'en écouter un (M4).
    **Calibration entre annotateurs** (§5) : case « ne masquer que mes réponses » dans le
    poste, et `blanci agreement --annotators a,b` : accord brut sur le label, accord
    blanci/non (« A. blanci ? » compte non), accord sur les positifs 2a/(2a+b+c), tableau
    croisé.

70. **Logits des congénères de Perch 2.0** (§2, §3, §5) : pendant `embed` avec perch_v2,
    l'adaptateur garde les logits d'*A. baeobatrachus*, *A. stepheni* et *A. surinamensis*
    (`logit_classes` de la config ; vérifié sur le vrai modèle) et `embed` les range dans
    `scores` (`<encodeur>:logit:<espèce>`), sans seconde inférence. `blanci candidates
    --congeners <perch_v2-…>` en tire une file : meilleure fenêtre par enregistrement, le
    meilleur de chaque micro d'abord. Logits non calibrés : classement seulement.

71. **Métriques du §6 complétées** : `evaluate.recall_by_group` (rappel au seuil de précision
    plancher par qualité A/B/C, site, tranche de RSB, avec Wilson), affiché par
    `blanci evaluate` ; `false_alarms_per_hour` (réservé aux ensembles exhaustifs : audit,
    jeu gelé) ; `sequential.note_snr_db` (énergie en bande pendant les notes contre les 0,5 s
    voisines). La qualité de l'annotation suit désormais la fenêtre jusqu'au jeu d'évaluation.
    La CLI écrit en UTF-8 : la console Windows en cp1252 plantait sur « ≥ ».

72. **Relevé des encodeurs, machine au repos** (i5-1145G7, CPU seul, bruit synthétique à
    48 kHz ; remplace les chiffres provisoires du n° 66). Campagne = une semaine de pose,
    575 h d'audio ; pas de la grille = ½ fenêtre ; `data/reports/debit.md` :

    | encodeur | f_e | fenêtre | dim | fenêtres/s | temps réel | campagne | heures de pic | mémoire |
    |---|---|---|---|---|---|---|---|---|
    | birdnet | 48 kHz | 3 s | 1 024 | 43,3 | ×65 | 9 h | 2,5 h | 2,0 Go |
    | protoclr | 16 kHz | 6 s | 384 | 17,5 | ×52 | 11 h | 3 h | 1,7 Go |
    | perch_v2 | 32 kHz | 5 s | 1 536 | 7,0 | ×17 | 33 h | 9 h | 4,1 Go |
    | birdmae_base | 32 kHz | 5 s | 768 | 3,7 | ×9 | 62 h | 17 h | 1,5 Go |
    | convnext_birdset | 32 kHz | 5 s | 1 024 | 3,7 | ×9 | 62 h | 17 h | 2,7 Go |
    | beats | 16 kHz | 5 s | 768 | 2,9 | ×7 | 81 h | 22 h | 2,2 Go |
    | naturebeats | 16 kHz | 5 s | 768 | 2,8 | ×7 | 82 h | 23 h | 2,3 Go |
    | perch_bird | 32 kHz | 5 s | 1 280 | 1,8 | ×4,5 | 129 h | 36 h | 2,4 Go |
    | birdmae (Huge) | 32 kHz | 5 s | 1 280 | 0,5 | ×1,2 | 494 h | 137 h | 4,2 Go |

    - **birdmae tel que bacpipe le charge (Bird-MAE-Huge) est hors de portée de l'i5**, même
      aux heures de pic seules. Variante `birdmae_base` ajoutée (Bird-MAE-Base, 768
      dimensions, le modèle que le §7 chiffrait) ; le benchmark dira si Base tient l'AP de
      Huge. Encoder le sous-ensemble du benchmark (28 h) avec Huge prendrait ~23 h ici : à
      faire sur le Mac, ou à remplacer par Base.
    - Seuil du risque 5 (> 15 h par campagne après leviers) : seuls birdnet (non déployable,
      licence et TensorFlow) et protoclr passent sans levier ; perch_v2 (33 h, 9 h aux
      heures de pic) passe avec les heures de pic ; les autres demandent aussi la
      quantification int8 et le sous-échantillonnage des fenêtres (§7).
    - perch_bird a échoué au premier téléchargement (délai dépassé chez Kaggle), réussi au
      second. Tous les encodeurs du §2 sont désormais installés (`data/models/bacpipe`, cache
      Hugging Face), 12 à 40 s de chargement chacun.

## 2026-09-23 (fin d'après-midi) — Jeu gelé, fusion, contrôle qualité, attentive, AnuraSet

73. **Fenêtres des encodeurs à 5–6 s : règle actuelle conservée** (décision de Léonard).
    Une fenêtre de la grille de l'encodeur hérite du label d'une annotation de 3 s si elle la
    contient entière (DECISIONS n° 4) ; vérifié : les 495 annotations tiennent entières dans
    une fenêtre de 3, 5 et 6 s. Pas de fenêtres recentrées sur les annotations.
    **Risque mesuré sur le contexte ajouté** : aucun des 150 négatifs n'est dans un
    enregistrement qui contient un positif annoté, mais pour 42 d'entre eux (5 s) et 28
    (6 s), Blancinet a une détection non écoutée de score ≥ 0,5 dans les 2–3 s ajoutées.
    Si du chant s'y trouve, la fenêtre est un négatif faux : la tête apprend à baisser le
    score d'un chant vrai, et le benchmark compte une fausse alarme là où l'encodeur avait
    raison. Le bruit touche les encodeurs à fenêtre longue, pas birdnet (3 s).

74. **Jeu gelé programmé** (`blanci/inputs/frozen.py`, `blanci freeze`, `blanci evaluate --frozen`).
    Une version = liste d'enregistrements figée (`paths.frozen_test/jeu_gele_<v>.csv`, en
    lecture seule, jamais réécrite). Ses enregistrements sont exclus de tout ce qui apprend
    ou règle : tête, seuil, benchmark, baselines, recherche de similarité, clustering C1,
    fusion, jetons. Chaque tête enregistre les versions exclues ; `evaluate --frozen` refuse
    une tête entraînée avant le gel. Mesures sur le jeu gelé : AP (enregistrement, fenêtre),
    rappel au seuil de la tête, fausses alarmes par heure (le jeu est écouté en entier), rappel
    par qualité et par site.

75. **Module séquentiel et fusion branchés** (§3). Débuts de notes calculés une fois par
    enregistrement pendant `embed` (l'audio est déjà en mémoire), rangés dans la table
    `onsets` (migration 2 de la base, ajout seul) ; `blanci onsets` pour les autres.
    `blanci fusion` : dans chaque pli par micro, une tête sans le micro score les fenêtres
    étiquetées et toutes les fenêtres de leurs enregistrements (persistance) ; fusion
    évaluée hors-pli contre la tête seule (bootstrap apparié par enregistrement), puis
    enregistrée en JSON avec son seuil. `blanci score --fusion` décide avec elle. Colonnes :
    `fusion.columns` (frac_ioi_blanci, onset_rate_hz, frac_windows, longest_run : 4, pour
    ~10 enregistrements positifs par coefficient). Le seuil de persistance est la frontière
    de la tête logistique (0).

76. **Calibration du contrôle qualité** (`blanci qc-calibrate`, lecture seule, 131
    enregistrements, 2 à 3 min). Indice « micro dans sac » = part d'énergie au-dessus de
    2 kHz : enregistrements avec fenêtres « micro dans sac » 0,001 à 0,30 ; enregistrements
    à A. blanci 0,265 à 1 (médiane 0,98) ; autres 0,30 à 1. **Le seuil actuel (0,02) ne
    signale qu'un enregistrement « dans sac » sur 8.** 0,2 en signalerait 6/8 avec une marge
    sous le positif le plus bas ; non appliqué, décision de Léonard. Pluie : trop peu de
    cibles (3 enregistrements) pour calibrer, seuil gardé. Silence et saturation : aucune
    fenêtre étiquetée. Rappel : l'inventaire a été fait sans contrôle audio (`--no-qc`),
    ces drapeaux ne sont donc calculés sur aucun enregistrement réel aujourd'hui.

77. **Attentive probing** (`blanci/heads/attentive.py`) : une requête apprise pondère les jetons
    d'une fenêtre avant le classement (2d + 1 paramètres). Entraîné avec torch, appliqué en
    numpy, sauvegardé sans pickle. Seul perch_v2 expose des jetons (16 temps × 4 fréquences
    × 1 536, moyennés sur la fréquence). `blanci tokens --encoder perch_v2` calcule les
    jetons des seules fenêtres du benchmark (~1 500) ; `blanci benchmark` ajoute alors la
    sonde « attentive ». Sur données synthétiques (note dans 1 jeton sur 16), AP hors-pli
    ~0,67–0,8 contre ~0,53 pour la moyenne des jetons.

78. **Pré-benchmark AnuraSet préparé** (`blanci/evaluation/anuraset.py`, `anuraset/anuraset.yaml`,
    commandes `anuraset prepare`, `anuraset profile`, `anuraset benchmark`). Licence
    **CC BY** (la feuille de route disait CC0). Téléchargé : `raw_data.zip` (7,2 Go,
    enregistrements bruts d'une minute) + `strong_labels.zip` (chants datés), pas
    `anuraset.zip` (extraits de 3 s, trop courts pour les encodeurs à 5–6 s). Base, stocks et
    rapports séparés des données ONF. Fenêtre positive = contient un chant entier de l'espèce
    (ou tient dans un chœur annoté d'un seul tenant) ; négative = aucun chant de l'espèce ;
    chant coupé = écartée ; négatifs tirés par site (1:20) ; plis par site (4 sites).
    Espèces à choisir avec `anuraset profile` (note brève, dominante 3–6 kHz, ≥ 300 chants,
    ≥ 2 sites). Encodage d'AnuraSet (~27 h d'audio) : avec celui des données ONF, ce week-end.

79. **Drapeaux : trois origines, une règle d'exclusion** (23/09/2026, avec Léonard). Un
    drapeau est une remarque sur un enregistrement (`recordings.qc_flags`, le fichier n'est
    jamais touché). Origines : inventaire (durée anormale, hors relevé), audio (silencieux,
    saturation, micro dans sac, pluie), **écoute** (nouvelle clé `annotated` : présente sur
    tout enregistrement annoté à la main, avec les drapeaux que l'annotateur y a posés —
    label `artefact_in_bag` → micro dans sac ; label `rain` ou mention de pluie → pluie).
    **Seuls silencieux, micro dans sac, durée anormale et hors relevé écartent du corpus**
    (jamais encodés) ; pluie et saturation sont des remarques : un micro sous la pluie
    enregistre son milieu, ces enregistrements font partie du jeu de données. **Un
    enregistrement où A. blanci a été entendu n'est jamais écarté** : l'écoute prime sur le
    calcul. Le contrôle audio se fait pendant `embed` sur l'audio déjà lu (option `--no-qc`
    pour s'en passer), une fois par enregistrement ; ses indices sont gardés, `blanci flag`
    réapplique un seuil changé sans relire l'audio et recalcule les drapeaux d'écoute (aussi
    recalculés à chaque label ajouté et après chaque import). Sur la base au 23/09 : 131
    enregistrements annotés, 8 micro dans sac (aucun avec un positif), 5 pluie (dont 2 avec
    un positif). Seuil « micro dans sac » inchangé (0,02) tant que Léonard n'a pas tranché
    (n° 76).

80. **Négatifs suspects** (23/09/2026, règle de Léonard). Un négatif annoté est jugé sur
    3 s ; les encodeurs à fenêtre de 5–6 s y ajoutent 1 à 3 s jamais écoutées. Règle : un
    négatif est **suspect** si A. blanci est détecté dans les fenêtres voisines (± 3 s, la
    fenêtre de 3 s de chaque côté, ce qui couvre tous les encodeurs du §2) ; sinon il reste
    négatif. « Détecté » = détection Blancinet de score ≥ 0,5 que personne n'a écoutée
    (aucune fenêtre annotée ne la contient), ou positif annoté. Jamais les scores de nos
    propres têtes (ils choisiraient les labels qui les jugent). Un suspect ne sert ni à
    l'entraînement ni à l'évaluation, **pour tous les encodeurs** (mêmes négatifs pour tous,
    benchmark comparable). Même règle pour les négatifs appariés présumés : aucun n'est tiré
    à moins de 3 s d'une détection non écoutée. Les détections Blancinet (74 785, dont
    74 286 jamais écoutées) sont rangées comme scores du détecteur « blancinet »
    (`blanci import-detections`), pas comme labels. Au 23/09 : **46 négatifs suspects sur
    149**. `candidates --suspects` tire les 52 détections voisines à écouter : une voisine
    écoutée et non-blanci lève le soupçon, une voisine blanci le confirme (et donne un
    positif). **Effet mesuré sur les baselines** (relancées, lecture seule) : écarter les
    46 suspects annotés change peu l'AP (meilleure baseline, fenêtres : 0,368 → 0,374) ;
    écarter aussi les négatifs présumés voisins d'une détection la fait passer à 0,543
    (enregistrements : 0,155 → 0,263). Deux lectures : bruit d'étiquette retiré (au même
    micro, même créneau, en saison, une détection Blancinet ≥ 0,5 est souvent un vrai
    chant) ou négatifs difficiles retirés (évaluation optimiste). Le jeu gelé, écouté en
    entier, tranchera. Revers constaté : les suspects annotés sont surtout les faux amis
    principaux (*A. andreae* 12 sur 22, Fourmilier tacheté 11 sur 23), qui chantent en
    continu et déclenchent Blancinet sur les fenêtres voisines : sans écoute des voisins,
    la tête perd la moitié de ses exemples de ces faux amis. Écouter les 52 voisins avant
    l'entraînement.

81. **Seuil « micro dans sac » : 0,02 → 0,2** (accord de Léonard, calibration n° 76).
    Signale 6 enregistrements « dans sac » annotés sur 8, aucun enregistrement à A. blanci
    (le plus bas est à 0,265). Contrôle audio pendant `embed` réglable
    (`qc.during_embed`), coupé pour AnuraSet : seuils calibrés sur les Song Meter ONF, et
    positifs AnuraSet absents de la table labels, donc non protégés.

82. **Commentaire accolé à chaque fenêtre annotée** (précision de Léonard sur le n° 79 : le
    « drapeau » voulu est le commentaire de l'annotateur, en plus du label). Il était déjà
    gardé tel quel dans chaque label (`conditions.comment`) ; il suit désormais la fenêtre :
    `current_labels` et le jeu d'apprentissage (fenêtres de grille héritières) portent une
    colonne `comment` ; `blanci export-labels` écrit toutes les fenêtres annotées avec label,
    qualité, espèce, commentaire et suspect. Au poste d'annotation, le commentaire est lu
    comme à l'import (conditions pluie, lointain, second plan ; espèces citées) : « pluie »
    écrit au poste pose le drapeau pluie. Les drapeaux d'écoute par enregistrement (n° 79)
    restent : ce sont eux qui écartent les 8 enregistrements « dans sac ».

83. **Réentraînement sans intervention** (§4, M5, `blanci retrain`). Nouvelle tête sur
    tous les labels hors jeu gelé, seuil hors-pli ; nouvelle tête et tête adoptée jugées
    sur le même jeu gelé, à leur seuil ; adoption si l'AP (enregistrements) et le rappel au
    seuil ne perdent pas plus de 0,02 (`retrain.tolerance`). Sans jeu gelé : pas
    d'adoption, sauf `--force`. Tête adoptée non jugeable (entraînée avant le gel) : la
    nouvelle est adoptée. Historique des adoptions dans `models` (kind `adoption`) ; une
    tête refusée reste en base. `blanci score` prend par défaut la tête adoptée (sinon la
    plus récente) : une tête refusée n'est jamais utilisée par mégarde. La fusion n'est pas
    réentraînée par cette commande.

84. **Courbes d'activité** (§6 niveau 3, M4, `blanci activity`). Depuis les décisions par
    enregistrement : indice horaire (part des enregistrements détectés par heure locale,
    l'effort au dénominateur) et probabilité journalière par mois (part des jours avec au
    moins une détection), intervalles de Wilson, par site ou par micro. Confrontées aux
    patrons de Courtois et al. 2025 (`activity.reference` : pics 7–9 h et 15–17 h, mois
    forts janvier–avril, creux juillet–octobre) : un patron est « retrouvé » si les
    intervalles de Wilson (pics contre autres heures, mois forts contre creux) sont
    disjoints — un rapport > 1 seul se lève au hasard sur une activité plate. Avec des
    courbes numérisées de la publication (`--reference-hours`, `--reference-months`),
    corrélation de Pearson, alerte sous 0,5 (§11, risque 6). Enregistrements « suspect »
    non comptés, sauf `--suspects-detected`. À lancer sur 2023 une fois la phénologie
    inventoriée, encodée et scorée.

85. **Négatifs suspects abandonnés** (23/09/2026, Léonard ; remplace la partie « négatifs
    annotés » du n° 80). Les voisins d'un négatif sont souvent le même faux ami (un
    Fourmilier tacheté qui chante sur une fenêtre chante sur les suivantes) : écouter les
    voisins en ferait écouter d'autres, sans fin. Règle : **un négatif annoté est négatif,
    quel que soit le contexte** ajouté par les encodeurs à fenêtre de 5–6 s. Si A. blanci ne
    chante pas pendant les 3 s écoutées, qu'elle commence juste après est peu probable.
    **Biais possible, gardé en tête** : quelques fenêtres « négatives » de 5–6 s peuvent
    contenir la fin d'un chant ; il pèserait sur les encodeurs à fenêtre longue. Supprimés :
    colonne `suspect`, `candidates --suspects`. **Gardé** (question distincte, non
    tranchée) : aucun négatif *présumé* n'est tiré à ± 3 s d'une détection Blancinet ≥ 0,5
    non écoutée ; les détections restent rangées comme scores. Baselines relancées sous
    cette règle.

86. **AnuraSet extrait et profilé** (23/09/2026, lecture et extraction seules, rien
    d'encodé). 1 612 enregistrements d'une minute, 4 sites, archive vérifiée. Aucune espèce
    ne remplit tous les critères (note ≤ 0,3 s, dominante 3–6 kHz, ≥ 300 chants, ≥ 2
    sites) : les durées sont celles des segments annotés, qui regroupent souvent plusieurs
    notes, et la plupart des espèces n'est présente que sur un site. Seule candidate
    multi-sites dans la bande : **DENMIN** (*Dendropsophus minutus*, 1 724 chants, 3 sites,
    ~5,3 kHz, segment médian 0,61 s). Mono-site dans la bande : LEPPOD (761, ~5,8 kHz,
    0,33 s), PHYDIS (419, ~5,3 kHz, 0,33 s), DENNAN (596, ~4,4 kHz, 0,44 s) ; elles
    imposeraient des plis par enregistrement, qui ne disent rien du transfert entre sites.
    Fréquence dominante mesurée sur l'intervalle annoté : peut être captée par des
    insectes (PHYCUV à 6,3 kHz est suspect). Choix des espèces (`anuraset.species`) :
    Léonard.

87. **Règle des négatifs présumés voisins supprimée** (23/09/2026, Léonard). Plus aucune
    règle liée aux détections Blancinet dans la construction des jeux : négatifs annotés et
    présumés sont tirés comme avant le n° 80. Les détections restent rangées comme scores
    du détecteur « blancinet » (`import-detections`), pour comparer Blancinet et nos têtes
    sur les mêmes fenêtres. Baselines relancées : de nouveau celles d'avant le n° 80.

## 2026-09-24 — Ajouts demandés par Léonard (liste du 24/09, après le point d'étape)

88. **Négatifs appariés : trois stratégies** (`benchmark.pairing`, `dataset.py`). Écart
    constaté : la règle « même créneau, autre jour » n'excluait pas le même jour ; comme les
    micros enregistrent toutes les 30 min, les enregistrements voisins du positif (± 30 min, le
    même jour) étaient tirés aussi. Désormais : `other_day` (défaut, autre jour vraiment),
    `same_day` (même jour, les enregistrements les plus proches à au moins
    `same_day_min_gap_min`, et au plus ce délai + la tolérance), `mixed` (moitié, moitié ; l'un
    complète l'autre). Colonne `pairing` sur chaque négatif présumé. Avis donné à Léonard : le
    même jour est le meilleur témoin du fond (météo, chœur, saison), mais A. blanci chante par
    épisodes : le voisin le plus proche est aussi le plus susceptible de la contenir. À
    trancher en écoutant une vingtaine de négatifs présumés de chaque stratégie. Le
    sous-ensemble à encoder (`embed --subset benchmark`) prend les candidats des deux
    stratégies : changer de stratégie ne demande pas de ré-encoder. **Les baselines et tout
    benchmark antérieur sont à relancer** (négatifs changés).

89. **Chevauchement ajustable des fenêtres** (`encoders.overlap`, `embed --overlap`,
    `grid.hop_for_overlap`). De 0 (fenêtres jointives, grille standard) à 0,99 ; pas =
    fenêtre × (1 − chevauchement), au centième (précision de `window_id`), jamais moins de
    0,01 s. Hors 50 %, stock séparé `<encodeur>@o<%>` : deux grilles ne se mélangent jamais,
    et les têtes, scores et décisions de chaque grille restent distincts. À 99 %, 50 fois plus
    de fenêtres qu'à 50 % : réservé au sous-ensemble du benchmark. `grid_hop_ratio` reste lu.

90. **Seuillage spectral en amont, fonctionnalités activables** (`blanci/prefilter.py`,
    section `prefilter`, option `--prefilter`, `blanci prefilter-bench`). Revient, à la
    demande de Léonard, sur le « pas de filtre amont » du §3 (jugement) : tout reste coupé par
    défaut, et le banc d'essai juge avant tout usage. Transformations (le son donné à
    l'encodeur change → autre encodeur, stock `<nom>+bp3-7k`, jugé par le benchmark) :
    passe-bande, débruitage par soustraction spectrale. Portes (une fenêtre arrêtée n'est pas
    encodée, embedding nul, colonne `gated` du stock, score `GATED_SCORE`, jamais apprise ; un
    positif arrêté compte comme manqué) : énergie en bande, contraste aux bandes voisines,
    débuts de notes, rythme ; combinaison « toutes » ou « une ». Banc d'essai : par porte et
    par seuil, négatifs arrêtés (calcul économisé) contre rappel plafond en enregistrements ;
    avec un encodeur, AP après la porte, et la combinaison configurée jugée fidèlement (tête
    réentraînée sans les fenêtres arrêtées) contre l'absence de porte.

91. **Plis communs et stock de scores hors-pli** (`dataset.folds_for`, `blanci/evaluation/oof.py`,
    `blanci sources`). Les plis étaient recalculés sur chaque jeu (StratifiedGroupKFold sur
    les fenêtres) : deux encodeurs, avec leurs grilles et leurs négatifs présumés, n'avaient
    pas exactement les mêmes micros tenus à l'écart. Désormais le pli de chaque micro est
    calculé une fois, sur les enregistrements annotés (hors jeu gelé), et partagé par
    encodeurs, têtes, baselines, fusion, ensembles et détecteurs. Chaque benchmark range ses
    scores hors-pli (`paths.reports/oof/`) au même format, avec une empreinte des labels et
    réglages : une source calculée sur d'autres labels est signalée au lieu d'être comparée
    en silence. Noms des sources : `<encodeur>/<tête>`, `baseline/<nom>/c<canal>`,
    `<encodeur>/fusion:<emplacement>/<méthode>`, `ensemble:<méthode>(…)`, `detector/<nom>`,
    `external/blancinet`.

92. **Toutes les têtes à comparer** (`head.py`, `pooling.py`, `head_benchmark.py`,
    `blanci heads`). Ajouts : recherche par l'exemple (plus proche positif ;
    `exemplar_medoid` = un seul exemple de référence, le positif le plus central), linear
    probe sur jetons résumés (`logistic:<pooling>` : moyenne, maximum, les deux, moyenne +
    écart-type, top-k, moyenne en fréquence et maximum en temps, et l'inverse), cascade
    (logistique puis attentive sur les 20 % meilleurs candidats, `head.cascade_fraction`). Les
    jetons de perch_v2 sont gardés en grille 16 temps × 4 fréquences (ils étaient moyennés sur
    la fréquence) ; l'attentive les prend à plat. Seul perch_v2 expose ses jetons : pour un
    transformer, `encoders.models.<nom>.token_grid` (à vérifier sur le code du modèle,
    notes.md) remet les jetons en grille. Le benchmark des têtes donne aussi le diagnostic du
    fond capté (prototype différentiel contre simple, apparié). Mécanisme d'attention propre
    au modèle et wrapper bacpipe : non programmés (rien de concret à brancher aujourd'hui).

93. **Courbe selon le nombre d'annotations du site cible** (`blanci heads-curve`). Hypothèse
    de Léonard : le prototype différentiel généraliserait mieux que le linear probe sur un site
    peu annoté. Protocole : par cible (micro tant que tous les positifs sont à Mataroni, site
    ensuite), moitié de test fixe tirée au hasard (positifs, négatifs annotés, négatifs
    présumés séparément) ; entraînement = autres cibles + négatifs présumés de la réserve (le
    fond du site, gratuit) + k enregistrements positifs de la réserve (k emboîtés, 0 = transfert
    pur) et une part proportionnelle de ses négatifs annotés ; 5 tirages ; C choisi une fois
    par cible sur les autres. Écart de chaque tête à la référence, apparié par (cible, tirage),
    intervalle bootstrap sur les cibles : l'hypothèse tient si différentiel − logistique > 0
    aux petits k.

94. **Fusion à N entrées** (`fusion.py`, `stacking.py`, `blanci fusion-bench`, `fusion.method`,
    `fusion.sources`). Entrées : score de la tête (hors-pli), descripteurs séquentiels, et
    toute autre source — tête d'un autre encodeur (`head:<id>`, apprise pli par pli, ramenée
    sur la grille principale), logits des congénères (`congeners:<id>`) : trois entrées ou plus.
    Méthodes : logistique (poids appris), poids fixés à la main (`fusion.weights` ; les entrées
    sans poids se partagent le reste), poids cherchés sur une grille, moyenne, moyenne des
    rangs, maximum (OU), minimum (ET), entrées orientées sur l'entraînement. La part de chaque
    entrée répond à « quel expert écouter en priorité ? ». `blanci fusion` enregistre la
    méthode, l'emplacement et les sources de la config ; une fusion déjà enregistrée dans
    l'ancien format reste lue.

95. **Module séquentiel ≠ seuillage spectral ; emplacement paramétrable**
    (`sequential.position`). Réponse à la question de Léonard : le seuillage agit sur le son
    avant l'encodeur ; le module séquentiel produit des descripteurs (rythme, sur l'audio ;
    persistance, sur les scores de la tête). Ils partagent leurs briques (enveloppe en bande,
    débuts de notes). Emplacements : `upstream` (le rythme sert de porte : c'est une porte du
    seuillage en amont, calculée sur les débuts de notes rangés, `sequential.gate`),
    `parallel` (rythme fusionné avec la tête), `downstream` (persistance, qui n'existe qu'après
    la tête), ou aucun. Défaut : parallèle + aval (la fusion d'avant). `fusion-bench` compare
    les emplacements × méthodes à la tête seule.

96. **Ensemble de modèles** (`ensemble.py`, `blanci ensemble`). Oui, il faut un outil : les
    grilles diffèrent (3 s, 5 s). Trois voies : combinaison par enregistrement de n'importe
    quelles sources du stock hors-pli (méthodes de la fusion, apprises pli par pli) ;
    combinaison par fenêtre = fusion à N entrées (production possible) ; concaténation des
    embeddings (blocs normalisés) pour des encodeurs de même grille. Un ensemble n'est jugé
    que contre sa meilleure source seule.

97. **Emplacements : distillation, modèle fait maison, fine-tuning / LoRA**
    (`blanci/heads/detectors/`, `blanci/heads/finetune.py`, `blanci detector-bench`). Contrat
    « détecteur » (audio → score, `fit` s'il apprend) et banc d'essai déjà branchés : hors-pli
    sur les plis communs, scores rangés dans le stock commun, donc repris par le benchmark
    complet et les ensembles. `band_contrast` montre que la chaîne marche ; `distilled` et
    `homemade` disent qu'ils sont réservés. Fine-tuning / LoRA : contrat de sortie (un
    encodeur en paquet ONNX) et contrainte d'évaluation (adaptation pli par pli, ou jugement
    sur le seul jeu gelé et les nouveaux sites). Conforme au §13.7 : rien avant M4.

98. **Benchmark complet des modèles** (`full_benchmark.py`, `blanci benchmark-all`). Toutes
    les sources au niveau enregistrement, sur les enregistrements évalués par toutes : AP [IC],
    rappels aux précisions plancher [Wilson], rappel par site, coût des encodeurs (dimension,
    débit), fraîcheur des labels ; comparaison appariée à la référence. Sources externes
    rangées à la demande sur les fenêtres des baselines : Blancinet (peut-être entraîné sur ces
    mêmes enregistrements : optimiste, à confirmer avec Biophonia), logits des congénères.

99. **Outil de sélection unique** (`selection.py`, `blanci select`, `cluster-status`,
    `cluster-label`, `yapat-export`, `yapat-import`). Méthodes : 60-20-20 à proportions
    réglables, similarité, couverture (k-centres : YAPAT fait maison automatique), groupes
    HDBSCAN puis étiquetage en bloc d'un groupe homogène (10 écoutes d'un même label,
    `selection.cluster_min_checked`), audit aléatoire, hasard, negative mining (scores hauts là
    où A. blanci est improbable, ou proches des faux amis annotés), gabarit phénologique
    (`selection.phenology`), détections isolées, congénères, Blancinet. Nouvelles sources de
    labels : mining, phenology, suspect, coverage, cluster, bulk (propagé sans écoute), yapat.
    YAPAT (Docker, PostgreSQL, embeddings BirdNET, licence non commerciale) n'est pas embarqué :
    export d'extraits + manifeste, relecture des réponses (format non documenté : à valider au
    premier essai, `selection.yapat_label_map`).

100. **Poste d'annotation refondu** (`blanci annotate`). Mode de sélection dans le panneau de
     gauche (file existante, ou file tirée sur place par n'importe quelle méthode), carte des
     embeddings où l'on entoure une zone à écouter (YAPAT fait maison, à la main), formulaire
     classe + qualité + espèce + commentaire et bouton « Envoyer ▶ » (label ajouté, fenêtre
     suivante), étiquetage en bloc des groupes homogènes. `app/streamlit_app.py` : voir n° 104.

## 2026-09-24 (soir) — Retours de Léonard sur les ajouts

101. **Négatifs appariés : la plus proche, même enregistrement compris, par défaut**
     (`benchmark.pairing: nearest`, décision de Léonard). Rappel : ce ne sont que des fenêtres
     *présumées* négatives (jamais écoutées, jamais écrites dans la table labels) ; les négatifs
     annotés entrent à part. Désormais, par enregistrement positif, les fenêtres les plus proches
     dans le temps d'une annotation positive, dans l'enregistrement lui-même d'abord — jamais une
     fenêtre qui chevauche une annotation positive, jamais une fenêtre encadrée de positifs
     (n° 102) —, puis le même jour, puis un autre jour (colonne `pairing` : same_recording,
     same_day, other_day). **Risque signalé** : si A. blanci chante tout l'enregistrement quand
     elle chante (H20, notes de Léonard), les voisines d'un positif en contiennent : négatifs
     faux, et la tête apprendrait à baisser le score d'un vrai chant. À mesurer en écoutant une
     vingtaine de ces négatifs ; `other_day` reste disponible. Les fenêtres des baselines
     incluent désormais celles des enregistrements positifs.

102. **Faux négatifs suspects : fenêtres négatives encadrées de positives** (idée de Léonard,
     miroir du « suspect » = positif isolé, faux positif probable). Critère commun
     (`sequential.surrounded_by_positives`) : un positif avant et un après, chacun à moins de
     `sequential.gap_radius_s` (6 s ≈ 4 notes). Usages : (1) jamais tirée comme négatif
     présumé ; (2) négatif annoté encadré de positifs annotés : colonne `suspect_fn` du jeu
     d'apprentissage, **il reste négatif** (n° 85 : l'écoute prime), à réécouter ; (3)
     persistance : descripteur `n_gaps` (trous dans une série positive) ; (4) sélection
     `gaps` : mode `scores` (fenêtres sous le seuil de la tête entre deux au-dessus : si
     A. blanci y chante, les positifs que le modèle rate, les plus utiles à annoter), mode
     `labels` (négatifs annotés entre deux positifs, réécoute). Source de label « gap ».

103. **Seuillage spectral et module séquentiel fusionnés** (demande de Léonard). Un seul
     module (`sequential.py` ; `prefilter.py` supprimé, `encoders/prefiltered.py` devient
     `encoders/upstream.py`), une seule section de config (`sequential` : `position`,
     `columns` — ex-`fusion.columns` —, `gap_radius_s`, `upstream` — ex-section `prefilter` —),
     un seul réglage de place : `position` dit où le module agit, `upstream.*.enabled` dit
     quelles fonctionnalités amont. Une seule détection des notes partout : les portes `notes`
     et `rhythm` se comptent sur les débuts de notes de l'enregistrement (`onset_counts`), à
     l'encodage comme dans la chaîne de décision et au banc d'essai ; `sequential.gate` est
     supprimé (la chaîne prend les portes de rythme activées, sinon `notes` à son seuil) ; les
     portes spectrales agissent à l'encodage (stock `+g-…`). `embed` n'applique l'amont que si
     `position` contient `upstream`, ou sur `--upstream a,b` (ex-`--prefilter`) ; banc
     d'essai : `blanci upstream-bench` (ex-`prefilter-bench`). Les anciennes clés
     (`prefilter`, `fusion.columns`) restent lues.

104. **`app/streamlit_app.py` supprimé** (décision de Léonard : désuet). Il écrivait les labels
     sans passer par la couche de service (ni validation, ni drapeaux d'écoute) ; le poste
     d'annotation est `blanci/annotation/app.py` (`blanci annotate`). La feuille de route (§13.2) le cite
     encore dans l'arborescence : document de référence, non modifié.

## 2026-09-25 — Notebooks d'exploration ; ce que contiennent les négatifs présumés

105. **Notebooks d'exploration** (demande de Léonard). `notebooks/01_explorer_une_fenetre`
     (un enregistrement et une fenêtre à travers la chaîne : écoute, grille, portes, module
     séquentiel en amont, en parallèle et en aval, négatif apparié, embeddings stockés ou
     calculés à la volée, prototype différentiel dans le son, sur la paire et sur le stock),
     `02_negatifs_apparies` (ce que tire chaque stratégie, feuille d'écoute et taux de
     contamination avec IC de Wilson), `03_module_sequentiel` (réglage de la détection des
     notes, valeurs et balayage des portes). Calculs dans `blanci/exploration/explore.py` (base ouverte en
     `mode=ro`, audio lu), graphiques dans `blanci/exploration/explore_plots.py` ; groupe `notebook`
     (ipykernel, matplotlib), installé avec `uv sync --inexact` pour garder research et app.
     Les négatifs d'une fenêtre sont tirés par `paired_negatives` (partie « même
     enregistrement » identique au benchmark, tirages au hasard possiblement différents).
     `detect_onsets` prend le lissage de l'enveloppe en paramètre (`smooth_s`, défaut 0,01 s
     inchangé). Sorties jamais committées : les lecteurs audio embarquent le son des
     enregistrements (`tests/exploration/test_notebooks.py`). Les trois notebooks tournent sur les
     données réelles (48 s, 3 min, 2 min).

106. **Mesure : « non annotée » ne veut pas dire « négative »** (25/09/2026, lecture seule,
     canal 0). 51 enregistrements positifs de 120 s : 3 fenêtres annotées en médiane (9 s sur
     120), aucune annotée négative. Les 345 positives sont toutes des détections BlanciNet
     validées ; 87 % des 1 695 tranches de 3 s non annotées de ces enregistrements sont aussi
     détectées (score médian 0,93). L'expert a validé une partie des détections, il n'a pas
     marqué tout le chant. `nearest` tire bien la fenêtre libre la plus proche d'une
     annotation, mais elle n'est pas écoutée : sur 1 020 négatifs `nearest`, 965 sont dans
     l'enregistrement positif et 80 % de ceux-là ont une détection BlanciNet ≥ 0,5 ; contraste
     en bande médian 6,1 dB (annotées 8,9 ; 7,6 à moins de 6 s d'une annotation, 5,4–5,9
     au-delà de 15 s). Les autres stratégies ne sont pas propres non plus : BlanciNet ≥ 0,5 sur
     46 % des négatifs `same_day` et 43 % des `other_day` (janvier à Mataroni : A. blanci
     chante aux mêmes heures d'un jour à l'autre). BlanciNet n'est pas la vérité (faux amis),
     l'écoute tranchera (`02_negatifs_apparies`, section 4). Options soumises à Léonard, non
     tranchées — le n° 87 a retiré toute règle liée à BlanciNet : (a) négatif présumé = ni
     annoté, ni détecté par BlanciNet au-dessus d'un seuil ; (b) jamais dans un enregistrement
     positif (« elle chante du début à la fin », notes de phénologie) ; (c) garder, et mesurer
     la contamination à l'écoute.

107. **Détection des notes trop stricte sur les données réelles** (mesure, réglage inchangé).
     Au réglage actuel (`onset_k_mad` 4, enveloppe lissée sur 10 ms), 0 à 14 notes par
     enregistrement positif de 2 min, 1 seule sur le plus annoté (36 fenêtres, chant tout du
     long) : le seuil médiane + 4 MAD (≈ +6 dB) morcelle les notes en éclats de 2 à 20 ms,
     rarement de 0,07–0,13 s. Sur 60 positives et 60 négatives (`03_module_sequentiel`),
     l'AUC du nombre de notes vaut 0,53 à ce réglage, 0,65–0,68 au mieux (k 1–2). Portes
     `notes` et `rhythm` : au seuil 1, 83 % des négatifs arrêtés mais 31 % des enregistrements
     positifs gardés — inutilisables en l'état ; `band_contrast` à 3 dB : 12 % arrêtés, 92 %
     gardés. Les descripteurs de rythme du module en parallèle sont presque toujours nuls.
     À reprendre avec Léonard (bande, durées, seuil) avant toute porte de rythme.

## 2026-09-25 (après-midi) — Régularisations des têtes

108. **Régularisations programmées, coupées par défaut** (tri de Léonard du 25/09 sur la liste
     R1–R84, `documentation/regularizations/regularizations.md`). `blanci/heads/regularization.py`, numéros conservés
     partout dans le code et la config. Une tête du benchmark les active dans son nom :
     `blanci heads --methods logistic,logistic+R18=16,logistic+R19` (« =v » remplace le réglage
     principal) ; le nom canonique (R triées) est celui des rapports et des scores hors-pli,
     `regularization.variants` en ajoute à la liste par défaut, `head.reference` les suit.
     Ordre : R19/R20 → R17 → R18 → R21 → tête. Programmées :
     - R13 (chaque micro pèse autant dans sa classe) et R15 (négatifs annotés × `hard_weight`,
       3 par défaut, face aux présumés — d'autant plus utile que les présumés sont contaminés,
       n° 106) : poids des fenêtres, de moyenne 1 par classe, `class_weight` inchangé ;
     - R17 (norme 1), R19 (− moyenne du micro), R20 (AdaBN : centrage et réduction par micro).
       R19/R20 lisent tout le stock du micro, partition par partition, sans labels
       (`store_domain_statistics`) : ce que la chaîne aura sur un nouveau site. Embedding par
       défaut seulement (pas les jetons résumés) ;
     - R18 (ACP) et R21 (retrait des directions du micro, sur les négatifs) ajustées dans
       chaque pli sur l'entraînement ; le C est ensuite choisi sur les fenêtres transformées.
       R21 `means` (défaut) retire les écarts entre moyennes des micros (≤ micros − 1
       directions, principe de LEACE) ; `inlp` itère un classifieur de micros, avec arrêt dès
       qu'il ne fait plus mieux que le hasard. Sur données simulées à micros très séparés, l'INLP
       retire presque toutes les dimensions, chant compris : d'où `means` par défaut ;
     - R27 (L1), R28 (Elastic Net) : `l1_ratio` de scikit-learn (`penalty` est déprécié depuis
       la 1.8 ; `penalty="elasticnet"` ajouté pour les versions antérieures), solveur saga ;
     - R22 : pooling `gem` des jetons (`logistic:gem`, p = 3 ; `gem2`, `gem5`… pour un autre p).
       Calculé sur x − min des jetons puis recentré (jetons non positifs) : p = 1 et p → ∞
       redonnent la moyenne et le maximum. p n'est pas appris : on compare quelques valeurs ;
     - R30 : tête `logistic_to_prototype`, perte ½‖w − w₀‖² + C·Σ perte logistique, w₀ = prototype
       différentiel dans l'espace standardisé, mis à l'échelle par une logistique à une variable
       (scipy L-BFGS). C petit : le prototype ; C grand : la logistique libre. Même grille de C ;
     - R31 : tête `lda_shrunk`, LDA à covariance rétrécie de Ledoit-Wolf (scikit-learn,
       `shrinkage="auto"`, qui rétrécit la matrice de corrélation), sans hyperparamètre.
     R30 et R31 rejoignent la courbe selon le nombre d'annotations (`head.curve.methods`). La
     courbe choisit désormais le C de chaque tête sur ses propres entrées (régularisations
     comprises), et non plus une fois pour toutes sur l'embedding par défaut.
     Combinaisons refusées : R19 + R20, R27 + R28, R21 + R19/R20 (n° 109), R19/R20 sur les
     jetons résumés, poids et pénalités hors des têtes logistiques. Hors du benchmark des têtes
     (`train`, `benchmark`, fusion, empilement), rien ne change : logistique L2 tant qu'aucune
     n'est retenue. Environnement : torch, bacpipe, streamlit, pytest et ruff manquaient le
     25/09 vers 11 h 40 (dates d'installation dans `.venv`) — signature d'un `uv sync` exact
     (le défaut de `uv sync`) lancé sans ces groupes ; `uv run`, lui, n'enlève jamais rien.
     Remis par `uv sync --inexact --group research --group app --group notebook`. Toujours
     `--inexact` et tous les groupes voulus.

109. **Mesure sur données simulées : la validation croisée sur un seul site récompense les
     raccourcis de micro** (`tests/heads/test_regularization.py`). Corpus : micros « riches » (50 %
     de positifs) et « pauvres » (5 %) séparés par un axe de fond commun ; 4 micros d'un
     « nouveau site » inversent la relation. AP sur le nouveau site, 4 tirages : sans
     régularisation 0,25–0,50 ; R21 0,50–0,79 ; R19 0,55–0,73 ; R20 0,51–0,69. En validation
     croisée sur les micros d'entraînement, où le raccourci reste vrai, l'ordre s'inverse :
     sans régularisation 0,68–0,82, R21 0,45–0,69, R19 0,50–0,62. Conséquences :
     (a) R19, R20 et R21 ne se jugent pas sur les plis groupés de Mataroni, mais sur un site
     tenu à l'écart (`blanci evaluate --holdout tresor`, `heads-curve --by site`) dès que Trésor
     ou Kaw auront des positifs ; d'ici là, un recul en validation croisée n'est pas un verdict ;
     (b) R19 + R21 : 0,29–0,41 partout — centrés par micro, les négatifs ne diffèrent plus que
     par le chant qui fuit dans la moyenne des micros riches, et R21 efface l'espèce : refusé ;
     (c) R19 retire aussi du chant là où A. blanci occupe beaucoup de fenêtres (la moyenne du
     micro en contient) : à surveiller sur les micros à chœur.

110. **R85, sonde à portes** (« attentive sur l'embedding », idée de Léonard ; tête `gated`,
     `blanci/heads/gated.py`). Score = w · (x̃ ⊙ g(x̃)) + b, porte g = σ(B·A·x̃ + c) de rang 8 : le
     poids de chaque dimension dépend de la fenêtre. Départ B = 0 → portes à ½, la tête part
     d'une logistique. AdamW (weight decay 1e-2, lr 0,05, 300 époques, lot entier), classes
     équilibrées ; torch (groupe research), écartée de la liste par défaut sans torch. Sur
     données simulées où le chant ne compte que selon le contexte : AP 0,84–0,87 sur un jeu
     d'essai contre 0,80 pour la logistique, mais AP ≈ 1 à l'entraînement et 0,58–0,81 en
     validation croisée pooled (logistique 0,73) : elle sur-apprend et l'échelle de ses scores
     varie d'un pli à l'autre. Ni réglée plus avant sur données jouets, ni arrêt précoce :
     à juger au benchmark réel ; R42 (arrêt précoce) serait le premier remède. Aucune
     régularisation par suffixe n'est refusée sauf les poids (R13/R15) et pénalités (R27/R28).

111. **Tableau des encodeurs de bacpipe** (`documentation/encodeurs-bacpipe.md`, demande de
     Léonard, R23) : pour les 8 encodeurs du projet et les 17 autres de bacpipe 1.3.5,
     framework, f_e, fenêtre, ce que bacpipe rend, accès aux jetons et aux couches
     intermédiaires. Relevé dans le code, sans exécuter les modèles (tailles de grille à
     mesurer). Bird-MAE : son `last_hidden_state` est la sortie *agrégée* (moyenne des
     patchs, jeton de classe exclu, puis `fc_norm`), malgré le nom — lu dans
     `modeling_bird_mae.py` ; une première version de ce numéro concluait à tort que bacpipe
     rendait ses jetons : le n° 77 était juste. Jetons par `output_hidden_states=True`
     (1 + 32 temps × 8 fréquences). BEATs et NatureBEATs : jetons par `avg_pooling = False`, sans hook ; BirdNET (Keras),
     ProtoCLR et ConvNeXt-BirdSet : faciles ; Perch v1 et v2 : couches intermédiaires
     difficiles (modèles exportés). Rien n'est encore branché dans l'adaptateur.

## 2026-09-25 (fin d'après-midi) — Pertes, échantillon versionné, poste d'annotation

112. **Benchmark des pertes** (R34, R35 ; `blanci/heads/losses.py`, têtes `loss:<nom>`,
     `blanci heads --methods losses`). Même tête linéaire (standardisation, classes
     équilibrées, L2, C par validation groupée), sept pertes contre la logistique : hinge et
     squared_hinge (`LinearSVC`), least_squares (`RidgeClassifier`, α = 1/2C), focal (γ = 2),
     gce (q = 0,7), sce (α = 0,1, β = 1, A = −4), sigmoid (1 − p, bornée). Les quatre
     dernières par L-BFGS, départ depuis la logistique de même C (gce, sce, sigmoid ne sont pas
     convexes). Gradients vérifiés par différences finies ; focal à γ = 0 = logistique,
     gce à q = 1 = sigmoid. Poids R13/R15 acceptés ; pénalités R27/R28 non (L2 seulement).
     Pourquoi les pertes robustes : les négatifs présumés sont contaminés (n° 106), un positif
     caché parmi eux pèse sans limite dans la logistique, au plus 1 dans sigmoid.

113. **Banc d'essai sur l'échantillon versionné** (`echantillon/`, 66 clips de Léonard ;
     `blanci/echantillon.py`, `blanci echantillon`). Fenêtre de l'encodeur centrée sur la
     fenêtre étiquetée, canal `audio.channel`, cache `data/echantillon/<encodeur>.npz` ; plis
     groupés par micro (5 : un par micro positif) ; R13/R19/R20/R21 groupés par **site** (1 à 3
     clips par micro) ; diagnostic « le site se lit-il dans l'embedding d'un micro jamais vu ».
     Poids de perch_v2 : `hf_hub_download` restait bloqué à 0 octet (transfert xet) ; archive
     récupérée par `curl --http1.1 -C -` depuis le même dépôt, extraite dans
     `data/models/bacpipe/perch_v2` (bacpipe la reconnaît ensuite).
     **Mesure perch_v2, 25/09** (10 positifs de 5 micros, tous à Mataroni ; 55 négatifs
     annotés, 3 sites ; aucun négatif présumé, donc plus facile que le vrai benchmark). La
     plupart des têtes font AP 0,86–0,96, intervalles de ~0,8 à 1 : **aucun classement**.
     Écarts appariés à la logistique (AP 0,957), bootstrap 2 000 :
     - R19 + R21 : −0,54 [−0,74 ; −0,22] — premier appui réel au mécanisme du n° 109 (b),
       exagéré ici (positifs = 10 des 48 clips de Mataroni, bien plus que dans le stock) ;
     - R20 : −0,23 [−0,49 ; −0,05] — écart-type estimé sur 3 à 14 clips par site : artefact
       de l'échantillon, R20 ne se juge pas ici ; R19 seul : −0,09 [−0,23 ; 0] ;
     - cascade : −0,45 [−0,69 ; −0,14] — ses scores sont remis à l'échelle dans chaque pli
       (plancher + 1 + …) ; l'AP poolée sur les plis la pénalise peut-être : à vérifier avant
       de conclure ;
     - `logistic:max` : −0,25 [−0,42 ; 0,00] (gem −0,23, non significatif) : sur les jetons
       16 × 4 de perch_v2, le maximum fait moins bien que la moyenne ;
     - `loss:hinge` : −0,10 [−0,25 ; −0,01] ;
     - R21 seul : scores changés (corrélation de rang 0,89), ordre positifs/négatifs identique :
       même AP.
     Site : exactitude 0,76 contre 0,69 au hasard (38 Mataroni, 14 CDR, 3 Patawa) : pas de
     conclusion à cet effectif. Ce qui est acquis : toute la chaîne (encodage, jetons, têtes,
     régularisations, pertes) tourne sur le vrai son.

114. **Poste d'annotation et R19 + R21** (demandes de Léonard). Streamlit reste : c'est le
     poste d'annotation (`blanci annotate`, `blanci/annotation/app.py`), pas l'ancien
     `app/streamlit_app.py` supprimé au n° 104. Les 4 échecs de `tests/annotation/test_app.py` venaient
     de Streamlit 1.64, qui résout un chemin relatif depuis le fichier de test : chemin absolu.
     Le refus de R19/R20 + R21 (n° 108–109) est levé : la combinaison se mesure au lieu d'être
     interdite ; le mécanisme du n° 109 (b) reste l'hypothèse à vérifier.

## 2026-09-26 — Tri de Léonard sur R33–R49 ; tableaux des benchmarks en images

115. **R39, voisins : k et pondération** (`head.nearest_similarity`, têtes `knn:k=…`,
     `exemplar:k=…`, suffixe `:w`). k > 1 : moyenne des cosinus aux k plus proches ; `:w` :
     moyenne pondérée par 1 / distance (distance euclidienne entre vecteurs de norme 1,
     `weights="distance"` de scikit-learn), entre k = 1 et la moyenne. Pour `knn`, les k
     voisins sont pris dans chaque classe séparément (un vote toutes classes confondues serait
     écrasé par les ~20 négatifs par positif). `knn:k=3` et `knn:k=5` rejoignent la liste par
     défaut de `blanci heads` ; `--methods neighbors` lance toutes les têtes par similarité
     (avec logistique et prototype) ; la courbe selon le nombre d'annotations ajoute
     `exemplar:k=3`, `knn`, `knn:k=3` (Léonard : ces baselines serviront à amorcer les
     nouveaux sites). Sur données simulées, avec six négatifs étiquetés positifs, k = 5 classe
     mieux que k = 1 (le voisinage d'un faux positif ne l'hérite plus). R19 se combine :
     `knn:k=5+R19`.

116. **R36 (poids des classes), R37 (biais par micro) ; R33 écartée, R38 en attente.**
     - R33 (norme maximale) : écartée, équivalente à la L2 pour une tête linéaire.
     - R36 : les têtes gardent `class_weight="balanced"` (Léonard : le déséquilibre doit être
       pris en compte). `+R36=power` remplace le poids n / (2·n_classe) de chaque classe par
       sa puissance `power` : 0 = aucun rééquilibrage (défaut de la variante, pour mesurer ce
       que l'équilibrage apporte), 0,5 = entre les deux. Passe par les poids des fenêtres :
       logistic, loss:<nom>, logistic_to_prototype, cascade ; le C est rechoisi avec.
     - R37 : un biais par **point** (site/micro, `regularization.by`), à la demande de
       Léonard (un biais par site ne suffirait pas). Un micro déplacé sur un autre site change
       de point, donc de biais : c'est ce qu'on veut pour l'ambiance du lieu. Colonnes
       indicatrices non standardisées, a priori N(0, σ²) sur chaque biais, σ = `scale` en
       logit, indépendant de C (colonne × σ/√C). Micro absent de l'entraînement : biais commun.
       logistic, loss:<nom>, cascade. **Mesure, données simulées du n° 109** (fond du micro
       qui prédit la présence, nouveau site où il ne tient plus), AP sur le nouveau site,
       3 tirages : sans R37 0,25–0,37 ; σ = 0,3 : 0,24–0,42 ; σ = 1 : 0,42–0,57 ; σ = 3 :
       0,56–0,68 ; σ = 10 : 0,61–0,71 ; R21 : 0,50–0,62. Contrairement à la liste du 25/09
       (« biais très pénalisé »), un biais fortement rétréci n'y change rien : w garde le
       raccourci. Défaut σ = 3, provisoire (n° 119).
     - R38 (écarts de w par micro) : non programmée. ~100 micros × 1 536 dimensions ; à
       reprendre après une ACP (R18), avec des effets croisés micro + site si l'on veut
       séparer le matériel (qui suit le micro) du lieu.

117. **Régularisations de l'attentive (R40–R42, R45–R47) et de la sonde à portes (R40, R42,
     R46)** (`attentive.fit_with_options`, `attentive.optimise`, partagé avec `gated`).
     Comportement par défaut inchangé (sorties identiques au bit près, vérifié). R40 : weight
     decay choisi sur `grid` par validation groupée interne (AP). R41 : AdamW, weight decay
     0,01. R45 : chaque jeton masqué avec la probabilité p avant l'attention (au moins un
     gardé). R46 : dropout de z (attentive) ou de x̃ ⊙ g (gated). R47 : w part de la
     logistique sur la moyenne des jetons (C choisi sur la même grille) et ½λ‖w − w₀‖²
     remplace le weight decay sur w. R42 : **mesure** — avec un seul pli de validation (1 à 2
     micros), la perte de validation est au plus bas à l'époque 6, puis à l'époque 1 en
     moyennant les plis, alors que l'AP de test progresse jusqu'à ~200 époques : la perte
     monte dès que la tête devient trop sûre d'elle, même quand son classement s'améliore.
     R42 relève donc la courbe du critère à chaque époque dans chaque pli groupé interne,
     retient l'époque qui optimise la courbe moyenne, puis réentraîne sur tout le pli
     d'entraînement ; critère `monitor: ap` par défaut (`loss` possible). AP hors-pli sur
     données simulées (note dans 1 jeton sur 16, 3 tirages) : sans régularisation 0,67 /
     0,74 / 0,79 ; R42 (loss) 0,54 / 0,65 / 0,78 ; R42 (ap) 0,66 / 0,74 / 0,79 ; R41, R45,
     R47 à ±0,03 de la tête seule ; R46 (p = 0,2) 0,63 / 0,69 / 0,74. Ces données ne font
     pas sur-apprendre la tête : à juger au benchmark réel. R43, R44, R48, R49 : expliquées à
     Léonard, non programmées.

118. **Tableaux des benchmarks en images** (`documentation/tableaux/`, demande de Léonard :
     les tableaux Markdown s'affichent mal dans Xcode). Un PNG par benchmark (protocole,
     encodeurs, têtes, poolings, pertes, voisins, régularisations, fusion, seuillage en
     amont, négatifs appariés, baselines, ensembles), ~600 Ko en tout, générés par
     `generer.py` (données + rendu Pillow, polices de matplotlib : rien à télécharger).
     Un nouveau benchmark = une entrée de `TABLEAUX`. `encodeurs-bacpipe.md` : tableaux
     remplacés par des fiches lisibles en texte brut. Le diagnostic de la cascade (n° 113)
     est abandonné (Léonard). Tests lancés depuis un conteneur en root :
     `test_freeze_writes_a_read_only_version_and_never_overwrites` y échoue (root écrit dans
     un fichier en lecture seule), sans rapport avec le code.

119. **Aucune conclusion sur les régularisations avant la base complète** (Léonard, 26/09). La
     base complète n'est pas accessible aujourd'hui. Les mesures des n° 109, 113, 116 et 117
     (données simulées, échantillon de 66 clips) vérifient que le code fait ce qu'il doit ;
     elles ne classent aucune régularisation. Les réglages par défaut qui en découlent sont
     provisoires et se choisiront sur la base : σ de R37 (comparer `logistic+R37=0.3`, `=1`,
     `=3`, `=10`) et le critère de R42, désormais réglage principal de son suffixe
     (`attentive+R42=ap` et `attentive+R42=loss` dans un même run, scores hors-pli
     distincts). Même prudence pour les avis théoriques donnés en discussion (R44 redondante
     avec le weight decay sur q, par exemple) : à vérifier, pas à appliquer.

120. **R50 : C de la fusion logistique par validation groupée ; index des régularisations.**
     Méthode de fusion `logistic+R50` (`fusion.choose_fusion_C`) : le C est choisi sur
     `fusion.C_grid` par l'AP moyenne sur des micros tenus à l'écart, dans chaque pli (sur les
     seuls micros d'entraînement du pli) et, pour le modèle de production, sur tout le jeu de
     développement ; le C retenu est enregistré avec le modèle. `logistic` garde son C fixé
     (`fusion.C`, 1) : `fusion-bench` compare les deux (`benchmark_methods`), la production
     reste `fusion.method: logistic` tant que rien n'est tranché (n° 119). Organisation
     (question de Léonard) : `blanci/heads/regularization.py` est l'index de toutes les
     régularisations programmées (tableau « où, comment l'activer ») et contient celles qui
     transforment les entrées, les poids ou la pénalité (R13–R21, R27, R28, R36, R37) ainsi
     que R40/R42, qui entourent l'entraînement des têtes torch (`fit_with_options`, déplacé
     depuis `attentive.py`, sorties identiques). Restent là où elles s'appliquent : les options
     de la boucle d'entraînement (R41, R45–R47 dans `attentive.py`, R46 dans `gated.py`), les
     têtes (R22 `pooling.py`, R30/R31/R39 `head.py`, R34/R35 `losses.py`, R85 `gated.py`) et
     la fusion (R50 `fusion.py`, R57 `stacking.py`). R51–R56 et R58 : expliquées à Léonard,
     en discussion.

121. **La mécanique des régularisations regroupée dans `blanci/heads/regularization.py` ; R59, R62,
     R63 ; tri de la section F** (demande de Léonard : ne pas se perdre dans un projet qui
     grandit). Le module contient désormais la mécanique de toutes les régularisations qui en
     ont une, et les autres modules l'appellent là où elle prend effet :
     - `grouped_search` : un réglage choisi par validation groupée, commun au C des têtes (R26,
       `head.select_C`), au weight decay des têtes torch (R40) et au C de la fusion (R50,
       `choose_fusion_C`, déplacée depuis `fusion.py`) — trois copies du même calcul en une ;
     - la boucle d'entraînement torch `optimise` (R41 ; R42 avec `validation_criterion` ; R59),
       `keep_mask` (R45), `dropout` (R46), `logistic_start` (R47), déplacées depuis
       `attentive.py` et `gated.py` ;
     - `nearest_similarity` (R39, depuis `head.py`), `group_bias_scale` (R37, σ/√C, auparavant
       écrit deux fois dans `head.py` et `losses.py`).
     Sorties identiques au bit près, vérifiées sur 12 têtes et variantes (logistique, R30,
     focal, R36, R37, knn:k=5:w, exemplar:k=3, gated+R40+R42, attentive seule et régularisée,
     fusion logistic+R50). Restent dans leur module, avec une ligne de l'index : les têtes qui
     *sont* la régularisation (R22 `pooling.gem`, R30, R31, R34/R35, R85), la L2 elle-même
     (R26, `fit_logistic`) et le hors-pli de la fusion (R57).
     Section F (tri du 26/09) : R59 programmée pour l'attentive et la sonde à portes
     (`+R59` : warm-up linéaire du pas sur `warmup` époques, norme du gradient écrêtée à
     `clip_norm`) — le reste de R59 (dropout, weight decay, arrêt précoce) l'était déjà ; R60 :
     le rang du LoRA est réglable dans la config ; R62 : `l2_sp_penalty`, utilisée dès
     aujourd'hui par R47 (même calcul, vers la logistique) et prête pour le fine-tuning ;
     R63 : `distillation_loss` (labels souples de l'enseignant, température), prête pour le
     détecteur distillé ; R65 accordée, à faire avec le modèle maison et le LoRA. R61, R64,
     R66 expliquées, en discussion ; R67 à explorer. Les emplacements réservés (`finetune.py`,
     `detectors/distilled.py`, `detectors/homemade.py`) nomment les fonctions à appeler.

122. **Sélection des réglages : R74, R75, R76, R77, R79** (tri de Léonard du 26/09, section H).
     Léonard : ne pas supposer 51 positifs pour toujours — d'autres annotations viendront, mais
     des sites resteront peu ou pas annotés ; tous les cas doivent tourner.
     - R75 (règle du « 1 écart-type ») **par défaut** (`head.selection_rule: one_se`, `best`
       pour l'ancien comportement) : parmi les valeurs dont l'AP moyenne des plis internes est à
       moins d'une erreur type de la meilleure, la plus régularisante (plus petit C ; plus grand
       weight decay). Vaut pour le C des têtes (R26), le weight decay (R40) et le C de la fusion
       (R50) : un seul calcul, `grouped_search`. Les C retenus changent donc par rapport aux
       benchmarks d'avant ce numéro.
     - R76 : `+R76` remplace la grille de C d'une tête par 13 valeurs log (facteur ~2,2 au lieu
       de 10) ; à réserver aux têtes rapides. `blanci train` écrit le chemin de régularisation
       (`chemin_C_<encodeur>_<version>.csv`, et `.png` si matplotlib est là).
     - R77 : `head.n_splits: lomo`, un pli par micro positif, les micros sans positif répartis
       entre ces plis ; vaut aussi pour les plis internes.
     - R74 : `blanci train` rapporte, à côté du rappel au seuil de précision plancher, le rappel
       et la précision d'un seuil choisi pli par pli sur les autres plis ; `blanci heads`
       rapporte l'AP par pli de la procédure « garder la meilleure variante » choisie sans voir
       le pli jugé, contre celle de la gagnante du tableau (R80 : ce que la gagnante doit à la
       chance). Le C, lui, était déjà choisi dans chaque pli sur les seuls micros
       d'entraînement : la note de la liste sur R74 était une inquiétude, pas un constat.
     - R79 : `+R79` (logistic, loss:<nom>, logistic_to_prototype), 20 têtes apprises sur des
       tirages bootstrap des **micros**, scores moyennés.

123. **Réseaux : R60, R61, R62, R63, R64** (section F). R64 : `+R64=ema` ou `+R64=swa` pour
     attentive, gated et dann (`WeightAverage`) ; la validation de R42 juge les poids moyens.
     R61 (`layerwise_lr_groups`, `unfreezing_schedule`, `unfreeze_top`) et R62 complet
     (`snapshot`, `l2_sp_model_penalty` : α vers les poids pré-entraînés, β vers 0 pour les
     poids nouveaux) : outils prêts et testés sur un petit réseau, réglages dans `finetune`,
     appelés par `finetune.py` quand il sera écrit. R60 : LoRA sur les couches hautes seulement
     (`finetune.lora.layers: 4`), spécifié dans `finetune.py`. R63 : la perte existe
     (`distillation_loss`) ; le détecteur distillé lui-même reste prévu après M4 (§13.7). R59 :
     l'écrêtage du gradient était déjà dans `optimise`, avec le warm-up.

124. **Nouvelles têtes : R66 (`dann`), R67 (`multiclass`), R81 (`+R81`).**
     - R66, `blanci/heads/dann.py` : h = tanh(A·x̃ + a), score = w·h + b ; un classifieur de micro
       branché sur h à travers une inversion du gradient (`grad_reverse`), dont la force monte
       de 0 à `strength` (`dann_strength`). Le micro n'est appris que sur les négatifs par
       défaut (`domain_on`), comme R21. R40, R42, R46, R59, R64 par suffixe. Données simulées du
       n° 109 (vérification du mécanisme, pas un verdict) : la part de micros reconnue dans h
       par une logistique neuve passe de 0,49–0,55 sans adversaire à 0,41–0,46 avec (hidden 8,
       600 époques) ; l'AP sur le nouveau site n'y dépasse pas celle de R21.
     - R67, tête `multiclass` (logistique multinomiale, classes équilibrées ; score = logit de
       P(A. blanci)). Choix laissés à Claude par Léonard : négatifs présumés → classe `fond`
       (aucun événement noté) ; classes de moins de 10 fenêtres (`R67.min_count`) → `autre`,
       puis `fond` si `autre` reste sous le seuil. Classes : `CLASS_OF_LABEL` (amphibien —
       congénères compris —, orthoptère, oiseau, pluie, fond, artefact, autre). Données
       simulées avec un faux ami : 0,83 d'AP contre 0,74 pour la logistique binaire.
     - R81, `+R81` : réservoir de 20 000 fenêtres non annotées tirées du stock (hors fenêtres
       du benchmark), des seuls micros d'entraînement par défaut (`pool_from: all` : aussi
       ceux du pli jugé, adaptation sans labels) ; pseudo-positifs = logit ≥ 3, au plus 1 % du
       réservoir, au poids 0,3 ; pas de pseudo-négatifs (même contamination que les négatifs
       présumés, n° 106).

125. **Tri des sections G et I ; R78 sur AnuraSet.** Écartées : R68 et R69 (Léonard compte se
     servir des fenêtres négatives prises entre des positives — les faux négatifs suspects du
     n° 102 — pour du « positive mining » ; lisser les scores effacerait justement ces
     contrastes), R71, R72 (les campagnes de 7 jours suffisent, d'après la phénologie de
     Courtois et al., à distinguer absence et présence non détectée), R82. R70 : le maximum,
     déjà le défaut de l'évaluation ; le travail porte sur les fenêtres. R80 : présélection des
     variantes plutôt que tout tester (et R74 pour mesurer la part de chance). R83 : à faire
     plus tard. R84 : en place (20 % aléatoire stratifié par micro et heure). R73 : en
     discussion (origine du plancher 0,1 et correction de la prévalence, voir la réponse du
     26/09). R78 : `blanci anuraset heads` (config `anuraset/anuraset.yaml`) — toutes les têtes
     sans jetons, un pli par site, réglages choisis par des plis internes eux aussi par site,
     régularisations groupées par site ; AP de chaque tête sur chaque site tenu à l'écart.
     Section E (R51–R56, R58) : en attente de Léonard.

## 2026-09-27 — Section E, R43, seuil du stock, site contre micro

126. **R43 et jugement par type de positif.** `attentive+R43=β` : entropie moyenne des poids
     d'attention ajoutée à la perte × β (`attention_entropy`) ; β > 0 pousse vers une attention
     piquée (une note brève, un ou deux jetons), β < 0 vers une attention diffuse (chœur).
     Léonard : comparer les approches sur les chœurs et sur les mâles seuls. `blanci heads`
     rapporte donc l'AP de chaque tête par label positif (`blanci_solo`, `blanci_chorus`,
     `blanci`), ses positifs de ce type contre tous les négatifs (`by_positive_type`). Les 345
     positifs importés sont tous « blanci » (H21, n° 3) : la comparaison vivra avec les
     prochaines annotations. R48 non programmée : dans l'attentive, query et weight n'ont que
     2 × d paramètres, rien à factoriser ; le rang faible sert déjà dans R85.

127. **Fusion : R52, R53, R56 ; R51, R54, R55, R58.** Variantes de la fusion logistique,
     combinables avec R50 : `+R52` (coefficients ≥ 0 sur les entrées orientées : pour classer,
     une combinaison convexe à l'échelle près), `+R53` (pénalité L1 : les entrées inutiles
     tombent à 0, exactement avec R52), `+R56` (s = w_tête·z_tête + cap·tanh(Σ autres / cap) :
     les descripteurs déplacent le logit d'au plus `fusion.R56.cap` = 2, aucun veto sur la
     tête). Mécanique : `regularization.fit_constrained_logistic`, `fusion_variant`.
     `fusion-bench` compare `logistic+R50`, `+R52`, `+R53`, `+R56`. R51 : remplacée par R52
     (Léonard) — R52 prend l'orientation de chaque entrée sur les données (signe de sa
     corrélation au label à l'entraînement), R51 l'aurait imposée ; sur un site encore peu
     annoté, un signe imposé est plus sûr qu'un signe appris : à garder en tête. R53 : Léonard
     fera un tri des descripteurs (redondances) ; la sélection se fait dans chaque pli. R54,
     R55 : réexpliquées, en discussion. R58 : écartée. `weighted` (poids fixés à la main,
     `fusion.weights`) reste la voie pour pondérer soi-même les entrées.

128. **R73 : le seuil pour la précision du stock.** La précision plancher (0,1) vient du §6 :
     la file hebdomadaire doit tenir en 1 h à 10 s par candidat, et le rappel prime (une fausse
     absence est irréversible, une fausse présence se corrige à l'écoute). Mais la précision
     dépend de la part de positifs : ~1/21 dans le benchmark, bien moins dans le stock. Démarche
     retenue, en chaîne : (1) `blanci prevalence` mesure la part de fenêtres positives sur les
     seules fenêtres écoutées au hasard (sources `random`, `audit`), avec son intervalle de
     Wilson ; (2) reportée dans `decision.prevalence`, `blanci train` choisit le seuil au plus
     grand rappel dont la précision attendue dans le stock, π·TPR / (π·TPR + (1 − π)·FPR),
     atteint le plancher (`threshold_at_prevalence`) — TPR et FPR ne dépendent pas de la part
     de positifs ; (3) `blanci train` écrit la courbe seuil → rappel, précision du benchmark,
     précision attendue dans le stock, part des fenêtres signalées (`seuils_<encodeur>_<v>`,
     CSV et PNG) et range la calibration de Platt (a, b) et la prévalence du benchmark avec
     la tête, pour des probabilités ramenées au stock (`prior_shift`). Sans prévalence, rien
     ne change. Hypothèse à garder en vue : les négatifs du benchmark (mêmes micros, mêmes
     heures que les positifs) représentent-ils ceux du stock ? La strate aléatoire le dira.

129. **Site contre micro.** Question de Léonard : la différence entre micros d'un même site
     n'est-elle pas déjà aussi forte qu'entre sites ? `blanci cluster --mode c0` rapporte
     désormais la décomposition de la variance des embeddings (normés) entre site, micro dans
     son site et intérieur d'un micro (`cluster.variance_partition`, ANOVA emboîtée, parts de
     somme 1), et l'AMI groupes/sites à côté de l'AMI groupes/micros. À lancer sur un
     échantillon qui couvre plusieurs sites.


130. **Fusion : R54 et R55, des descripteurs courbés.** Accord de Léonard (27/09) ; le tri
     des descripteurs viendra après. Les deux ne touchent que les descripteurs du module
     séquentiel (`regularization.is_descriptor`) : le score de la tête et les autres sources
     restent des droites. Chaque descripteur standardisé devient un bloc de colonnes
     (`DescriptorBasis`), la fusion reste une logistique pénalisée
     (`fit_constrained_logistic`, appelée par `fit_fusion_logistic`).
     - `logistic+R54` : `fusion.R54.bins` = 5 classes de même effectif, soit 4 marches aux
       quantiles de l'entraînement ; la hauteur de chaque marche est un coefficient ≥ 0 sur
       le descripteur orienté (signe de sa corrélation au label, comme R52) : la courbe ne
       fait que monter dans le sens « plus A. blanci ». Robuste aux descripteurs asymétriques
       (une longue queue ne tire plus la droite) ; incapable d'une forme en U, par
       construction.
     - `logistic+R55` : modèle additif (GAM) par P-splines (Eilers et Marx 1996, pas de
       dépendance : scipy suffit, mgcv est en R et pygam serait à télécharger). B-splines
       cubiques sur `fusion.R55.segments` = 8 intervalles égaux entre les quantiles 1 % et
       99 % ; au-delà la courbe reste plate : pas d'extrapolation sur un site aux valeurs
       jamais vues. Pénalité ½ βᵀ(G + `smoothness` · DᵀD)β : G = BᵀB / n mesure la taille de
       la courbe (pour une droite, la même L2 qu'une entrée linéaire : quand C baisse, tout
       rétrécit ensemble, rien ne s'échappe) ; DᵀD, différences secondes des coefficients,
       mesure la courbure (`smoothness` grand → une droite). `smoothness` = 1 est provisoire.
     C par validation groupée avec R50 (`logistic+R50+R54`, `+R50+R55`, ajoutées à
     `fusion.benchmark_methods`) ; « =v » remplace le réglage principal (`R54=3`, `R55=10`,
     et désormais `R56=1` pour le plafond, `FUSION_PARAMETER`). Combinables avec R52
     (entrées linéaires ≥ 0) et R56 (plafond sur la somme des courbes, centrées) ; R54 et R55
     s'excluent, R53 et R55 aussi (la L1 ne lisse pas une courbe). La « part » d'une entrée
     (`FusionModel.weights`) devient l'amplitude de sa courbe de −2 à +2 écarts-types. Plus
     de coefficients par descripteur (5 à 11 au lieu de 1) : c'est la pénalité, pas le nombre
     de colonnes, qui fixe ce que la fusion peut apprendre de ~10 positifs par degré de
     liberté (n° 94). Données simulées (un descripteur à seuil, un en U) : AP hors-pli 0,60
     (logistique), 0,60 (R54), 0,76 (R55) — ce que la mécanique sait faire, pas un verdict.
     Les ajustements contraints limitent BLAS à un fil (`threadpoolctl`, déjà là avec
     scikit-learn) : sur ces petites matrices, le multi-fil coûtait jusqu'à 100×.

131. **R37 en GLMM.** Accord de Léonard (27/09), pour le programme, sans usage immédiat. R37
     est déjà l'effet aléatoire « micro » d'un GLMM (biais b_micro ~ N(0, σ²), n° 116) ; il
     lui manquait ce qui fait un GLMM : σ estimé sur les données. `logistic+R37=glmm` (ou
     `R37.scale: glmm`) choisit σ parmi `R37.glmm_grid` en maximisant la vraisemblance
     marginale approchée (Laplace sur les biais, poids de la tête à leur valeur ajustée) :
     − perte − ½‖w‖²/C − Σ b̂²/2σ² − ½ log det(I + Σ H), H la courbure de la perte le long des
     biais (`glmm_evidence`, `glmm_scales`). σ grand laisse chaque micro coller à ses données
     mais se paie en log det ; σ petit coûte en ajustement si les micros diffèrent vraiment.
     Sur données simulées, σ vrai 0 → 0,3 retenu ; 0,5 → 1 ; 1,5 → 1 ; 3 → 3. Mais sur le
     corpus « raccourci » du n° 116 (le fond du micro prédit la présence, pas sur le nouveau
     site), le GLMM retient σ = 1 et fait AP 0,42–0,58 sur le nouveau site, contre 0,58–0,69
     pour σ = 3 et 0,21–0,37 sans R37 : le GLMM choisit le σ qui **décrit** le mieux les
     micros de l'entraînement, pas celui qui **protège** le mieux un site nouveau (des biais
     plus libres absorbent davantage le niveau du micro, w en garde moins). Les deux se
     comparent dans le benchmark (`logistic+R37=glmm` contre `=3`), jugés sur un site tenu à
     l'écart (n° 109). Niveau site en
     option (`R37.site_scale`, σ fixé ou « glmm ») : biais de site + biais de micro dans son
     site, emboîtés ; un micro nouveau d'un site connu hérite du biais de son site, un site
     nouveau du biais commun. Logistique et cascade seulement (pas `loss:`). Coût : un
     ajustement par valeur essayée (5 par défaut, jusqu'à 20 avec deux niveaux). Ce que le
     GLMM lisse vers la moyenne : les biais de la tête (le niveau de score propre à un micro,
     estimé sur ses fenêtres **annotées**), pas une probabilité de présence par point
     d'écoute — ce dernier usage (occupation, abondance) est celui de R72, écartée : le projet
     cherche des sites, une détection attestée suffit à déclencher la visite.

132. **Graphe de l'ACP : `blanci pca`.** Demande de Léonard : la part d'information perdue
     selon le nombre de dimensions gardées. Sur un échantillon du stock de l'encodeur
     (`cluster.c0_sample`, fenêtres arrêtées exclues), `cluster.pca_information_curve` donne,
     pour k = 1 … d, la part de variance des embeddings que perdent les k premières
     composantes : telles quelles (R18 seule) et après centrage par micro (R19 puis R18),
     chacune en % de sa propre variance totale, avec la part que le centrage retire (les
     différences entre micros). CSV et PNG `rapports/acp_<encodeur>` (abscisse logarithmique,
     repères aux k de R18 — 16, 32, 64, 128 — et du clustering, `cluster.pca_components`) ;
     la commande affiche combien de composantes gardent 80, 90, 95, 99 % et ce que perdent
     16, 32, 64, 128. Variance n'est pas information utile : la note occupe 2–3 % d'une
     fenêtre, une direction de faible variance peut porter le chant. La courbe dit combien de
     dimensions décrivent le paysage sonore ; ce qu'elles valent pour A. blanci se lit dans
     le benchmark (`logistic+R18=16`, `=32`, `=64`).

133. **L'AP poolée pénalise les réglages choisis pli par pli (mécanisme confirmé).** Le n° 113
     le soupçonnait pour la cascade. Sur données simulées, `logistic+R50` (fusion) : AP poolée
     0,48 contre 0,60 pour `logistic` (C fixé), mais AP moyenne par pli 0,59 contre 0,61.
     Cause : les plis qui retiennent un C petit rendent des logits resserrés (écart-type 0,6
     au lieu de 2,2) ; mis bout à bout, les scores de plis différents ne sont plus sur la
     même échelle et le classement commun se dégrade, sans que le classement dans chaque pli
     change. Même effet possible pour toute tête dont C (R26), σ (R37=glmm), les époques
     (R42) ou le nombre de composantes varient d'un pli à l'autre. À garder en tête en
     lisant `heads` et `fusion-bench` : un écart d'AP poolée entre une méthode à réglage fixe
     et une méthode à réglage choisi n'est pas forcément un écart de classement. Remèdes :
     l'AP moyenne par pli, et la recalibration de chaque pli avant la mise bout à bout, tous
     deux programmés au n° 135. Le seuil (R74, sur les scores hors-pli des autres plis) subit
     le même effet.

134. **`regularization.py` devient un paquet.** À ~2 100 lignes, découpé comme Léonard
     l'avait demandé (« si c'est trop long, des sous-modules ») : `blanci/heads/regularization/`
     avec `names` (noms des têtes, validation), `windows` (R13, R15, R17–R21, R36,
     indicatrices R37), `assembly` (`Context`, `Regularizer`, `regularizer_for`), `selection`
     (R26, R40, R74–R76, R79, R81), `torch_training` (R40–R47, R59, R61–R64, R66), `glmm`
     (R37), `fusion_logistic` (R50, R52–R56), `decision` (R73), `classes` (R67), `neighbors`
     (R39). Le code est déplacé tel quel (découpage par sections, imports entre sous-modules
     calculés, aucun cycle) ; `__init__.py` garde l'index et réexporte tout :
     `from blanci.heads.regularization import …` ne change nulle part. Suite de tests identique
     avant et après (793 réussis).

135. **AP moyenne par pli et recalibration par pli (remèdes du n° 133).** Accord de Léonard
     (28/09), sans conclusion avant les grands jeux (base complète, AnuraSet).
     (a) Toujours là : `ap_fold_mean`, l'AP de chaque pli (ses scores entre eux) moyennée
     sur les plis à deux classes (`evaluate.fold_mean_ap`, option `folds` de `evaluate`),
     dans `heads`, `fusion-bench`, `anuraset heads`, le benchmark des encodeurs et
     AnuraSet. Elle ne met jamais bout à bout deux modèles : elle ne voit que le classement.
     Plus bruitée (peu de positifs par pli) : elle se lit à côté de l'AP poolée. Un grand
     écart entre les deux signale des plis sur des échelles différentes.
     (b) Option `benchmark.fold_calibration: platt` (défaut `none`) : chaque pli choisit son C
     une fois, puis 3 plis internes (par micro, `calibration_splits`) donnent, **avec ce même
     C**, des scores hors-pli de son entraînement ; une calibration de Platt à classes
     équilibrées apprise dessus (`regularization.fold_platt`) ramène les scores du pli sur une
     échelle commune (la cote « A. blanci contre fond », indépendante de la part de positifs
     du pli) avant la mise bout à bout. Les réglages restent choisis pli par pli : seule
     l'échelle change, jamais le classement dans un pli (pente > 0). Une Platt apprise sur
     les scores d'entraînement du pli eux-mêmes ne marche pas ici : avec plus de dimensions
     que de fenêtres, l'entraînement est séparable et la pente part à l'infini. Non
     recalibrées : les têtes par similarité (rien n'y est choisi pli par pli) et les règles
     de fusion fixes. `ap_raw` garde l'AP poolée d'avant ; les scores rangés (stock hors-pli,
     entrées de la fusion) sont les recalibrés. Coût : 1 + 3 entraînements par pli.
     Mesuré sur données simulées : fusion `logistic+R50`, AP poolée 0,48 → 0,58 (par pli
     0,59) ; `logistic` à C fixé 0,601 → 0,601 ; `logistic+R50+R55` 0,756 → 0,761 ; têtes
     logistiques (C par pli, 3 tirages) 0,189 → 0,223, 0,175 → 0,173, 0,139 → 0,129 ; une
     première version qui recalibrait aussi le prototype le faisait perdre (0,243 → 0,218),
     d'où l'exclusion des têtes par similarité. Lecture : la recalibration répare un vrai
     écart d'échelle, et ne coûte, ailleurs, que le bruit de sa propre estimation. `blanci
     train` n'est pas touché (seuil sur les scores hors-pli bruts) : à reprendre si l'option
     est retenue.

136. **Campagne AnuraSet d'un seul tenant : `blanci anuraset campaign`.** Demande de Léonard
     (28/09) : choisir les espèces, encoder, benchmarker, essayer les régularisations. Dans
     cet environnement cloud, l'accès réseau est restreint : zenodo.org (les données
     AnuraSet), huggingface.co et kaggle.com (les poids des encodeurs) sont refusés par le
     proxy ; seuls PyPI et GitHub passent. Aucune donnée AnuraSet n'a donc pu être
     téléchargée ici, et aucun résultat réel n'existe encore. Tout est prêt pour qu'une seule
     commande fasse la campagne dès que les données sont là (cloud avec zenodo.org et
     huggingface.co autorisés, ou un poste bien connecté) :
         uv run blanci --config anuraset/anuraset.yaml anuraset campaign --encoders perch_v2
     Étapes, reprenables : extraction et inventaire (`prepare`) ; profil des espèces et
     fréquence dominante ; choix des espèces (`choose_species`, sauf `anuraset.species` ou
     `--species`) par niveaux : d'abord note ≤ 0,3 s en 3–6 kHz avec ≥ 300 chants (proche
     d'A. blanci), puis élargi (≤ 0,5 s, 2–7 kHz, ≥ 150 chants), puis multi-sites (≤ 1 s,
     ≥ 100 chants) ; toujours ≥ 2 sites, classées par nombre de sites puis de chants ;
     embeddings de chaque encodeur ; sondes du §3 par encodeur et espèce, encodeurs classés
     par l'AP moyenne de la logistique (`rank_encoders`) ; sur le meilleur, les têtes de
     `CAMPAIGN_HEADS`, par priorité : logistique (référence) ; prototype et knn:k=5
     (amorcer avec quelques exemples) ; R19, R20 (fond du site retiré par le stock non
     annoté : l'hypothèse de Léonard pour amorcer un site) ; R21 et dann (directions du site
     effacées) ; R37, R37=glmm, R19+R37 (biais par site) ; R13, loss:focal (déséquilibre) ;
     R18=64 (l'ACP jette-t-elle le chant ?) ; lda_shrunk. Toutes jugées un site à la fois ;
     classement par rang moyen sur les espèces (`rank_heads`), AP poolée et AP moyenne par
     site, écarts appariés à la logistique, AP de chaque site tenu à l'écart.
     Rapport `anuraset_campagne.md` (et les rapports détaillés). Répétition générale sur un
     AnuraSet simulé (4 sites × 6 enregistrements d'une minute, 3 espèces, fond propre à
     chaque site, encodeur factice ; 1 à 2 min) : toute la chaîne tourne, les 14 têtes
     comprises. Elle a trouvé un défaut : une espèce qui chante dans chaque enregistrement
     d'une minute rend l'AP par enregistrement indéfinie (aucun enregistrement négatif) ; le
     classement se fait donc au niveau fenêtre, et les écarts appariés à la logistique sont
     aussi calculés par fenêtre (`window_comparisons`, bootstrap par enregistrement). Sur ce
     jeu factice (trop facile : AP par site de 1 partout), l'AP poolée de la logistique
     tombait à 0,94 et celle de R37 à 0,81 alors que chaque site était parfaitement classé,
     et R19 la ramenait à 1 : l'effet d'échelle entre plis du n° 133, ici entre sites. Rien
     sur les vraies espèces.

137. **Campagne AnuraSet du 28/09 : premiers résultats réels (perch_v2).** Réseau ouvert par
     Léonard (zenodo.org, huggingface.co, kaggle.com). Données : `raw_data.zip` (7,2 Go,
     téléchargé par 12 plages en parallèle), 1 612 enregistrements d'une minute, 4 sites.
     Encodés : les 1 206 aux chants datés et les 393 sans aucune espèce (vrais négatifs,
     labels faibles) ; 13 écartés (espèce signalée sans chant daté). perch_v2 via bacpipe
     installé sans ses dépendances lourdes (ONNX, ni TensorFlow ni CUDA) ; fenêtres de 5 s
     jointives (`encoders.overlap: 0` dans `anuraset/anuraset.yaml`) : 19 080 fenêtres en 37 min
     (15,5 fenêtres/s, 4 cœurs).
     Espèces : les 5 présentes sur au moins 2 sites sont DENMIN, LEPLAT, PITAZU, BOAFAB,
     PHYCUV. Retenues : DENMIN (5,3 kHz, 3 sites, l'analogue le plus proche d'A. blanci),
     BOAFAB (2 sites équilibrés), PHYCUV (3 sites), PITAZU (note de 0,3 s ; 1 386 fenêtres
     positives à INCT17 contre 29 à INCT41 : un test d'amorçage). LEPLAT (1 270 contre 79)
     laissée pour une prochaine passe. Aucune n'est dans la bande 3–6 kHz avec une note
     ≤ 0,3 s : `choose_species` aurait pris PITAZU, DENMIN, PHYCUV.
     Protocole : un pli par site, 20 négatifs par positif et par site, C sur la grille
     complète dans chaque pli, 14 têtes (`CAMPAIGN_HEADS`), 4 espèces en parallèle (5 à
     15 min chacune). Rapport complet : `documentation/anuraset/`, tableau
     `documentation/tableaux/anuraset.png`.
     Ce que ça dit (indicateur, pas verdict ; A. blanci n'y est pas) :
     - la logistique domine les têtes par similarité d'un site à l'autre : prototype et
       knn:k=5 s'effondrent sur DENMIN (AP poolée 0,27 et 0,44 contre 0,91) et PHYCUV
       (0,68 et 0,57 contre 0,87) ; amorcer un site par similarité à quelques exemples
       d'ailleurs ne suffit pas ;
     - R37 (et R37=glmm, identique) : seule régularisation significativement meilleure, sur
       PITAZU (+0,045 d'AP poolée), l'espèce au site presque vide ; −0,01 à −0,02 ailleurs ;
     - R18=64 (ACP) : égale ailleurs, et la meilleure sur le site presque vide de PITAZU
       (AP 0,53 contre 0,28 à INCT41) ; l'ACP n'a pas jeté le chant ici ;
     - R19, R20 : meilleures site par site sur DENMIN et PITAZU (AP par site 0,69 et 0,67
       contre 0,67 ; 0,56 et 0,58 contre 0,52) mais l'AP poolée s'effondre (DENMIN 0,59 et
       0,31 contre 0,91). Mécanisme probable : le centrage par site retire une part du chant
       là où l'espèce est très abondante (DENMIN dans ~38 % des fenêtres d'INCT17). A. blanci
       est rare dans le stock : l'hypothèse de Léonard (R19/R20 pour amorcer un site) reste
       à juger sur les données ONF, où ce mécanisme devrait peser bien moins ;
     - R13, R21, DANN, lda_shrunk, loss:focal : aucun gain ; R13 perd (DENMIN −0,04,
       PITAZU −0,09) ;
     - BOAFAB est saturée (≈ 0,96 partout) ; DENMIN sur INCT20955 reste à ≈ 0,14 pour toutes
       les têtes : le cas dur, où rien ne généralise.
     Recalibration par pli (n° 135) sur 5 têtes : elle **dégrade** l'AP poolée (DENMIN :
     logistique 0,91 → 0,86, R37 0,90 → 0,60, R19 0,59 → 0,23). Les plis internes sont ici
     d'autres sites : la calibration apprise sur eux se transpose mal au site testé, et elle
     efface des écarts de niveau entre sites qui étaient justes (un site où l'espèce abonde
     mérite des scores plus hauts). Décision provisoire : `fold_calibration: none` reste le
     défaut ; l'AP par site (`ap_fold_mean`) est le complément utile.

## 2026-09-28 (soir) — Corrections de l'audit : AnuraSet, comparaisons, rappel

138. **AnuraSet : chants coupés aux jonctions, espèces signalées sans chant daté.** Audit du
     28/09 (sous-agents, vérifié dans le code et sur les données). Deux défauts de
     `window_labels` faussaient la campagne du n° 137 :
     - fenêtres jointives : un chant à cheval sur la jonction de deux fenêtres les rendait
       toutes deux NaN, il n'avait plus aucune fenêtre positive (biais vers les chants
       centrés). Désormais, un chant qu'aucune fenêtre ne contient entier rend positive celle
       qui en porte la plus grande part (la première à égalité), l'autre reste écartée. Grille
       jointive de 5 s sur tous les fichiers aux chants datés : fenêtres positives DENMIN
       1 813 → 2 001, PHYCUV 905 → 984, PITAZU 1 415 → 1 475, BOAFAB 1 814 → 1 857. Grille à
       50 % : rien ne change pour les chants de 2,5 s ou moins ;
     - labels faibles : un fichier gardé pour les chants datés d'une espèce, qui en signale une
       autre sans aucun chant daté d'elle, comptait comme négatif pour l'autre (positifs
       cachés). Fichiers gardés concernés : PITAZU 25, PHYCUV 20, LEPLAT 12, DENMIN 9,
       BOAFAB 6. Leurs fenêtres sont désormais écartées pour cette espèce-là, à l'évaluation
       (`weak_only_files`, `window_labels(unsure_files=…)`) ; la docstring de
       `campaign_recordings` le promettait sans le faire.
     `anuraset.weak_labels` donné mais absent : erreur explicite (avant : aucun vrai négatif,
     sans un mot). En-tête de `anuraset/anuraset.yaml` : le stock s'appelle
     `perch_v2-bacpipe1.3.5@o0`. Pas de réencodage : seuls les labels et l'évaluation
     changent. Les chiffres du n° 137 sont à refaire.

139. **Comparaisons appariées : micros tirés entiers, Holm, référence fixée d'avance.**
     - Bootstrap par micro : les enregistrements d'un même micro partagent fond et faune ; les
       tirer un à un rend l'intervalle trop étroit, et « A meilleur que B » trop fréquent,
       quand les micros sont peu nombreux. `evaluate(clusters=…)`, `compare_encoders`,
       `compare_to_reference`, `background_diagnostic` et `benchmark-all` tirent désormais des
       points (site/micro) sur les données ONF. AnuraSet garde l'enregistrement : 2 à 4 sites
       ne se tirent pas ; l'intervalle y reste optimiste, l'AP par site tenu à l'écart dit la
       variabilité.
     - Holm : `paired_bootstrap` rend une p-valeur bilatérale (2 × min(P(Δ ≤ 0), P(Δ ≥ 0))) ;
       `with_holm` ajoute `p_holm` et `significant_holm` à chaque tableau de comparaisons
       (encodeurs, têtes, benchmark complet ; AnuraSet : toutes les têtes et toutes les
       espèces d'un même niveau). 13 têtes × 4 espèces jugées à 5 % donnent 2 à 3 victoires
       par hasard : « R37 significativement meilleure sur PITAZU » (n° 137) se rejuge sur
       `significant_holm`. `significant` (chaque intervalle, seul) reste pour mémoire.
     - Référence du benchmark complet : `benchmark.reference` ou `--reference` ; à défaut la
       meilleure AP, signalée comme choisie après coup (`reference_post_hoc`) : elle doit une
       part de sa place à la chance. Le tableau ajoute `ap_fold_mean`.

140. **Rappel à précision plancher : seuil choisi sur les autres plis.** Le seuil choisi et
     jugé sur les mêmes scores est un oracle : il connaît les labels qu'il mesure. Avec des
     plis (`evaluate(folds=…)` ; benchmark complet par la colonne `fold` du stock hors-pli),
     `recall@p…` est désormais le rappel au seuil choisi pli par pli sur les autres plis
     (`evaluate.cross_fitted_recall`, le calcul de R74 déplacé depuis `regularization`) ;
     l'oracle reste en `recall@p…_oracle`, et `threshold@p…` est toujours le seuil qu'on
     déploierait (choisi sur tous les scores). Le rappel par site du benchmark complet suit
     le même seuil. Sans plis (baselines, sources externes), seul l'oracle se calcule : il
     reste en `recall@p…`. Les rappels publiés avant ce numéro sont optimistes.

141. **Campagne AnuraSet refaite après l'audit ; premier rapport de benchmark.** Branche
     `corrections-audit` fusionnée (n° 138–140), 803 tests réussis (le seul échec, le jeu gelé,
     tient à l'exécution en root). Les 9 enregistrements d'INCT20955 de 26 à 57 s, écartés par
     le drapeau `duration_off` (durée attendue 60 s, celle des Song Meter ONF), sont réintégrés
     sur décision de Léonard : la campagne n'écarte plus aucun drapeau QC sur AnuraSet
     (`select_recordings(exclude_flags=())`) ; 86 fenêtres de plus, 1 599 enregistrements,
     19 166 fenêtres. Relance sur le stock perch_v2 existant, 5 espèces (LEPLAT ajoutée),
     14 têtes, un pli par site. Rapport complet, figures et données :
     `documentation/benchmarks/2026-09-28_anuraset_perch_v2/` ; modèle réutilisable :
     `documentation/benchmarks/MODELE_RAPPORT.md`. Ce qui change par rapport au n° 137 :
     - positifs : DENMIN 1 806 → 2 002, PITAZU 1 415 → 1 475, PHYCUV 905 → 984 ; AP poolée
       moyenne sur les têtes +0,04 (PITAZU), +0,01 ailleurs ; haut et bas du classement
       inchangés ;
     - R37 (et R37=glmm) sur PITAZU : +0,05, survit à Holm, et meilleure sur ses deux sites
       tenus à l'écart ; neutre ailleurs ;
     - R19, R20 : l'effondrement de l'AP poolée (DENMIN, PITAZU, LEPLAT) suit la part de
       fenêtres positives du site (26 à 41 %) ; rien sur PHYCUV (2 à 12 %). Hypothèse : le
       centrage retire le chant là où l'espèce est partout ; à vérifier sur les données ONF,
       où A. blanci est rare ;
     - rappel à précision 0,5 au seuil choisi sur les autres sites (n° 140) : PITAZU 0,01
       contre 0,92 au seuil choisi sur place ; un nouveau site demandera son propre seuil ;
     - recalibration par pli : jamais meilleure (DENMIN logistique 0,91 → 0,86) ; défaut
       `none` confirmé.

142. **Import des annotations et des détections : cellules vides, verdicts nuancés, bornes.**
     - Décalage vide (NaN) : `parse_offset` le refuse (« décalage illisible », ligne par
       ligne). Avant, NaN passait le contrôle de bornes (toute comparaison à NaN est fausse),
       puis l'INSERT (`offset_s NOT NULL`) faisait échouer tout l'import.
     - Verdict nuancé (« blanci ? », « blanci sans doute » — qui veut dire *probablement* —,
       « blanci pas sûr », « peut-être ») : `parse_verdict` ne tranche plus (`VERDICT_DOUBT`).
       Avant, « blanci ? » devenait un positif ferme et « blanci sans doute » un négatif. La
       ligne est signalée comme tout verdict illisible, l'import attend. « probablement pas
       blanci » reste un négatif « uncertain ».
     - `import-detections` : même contrôle de bornes que les labels (`out_of_range`, compté),
       score à virgule décimale lu (« 0,87 »), score illisible compté au lieu d'arrêter tout
       l'import. Côté labels, un score illisible de l'ancien modèle est gardé en texte.

143. **Stocks d'embeddings : canal, checkpoint et transformations vérifiés à la reprise.** Le
     nom d'un stock ne dit ni le canal lu, ni le checkpoint bacpipe (`birdmae_base`), ni
     l'ordre du passe-bande ou le plancher du débruitage : reprendre `embed` avec un autre
     réglage ajoutait au même stock des embeddings qui ne se comparent pas aux siens, et
     `params_json`, écrasé, effaçait la trace du mélange. Désormais ces réglages sont rangés
     avec l'encodeur (`stock_identity`) et `embed` refuse de compléter un stock encodé
     autrement (`check_stock_identity`). Pas de suffixe de canal dans le nom : les stocks
     existants gardent le leur. L'ordre du passe-bande et le plancher du débruitage entrent
     dans l'étiquette quand ils ne sont pas à leur défaut (`bp3-7k-o6`, `dn1-f0.2`). Reste
     ouvert : la révision Hugging Face du checkpoint n'est pas figée.

144. **Inventaire et configuration.**
     - Copie laissée d'un relevé : après un changement de `paths.raw`, un fichier du relevé 1
       vu sur le disque du relevé 2 passait pour « déplacé », et prenait le chemin et le
       **site** du relevé 2. Un déplacement garde désormais son site ; sous un autre site,
       c'est un doublon, écarté et signalé.
     - Ordre naturel des dossiers (« RELEVE 2 » avant « RELEVE 10 ») : la règle « le premier
       inventorié l'emporte » suit l'ordre des relevés même quand un scan en couvre plusieurs.
     - Chemins relatifs de la configuration résolus depuis la racine du projet
       (`project_path`), plus depuis le dossier de lancement : lancée depuis D:\, une commande
       créait une base vide sur le disque externe.
     - Section vide dans `local.yaml` (`qc:` suivi de commentaires) : la section par défaut
       est gardée, au lieu d'un None qui cassait plus loin.

145. **Décisions et têtes.**
     - `score --site B` ne remplace plus que les décisions des enregistrements scorés : celles
       du site A, même tête et même seuil, restent (et le CSV des points les garde).
     - `queue`, `fusion` et `score` prennent la même tête par défaut : l'adoptée par
       `retrain`, sinon la plus récente. `queue` et `fusion` prenaient la plus récente et
       échouaient après un `retrain` non adopté.
     - Tête multi-classes (R67) : le logit de P(A. blanci) se calcule directement
       (z_blanci − logsumexp des autres) en float64. En float32, P arrondie à 1 dès un logit
       de ~17 donnait un score +inf, et l'AP refusait les scores.
     - Fusion de première version (`blanci fusion`, `score --fusion`) : un descripteur
       manquant vaut la moyenne de l'entraînement (0 une fois standardisé, comme
       `FusionModel`), plus 0 en unités brutes (`ioi_cv` = 0 : « rythme parfaitement
       régulier », un biais vers A. blanci). Une fusion enregistrée avant ce numéro est à
       réapprendre.

146. **Hygiène : dépendances de recherche, chemin ONNX testé, socle torch commun, tests.**
     - Groupe `research` : `torch` (importé directement par attentive, R85, R66,
       `torch_training`, l'export) et `onnx` (< 1.18 : les suivantes veulent ml-dtypes ≥ 0.5,
       TensorFlow 2.15 le fige en 0.3). Sans `onnx`, `torch.onnx.export` échouait : l'export
       du livrable ne pouvait pas tourner.
     - `tests/embedding/test_onnx_export.py` : export d'un petit réseau, équivalence torch/ONNX
       (cosinus > 0,999), axe de lot dynamique, rééchantillonnage, paquet corrompu ou
       manifeste incomplet refusés, quantification int8 (cosinus moyen > 0,99).
     - `regularization.torch_training` : `import_torch`, `feature_scaling`, `balanced_bce`,
       le socle que recopiaient attentive, R85 et R66 (mêmes calculs).
     - Tests : marqueur `slow` (suite rapide : `pytest -m "not slow"`, ~3 min 30 contre ~6 min),
       `-rs` liste les tests sautés faute de torch ou de streamlit, `quick_cfg` et `ToyEncoder`
       dans `conftest.py`. `test_attentive` : note de force 4 (écart ≥ 0,27 pour un seuil de
       0,1 sur quatre graines ; 0,13 à force 3).
     Laissés de côté : réécrire l'historique pour en retirer `pheno-blanci.pdf` (5,2 Mo)
     forcerait tout clone à repartir de zéro ; la révision Hugging Face du checkpoint (n° 143).

## 2026-09-29 — Benchmark 04 : birdmae_base sur AnuraSet

147. **birdmae_base (Bird-MAE-Base gelé) sur le protocole du benchmark 01 : indicateur.**
     Encodé dans le cloud, CPU seul : 1 599 enregistrements, 19 166 fenêtres de 5 s jointives,
     7,3–7,9 fenêtres/s (≈ 45 min) ; stock et base sur la branche
     `donnees-anuraset-birdmae_base`. Rapport :
     `documentation/benchmarks/2026-09-29_anuraset_birdmae_base/RAPPORT.md`.
     - Logistique, AP poolée : 0,31 à 0,70 sur 4 espèces sur 5 (perch_v2 : 0,74 à 0,91) ;
       BOAFAB 0,94. Poids vérifiés (152 tenseurs identiques au checkpoint).
     - `lda_shrunk` mène le rang moyen (+0,23 PHYCUV, +0,10 PITAZU, Holm), comme avec
       protoclr, alors qu'elle perdait 0,39 sur LEPLAT avec perch_v2 : le choix de la tête
       dépend de l'encodeur.
     - Amorçage d'un site peu annoté (seuil de précision 0,5 choisi sur les autres sites) :
       PITAZU/INCT41, AP 0,04, rappel 0 ; LEPLAT/INCT4, AP 0,25, rappel 0,71 à précision 0,13
       (perch_v2, refait à l'identique : 0,78 ; 0,84 à précision 0,40).
     - Pas de verdict sur Bird-MAE : une tête sur jetons (sondage par prototypes, celui que
       proposent ses auteurs) reste à essayer avant de l'écarter.
     - Environnement : transformers < 5 (la 5.17 ne charge pas le code distant du modèle) ;
       torch CPU ; une espèce par processus avec `OMP_NUM_THREADS=1` (sans limite, 5 processus
       saturaient 4 cœurs, charge 32). Le conteneur a redémarré deux fois pendant l'encodage :
       reprise sans perte (`embed_recordings` saute ce qui est fait).

## 2026-09-29 — Benchmarks 02, 03, 05 : protoclr, perch_bird, birdnet sur AnuraSet

148. **Encodeurs légers sur le protocole du benchmark 01 : indicateurs.** Encodés dans le
     cloud, CPU seul, l'un après l'autre, même sélection que le n° 141 (1 599 enregistrements,
     fenêtres jointives à la durée native de chaque modèle) ; stocks et base sur les branches
     `donnees-anuraset-<encodeur>`. Têtes : les 14 de `CAMPAIGN_HEADS`, 5 espèces, un pli par
     site, une espèce par processus (1 thread BLAS), Holm sur toutes les espèces. Rapports :
     `documentation/benchmarks/2026-09-29_anuraset_{protoclr,perch_bird,birdnet}/` ; outils
     communs (encodage, une espèce par processus, rassemblement, modèle de `generer.py`) :
     `anuraset/`.
     - Logistique, AP poolée (DENMIN / PITAZU / PHYCUV / LEPLAT / BOAFAB) : perch_v2 0,91 /
       0,74 / 0,89 / 0,79 / 0,97 (n° 141) ; perch_bird 0,93 / 0,76 / 0,84 / 0,75 / 0,98 ;
       birdnet 0,92 / 0,68 / 0,58 / 0,34 / 0,96 ; protoclr 0,57 / 0,24 / 0,14 / 0,33 / 0,85.
       Runs séparés (tirages de négatifs indépendants, grilles de 3, 5 et 6 s) : pas d'écart
       apparié entre encodeurs.
     - birdnet décroche sur les deux espèces aux chants les plus longs (PHYCUV, LEPLAT, p90
       3,6 et 2,7 s) : hypothèse de la fenêtre de 3 s, à tester (niveau enregistrement ou
       overlap 0,5). A. blanci, notes ≤ 0,3 s, ressemble plutôt à DENMIN et PITAZU, où birdnet
       égale perch_v2.
     - Têtes : avec les trois encodeurs forts, famille logistique en tête et R37 (ou
       R37=glmm) meilleure sur PITAZU (+0,025 à +0,034, Holm) ; R19, R20 : même chute de l'AP
       poolée là où l'espèce est partout. Avec protoclr (comme birdmae_base, n° 147),
       `lda_shrunk` mène le rang moyen : le classement des têtes dépend de l'encodeur.
     - Amorçage (seuil de précision 0,5 choisi sur les autres sites) : LEPLAT/INCT4 avec
       perch_bird, rappel 0,79 à précision 0,66 ; PITAZU/INCT41, échec partout (précision
       ≤ 0,18 pour la logistique). Le seuil voyage quand le site ressemble aux autres.
     - Environnement : `bacpipe` 1.3.5 sans dépendances, puis les modules manquants ; torch,
       torchvision, torchaudio en roues CPU (les roues CUDA remplissaient le disque) ;
       tensorflow 2.21 (roue CPU) pour birdnet et perch_bird. Durées d'encodage : protoclr
       25 min, perch_bird 88 min (3,7 fenêtres/s, TensorFlow), birdnet 23 min.

## 2026-09-29 — Benchmark 06 : birdmae_huge sur AnuraSet

149. **birdmae_huge (Bird-MAE-Huge gelé) sur le protocole du benchmark 01 : indicateur.**
     Nouvelle entrée `encoders.models.birdmae_huge` : le calcul de `birdmae` (bacpipe :
     extracteur de Base, réseau Huge ; embeddings vérifiés identiques), avec le checkpoint
     explicite pour que le stock porte un nom distinct de `birdmae_base`. Encodé dans le cloud,
     CPU seul : 1 599 enregistrements, 19 166 fenêtres, 1,1 fenêtre/s (4 h 50) ; stock et base
     sur la branche `donnees-anuraset-birdmae_huge`. Les 14 têtes de `CAMPAIGN_HEADS`, une
     espèce par processus. Rapport :
     `documentation/benchmarks/2026-09-29_anuraset_birdmae_huge/RAPPORT.md`.
     - Logistique, AP poolée : 0,33 à 0,63 sur les 4 espèces non saturées (perch_v2 : 0,74 à
       0,91), soit −0,29 à −0,42 ; l'AP par site perd autant. BOAFAB 0,93.
     - Huge ne fait pas mieux que Base (n° 147) : écarts de −0,07 à +0,08 selon l'espèce, sans
       tendance, pour un encodage 7 fois plus lent.
     - `lda_shrunk` mène encore le rang moyen (PHYCUV +0,19 sur la logistique, Holm). Les
       embeddings sont presque alignés (cosinus médian 0,997 avec leur moyenne, norme
       constante) : hypothèse, le chant tient dans des directions de faible variance, que la
       LDA décorrèle et que la logistique standardisée seule ne dégage pas. Test proposé :
       logistique sur embeddings blanchis.
     - R37 sur PITAZU : +0,07 (Holm), comme avec perch_v2 (+0,05).
     - Amorçage : PITAZU/INCT41, AP ≈ 0,05 pour toutes les têtes (le hasard) ; LEPLAT/INCT4,
       0,12 (perch_v2 0,78).
     - Pas de verdict sur Bird-MAE (comme au n° 147) : blanchiment et sonde sur les jetons
       restent à essayer. En attendant, perch_v2 reste la référence, et Base est préférable à
       Huge pour la même AP.

## 2026-09-29 — Benchmark 07 : six encodeurs côte à côte, amorçage d'un site

150. **Encodeurs comparés en apparié, et courbe d'amorçage sur AnuraSet : indicateur.** Les
     stocks des six encodeurs (branches `donnees-anuraset*`) sont réunis dans une même base. Têtes
     réapprises sur un protocole commun : un site tenu à l'écart à la fois, et **toutes** ses
     fenêtres jugées, à la prévalence réelle (les benchmarks 01 à 06 tiraient 20 négatifs par
     positif). Unité commune : la minute (max des fenêtres), seule identique pour des grilles de
     3, 5 et 6 s ; référence fixée d'avance, perch_v2 + logistique ; Holm. Rapport :
     `documentation/benchmarks/2026-09-29_anuraset_global/RAPPORT.md` ; outils :
     `anuraset/global_bench.py`, `rassembler_global.py` ; sorties brutes : branche
     `resultats-anuraset-07`.
     - AP moyenne par site (minute, moyenne des 5 espèces) : perch_v2 0,79, perch_bird 0,78
       (aucun écart significatif, Holm), birdnet 0,68 (égal sur DENMIN, PITAZU, BOAFAB ; −0,15
       sur PHYCUV, −0,35 sur LEPLAT, Holm), birdmae_base 0,60, birdmae_huge 0,56, protoclr
       0,43 (meilleure version de chacun).
     - **Encodeur retenu : perch_v2** (égal à perch_bird, 2,4 fois plus rapide, jetons
       disponibles). perch_bird et birdnet restent à départager sur les données ONF.
     - Amorçage (9 sites cibles, k enregistrements positifs du site ajoutés) : perch_v2 +
       logistique 0,81 (k = 0) → 0,85 (5) → 0,89 (10) → 0,91 (moitié du site). Le gain vient
       des sites où le transfert échoue (DENMIN/INCT20955 0,12 → 0,60 à 10). 1 ou 2
       enregistrements n'apportent presque rien (+0,01).
     - **Même tête dans les deux pipelines : la logistique.** R20 : +0,01 à k = 0 seulement,
       puis −0,02 à −0,03. Prototype : 0,73 à 0,76 à tout k (l'hypothèse du `notes.md`, prototype
       meilleur quand le site est peu annoté, ne tient pas ici). R37=glmm : égale à la
       logistique en AP par site, mais fait baisser l'AP poolée (BOAFAB 0,96 → 0,90) : à écarter
       d'une file d'annotation qui mélange des sites, tant que le biais d'un site neuf n'est pas
       réglé. La pipeline d'amorçage se distingue par sa stratégie d'annotation (au moins 5 à 10
       enregistrements positifs du site) et par son seuil, fixé sur place.
     - Seuil de précision 0,5 choisi sur les autres sites : tient sur 5 couples espèce × site
       sur 12. Aucune alerte pour les deux espèces présentes sur deux sites seulement (PITAZU,
       LEPLAT) : c'est le cas d'A. blanci tant que Mataroni sera seul annoté.
     - Encodeurs : 5 des 26 modèles de bacpipe testés (Bird-MAE en deux tailles). État de l'art et priorités dans
       `documentation/encodeurs-bacpipe.md` : naturebeats, birdnet_v3 (préversion, licence
       déployable) et esp-aves2 (ESP, ICLR 2026, hors bacpipe) d'abord. mix2 a été appris sur
       AnuraSet : il est exclu de ces benchmarks, mais reste candidat sur les données ONF.

## 2026-09-29 (après-midi) — Tête adaptée à chaque encodeur ; encodeurs à benchmarker

151. **Un encodeur se juge avec la tête qui correspond à sa sortie.** Demandé par Léonard :
     un test cassé fait écarter un bon modèle. Les transformers auto-supervisés (Bird-MAE, BEATs,
     NatureBEATs, EAT) ne donnent leur mesure qu'avec une tête sur leurs jetons (BEATs : 94,10 →
     97,98 AUROC sur BEANS en passant du linéaire à l'attentif, revue de Schwinger et al.).
     Règle, avec un tableau par encodeur (sortie, tête des auteurs, tête à essayer), dans
     `documentation/encodeurs-bacpipe.md` (« Règle ») ; rappel dans `MODELE_RAPPORT.md`.
     - Avant d'écarter un encodeur : sa tête d'origine ou son équivalent a été essayée ; on lit
       bien la sortie que ses auteurs évaluent (couche, agrégation, f_e, fenêtre ; n° 111) ; le
       témoin BOAFAB passe (AP moyenne par site, minute, ≥ 0,85 pour les six encodeurs du
       benchmark 07 : nettement en dessous, c'est d'abord le tuyau qu'on soupçonne).
     - Vocabulaire des rapports : « en retrait en sondage linéaire » tant que la tête adaptée
       n'a pas été essayée, « écarté » seulement ensuite. Bird-MAE : verdict suspendu (n° 147,
       149) ; le rapport 07 est corrigé en ce sens.
     - Prérequis : l'adaptateur n'extrait les jetons que de perch_v2. Pour les transformers,
       l'extraction des jetons précède leur benchmark. Les jetons restent sur la machine qui
       encode ; elle calcule les têtes sur jetons et ne pousse que les résultats.

     **Encodeurs à benchmarker** (décision de Léonard, `encodeurs-bacpipe.md`) : naturebeats,
     beats, convnext_birdset, birdnet_v3, rcl_fs_bsed, audioprotopnet, avesecho_passt,
     biolingual, insect66, insect459, esp-aves2, MetaPerch, et birdmae_large. Bird-MAE-Large et
     esp-aves2 (variantes `-all` comprises) : priorité 1 comme références de benchmark, pas
     comme livrables. Gros plan sur birdnet_v3. Entrées ajoutées à `encoders.models` pour les
     modèles de bacpipe (birdmae_large, birdnet_v3, audioprotopnet, avesecho_passt, biolingual,
     insect66, insect459, rcl_fs_bsed) ; esp-aves2 (bibliothèque AVEX) et MetaPerch demandent
     un adaptateur.
     - Vérifié le 29/09 : les dix points de contrôle esp-aves2 sont sous CC-BY-NC-SA-4.0,
       variantes `-all` comprises (non déployables). BirdNET+ V3.0 : CC BY-SA 4.0, conditions
       d'utilisation lues (commercial permis ; interdits : braconnage, usage militaire),
       déployable. Ses 11 560 classes comptent 647 amphibiens, dont 4 de nos 5 espèces AnuraSet
       (pas PITAZU) : son classifieur se jugera sans entraînement. A. blanci n'y est pas, mais
       son congénère *A. baeobatrachus* y est. Ses sources d'entraînement ne sont pas publiées :
       AnuraSet y est peut-être, et un score spectaculaire se revérifiera sur les données ONF.
     - MetaPerch : d'après l'article, la localisation et la date servent de pertes auxiliaires
       à l'entraînement seulement ; rien à fournir à l'inférence. Poids non vérifiés.

152. **Prérequis de la vague d'encodage 2 : jetons, têtes sur jetons, classifieur de BirdNET 3,
     esp-aves2.** Codés et testés sur de vrais enregistrements d'AnuraSet (plan des sessions :
     `anuraset/VAGUE_ENCODAGE_2.md`).
     - Jetons (`BacpipeEncoder.embed_tokens`) : Bird-MAE (dernière couche cachée, jeton de
       classe retiré, 32 temps × 8 fréquences), BEATs et NatureBEATs (jetons avant la moyenne de
       bacpipe, 31 × 8), AudioProtoPNet (carte de la dernière couche, 19 × 8, celle que lit sa
       tête à prototypes ; orientation vérifiée : entrée 256 bandes × 626 trames). Contrôle :
       cosinus embedding / moyenne des jetons 0,999–1 pour Bird-MAE et BEATs. `jetons.py`
       extrait tout un stock (Bird-MAE-Base : ~2,5 s par enregistrement, ≈ 1 Go en float16,
       moyenne sur la fréquence).
     - Têtes sur jetons (`global_bench.py --tokens`, transfert seul) : `attentive`,
       `logistic:max`, et `proto_probe`, nouvelle sonde à prototypes (`blanci/heads/proto_probe.py` :
       K = 8 prototypes appris, cosinus avec chaque jeton, maximum sur la fenêtre, couche
       linéaire ; version simplifiée du sondage par prototypes des auteurs de Bird-MAE).
       `simple_prototype` rejoint les têtes du transfert (tête d'origine de protoclr et de
       rcl_fs_bsed).
     - birdnet_v3 : les probabilités de son classifieur (`model.predictions`) sont rangées à
       l'encodage comme les logits de perch_v2 (`logit_classes` : DENMIN, LEPLAT, PHYCUV,
       BOAFAB sous « Hypsiboas faber », *A. baeobatrachus*) ; `global_bench.py --native` les
       juge sans entraînement. 32 kHz, fenêtres de 3 s, 1 280 dimensions (vérifié).
     - esp-aves2 : `blanci/embedding/encoders/avex_encoder.py` (`backend: avex`, dix entrées dans
       `encoders.models`), fenêtres de 5 s (grille de perch_v2). Embedding : l'agrégation d'AVEX
       (jeton de classe pour EAT, moyenne pour BEATs et EfficientNet) ; jetons en grille, sans
       les lignes de temps du complément à 10 s d'EAT (vérifié : elles ne varient plus d'une
       fenêtre à l'autre après la 32ᵉ).
     - MetaPerch : bloqué. Le dépôt `google-research/perch` n'a qu'un README (« will be
       provided here soon ») et Kaggle n'a rien ; la session 15 attend.
     - `importer_stock.py` ajoute le stock d'une branche de données à la base locale sans
       l'écraser. Tests : 790 réussis (le seul échec, le jeu gelé, tient à l'exécution en root).

## 2026-09-29 — Inventaire de la phénologie 2023 terminé

153. **Inventaire 2023 complet** (reprise du 23/09, `ingest --no-qc --no-hash` dossier par
     dossier, relevés dans l'ordre chronologique). « Projet Phénologie blanci », 6 relevés
     (déc. 2023 → nov. 2024), 3 sites : 66 779 enregistrements, dont 66 754 de 120 s (2 225 h,
     1 538 Go) ; avec 2026, la base compte 96 292 enregistrements, 3 204 h de 120 s, 2 216 Go
     (tableau dans le README). Aucune erreur à la reprise ; `blanci flag` relancé. Contrôles :
     - **fichiers illisibles** : 2 (pas 3), SMA14163 à Molokoi et SMA13417 à Trésor, datés du
       25/10/2023 (avant la pose) : 262 144 octets de zéros, fichiers vides de l'enregistreur,
       rien à récupérer ;
     - **horaires hors programme** (107 hors 5 h–20 h locales) : 5 sont des déclenchements
       de test de quelques secondes avant la pose (juillet–octobre 2023), déjà écartés pour
       durée anormale ; les 102 autres viennent de **Molokoi SMA14636 au relevé 3** (avril
       2024), dont les **1 520 fichiers** ont une heure GUANO en avance d'une heure sur le nom
       de fichier. Le nom (5 h–19 h 30) suit le programme, le GUANO (6 h–20 h 30) non : le
       fuseau de l'enregistreur était sans doute réglé sur UTC−4. Ailleurs, l'heure du nom et
       celle du GUANO concordent (sauf un fichier de test par enregistreur). **Non corrigé** :
       `start_utc` de ces 1 520 enregistrements est à décaler d'une heure (heure d'activité,
       négatifs appariés au créneau), sur accord de Léonard.

154. **Horloge douteuse : enregistrements écartés, en file d'écoute** (décision de Léonard,
     29/09/2026). Nouveau drapeau d'inventaire `clock_off`, qui écarte (`EXCLUDING_FLAGS`) :
     l'heure du nom de fichier (heure locale de l'enregistreur) et celle de l'en-tête GUANO
     diffèrent de plus de `qc.clock_tolerance_min` (5 min). Il lève 1 599 enregistrements :
     les 1 521 de Molokoi SMA14636 en avril 2024 (1 520 du relevé 3, un resté sur la carte du
     relevé 4 ; en-tête en avance d'une heure, n° 153) et 78 déclenchements de test de
     quelques secondes, déjà écartés pour leur durée (un par enregistreur en 2026, en-tête en
     UTC). Pas de correction de l'heure pour l'instant. Les 1 521 sont dans la file
     `data/reports/candidats_horloge_SMA14636.csv` (`candidates --drapeau clock_off`),
     enregistrements entiers, source de label « flag » — nouvelle source, pour ne jamais les
     mêler à l'audit aléatoire qui mesure le rappel. La file ne prend que les enregistrements
     écartés par ce seul drapeau. Un enregistrement où A. blanci est entendu n'est jamais
     écarté de l'encodage (`select_recordings`) : s'il y chante, il revient dans le corpus,
     avec son heure à corriger d'une heure (`start_utc` − 1 h) avant toute analyse par heure
     ou tout appariement au créneau.

## 2026-09-29 (soir) — Entretien avec Élodie : objectifs révisés, feuille de route V5

155. **Feuille de route V5** (entretien de Léonard avec Élodie, 29/09/2026).
     `documentation/feuille-de-route-V5.md` remplace la V4, archivée dans `documentation/old/`
     (la V5 a été fusionnée dans le cadre en tête de ce fichier le 01/10/2026, n° 162).
     Numéros de section conservés : les renvois des entrées précédentes restent valables.
     Nouveaux : §0 (objectifs, ordre des critères, définition de « libre ») et §14 (pour aller
     plus loin : article de phénologie, gabarit multi-espèces). Ordre des critères (Élodie) :
     licence libre (filtre), puis performance, puis facilité d'utilisation, puis durée
     d'encodage ; un naturaliste n'écrit pas plus d'une ou deux lignes de code, zéro visé.

156. **Encodeur libre d'accès impératif** (V5 §0, §2). Définition de travail, à confirmer par
     Élodie : poids publics **et** licence qui permet l'usage par l'ONF (établissement public à
     caractère industriel et commercial) ; clause non commerciale ou licence absente = non libre.
     Relevé du 29/09 (fiches Hugging Face) : AudioProtoPNet CC BY-NC 4.0 ; NatureLM-audio (source
     probable de naturebeats) CC BY-NC-SA 4.0 ; esp-aves2 CC BY-NC-SA 4.0 (n° 151) ; **aucune
     licence déclarée** pour Bird-MAE (Base, Large, Huge), ConvNeXt-BirdSet et BioLingual ;
     BEATs : dépôt `microsoft/unilm` sous MIT. Libres : perch_v2 (Apache 2.0), birdnet_v3
     (CC BY-SA 4.0), beats et perch_bird sous réserve. Après AnuraSet, **au plus deux non
     libres** restent pour le benchmark ONF, et seulement s'ils font mieux que le meilleur libre ;
     au 29/09, aucun ne le fait (perch_v2 en tête du benchmark 07). La vague 2 d'AnuraSet se
     clôt le **09/10/2026** ; ce qui n'est pas fini ce jour-là est abandonné.

157. **Annotations reprises de zéro** (V5 §5). Les 345 positifs et 150 négatifs Blancinet
     (n° 34) restent dans la base (ajout seul) mais sortent de l'entraînement et de
     l'évaluation. Jeu v1 tiré par plan, avant écoute, sans aucun détecteur : partition des
     points (un site 2026 entier, 20 % des points des autres sites, une station 2023 sur deux
     tenus à l'écart) ; lot d'entraînement 1 d'environ 450 extraits de 30 s (3 par point 2026,
     50 par station 2023 ; 60 % aux heures de pic) ; jeu d'évaluation d'environ 250
     enregistrements entiers, probabilités de tirage notées et métriques repondérées ; labels
     par intervalles, dont se déduisent les fenêtres de chaque grille. Avis sur les
     propositions d'Élodie (février–mars, pics journaliers) et questions pour Sylvain : V5 §5.8.
     À coder (V5 §13, liste v5) : `candidates --plan`, mode « extrait + intervalles »,
     filtre des sources `import`, `active` et `similarity`, pondération des métriques.

158. **Aucun benchmark sur les données ONF avant le go d'Élodie et Benoît** (V5 §5.7).
     Paquet de vérification à l'aveugle : tous les incertains, tous les positifs du jeu
     d'évaluation, 20 % de ses négatifs, 60 positifs et 100 négatifs d'entraînement, 30 éléments
     écoutés par les deux experts (≈ 2 h 30 chacun). Go si 0 erreur sur 60 positifs et au plus
     un chant manqué sur 100 négatifs (bornes unilatérales à 95 % : ≤ 5 % et ≤ 4,7 %) ; sinon
     réécoute de la strate fautive. Le go s'écrira ici, avec ses chiffres. Benchmark ONF borné
     ensuite : choix de l'encodeur le **20/11/2026**, puis plus aucun benchmark d'encodeur.

159. **Livrable : application Windows installable, sans code** (V5 §7). PyInstaller + Inno
     Setup, construite par GitHub Actions ; interface locale (poste Streamlit repris, Gradio en
     repli), sur le modèle de BirdNET-Analyzer ; étape intermédiaire et repli : `uv`, deux
     lignes PowerShell. Essai d'empaquetage du poste actuel en S4. La règle du §13.7 « ne pas
     toucher à la GUI avant M4 » est levée : l'application avance pendant que les experts
     vérifient. PAMGuard est écarté.

160. **Classification à trancher avant le lot 1** (demande de Léonard ; V5 §5.11). Options :
     une seule classe, trois niveaux de détection (qualité A/B/C), chœur contre solo,
     hiérarchie. Recommandation : **annoter fin, décider gros**. À l'écoute : sous-classe
     (solo, chœur, indécis) et qualité par intervalle positif. En sortie : une classe binaire.
     La qualité est un attribut (rappel par qualité), pas une classe. Les sous-classes deviennent
     des tâches auxiliaires (une régularisation, du même type que R67) seulement si chacune
     compte au moins 30 enregistrements indépendants, et elles ne sont adoptées que si l'AP
     binaire progresse sur les points tenus à l'écart. À valider avec Élodie, Benoît (H21 : solo
     et chœur s'entendent-ils ?) et Sylvain.

## 2026-09-30 — Benchmark 08 : 22 encodeurs sur AnuraSet, têtes adaptées

161. **Pré-benchmark AnuraSet des encodeurs : indicateur.** Protocole 07 (un site tenu à
     l'écart, toutes ses fenêtres jugées, minute commune), 23 encodeurs à cinq espèces
     complètes, chacun avec les têtes sur embedding et, pour ceux qui ont des jetons, les têtes
     sur jetons (n° 151, 152) ; perch_v2 réévalué sur ses jetons spatiaux (16 × 1 536) pour une
     comparaison équitable. Rapport : `documentation/benchmarks/2026-09-30_anuraset_encodeurs/` ;
     outils : `rassembler_08.py`, `apparie.py` (bootstrap apparié commun aux 07 et 08).
     - AP moyenne par site (minute, meilleure tête, choisie après coup) : naturebeats + sonde à
       prototypes 0,84 ; perch_v2 + sonde à prototypes 0,81 ; convnext_birdset + logistique
       0,81 (ajouté le 30/09 au soir) ; perch_v2 + logistique 0,79 ; aves2
       sl_beats_bio 0,79 ; perch_bird 0,78 ; Bird-MAE-Base + prototypes 0,74 ; birdnet_v3 0,74.
       Aucun autre encodeur ne bat perch_v2 + logistique après Holm (230 comparaisons).
     - **Meilleur libre : perch_v2 + sonde à prototypes.** Aucun non libre ne le bat sur une
       espèce après Holm (naturebeats : +0,14 sur LEPLAT, −0,10 sur PITAZU). Au sens strict du
       n° 156, aucun non libre ne gagne sa place au benchmark ONF ; naturebeats (sonde à
       prototypes) et convnext_birdset (logistique ; aucune licence déclarée, à demander au
       laboratoire DBD), au même niveau, sont proposés à Élodie comme les deux non libres
       possibles.
     - La sonde à prototypes bat la logistique sur les cinq espèces pour 7 des 10 transformers
       (Bird-MAE-Base 0,51 → 0,74, naturebeats 0,69 → 0,84) ; pas pour les EAT affinés sur
       étiquettes, à peine pour perch_v2 (+0,03, 2 espèces sur 5). Elle rejoint les têtes de
       référence. La sonde attentive n'aide presque jamais.
     - Site sans annotation (courbe, logistique) : convnext_birdset 0,83, perch_v2 0,81 (premier
       des libres), naturebeats 0,71, qui rejoint perch_v2 vers 20 enregistrements annotés.
     - birdnet_v3 : 0,74 (birdnet 2.4 : 0,69), sous perch_v2 ; son classifieur sans entraînement
       égale la logistique sur BOAFAB (0,99). Insectes (0,40–0,41), rcl_fs_bsed (0,47, témoin
       non passé) derniers ; EfficientNet AudioSet seul (0,50) sous la version bioacoustique
       (0,59) ; bio ≈ all pour esp-aves2 ; Bird-MAE Base ≥ Large ≥ Huge.
     - Manquent avant la clôture de la vague 2 (09/10, n° 156) : audioprotopnet (session 8),
       avesecho_passt, biolingual (session 9), relancées le 30/09 et à ajouter au benchmark 08
       quand elles auront fini ; MetaPerch, seulement annoncé (poids pas encore publiés au
       30/09).
     - Le 30/09, une fusion (branche `tmp-merge`) avait versé dans `main` les 340 sorties brutes
       de `resultats-anuraset-07` (≈ 89 Mo) : retirées de l'arbre (6f2ff13), mais encore dans
       l'historique ; règle ajoutée à `VAGUE_ENCODAGE_2.md`.

## 2026-10-01 — Rangement du dépôt pour les tuteurs

Le dépôt est public (`Rumble-Paw-Patrol/Grenouille`) et ses lecteurs sont les tuteurs : décisions
de Léonard, 01/10/2026.

162. **Feuille de route V5 fusionnée ici et supprimée.** Léonard ne s'en sert plus : elle servait
     à amorcer le projet avant le début du stage. Ce qui reste utile (objectifs et critères,
     données, protocole d'annotation et d'évaluation, livrable, planning, risques, questions et
     hypothèses ouvertes, règles d'implémentation) forme le **cadre** en tête de ce fichier, avec
     les numéros de section conservés. Abandonnés : journaux v2 → v5, montée en compétence,
     hypothèses déjà levées, texte des anciennes versions de §2 ; la bibliographie citée est dans
     `documentation/biblio/biblio.md` (§6). Version complète : `git show
     ad43369:documentation/feuille-de-route-V5.md`. Les feuilles V1 à V4 restent dans
     `documentation/old/`.

163. **Échantillon versionné retiré** (n° 113). Les 66 clips d'`echantillon/` (60 Mo de FLAC tirés
     des enregistrements de l'ONF et de Biophonia) sortent de l'arbre de travail, avec
     `blanci/echantillon.py`, la commande `blanci echantillon` et `tests/test_echantillon.py` :
     les enregistrements ne sont pas à publier, et Léonard dispose maintenant du jeu complet. Ils
     restent dans l'historique git (dernier commit qui les contient : `ad43369`) ; les
     3 extraits WAV de `documentation/prez/presentation-suivi-2/audio/` en viennent
     (`generer_figures.py` explique comment les restaurer). Les mesures du n° 113 et suivants qui
     s'appuyaient sur l'échantillon restent écrites ici comme résultats passés, sans commande
     pour les reproduire. `pheno-blanci.pdf` (rapport de Courtois et al., 2025) sort aussi du
     dépôt : la référence complète, avec lien, est dans `documentation/biblio/biblio.md` ainsi que
     celles de l'UICN et du plan national d'actions.

164. **Réorganisation des dossiers.**

     | Avant | Après |
     |---|---|
     | `documentation/feuille-de-route-V5.md` | fusionnée dans le cadre de ce fichier (n° 162) |
     | `documentation/regularizations/tableaux/` | `documentation/tableaux/` (tableaux de tous les benchmarks) |
     | `documentation/regularizations/regularisation.md` | `documentation/regularizations/regularizations.md` |
     | `documentation/benchmarks/outils_anuraset/` | `scripts/anuraset/` (du code, pas de la documentation), puis `anuraset/` (n° 166) |
     | `LISEZMOI.md` (racine) | supprimé : reste d'une branche de résultats, chemins disparus |
     | `documentation/Offre de stage_VF.pdf` | supprimé : doublon de `documentation/biblio/` |
     | `documentation/prez/Presentation_suivi_2.key` | supprimé : le `.pptx` fait foi |
     | `documentation/notes.md` | sorti du dépôt : brouillon personnel, reste en local |
     | README (commandes) | `documentation/commandes.md` ; README refait pour un lecteur extérieur |

     Branches distantes `claude/benchmark-regularisations-pertes-jt41ye`, `corrections-audit` et
     `resultats-anuraset-07` supprimées (déjà fusionnées dans `main`). Les chemins cités dans les
     entrées plus haut qui ont changé sont ceux du tableau ci-dessus.

165. **Branches `donnees-anuraset-*`** : conservées hors de `main`. Elles portent les embeddings et
     la base AnuraSet (6 branches, 15 à 70 Mo chacune, ≈ 280 Mo non compressés en tout) que
     `anuraset/importer_stock.py` lit ; ce sont des données publiques (AnuraSet, CC BY).
     Hébergement à trancher (Git LFS, Zenodo ou pièce jointe de version) ; en attendant elles
     n'encombrent pas l'arbre de `main`. Le pack git local pèse 580 Mo parce que l'historique
     contient aussi les anciennes sorties brutes (n° 161), l'échantillon audio, les
     présentations et les PDF.

166. **`blanci/` et `tests/` rangés par étape de la chaîne** (question de Léonard : 69 modules par
     ordre alphabétique, illisibles pour un encadrant). Un sous-paquet par étape, dans l'ordre où
     les données la traversent : `core` (config, db, audio), `inputs` (ingest, qc, labels,
     dataset, frozen), `embedding` (grid, encoders/, embed, store, index), `heads` (têtes,
     pooling, losses, dann, sequential, baselines, cluster, finetune, detectors/,
     regularization/), `combination` (stacking, fusion, ensemble), `evaluation` (evaluate, oof,
     benchmarks, throughput, qc_calibration, anuraset), `annotation` (selection, active,
     workbench, app), `results` (aggregate, activity), `exploration` (modules des notebooks) ;
     `cli` et `service` restent à la racine. Le `__init__.py` de chaque sous-paquet donne le rôle
     de chacun de ses modules ; `tests/` suit la même arborescence. Noms anglais comme le code et
     les sous-paquets existants ; `inputs` plutôt que `data` (ignoré par `.gitignore`, confondu
     avec `data/`), `combination` et `exploration` plutôt que `fusion` et `explore` (noms de
     modules qu'ils contiennent). Les imports `blanci.<module>` deviennent
     `blanci.<étape>.<module>`, partout (code, tests, notebooks, documentation, journal).
     AnuraSet, terminé, est isolé : `scripts/anuraset/` et `config/anuraset.yaml` réunis dans
     `anuraset/` à la racine (gardés pour reproduire les rapports 02 à 08), commandes regroupées
     sous `blanci anuraset prepare|profile|benchmark|heads|campaign` (hors de l'aide principale).
     Au passage : les `generer.py` des rapports cherchaient `documentation/benchmarks/tableaux/`,
     déplacé en `documentation/tableaux/` (n° 164).
