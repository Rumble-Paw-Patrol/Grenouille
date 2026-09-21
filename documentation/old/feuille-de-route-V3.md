# Feuille de route — Stage ONF Guyane
## Détection acoustique automatique d'*Anomaloglossus blanci* — 15/09/2026 → 14/03/2027

Version 2 (17/09/2026). `[HYPOTHÈSE Hn]` = information manquante, liste en §12 ; `[À VÉRIFIER]` = hors bibliographie fournie et non vérifié ; « établi » = bibliographie fournie ou source vérifiée (liste en fin) ; « jugement » = arbitrage d'ingénieur ; « tuteur » = confirmé en réunion.

**Journal v1 → v2**
- Données réelles intégrées : jeux 2023 et 2026, cycle 2 min / 30 min, plage 5 h–19 h 30, 345 fenêtres positives et 158 négatifs durs annotés (§1, §2, §5, §6).
- Chant continu d'*A. blanci* : l'enregistrement de 2 min devient l'unité de décision principale, la persistance le premier discriminant, la détection isolée un signal suspect (§1, §3, §6).
- Amorçage few-shot rétrogradé en baseline ; probing linéaire dès S3 (§3, §8).
- Solo contre chœur ajouté au module séquentiel (§3, §5).
- Saisonnalité et heure : usage encadré pour éviter d'apprendre la phénologie au lieu de l'acoustique (§3, §6).
- Classes de qualité du prestataire : piste à évaluer ; SNR estimé comme stratification par défaut (§5, §6).
- Stacking reformulé (§3) ; planning tranché (§8) ; document raccourci.

**Positions des encadrants** `[HYPOTHÈSE H4]` (verbatim non fournis)
- A : contre BirdNET (TensorFlow, base de code BirdNET-Analyzer) ; pour BEATs/NatureLM-audio et Bird-MAE via HuggingFace ; Streamlit ou Gradio pour la réannotation.
- B : Perch 2.0 via bacpipe, probing linéaire, apprentissage actif ; objection des 16 kHz contre NatureLM.
- Arbitrage : aucun TensorFlow à l'exécution dans le livrable ; encodeur choisi par benchmark sous filtre de licence et de déployabilité ; 16 kHz tranché empiriquement (§2).

---

## 1. Reformulation du problème

**Cadre (tuteur)**
- Objectif : détecteur robuste d'*A. blanci*, fonctionnant sur de nouveaux sites.
- Métriques : average precision, rappel.
- Contraintes : six mois, un portable, logiciel prenable en main par un naturaliste.
- Critères de comparaison : score, vitesse de calcul, prise en main.

**Données connues**
- Jeu 2023 : 6 micros, 2 par site (Trésor, Kaw, Molokoï), un an, 3 min toutes les 30 min.
- Jeu 2026 : Trésor et Kaw, 10 à 20 micros chacun, durée `[HYPOTHÈSE H16]` ; Mataroni (site neuf), > 70 micros, 5 jours ; 2 min toutes les 30 min.
- Plage horaire : 5 h → 19 h 30, soit 29 créneaux par jour ; 87 min/jour/micro en 2023, 58 en 2026.
- Cycle utile : 10 % en 2023, 6,7 % en 2026 (4 min/h, et non 2 comme en v0).
- Arithmétique : 6 micros × 29 × 365 ≈ 63 000 fichiers de 3 min ≈ 3 200 h si les micros ont tourné toute l'année ; Mataroni ≈ 10 000 fichiers ≈ 340 h. À concilier avec « des milliers d'enregistrements » `[HYPOTHÈSE H16]` : sous-ensemble extrait, ou micros non continus.
- Annotations : 345 fenêtres de chant validées par un expert ; 158 fenêtres de negative mining sur espèces proches. Durée des fenêtres, nombre d'enregistrements et de sites d'origine `[HYPOTHÈSE H2]`.
- Correspondance avec l'offre : jeu 2023 = entraînement, jeu 2026 = généralisation `[HYPOTHÈSE H3]`.

**Phénologie (tuteur)**
- *A. blanci* chante sans s'arrêter : présente, elle chante du début à la fin des 2 min. Une détection isolée est donc probablement un faux positif, sauf chant ponctuel avéré `[HYPOTHÈSE H20]`.
- Forte saisonnalité ; fort effet de l'heure.
- Un individu seul et un chœur ne produisent pas la même signature `[HYPOTHÈSE H21]`.

**Problème d'apprentissage**
- Détection d'une espèce à note brève (0,09–0,10 s, 4,5–5,4 kHz) mais à chant continu, dans une bande saturée d'autres espèces, avec 345 fenêtres positives issues vraisemblablement de peu d'enregistrements et de peu de sites.
- Posé proprement : classement de fenêtres sous fort déséquilibre, puis décision par enregistrement, puis par point ; vérification humaine structurelle ; rappel prioritaire.
- Effectif effectif : si une fenêtre positive en implique 47 autres dans le même enregistrement, 345 fenêtres ≈ 10 enregistrements. C'est le nombre d'enregistrements et de sites distincts qui gouverne la généralisation, pas 345.

**Unité de décision (jugement)**

| Niveau | Rôle | Règle |
|---|---|---|
| Fenêtre (3–5 s, pas ≤ 2,5 s) | Apprentissage, score, vérification | Métriques de développement |
| Enregistrement (2 min) | **Unité de décision principale** | Score = fraction de fenêtres au-dessus du seuil ; positif si ≥ 50 % (jugement) ; 1–2 fenêtres isolées = file « suspect », traitée à part |
| Point × période | Décision gestionnaire `[HYPOTHÈSE H5]` (règle du stagiaire) | « À vérifier » dès qu'un enregistrement est positif ; points classés par force (fraction, nombre d'enregistrements, jours distincts) ; « présence confirmée » après validation humaine ; sinon « non détecté », jamais « absent » |

**Écart entre métrique et décision**
- Asymétrie : fausse absence irréversible, fausse présence réversible → rappel.
- Chant continu : un seul créneau pendant l'activité suffit à détecter ; la contrainte n'est plus le cycle utile mais la **saison** : un site doit être échantillonné en période de chant. Le déploiement long couvre cette contrainte.
- La règle par point ne borne plus la charge : c'est la précision plancher du §6 et le classement qui la bornent.

**Échelle : note de 0,09 s contre fenêtre de 3–5 s**
- Pour un détecteur amont : évite la dilution dans l'agrégation ; mesure durée, fréquence, intervalles.
- Contre : bande saturée (tuteur), le détecteur déclenche en continu ; encodeurs et revue (établi : Schwinger et al. 2026) évalués sur fenêtres pleines ; chant continu → la fenêtre pleine contient le rythme, et l'enregistrement la persistance. L'attentive probing (tête d'attention sur les tokens pris avant agrégation, même couche que l'embedding) répond à la parcimonie sans détecteur amont (établi : nécessaire pour les transformers).
- Verdict provisoire : fenêtres pleines ; le seuillage spectral ne sert qu'à mesurer les onsets.
- Protocole (S3–S5, plis par site) : A grille pas 2,5 s ; B pas 1 s ; C fenêtres centrées sur onsets ; E agrégation des tokens moyenne / max / attention. C n'est adopté que s'il dépasse A/B de plus que l'incertitude.

---

## 2. Benchmark initial des embeddings (S1–S3)

- Colonnes : licence ; exécution sur M4 (native / ONNX / déportée / impossible) ; vitesse (temps pour 1 h d'audio) ; prise en main ; tokens accessibles ; AP ; rappel à précision 0,5 et 0,1 ; kNN top-1.
- Pré-benchmark AnuraSet (S2, établi : Cañas et al. 2023) : 93 000 fenêtres, 42 anoures, 4 sites, CC0. Deux ou trois espèces à note brève en 3–6 kHz, plis par site. Donne le classement des encodeurs sur des anoures en paysage sonore néotropical avec une puissance statistique que les données ONF n'ont pas. Indicateur, pas garantie (biomes, enregistreurs, aucun *Anomaloglossus*).
- Jeu annoté v0 : les 345 positifs et 158 négatifs durs, plus des négatifs **appariés** (mêmes micros, mêmes heures, mêmes mois, sans chant), ratio 1:20 à 1:50. Contrôle : un *Anomaloglossus* dans les classes de Perch 2.0 `[HYPOTHÈSE H8]` ?
- Extraction bacpipe : `birdnet`, `perch_v2`, `birdmae`, `beats`, `naturebeats`, `protoclr` (+ `perch_bird`, `convnext_birdset` si sans effort). Relevé : débit, mémoire, dimension, tokens.
- Sondes : kNN cosinus (k ∈ {1, 3, 5}), prototype différentiel, régression logistique L2. Validation **groupée par site** (jeu 2023 : 3 sites → 3 plis ; par micro : 6) ; si les positifs viennent d'un seul site, leave-one-recording-out et généralisation annoncée non mesurée. Comparaisons appariées sur les mêmes plis, bootstrap par enregistrement.
- Métriques trompeuses : exactitude ; AUROC ; F1 au seuil 0,5 ; AMI/ARI. Moins de 0,1 d'AP d'écart = égalité, départagée par licence, vitesse, prise en main.
- `perch_v2` (TensorFlow, GPU requis, établi) : (1) wrapper CPU `[À VÉRIFIER]` ; (2) export ONNX — un `perch_v2_no_dft.onnx` circule dans un notebook de BirdCLEF+ 2026 (vérifié le 16/09/2026), à valider contre des embeddings de référence ; (3) extraction déportée pour le benchmark seul ; (4) variante CPU officielle si elle sort. `naturebeats` : encodeur seul, jamais le LLM `[À VÉRIFIER]`.
- Désaccords → mesures : AP(birdnet) contre les autres ; AP(beats) à 16 kHz contre AP(birdmae) à 32 kHz **avec contrôle** birdmae sur audio filtré à 8 kHz ; AP(perch_v2) sur AnuraSet puis ONF.
- BirdCLEF+ 2026 (vérifié) : Pantanal, multi-taxons, 234 espèces, ROC-AUC macro sur 5 s, CPU 90 min, fin le 03/06/2026 ; working notes après CLEF (21–24/09), fils Kaggle déjà lisibles. Transférable : pseudo-étiquetage, lissage temporel, focal → paysage sonore, calibration. Non transférable : ensembles lourds, entraînement massif.
- Sortie : deux encodeurs au plus pour P1.

---

## 3. Cartographie des approches

| Approche | Principe | Annotations | Robustesse au décalage | M4 | Verdict |
|---|---|---|---|---|---|
| Template matching | Corrélation croisée de spectrogrammes (scikit-maad `[À VÉRIFIER]`) | 1–10 | Faible | Triviale | Baseline obligatoire ; conserve la note, indépendant de l'encodeur |
| Seuillage spectral 4,4–5,5 kHz | Passe-bande, enveloppe, seuil, durée 0,08–0,11 s | 0 | Faible (bande saturée) | Triviale | Onsets pour le module séquentiel ; pas de filtre amont |
| Indices acoustiques | Statistiques par enregistrement | 0 | — | Triviale | Contrôle qualité seulement |
| Prototype simple | Cosinus au centroïde des positifs | 5–30 | Sensible au fond partagé | Oui | Baseline de similarité (`index`) |
| Prototype différentiel | $\langle x, \mu_+ - \mu_-\rangle + b$, négatifs appariés | 10–50 | Retire le fond partagé | Oui | Baseline permanente ; forme fermée de la régression logistique |
| Linear probing | Régression logistique L2 sur embeddings gelés | Dizaines → centaines | Selon l'encodeur ; Perch 2.0 couvre des amphibiens (établi) | Oui (Perch conditionnel) | **Approche principale dès S3** (345 positifs) |
| Attentive probing | Tête d'attention sur tokens pré-agrégation | ≥ 150–200 fenêtres, effectif effectif à vérifier | Idem | Oui | P2 si tokens accessibles (établi : nécessaire pour les transformers) |
| Few-shot contrastif | Prototypes dans l'espace d'un encodeur contrastif (ProtoCLR, établi) | Dizaines | Conçu pour focal → paysage | Oui | Second candidat |
| Clustering | HDBSCAN sur ACP ; UMAP pour voir | 0 | — | Oui | §5 bis ; détecteur seulement si C1 réussit |
| LoRA / fine-tuning | Adaptation partielle ou totale de l'encodeur | Centaines à milliers ; AnuraSet | Sur-apprentissage au site | Partiel | Conditionné aux effectifs ; hors chemin critique |
| Distillation (modèle maison) | Petit CNN bande 3–7 kHz imitant le pipeline gelé sur les fenêtres non annotées | 0 | Hérite de l'enseignant | Oui | Livrable léger possible ; S13 au plus tôt ; hors chemin critique |

**Prototype différentiel**
- $x \approx c_{\text{site}} + c_{\text{espèce}} + \varepsilon$ ; $\mu_+$ garde le fond partagé, $\mu_-$ apparié aussi ; $w = \mu_+ - \mu_- \approx c_{\text{espèce}}$.
- Centroïde le plus proche sous variance commune = analyse discriminante linéaire à covariance identité = solution fermée de la régression logistique. Aucun hyperparamètre.
- Condition : négatifs **appariés** (mêmes micros, heures, mois). Non appariés → $w$ pointe vers un artefact de site.
- Rôle en v2 : baseline et diagnostic (l'écart entre $\cos(x,\mu_+)$ et $\langle x, w\rangle$ mesure le fond capté). Première instance de l'arithmétique d'embedding (tuteur, long terme, §11).

**Module séquentiel** (descripteurs calculés depuis l'audio, hors encodeur)
- Rythme intra-fenêtre : onsets en bande ; IOI ; fraction d'IOI < 0,1 s (trains des congénères) ; régularité. IOI d'*A. blanci* `[HYPOTHÈSE H9]`.
- **Persistance** : fraction de fenêtres au-dessus du seuil dans l'enregistrement ; continuité entre fenêtres voisines ; présence dans les créneaux voisins du même micro. Sépare le chant continu du chant ponctuel (fourmilier tacheté `[À VÉRIFIER : nom scientifique]`) et des détections isolées.
- **Solo contre chœur** `[HYPOTHÈSE H21]` : densité d'onsets, proportion de notes chevauchées, étalement spectral. Deux sous-classes de positifs, parce qu'un chœur brouille les IOI et change le SNR.
- Heure et saison : **pas dans le classifieur par défaut**. Risque d'apprendre la phénologie de Trésor et de la plaquer sur Mataroni. Usage : échantillonnage, classement des points, drapeau de plausibilité. Entrée du classifieur seulement si validée sur Mataroni.

**Tête de fusion (stacking)**
- Niveau 1 : `head` (régression sur embedding) et `sequential` (descripteurs). Niveau 2 : régression logistique sur (score de `head` **hors-pli**, 2–4 descripteurs).
- Hors-pli : chaque exemple est scoré par une version de `head` entraînée sans lui (validation croisée) ; sinon la fusion apprend sur des scores mémorisés. Les 5 versions ne sont pas « empilées » : elles fabriquent une colonne ; à l'inférence, `head` réentraînée sur tout ou moyenne des 5.
- Effectif : ≈ 10 positifs indépendants par coefficient (règle des événements par variable `[À VÉRIFIER]`) ; compter en enregistrements, pas en fenêtres.
- Le score séquentiel module ; il ne met jamais de veto sur un score d'embedding élevé.

---

## 4. Architecture

```
audio brut (format, f_e : H1)
  └─ ingest      inventaire, métadonnées (site, micro, horodatage, jeu 2023/2026) → SQLite
  └─ decode      forme d'onde float32 mono, f_e native (soundfile ; ffmpeg si exotique)
  └─ grid        fenêtres (recording_id, offset_s, dur_s), indépendantes de l'encodeur
  └─ encoder[k]  rééchantillonnage vers f_e(k) ; spectrogramme interne au modèle ; E_k ∈ R^{N×d_k}
  └─ store       Parquet partitionné encoder_id / site / mois, float16 (embeddings seulement, audio intact)
  └─ index       cosinus, exhaustif par fragments ; requêtes positives et négatives empilées ; FAISS optionnel
  └─ head[v]     régression logistique → score par fenêtre
  └─ sequential  rythme, persistance, solo/chœur → fusion avec head
  └─ aggregate   fenêtre → enregistrement (fraction) → point (classement)
  └─ queue       file de vérification → labels append-only → ré-entraînement de head
```

- `decode` s'arrête à la forme d'onde ; le log-mel est calculé dans chaque encodeur avec ses paramètres. Le spectrogramme n'apparaît que dans l'interface et dans les approches DSP.
- Découplage : labels attachés à (enregistrement, décalage), jamais aux embeddings → survivent au changement d'encodeur. Encodeur derrière un `Protocol` (`embed`, `embed_tokens`, `sample_rate`, `window_s`, `dim`), choisi par configuration. GUI → couche de service, jamais les modèles.
- Volume : jeu 2023 ≈ 3 200 h si continu → ≈ 4,5 M fenêtres à pas 2,5 s ≈ 7 Go en 768-d float16 ; ≈ 14 Go en 1536-d. Si « des milliers d'enregistrements » est exact, dix fois moins. SSD externe dans les deux cas.
- Index : requêtes (positifs + négatifs étiquetés) en matrice $n \times d$ multipliée par la base $d \times N$, une passe ; secondes sur MPS, lecture disque dominante. Ajout seul, jamais de suppression.
- Changement d'encodeur : ré-encodage en tâche de fond, tête réentraînée sur les mêmes labels, rapport de migration sur le jeu gelé avant bascule ; chaque décision porte (`encoder_id`, `head_version`, `threshold_id`).
- Croissance : encodeur figé ; index en ajout ; tête réentraînée de zéro sur tous les labels (réentraînement périodique, pas apprentissage continu au sens technique) ; seuils recalibrés.
- Inspiration YAPAT : embeddings dans la base, recherche = requête ; `sqlite-vec` ou Parquet ici, sans serveur.

---

## 5. Annotation et apprentissage actif

**Point de départ** : 345 fenêtres positives, 158 négatifs durs (espèces proches), classes de qualité éventuelles `[HYPOTHÈSE H19]`, instants de détection de l'étude de phénologie s'ils sont récupérables `[HYPOTHÈSE H7]`. Première tâche : documenter leur origine (enregistrements, micros, sites, durée de fenêtre, annotateur).

**Annotation par enregistrement (chant continu).** L'expert décide sur 20 s d'écoute si l'enregistrement de 2 min contient *A. blanci* ; toutes ses fenêtres héritent du label. Cinquante fois moins cher que la fenêtre. Bruit toléré aux bords ; à valider sur 20 enregistrements écoutés en entier `[HYPOTHÈSE H20]`.

**Schéma de labels** : {blanci-solo, blanci-chœur, congénère, fourmilier, autre déclencheur, orthoptère, fond, incertain}. Qualité : SNR estimé automatiquement par défaut ; classes du prestataire évaluées comme piste si ses annotations existent.

**Apprentissage actif dès S3**, en parallèle de tout :
- File = 60 % incertains, 20 % scores maximaux, 20 % aléatoire stratifié (micro, heure, mois). La strate aléatoire ne sert pas à entraîner : elle mesure les faux négatifs confiants qu'une file d'incertains ne remonte jamais.
- Lots de 100–150 fenêtres ou 30–50 enregistrements ; sélection par grappes contre les quasi-doublons d'un même enregistrement.
- Arrêt : gain d'AP sur le jeu gelé < 0,02 sur deux tours (jugement).

**Negative mining ciblé (tuteur)**
- Congénères (*A. degranvillei*, *dewynteri*, *surinamensis*, *baeobatrachus*, *saramaka*) : motifs et heures différents. Fourmilier tacheté : chant ponctuel. Autres déclencheurs `[HYPOTHÈSE H18]`. Les 158 négatifs existants sont le point de départ, complétés par des enregistrements de référence s'ils existent `[HYPOTHÈSE H14]` ; labels par espèce conservés pour évaluation hiérarchique.
- Négatifs durs : vérification des fenêtres les mieux classées qui se révèlent négatives. Négatifs faciles en volume : clustering (§5 bis, C2). Les deux, pour des raisons différentes.

**Échantillonnage phénologique** : priorité aux heures et mois de chant `[HYPOTHÈSE H15]`, stratification par micro ; 20 % hors strates.

**Établi d'annotation** : YAPAT (établi : Kath et al. 2024) essayé en S2, une demi-journée ; conditions : BirdNET contournable, mise en place ≤ 1 j ; sinon Streamlit minimal ou Whombat (établi). Outils de travail, pas livrable.

**Budget** : 250–350 fenêtres/h ou 80–120 enregistrements/h. Stagiaire ≈ 66 h ; expert `[HYPOTHÈSE H12]` : 3 h de calibration (accord inter-annotateurs mesuré) puis 1 h toutes les deux semaines. Un label `blanci` n'entre en entraînement qu'après validation experte.

**Après déploiement** : file hebdomadaire ≈ 150 fenêtres ou 40 enregistrements ; réentraînement local de la tête, comparaison affichée sur le jeu gelé embarqué, nouvelle version seulement si l'AP ne baisse pas ; labels immuables ; encodeur changé uniquement par migration. Démonstration réelle seulement si un rapatriement a lieu avant février `[HYPOTHÈSE H13]`.

---

## 5 bis. Voie non supervisée : clustering

- Deux fonctions : négatifs en volume à bas coût ; exploration. Statut de détecteur tranché par C1.
- Méthode (jugement) : HDBSCAN sur ACP (≈ 50 composantes), pas sur UMAP (densités déformées) ; UMAP pour visualiser.
- C0 (S3, 1 h) : clustering global ; AMI entre groupes et micros/sites ; sert à repérer anomalies (micro dérivant, saturation, espèce inattendue).
- C1 (S3, 1 j) : site riche en positifs, positifs connus + 5 000 fenêtres aux mêmes heures ; fenêtres pleines contre fenêtres recadrées. Seuils (jugement) : rappel du meilleur groupe ≥ 0,5 ; enrichissement ≥ 20 ; AMI avec le micro faible. Réussite → détecteur candidat au §6 ; échec → outil d'annotation seulement.
- C2 (S4–S7) : étiquetage en bloc des groupes homogènes après dix écoutes ; négatifs faciles.
- C3 (S6–S9) : sub-clustering par micro sur les heures de pic ; récolte de positifs sur micros sans présence connue.
- Limite : 2 % de fenêtre pour la note → les groupes sont des paysages sonores, pas des espèces ; le chant continu améliore le rapport (plusieurs notes par fenêtre), le recadrage encore plus.

---

## 6. Évaluation

- Métriques (tuteur) : AP et rappel. Ajouts : rappel à précision ≥ 0,1 et ≥ 0,5 ; fausses alarmes par heure ; **AP et rappel par enregistrement** (unité principale) ; accord du classement des points avec l'expert. Rejetées : exactitude, AUROC (annexe), F1 au seuil 0,5, kappa.
- Stratification : par SNR estimé (énergie en bande pendant les notes contre 0,5 s voisines), par solo/chœur, par classe de qualité si disponible. Un rappel global de 0,9 peut cacher 0,3 sur les chants faibles, ceux des sites à faible densité.
- Séparation : plis par site sur le jeu 2023 (Trésor, Kaw, Molokoï ; par micro si besoin) ; **jeu 2026 réservé**, et Mataroni en test de site neuf `[HYPOTHÈSE H3]`. Aucun label 2026 en entraînement avant P4, sauf audit aléatoire.
- Jeu gelé (S6–S9) : ≥ 60 enregistrements écoutés en entier sur tous les micros 2023 disponibles, tous positifs connus de micros tenus à l'écart, SNR et solo/chœur renseignés. Versionné, jamais entraîné.

| Critère (tuteur) | Mesure | Comment |
|---|---|---|
| Score | AP, rappel par enregistrement et par SNR | Plis par site, bootstrap par enregistrement |
| Vitesse | Temps pour 1 h d'audio sur le M4 (CPU, puis MPS) | Même fichier, encodage + tête |
| Prise en main | Installation, dépendances, taille, documentation | Grille à 5 points, stagiaire puis naturaliste en P4 |

- Rappel prioritaire (site manqué irréversible) ; précision plancher : à 10 s par candidat et précision p, un vrai positif coûte 10/p s ; file hebdomadaire ≤ 1 h (jugement). Cible : rappel ≥ 0,9 au seuil le plus élevé gardant précision ≥ 0,1 sur sites tenus à l'écart. L'agrégation par enregistrement (fraction de fenêtres) devrait relever fortement la précision ; à mesurer.
- Calibration : seuils sur scores hors-pli ; par jeu si décalage confirmé ; isotonique ou Platt seulement au-delà de ≈ 200 positifs indépendants.
- Intervalles : Wilson sur le rappel, comptés en **enregistrements** (n = 10 et rappel 0,9 → [0,60 ; 0,98] ; n = 30 → [0,74 ; 0,97]) ; bootstrap par enregistrement pour l'AP ; comparaisons appariées ; « A meilleur que B » seulement si l'intervalle apparié exclut zéro.
- Benchmark contre l'existant `[HYPOTHÈSE H10]` : template matching ; détecteur en bande ; BirdNET + sonde (référence non déployable) ; méthode du prestataire si récupérable `[HYPOTHÈSE H19]` ; pipeline retenu. Phase 4 : temps de détection, deux naturalistes, avec et sans l'outil ; indicatif.
- Combinaisons (décision du stagiaire, conditionnées) : concaténation d'embeddings sous une régression et fusion séquentielle d'abord ; soft voting, pondéré, stacking d'encodeurs, sélection gloutonne seulement si le jeu gelé compte ≥ 200 positifs indépendants ; chaque encodeur ajouté double l'inférence et ajoute 200–400 Mo.
- Boucle : courbe d'apprentissage (AP sur jeu gelé contre labels cumulés) et ablation à budget égal contre tirage aléatoire.

---

## 7. Outil livrable

| Option | Taille | Coût | Remarques |
|---|---|---|---|
| A. Bundle Python + ONNX Runtime + GUI | 300–600 Mo | 2–3 sem. + une construction par OS | Hors ligne ; ni TensorFlow ni PyTorch à l'exécution |
| B. Idem avec PyTorch CPU | +300–800 Mo | 2–3 sem. | Conversion évitée, plus lourd |
| C. PAMGuard comme hôte | Logiciel existant + modèle ONNX + configuration | 1–2 sem. | Voir ci-dessous |
| D. Dossier Python portable + script | 300–600 Mo | 1 sem. | Repli minimal |

- PAMGuard (établi : pamguard.org, JASA 2026 `[À VÉRIFIER auteurs]`) : bureau Java multiplateforme, sans code ; module deep learning chargeant un modèle générique (PyTorch JIT, TensorFlow, ONNX via DJL), segmentation, transformations, affichage, gestion de données, export ; support terrestre. Encodeur + tête fusionnés en un graphe ONNX → livrable = modèle + configuration. Limites : pas de réentraînement en place ; module séquentiel à porter ou à sacrifier ; ergonomie d'acousticien. Décision en S13 : A contre C, chacune testée par un utilisateur ONF.
- Poids : Bird-MAE-Base ≈ 185 Mo en float16 ; BEATs comparable `[À VÉRIFIER]` ; Perch 2.0 ≈ 50 Mo si voie CPU ; modèle distillé : quelques Mo.
- GUI (option A) : Gradio / NiceGUI / PySide6, choisie en S13 ; Streamlit pour le prototype. YAPAT exclu du livrable (PostgreSQL, Celery : trois services en arrière-plan).
- Écrans : Analyser (dossier, reprise idempotente) ; Vérifier (file, spectrogramme 2–8 kHz, boutons du schéma de labels, raccourcis) ; Résultats (points classés, export CSV) ; Modèle (versions, réentraînement, seuil).
- Mise à jour : tête dans l'application ; encodeur par paquet (`manifest.json` : nom, version, SHA-256, licence, f_e, fenêtre).
- Documentation : README ; « Interpréter un score » ; « Vérifier une file » ; guide du mainteneur ; `LICENSES.md`.
- Licences (sans avis juridique) : tout composant redistribué doit pouvoir l'être par l'ONF, confirmé avant fin S12 ; BirdNET (CC BY-NC-SA) exclu par défaut ; Perch 2.0 (Apache 2.0) et AnuraSet (CC0) compatibles ; autres `[À VÉRIFIER]`.

---

## 8. Planning (26 semaines, S1 = 14–18/09/2026)

Six étapes manuscrites reprises, avec trois arbitrages : apprentissage actif dès S3 ; indices acoustiques = contrôle qualité ; combinaisons conditionnelles.

| Phase | Semaines | Contenu | Livrable | Go / no-go |
|---|---|---|---|---|
| P0 Cadrage, données, benchmark | S1–S3 (15/09–02/10) | Lecture (ONF, Fouquet 2018, Courtois 2025, Kath 2024, fils Kaggle) ; inventaire des deux jeux et des 345 + 158 annotations ; négatifs appariés ; protocole d'évaluation figé ; essai YAPAT ; pré-benchmark AnuraSet ; benchmark ONF ; C0–C1 ; questions §10 | Tableau de benchmark à trois critères ; inventaire ; verdict C1 | ≥ 1 encodeur local avec AP ≥ 0,3 et rappel ≥ 0,8 à précision ≥ 0,1 en plis par site ; sinon DSP + template matching en primaire |
| P1 Pipeline et apprentissage actif | S4–S7 (05/10–30/10) | Pipeline CLI ; probing linéaire en service ; tours 0–1 ; annotation par enregistrement ; C2 ; expérience d'échelle ; calibration expert | Dépôt Git ; jeu annoté v1 (≥ 30 enregistrements positifs de ≥ 3 micros, négatifs par espèce) | ≥ 30 enregistrements positifs validés sur ≥ 2 sites ; sinon risque 1 |
| P2 Optimisation et évaluation | S8–S12 (02/11–04/12) | Tours 2–5 ; jeu gelé ; module séquentiel (persistance, solo/chœur) et fusion ; C3 ; attentive probing si tokens ; choix d'encodeur ; licences | Jeu gelé ; courbes d'apprentissage ; rappel par SNR ; note d'arbitrage | Rappel ≥ 0,85 à précision ≥ 0,1 sur sites tenus à l'écart ; sinon pivot (§9). **Fin S12 = limite de pivot méthodologique** |
| P3 Outil v1 | S13–S18 (07/12–15/01 ; S15–S16 réduites `[HYPOTHÈSE H17]`) | Expérimentation S13 : A contre PAMGuard ; export ONNX ; GUI ; test machine ONF `[HYPOTHÈSE H6]` ; distillation si enseignant stable | Application ou configuration PAMGuard testée sur deux machines ; guide v0 | 1 h d'audio en < 15 min hors ligne sur machine ONF ; sinon option D |
| P4 Généralisation et combinaisons | S19–S22 (18/01–12/02) | Jeu 2026 et Mataroni ; test utilisateurs (trois critères) ; boucle ; combinaisons et LoRA/AnuraSet si effectifs ; documentation | Rapport d'évaluation ; v1.1 ; documentation | Gel fin S22 |
| P5 Rapport | S23–S26 (15/02–12/03) | Rédaction, soutenance, transfert | Rapport `[HYPOTHÈSE H11]` ; soutenance ; dépôt transféré | — |

- Rédaction continue dès S17 (½ j/sem.). Chaque réunion hebdomadaire : avancement, chiffre clé (enregistrements positifs validés, AP sur jeu gelé), décision demandée.
- Chemin critique : données (S1) → benchmark (S3) → enregistrements positifs validés (S4–S7) → jeu gelé (S9) → encodeur (S12) → packaging (S13–S16) → test utilisateurs (S19–S20) → gel (S22). Marge : une semaine en P2, une en P4. Après S12 : plus de changement de famille ; après S16 : plus de changement d'encodeur.

---

## 9. Risques

| Risque | Signal | Seuil | Repli |
|---|---|---|---|
| 1. Positifs concentrés sur 1–2 sites / peu d'enregistrements | Inventaire des 345 fenêtres | < 3 sites ou < 10 enregistrements distincts | Généralisation annoncée non mesurable sur 2023 ; Mataroni seul juge ; annotation par enregistrement pour élargir ; prototype différentiel comme classifieur de secours |
| 2. Confusion avec les déclencheurs | > 30 % des 100 meilleurs candidats = congénères / fourmilier après le tour 3 | Précision < 0,1 à rappel 0,85 sur micros riches en déclencheurs fin S11 | Poids accru de la persistance et du rythme ; négatifs par espèce ; sortie « *Anomaloglossus* sp. » avec avertissement |
| 3. Chant ponctuel avéré (H20 fausse) | Enregistrements positifs validés avec < 50 % de fenêtres positives | > 10 % des positifs validés | Abandon du seuil de fraction ; retour au max par enregistrement ; file « suspect » réévaluée |
| 4. Bande saturée : détecteur amont inutile (tuteur) | Candidats/h en S3 | > 500/h aux heures de pic | Fenêtres pleines ; seuillage limité aux onsets |
| 5. Décalage 2023 → 2026 (micros, sites) | AP sur audit 2026 | < 50 % de l'AP en plis par site (S19) | Recalibration par jeu ; négatifs durs 2026 avec partie tenue à l'écart ; normalisation par micro |
| 6. Volume ingérable | Débit mesuré S1 | Encodage projeté > 10 jours-machine | Heures et mois de chant d'abord ; pas plus grossier ; encodeur léger ; extraction déportée `[À VÉRIFIER]` |
| 7. f_e ou format incompatibles | Inventaire S1 | f_e < 11 kHz ou propriétaire | Rééchantillonnage dans les wrappers ; ffmpeg/sox |
| 8. Pas de Perch 2.0 sur CPU | Échec des replis (1)–(2) | Fin S3 | Référence déportée seulement ; déploiement Bird-MAE / BEATs / ProtoCLR |
| 9. AnuraSet non représentatif | Classement inversé sur ONF | Premier ↔ dernier | AnuraSet réservé à l'adaptation, pas au choix |
| 10. Expert indisponible | Calibration non planifiée fin S4 | Aucun label validé fin S6 | « Incertain » strict ; validation asynchrone par lots |
| 11. Perte machine / données | Machine unique | — | SSD externe quotidien, copie ONF hebdomadaire, labels sous Git |

---

## 10. Questions aux encadrants (par priorité)

1. « Les 345 fenêtres et les 158 négatifs : combien d'enregistrements, quels micros, quels sites, quelle durée de fenêtre, quel annotateur, avec quelles classes de qualité ? » — Fixe l'effectif effectif et le protocole de validation.
2. « Quels enregistreurs, f_e, format, nommage ; volume réel des deux jeux ; durée des déploiements 2026 à Trésor et Kaw ; accès aux données cette semaine ? » — Chemin critique.
3. « *A. blanci* chante-t-elle parfois de façon ponctuelle ? Un chœur et un mâle seul sont-ils distinguables à l'oreille ? » — Décide la règle d'agrégation et les sous-classes.
4. « Quels sites et micros ont une présence confirmée en 2023 et en 2026 ? Mataroni est-il occupé ? » — Décide la généralisation.
5. « Liste des espèces déclencheuses, leurs heures et motifs ; enregistrements de référence du fourmilier et des congénères ? » — Negative mining et module séquentiel.
6. « L'étude de phénologie 2025 repose-t-elle sur le jeu 2023, et ses instants de détection sont-ils disponibles ? » — Peut multiplier les positifs.
7. « Taux d'émission et intervalles entre notes d'*A. blanci* ? » — Module séquentiel.
8. « Objection des 16 kHz : harmoniques, résolution temporelle, ou indices haute fréquence ? » — Benchmark (contrôle passe-bas).
9. « Qui vérifie, combien de temps par semaine, quelle séance de calibration ; quel volume de points l'ONF peut-il absorber ? » — Budget et précision plancher.
10. « Machines des utilisateurs ; connexion pour un téléchargement initial ? » — Packaging, option PAMGuard.
11. « Qui tranche les licences, à quelle échéance ? » — Filtre encodeurs avant S12.
12. « Outils actuels de dépouillement ; nouvelles données pendant le stage ; dates de rapport et soutenance ? » — Baseline, boucle réelle, P5.

---

## 11. Montée en compétence

- Signal (S1–S4, ≈ 14 h) : échantillonnage, Nyquist ; STFT et compromis fenêtre/pas (note de 90 ms : fenêtre 16–25 ms, pas ≤ 10 ms) ; log-mel ; passe-bande, enveloppe, onsets ; SNR sur enveloppe en bande ; corrélation croisée. Ressources : docs scikit-maad, librosa ; Smith, DSP Guide `[À VÉRIFIER]` ; Müller, FMP `[À VÉRIFIER]`.
- Écologie (S1–S6, ≈ 18 h) : vocabulaire du chant, température ; *Anomaloglossus*, sympatrie ; phénologie ; occupation et détection ; déclencheurs. Fouquet 2018 (S1) ; Courtois 2025 ; Cañas 2023 ; PNA `[À VÉRIFIER]` ; Köhler 2017 `[À VÉRIFIER]` ; écoute avec l'expert.
- Apprentissage actif (S2, 4 h) : Kath et al. 2024 ; Hamer, Laber, Denton 2023 ; Rauch et al. 2024 `[À VÉRIFIER]`.
- Arithmétique d'embedding (après S16, tuteur) : directions et décalages dans l'espace de représentation ; prototype différentiel en premier cas ; solo/chœur et distance comme décalages.
- Placement : première heure des journées S1–S4, puis 1 h/sem. ; rien sur le chemin critique après S6.

---

## 12. Hypothèses

| Id | Hypothèse | Levée par |
|---|---|---|
| H1 | Enregistreurs autonomes, WAV PCM mono, f_e ≥ 32 kHz | Q2 |
| H2 | Les 345 fenêtres proviennent de ≈ 10–20 enregistrements et de 1–2 sites ; fenêtres de 3–5 s | Q1 |
| H3 | Jeu 2023 = entraînement, jeu 2026 = généralisation, Mataroni site neuf occupé | Q4 |
| H4 | Positions des encadrants reconstituées | Relecture |
| H5 | Décision gestionnaire = planification forestière à l'échelle du point ou du bassin | Q9 |
| H6 | Utilisateurs sur portables Windows sans GPU, 8–16 Go | Q10 |
| H7 | L'étude de phénologie 2025 repose sur le jeu 2023, détections récupérables | Q6 |
| H8 | *A. blanci* absent des classes de Perch 2.0 | Liste de classes |
| H9 | IOI de l'ordre de la seconde ; plusieurs notes par fenêtre | Q7 |
| H10 | Dépouillement actuel par écoute et spectrogrammes ; Mac 16 Go | Q12 |
| H11 | Rapport en fin de stage, soutenance mars–avril 2027 | Q12 |
| H12 | Expert : 3 h puis 1 h toutes les deux semaines | Q9 |
| H13 | Un rapatriement de terrain avant février 2027 | Q12 |
| H14 | Enregistrements de référence des déclencheurs accessibles au-delà des 158 fenêtres | Q5 |
| H15 | Pluies de décembre à juillet, saison sèche d'août à novembre `[À VÉRIFIER]` | Calendrier ONF |
| H16 | Jeu 2023 ≈ 3 200 h si continu ; durée 2026 Trésor/Kaw de quelques semaines | Q2 |
| H17 | Deux semaines réduites aux fêtes | Calendrier |
| H18 | Déclencheurs < 10 espèces, heures partiellement disjointes | Q5 |
| H19 | Les 345 + 158 sont les annotations du prestataire, classes de qualité récupérables | Q1 |
| H20 | *A. blanci* ne chante jamais de façon ponctuelle ; détection isolée = faux positif | Q3 |
| H21 | Solo et chœur séparables par densité d'onsets et chevauchements | Q3 |

---

## Angles morts

- **Effectif effectif.** 345 fenêtres corrélées peuvent valoir 10 enregistrements ; tout le document raisonne en fenêtres là où il devrait compter en enregistrements et en sites.
- **H20 porte l'agrégation.** Si *A. blanci* chante parfois brièvement, la règle de fraction crée des faux négatifs exactement là où le rappel importe.
- **Phénologie comme piège.** Saison et heure sont d'excellents prédicteurs sur Trésor ; les mettre dans le classifieur reviendrait à apprendre la phénologie d'un site et à la plaquer sur Mataroni.
- **Mataroni : 5 jours.** Un site neuf en instantané : test de généralisation spatiale, mais aucune information sur sa saisonnalité ; une non-détection y est faible.
- **Charge de vérification.** La règle « tout point positif remonte » n'est bornée que par la précision plancher ; la capacité de l'ONF n'est pas connue.
- **Informations orales.** Fourmilier, congénères, chant continu, solo/chœur : rapportées par le tuteur, non vérifiées sur enregistrement.
- **AnuraSet et C1 sont des indicateurs faibles.** Autres biomes pour l'un, 30 positifs pour l'autre.
- **Après le 14 mars.** Vérification hebdomadaire non promise ; mainteneur non désigné.
- **Sur-ingénierie.** Deux voies, combinaisons, distillation, migrations : chaque heure d'infrastructure est retirée à l'annotation et à l'évaluation.
- **Occupation.** La non-détection devrait passer par un modèle d'occupation `[À VÉRIFIER]` ; hors périmètre.
- **Variation intraspécifique.** Paramètres acoustiques issus de peu d'individus ; température et taille peuvent sortir des filtres.

---

## Sources vérifiées (16–17/09/2026, hors bibliographie fournie)

- Cañas et al. (2023), AnuraSet, *Scientific Data* ; arXiv:2307.06860 ; github.com/soundclim/anuraset ; Zenodo, CC0.
- Kath, Serafini, Campos, Gouvêa & Sonntag (2024), *Ecological Informatics* 82, 102710 ; YAPAT : yapat.readthedocs.io.
- Hamer, Laber & Denton (2023), Agile Modeling for Bioacoustic Monitoring, tutoriel Climate Change AI ; Zenodo 10.5281/zenodo.11585179.
- Martínez Balvanera et al. (2025), Whombat, *Methods in Ecology and Evolution*.
- PAMGuard : pamguard.org (module Deep Learning, tutoriel) ; JASA 159(1), 2026 `[À VÉRIFIER auteurs]` ; Macaulay & Gillespie (2022), JASA 151(4) suppl.
- BirdCLEF+ 2026 : page Kaggle ; notebook communautaire mentionnant `perch_v2_no_dft.onnx` `[À VÉRIFIER]`.
- Rauch et al. (2024), arXiv:2406.18621.
