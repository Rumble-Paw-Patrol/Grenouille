# Feuille de route — Stage de fin d'études ONF Guyane
## Détection acoustique automatique d'*Anomaloglossus blanci* — 15/09/2026 → 14/03/2027

Document de travail interne, v0. Conventions : `[HYPOTHÈSE Hn]` = information manquante remplacée par une hypothèse explicite (liste en §12) ; `[À VÉRIFIER]` = fait ou source hors bibliographie fournie ; « établi » = appuyé sur la bibliographie fournie ; « jugement » = arbitrage d'ingénieur.

**Lecture des positions** (`[HYPOTHÈSE H4]` : extraits verbatim non fournis, positions reconstituées à partir des notes de préparation). Encadrant A : déconseille les embeddings BirdNET (dépendance TensorFlow, base de code BirdNET-Analyzer), oriente vers des modèles PyTorch sur HuggingFace — encodeur BEATs de NatureLM-audio, Bird-MAE — en citant la revue de Schwinger et al., vraisemblablement sa v1, et propose Streamlit ou Gradio pour la réannotation. Encadrant B : embeddings Perch 2.0 en première intention, extraits via bacpipe, probing linéaire et apprentissage actif ; objecte que NatureLM-audio travaille à 16 kHz. **Arbitrage** : (i) aucun composant TensorFlow à l'exécution dans le livrable — pas d'accélération sur Apple Silicon (établi par la carte de modèle citée), packaging lourd ; (ii) l'encodeur est choisi par le benchmark de la semaine 1 sur les données du projet, sous filtre de licence et de déployabilité ; (iii) l'objection des 16 kHz devient un test empirique (§2, §10).

---

## 1. Reformulation du problème

**Objectif écologique.** Établir, pour chacun des 86 points d'écoute, si *A. blanci* (En danger, Plan national d'actions) y chante, pour que la gestion forestière — exploitation, dessertes, zones tampons autour des criques — en tienne compte `[HYPOTHÈSE H5]`.

**Problème d'apprentissage.** Détection d'un événement rare et très bref (note de 0,09–0,10 s, énergie dominante entre 4,5 et 5,4 kHz) dans un flux à faible cycle utile (2 min par heure, 3,3 % du temps), avec quelques dizaines de positifs, une classe négative immense et structurée (congénères, orthoptères, pluie, cours d'eau) et un décalage de domaine attendu entre sites et entre jeux. Posé proprement : un problème de **classement** sous très fort déséquilibre. Le modèle ordonne des fenêtres par vraisemblance de présence, un humain vérifie le haut de la liste, la décision gestionnaire agrège les vérifications ; à prévalence très faible, 99 % de spécificité donnent une précision de l'ordre du pourcent, donc la vérification humaine est structurelle et le rappel prime.

**Unité de décision.** Cinq niveaux : note, fenêtre d'analyse (3 ou 5 s, imposée par l'encodeur), enregistrement (2 min), point d'écoute, session. Choix (jugement) :

| Niveau | Rôle | Conséquence |
|---|---|---|
| Fenêtre, grille glissante à pas ≤ 2,5 s (aucune note coupée) | Apprentissage, score, vérification humaine | Métriques de développement : AP, rappel à précision fixée |
| Enregistrement de 2 min | Restitution : présence confirmée / candidat non vérifié / rien | Score = max ou top-k des fenêtres |
| Point × période | Décision gestionnaire | « Présence confirmée » si ≥ k notes validées par un humain sur ≥ n enregistrements de jours distincts (k = 3, n = 2, à négocier) ; sinon « non détecté », jamais « absent » |

**Écart entre métrique optimisée et décision.** (1) L'AP mesure un classement ; la décision est un statut par point avec asymétrie de coût : une fausse absence peut détruire un habitat (irréversible), une fausse présence coûte du temps de vérification (réversible). (2) Une absence ne se démontre pas. Deux pics d'une heure à 2 min par heure : 4 min par jour dans la fenêtre favorable. Si p est la probabilité qu'un échantillon de 2 min contienne un cri détecté (probabilité de chant × rappel), la probabilité d'au moins une détection sur D jours vaut 1 − (1 − p)^(2D) : pour p = 0,05, 76 % à 14 jours, 95 % à 30 jours. Le déploiement permanent rend la non-détection crédible ; le rappel entre multiplicativement dans p. (3) Seuil et règle (k, n) ne sont pas optimisés par le modèle : fixés avec l'ONF, figés avant l'évaluation finale.

**Problème d'échelle.** Une note de 0,09 s occupe 1,8 % d'une fenêtre de 5 s (Perch 2.0, Bird-MAE) et 3 % d'une fenêtre de 3 s (BirdNET). L'embedding — vecteur résumant la fenêtre par agrégation de représentations locales — est dominé par tout ce qui n'est pas la grenouille.

*Pour un détecteur d'événements en amont* : il évite la dilution de la note dans l'agrégation temporelle et la variance due à sa position dans la fenêtre ; il mesure directement durée, fréquence et intervalles ; il réduit le nombre de fenêtres à encoder.

*Contre* : le rappel du détecteur plafonne celui du système ; dans cette bande, les orthoptères déclenchent presque en permanence ; les encodeurs et les résultats de la revue (établi : Schwinger et al. 2026) portent sur des fenêtres pleines de 3–5 s, pas sur 0,1 s complétée de zéros ; surtout, le caractère diagnostique d'*A. blanci* est la **répétition de notes isolées** : si plusieurs notes tombent dans 5 s `[HYPOTHÈSE H9]`, la fenêtre pleine contient le rythme qui la distingue des trains soudés des congénères. Enfin, l'*attentive probing* — petite tête d'attention entraînée sur les représentations locales (tokens) d'un encodeur gelé, qui apprend où regarder, le gradient ne traversant que la tête — est nécessaire pour exploiter les transformers (établi) et répond à la parcimonie de l'événement sans détecteur amont (jugement).

*Protocole pour trancher (S2–S4, même jeu annoté, validation par site).* A = grille 5 s, pas 2,5 s ; B = pas 1 s ; C = fenêtres centrées sur les événements du détecteur DSP ; D = fenêtres de 1 s complétées de zéros ; E = pour Bird-MAE et BEATs, agrégation des tokens par moyenne, maximum ou attention. Mesures : AP, rappel à précision ≥ 0,1, rappel propre du détecteur sur les notes annotées (≥ 0,95 exigé, jugement). Règle : C n'est adopté que s'il dépasse A/B de plus que l'incertitude (≈ 0,1 d'AP avec 30 positifs, §6) ; sinon le détecteur ne sert qu'au post-traitement séquentiel (§3).

---

## 2. Semaine 1 : le benchmark d'embeddings via bacpipe

**J1–J2 — inventaire et jeu annoté v0.** Inventaire (format, fréquence d'échantillonnage, enregistreur, nommage, sites, calendrier) ; conversion en WAV PCM mono à fréquence native (ffmpeg ou sox si format propriétaire `[À VÉRIFIER]`). Positifs = fenêtres contenant intégralement au moins une note labellisée. Négatifs de trois origines tracées : (a) tirage aléatoire aux mêmes heures et sites ; (b) fenêtres à forte énergie en bande 4,4–5,5 kHz sans note labellisée, prises à des sites sans présence connue pour limiter les positifs non annotés `[HYPOTHÈSE H2]` ; (c) congénères s'ils existent `[HYPOTHÈSE H14]`. Ratio 1:20 à 1:50 (jugement) ; chaque fenêtre porte site et enregistrement. Contrôle annexe : un *Anomaloglossus* dans les classes de Perch 2.0 `[HYPOTHÈSE H8]` ?

**J2–J3 — extraction.** `bacpipe` pour `birdnet`, `perch_v2`, `birdmae`, `beats`, `naturebeats`, `protoclr`, plus `perch_bird` et `convnext_birdset` s'ils tournent sans effort. Relevé par modèle : exécutable ou non, débit (fenêtres/s), mémoire, dimension — il alimente le risque 4.

**J3–J4 — évaluation.** Deux sondes : k plus proches voisins (cosinus, k ∈ {1, 3, 5}) et régression logistique L2 (C par validation interne, classes rééquilibrées). bacpipe fournit probing linéaire et kNN (établi) ; son découpage est `[À VÉRIFIER]` — s'il est aléatoire, un script maison impose une validation **groupée par site** : leave-one-site-out si les positifs couvrent au moins trois sites, sinon leave-one-recording-out en signalant que la généralisation à de nouveaux sites n'est pas mesurée. Vingt sous-échantillonnages des négatifs ; comparaison **par paires sur les mêmes plis**.

**J5 — tableau de décision** : modèle × (licence, exécution sur M4 : native / ONNX / déportée / impossible, débit, dimension, AP avec intervalle interquartile, rappel à précision 0,5 et 0,1, kNN top-1). Une UMAP peut être montrée ; elle ne décide de rien.

**Métriques qui concluent** : AP (aire sous la courbe précision–rappel) ; rappel à précision fixée ; différences appariées avec bootstrap par enregistrement ; stabilité du classement des modèles entre plis. **Trompeuses à cet effectif** : exactitude ; AUROC — avec 30 positifs contre 1 500 négatifs, 0,95 d'AUROC est compatible avec une précision catastrophique et 0,02 d'écart est du bruit ; F1 au seuil 0,5 ; AMI/ARI sur deux classes minuscules. Un positif vaut 3,3 points de rappel : moins de 0,1 d'AP d'écart = **égalité**, départagée par licence, déployabilité et débit.

**Modèles qui ne tourneront pas.** `perch_v2` : TensorFlow 2.20 et GPU requis (établi), sans accélération sur Apple Silicon. Replis : (1) wrapper bacpipe sur CPU `[À VÉRIFIER]` ; (2) export ONNX via tf2onnx puis onnxruntime CPU, selon les opérateurs du graphe `[À VÉRIFIER]` ; (3) extraction déportée sur une session GPU tierce (notebook gratuit ou machine institutionnelle `[À VÉRIFIER]`), limitée au jeu de benchmark (quelques milliers de fenêtres), embeddings rapatriés en `.npy` ; (4) variante CPU officielle si elle sort. Le repli (3) suffit au benchmark, pas au déploiement (risque 6). `birdnet` : TFLite sur CPU, débit acceptable. `naturebeats` : encodeur seul, jamais le LLM de 8 milliards de paramètres `[À VÉRIFIER que bacpipe le sépare]`. Bird-MAE-Base (92,9 M paramètres) tient en mémoire unifiée `[HYPOTHÈSE H10]`.

**Du désaccord à la question empirique.** « BirdNET plafonne » → AP(birdnet) contre les autres. « 16 kHz est trop bas » → AP(beats, naturebeats) à 16 kHz contre AP(birdmae) à 32 kHz, **avec un contrôle** : birdmae alimenté par le même audio filtré passe-bas à 8 kHz puis rééchantillonné à 32 kHz, pour isoler l'effet de la bande passante. « Perch 2.0 est le meilleur » : établi pour BirdSet et BEANS en v2 de la revue, non établi pour une note d'anoure en crique guyanaise → AP(perch_v2) sur nos données, sous réserve d'exécutabilité. Sortie : deux encodeurs au plus pour la phase 1, après filtre de licence (§7).

---

## 3. Cartographie des approches candidates

| Approche | Principe | Coût | Annotations | Robustesse au décalage de domaine | Faisabilité M4 | Licence | Verdict |
|---|---|---|---|---|---|---|---|
| Template matching | Corrélation croisée d'un spectrogramme de référence de la note avec celui de l'enregistrement (scikit-maad `[À VÉRIFIER]`) | 2–3 j | 1–10 références | Faible (SNR, réverbération, variation individuelle) | Triviale | Libre | Baseline obligatoire (§6) ; insuffisant seul contre des congénères de même bande |
| Seuillage spectral 4,4–5,5 kHz | Passe-bande, enveloppe d'énergie, seuil adaptatif, filtre de durée 0,08–0,11 s | 2–3 j | 0 (Fouquet et al. 2018) | Faible : déclenche sur les orthoptères | Triviale | Libre | Candidats et instants de notes pour le post-traitement ; jamais décideur |
| Indices acoustiques | Statistiques résumant un enregistrement | 1–2 j | 0 | Sans objet | Triviale | Libre | Rejeté pour la détection ; contrôle qualité (pluie, panne, saturation) |
| Linear probing sur embeddings gelés | Régression logistique sur l'embedding d'un encodeur fixé | 1 sem. avec bacpipe | Dizaines → centaines | Dépend de l'encodeur ; Perch 2.0 couvre > 14 500 espèces dont des amphibiens (établi) | PyTorch : oui ; Perch 2.0 : conditionnel (§2) | Par encodeur | **Approche principale** |
| Attentive probing | Tête d'attention sur les tokens d'un encodeur gelé | 1–2 sem. (accès aux tokens) | ≥ 150–200 positifs (jugement) | Idem | Oui | Idem | Phase 2 si l'effectif le permet (établi : nécessaire pour les transformers) |
| Few-shot contrastif | Prototypes de classe et distances dans l'espace d'embedding | 1–2 sem. | Dizaines | ProtoCLR conçu pour le transfert focal → paysages sonores (établi : Moummad et al. 2024) | Oui, petit modèle | `[À VÉRIFIER]` | Second candidat ; fine-tuning seulement si le probing plafonne |
| Clustering non supervisé | UMAP + HDBSCAN ou k-means sur embeddings | 2–3 j | 0 | — | Oui | Libre | Exploration et negative mining ; aucune décision |
| Fine-tuning | Mise à jour de tout ou partie de l'encodeur | 2–4 sem. | Centaines à milliers | Risque élevé de sur-apprentissage au site | Partiel (derniers blocs) ; complet non raisonnable | Par encodeur | Pas avant la phase 4 ; hors chemin critique |

**Structure temporelle du chant : post-traitement explicite, oui (jugement).** Elle encode directement le caractère diagnostique — notes isolées répétées contre trains de 6–11 notes de 0,028 s chez *A. saramaka* et trains soudés chez d'autres congénères —, elle est lisible par un naturaliste, elle coûte peu. Méthode : débuts de notes (onsets) sur l'enveloppe de la bande 4,4–5,5 kHz ; intervalles entre débuts successifs (IOI) ; descripteurs : fraction d'IOI < 0,1 s (signature de train), distribution des durées, régularité ; IOI d'*A. blanci* attendus de l'ordre de la seconde `[HYPOTHÈSE H9]`. Fusion par régression logistique sur (score d'embedding, descripteurs), mêmes plis. Limites : chœurs, onsets masqués par pluie ou stridulation continue. Le score séquentiel module ; il n'oppose jamais de veto à un score d'embedding élevé.

---

## 4. Architecture cible

```
audio brut (format et f_e inconnus, H1)
  └─ ingest      inventaire, métadonnées (site, horodatage, enregistreur) → SQLite
  └─ decode      float32 mono, f_e native conservée (soundfile ; ffmpeg si exotique)
  └─ grid        fenêtres (recording_id, offset_s, dur_s), INDÉPENDANTES de l'encodeur
  └─ encoder[k]  rééchantillonnage vers f_e(k) ; E_k ∈ R^{N×d_k}
  └─ store       Parquet partitionné encoder_id / site / mois, float16
  └─ index       similarité cosinus, exhaustive par fragments (FAISS optionnel)
  └─ head[v]     régression logistique → score par fenêtre
  └─ sequential  onsets en bande → descripteurs → fusion
  └─ aggregate   fenêtre → enregistrement → point × période
  └─ queue       file de vérification (GUI) → labels append-only → ré-entraînement de head
```

**Interfaces (signatures, pas de code).**
- `class Encoder(Protocol)` : `name`, `version`, `sample_rate`, `window_s`, `dim` ; `embed(wav: np.ndarray, sr: int) -> np.ndarray` (n_fenêtres, dim) ; `embed_tokens(...) -> np.ndarray | None` (n_fenêtres, n_tokens, dim) pour l'attentive probing. Implémentations : `BacpipeEncoder` (recherche), `OnnxEncoder` (livrable).
- `train_head(X, y, groups, C_grid) -> Head` (scikit-learn) ; `Head.score(E) -> np.ndarray`
- `detect_onsets(wav, sr, band=(4400, 5500)) -> np.ndarray` (scikit-maad, scipy)

**Points de découplage.** (1) La grille est définie sur (enregistrement, décalage) : les labels s'y rattachent, pas aux embeddings, et survivent à tout changement d'encodeur. (2) L'encodeur est derrière le protocole, sélectionné par un fichier de configuration (encodeur, version de tête, seuils). (3) La GUI parle à une couche de service, jamais aux modèles.

**Stockage et volume.** 86 points × 48 min/jour ≈ 69 h d'audio par jour `[HYPOTHÈSE H16]` ; à pas 2,5 s, ≈ 3 millions de fenêtres par mois, soit ≈ 4,5 Go/mois en 768-d float16 (Bird-MAE) ou ≈ 9 Go/mois en 1536-d (Perch 2.0) : tenable sur un portable pour une saison, SSD externe pour un an. Parquet, colonne de taille fixe en float16, un fichier par (encodeur, site, mois) ; métadonnées, labels, versions et décisions dans SQLite.

**Index de similarité.** À 10⁶–10⁷ vecteurs, une recherche exhaustive en cosinus par fragments (produit matriciel float16 sur MPS) prend quelques secondes (jugement) ; FAISS seulement si la GUI exige moins d'une seconde. *Append-only* : chaque partition est indexée à l'arrivée, aucune suppression n'est nécessaire.

**Changement d'encodeur.** Embeddings clés par `encoder_id`. Un changement déclenche une tâche de fond qui ré-encode le corpus (quelques heures, jugement), ré-entraîne la tête sur les mêmes labels et produit un rapport de migration (AP ancienne contre nouvelle sur le jeu gelé) avant basculement. Chaque décision porte (`encoder_id`, `head_version`, `threshold_id`) : une analyse passée reste reproductible ; seules les fenêtres non vérifiées sont rescorées, avec journal des changements de statut.

---

## 5. Amorçage et apprentissage continu

**Tour 0 (S2–S3) — recherche par similarité.** Embeddings des cris de référence ; requêtes par exemple et par prototype (moyenne normalisée des positifs) sur les fenêtres des heures de pic (6–7 h, 16–17 h) des mois humides d'abord, tous sites du jeu 1. File : 300 meilleures similarités + 100 fenêtres aléatoires des heures de pic + 100 négatifs durs en bande. Rendement attendu : 20 à 80 nouveaux positifs `[HYPOTHÈSE H2, H9]`.

**Negative mining ciblé.** Congénères : si des enregistrements d'*A. degranvillei*, *A. dewynteri*, *A. surinamensis*, *A. baeobatrachus* et *A. saramaka* sont accessibles `[HYPOTHÈSE H14]`, ils servent de requêtes et reçoivent un label propre `congénère`, conservé pour une évaluation hiérarchique (genre puis espèce). Orthoptères : fenêtres à forte énergie en bande et structure pulsée (densité d'onsets élevée, IOI < 50 ms), regroupées par HDBSCAN sur les embeddings ; un cluster est labellisé en bloc après contrôle de dix exemples. Labels : {blanci, congénère, orthoptère, autre, incertain}. Clustering exploratoire : UMAP + HDBSCAN sur 100 000 fenêtres d'heures de pic, inspection des clusters voisins du prototype positif et des clusters inattendus.

**Phénologie.** Priorité aux heures de pic et aux mois humides `[HYPOTHÈSE H15]`, stratification par site ; 20 % de chaque file est tirée hors de ces strates (nuit, saison sèche) pour ne pas enfermer le modèle dans les conditions de pic.

**Apprentissage actif (tours 1–5, S4–S12).** Après chaque ré-entraînement de la tête (secondes) : file = 60 % de fenêtres incertaines, 20 % de scores maximaux, 20 % d'échantillon aléatoire stratifié (estimation non biaisée, détection de dérive). Lots de 100–150 fenêtres, sélection par grappes contre les quasi-doublons. Arrêt : gain d'AP sur le jeu gelé < 0,02 sur deux tours consécutifs (jugement).

**Budget d'annotation.** 250–350 fenêtres/h avec raccourcis clavier et spectrogramme (jugement). Stagiaire : 6 h/sem. en S2–S6, 4 h/sem. en S7–S12, 2 h/sem. en S17–S22 ≈ 66 h, soit 16 000 à 20 000 fenêtres. Expert (naturaliste ONF ou auteurs du rapport de phénologie `[HYPOTHÈSE H12]`) : 3 h de calibration initiale (100 fenêtres écoutées en commun, accord inter-annotateurs mesuré), puis 1 h toutes les deux semaines sur les files `incertain` et `congénère` ≈ 14 h. Un label `blanci` n'entre en entraînement qu'après validation experte.

**Boucle après déploiement.** Qui : les naturalistes ONF, sur une file hebdomadaire de ≈ 150 fenêtres (30–45 min) en saison de chant ; l'application construit la file et journalise auteur et horodatage. Mise à jour : ré-entraînement local de la tête, comparaison affichée avec la version en service sur le jeu gelé embarqué, nouvelle version seulement si l'AP n'a pas baissé. Invariants : labels validés immuables ; analyses passées gardant leur triplet (encodeur, tête, seuils) ; encodeur changé uniquement par un mainteneur via la migration du §4. Calendrier : le stage commence en saison sèche `[HYPOTHÈSE H15]` ; la boucle est démontrée en simulation (§6), et en conditions réelles seulement si un rapatriement a lieu avant février `[HYPOTHÈSE H13]`.

---

## 6. Protocole d'évaluation

**Séparation par site.** Chaque point a son paysage sonore et souvent les mêmes individus ; un découpage aléatoire met des fenêtres d'un même enregistrement des deux côtés, le modèle apprend le site et le score ne mesure pas ce que l'ONF demande : de nouveaux points. Donc validation groupée par site (5 plis) sur le jeu 1, et **jeu 2 réservé à la généralisation** : hors entraînement avant la phase 4, et seulement par ses labels d'audit aléatoire `[HYPOTHÈSE H3]`.

**Jeu de test gelé.** Construit en S5–S8 : au moins 10 sites du jeu 1, 60 enregistrements de 2 min écoutés intégralement (≈ 2 h d'audio, 4–6 h d'annotation), plus tous les enregistrements positifs connus de sites tenus à l'écart. Versionné, jamais utilisé pour l'entraînement.

**Métriques retenues** : AP par fenêtre ; rappel à précision ≥ 0,1 (plancher) et ≥ 0,5 ; fausses alarmes par heure d'audio au seuil d'exploitation ; rappel par enregistrement ; accord du statut par point avec le jugement expert. **Rejetées** : exactitude, AUROC (annexe seulement), F1 au seuil 0,5, kappa.

**Priorité au rappel et plancher de précision.** Le rappel parce qu'une vérification corrige une fausse alarme et que rien ne corrige un site manqué. Le plancher parce que vérifier doit coûter moins qu'écouter directement les heures de pic : à 10 s par candidat et précision p, un vrai positif coûte 10/p secondes, et la file hebdomadaire doit tenir en une heure (jugement). Cible : rappel ≥ 0,9 au seuil le plus élevé maintenant une précision ≥ 0,1 sur sites tenus à l'écart ; sinon, présenter la frontière précision–rappel et laisser l'ONF choisir.

**Calibration des seuils.** Les scores ne sont pas des probabilités (classes rééquilibrées, effectif minuscule) : seuils choisis sur les scores hors-pli pour le rappel cible ; recalibration par jeu si le décalage de domaine est confirmé ; régression isotonique ou de Platt seulement au-delà de ≈ 200 positifs (jugement).

**Intervalles de confiance.** Wilson sur le rappel : n = 30 et rappel 0,9 → [0,74 ; 0,97] ; n = 100 → [0,83 ; 0,94]. Bootstrap par enregistrement (1 000 tirages) pour l'AP ; bootstrap apparié entre modèles ; McNemar sur les détections par positif. « A meilleur que B » n'est écrit que si l'intervalle apparié exclut zéro.

**Benchmark contre l'existant.** Méthodes actuelles `[HYPOTHÈSE H10]` : écoute et spectrogrammes (Audacity, Raven), éventuellement Kaleidoscope ou BirdNET-Analyzer. Sur le jeu gelé : (a) template matching ; (b) détecteur en bande seul ; (c) embeddings BirdNET + sonde, référence non déployable ; (d) pipeline retenu — AP, rappel à précision fixée, fausses alarmes/h. Phase 4 : test de temps de détection avec deux naturalistes, avec et sans l'outil ; indicatif.

**Démontrer que la boucle améliore quelque chose.** Courbe d'apprentissage : tête ré-entraînée avec les labels cumulés après chaque tour, AP sur le jeu gelé en fonction du nombre de labels ; ablation à budget égal en rejouant hors ligne un tirage aléatoire de même taille dans le pool de labels acquis. La boucle fonctionne si la courbe active domine la courbe aléatoire et si l'AP croît au-delà du bruit de l'intervalle.

---

## 7. Outil livrable

| Option | Taille indicative | Coût | Remarques |
|---|---|---|---|
| A. Bundle Python (PyInstaller ou équivalent) + ONNX Runtime + GUI | 300–600 Mo | 2–3 sem. + une construction par OS | Hors ligne, ni TensorFlow ni PyTorch à l'exécution ; conversion ONNX à valider par encodeur |
| B. Idem avec PyTorch CPU | +300–800 Mo | 2–3 sem. | Conversion évitée, taille et démarrage pénalisés |
| C. Electron/Tauri + service Python | 400–700 Mo | ≥ 4 sem. | Plus soigné, hors budget temps |
| D. Dossier Python portable + script de lancement | 300–600 Mo | 1 sem. | Repli minimal si A échoue |

Poids embarqués : Bird-MAE-Base 92,9 M paramètres ≈ 370 Mo en float32, ≈ 185 Mo en float16 ; encodeur BEATs comparable `[À VÉRIFIER]` ; ProtoCLR nettement plus petit `[À VÉRIFIER]` ; Perch 2.0 ≈ 12 M paramètres ≈ 50 Mo si une voie CPU existe. Recommandation (jugement) : option A avec ONNX Runtime, et une expérimentation de packaging de deux jours en S13 pour choisir la GUI entre Gradio (composants audio natifs, proposition de l'encadrant A), NiceGUI (packaging documenté) et PySide6 (natif) ; Streamlit convient au prototype de recherche dès S3, son packaging est plus incertain `[À VÉRIFIER]`. Test sur une machine Windows de l'ONF `[HYPOTHÈSE H6]` en S14 et S20 (pas de construction croisée : machine réelle ou virtuelle nécessaire). HuggingFace distribue les poids en phase de recherche, jamais à l'exécution.

**Interface.** *Analyser* (dossier, reprise idempotente par empreinte de fichier, progression) ; *Vérifier* (file triée, spectrogramme 2–8 kHz avec repère de durée de note, boutons blanci / congénère / orthoptère / autre / incertain, raccourcis clavier) ; *Résultats* (tableau par point et période, statut, notes validées, export CSV) ; *Modèle* (versions, mise à jour de la tête, seuil).

**Mise à jour chez l'utilisateur.** Tête : ré-entraînée dans l'application (§5). Encodeur : paquet de modèle (`manifest.json` : nom, version, empreinte SHA-256, licence, fréquence, durée de fenêtre) déposé dans `models/` ; manifeste inconnu refusé, migration du §4 proposée.

**Documentation minimale viable.** README d'installation ; « Interpréter un score » (2 pages) ; « Vérifier une file » (1 page) ; guide du mainteneur (ré-entraîner, remplacer l'encodeur, exporter, sauvegarder) ; CHANGELOG ; `LICENSES.md` ; cartes des modèles embarquées.

**Licences.** Tableau composant → licence → obligations (attribution, partage à l'identique, non commercial). Règle du projet, sans avis juridique : tout composant redistribué doit pouvoir l'être par l'ONF dans son contexte réel, ce que l'ONF confirme par sa voie administrative avant la fin de S12, départ du packaging. BirdNET (CC BY-NC-SA 4.0) exclu du bundle par défaut ; Perch 2.0 (Apache 2.0) compatible ; Bird-MAE, BEATs, NatureLM-audio et ProtoCLR à relever sur leurs cartes de modèle en S1 `[À VÉRIFIER]`. À poser plutôt qu'à trancher : le statut d'une tête entraînée sur les embeddings d'un modèle non commercial.

---

## 8. Planning (26 semaines, S1 = 14–18/09/2026)

| Phase | Semaines | Objectif | Livrable vérifiable | Go / no-go à la réunion hebdomadaire de fin de phase |
|---|---|---|---|---|
| P0 Cadrage et benchmark | S1–S2 (15/09–25/09) | Inventaire, jeu annoté v0, benchmark bacpipe, questions du §10 | Tableau de benchmark ; inventaire ; hypothèses mises à jour | ≥ 1 encodeur exécutable localement (natif ou ONNX) avec AP ≥ 0,3 et rappel ≥ 0,8 à précision ≥ 0,1 en validation par site ; sinon DSP + template matching en primaire |
| P1 Pipeline v0 et amorçage | S3–S6 (28/09–23/10) | Pipeline CLI bout en bout (heures de pic du jeu 1), tours 0–1, expérience d'échelle (§1), calibration expert | Dépôt Git ; jeu annoté v1 (≥ 100 positifs validés, ≥ 1 000 négatifs dont ≥ 300 durs) ; note sur l'échelle | ≥ 60 positifs validés sur ≥ 3 sites ; sinon risque 1 |
| P2 Apprentissage actif et évaluation | S7–S12 (26/10–04/12) | Tours 2–5, negative mining, jeu gelé, post-traitement séquentiel, choix d'encodeur, position ONF sur les licences | Jeu gelé documenté ; courbes d'apprentissage ; note d'arbitrage encodeur | Rappel ≥ 0,85 à précision ≥ 0,1 sur sites tenus à l'écart ; sinon pivot (§9). **Fin S12 = date limite de pivot méthodologique** |
| P3 Outil v1 | S13–S18 (07/12–15/01 ; S15–S16 réduites `[HYPOTHÈSE H17]`) | Export ONNX, GUI, packaging, test sur machine ONF | Application installable testée sur deux machines ; guide utilisateur v0 | Tourne hors ligne sur une machine ONF et analyse 1 h d'audio en moins de 15 min (jugement) ; sinon repli option D |
| P4 Validation et généralisation | S19–S22 (18/01–12/02) | Jeu 2, test utilisateurs, démonstration de la boucle, robustesse, documentation | Rapport d'évaluation ; application v1.1 ; documentation complète | Gel des résultats fin S22 |
| P5 Rapport et soutenance | S23–S26 (15/02–12/03) | Rédaction, soutenance, transfert | Rapport de stage `[HYPOTHÈSE H11]` ; support de soutenance ; dépôt transféré | — |

Rédaction continue dès S17 (une demi-journée par semaine). Chaque réunion hebdomadaire reçoit avancement, chiffre clé de la semaine (positifs validés, AP sur jeu gelé) et décision demandée.

**Chemin critique** : accès aux données (S1) → benchmark (S2) → positifs validés (S3–S6) → jeu gelé (S8) → choix d'encodeur (S12) → export et packaging (S13–S16) → test utilisateurs (S19–S20) → gel (S22). Marge : une semaine en P2, une en P4. Un retard de plus d'une semaine sur les données comprime P3–P4 et ferait sauter le test utilisateurs — d'où la demande d'accès avant le 15/09 (§10). Après S12, changer de famille de méthode n'est plus possible ; après S16, changer d'encodeur non plus.

---

## 9. Risques et plans B

| Risque | Signal d'alerte précoce | Seuil de déclenchement | Repli concret |
|---|---|---|---|
| 1. Trop peu de positifs | Rendement faible du tour 0 ; positifs concentrés sur un site | < 60 positifs validés sur ≥ 3 sites fin S6 | (a) Enregistrements de l'étude de phénologie `[HYPOTHÈSE H7]` ; (b) détecteur DSP + template matching en primaire ; (c) augmentation par mixage des notes validées dans des fonds sonores de sites, entraînement seulement ; (d) livrable = outil de tri par similarité sans tête entraînée |
| 2. Confusion non résolue avec les congénères | > 30 % des 100 meilleurs candidats sont des congénères après le tour 3 | Précision < 0,1 à rappel 0,85 sur sites riches en congénères fin S10 | Fusion embedding + structure séquentielle (§3) ; jeu de négatifs congénères dédié ; sortie hiérarchique « *Anomaloglossus* sp. » avec avertissement si l'espèce reste indécidable |
| 3. Décalage acoustique entre les deux jeux | AP sur un audit du jeu 2 (300 fenêtres aléatoires + 300 meilleures) très inférieure | AP jeu 2 < 50 % de l'AP en validation par site, mesurée S19 | Recalibration des seuils par jeu ; négatifs durs du jeu 2 dans l'entraînement en gardant une partie à l'écart ; normalisation par enregistreur |
| 4. Volume ingérable | Débit mesuré en S1 | Encodage complet projeté > 10 jours-machine | Priorité heures de pic et mois humides (÷ 12), pas plus grossier, encodeur plus léger, lots nocturnes ; extraction déportée si une machine institutionnelle existe `[À VÉRIFIER]` |
| 5. Fréquence d'échantillonnage ou format incompatibles | Inventaire S1 | f_e native < 11 kHz (improbable) ou format propriétaire | Rééchantillonnage dans les wrappers (bacpipe) ; conversion ffmpeg/sox ; compression avec pertes tolérée sous 8 kHz (jugement) |
| 6. Pas de variante CPU de Perch 2.0 | Échec des replis (1)–(2) du §2 | Non exécutable localement fin S2 | Perch 2.0 comme référence déportée seulement ; déploiement sur Bird-MAE, BEATs ou ProtoCLR ; interface `Encoder` prête pour une version CPU ultérieure |
| 7. Expert indisponible | Calibration non planifiée fin S3 | Aucun label validé fin S5 | Politique stricte « incertain » ; validation asynchrone par lots de 50 extraits ; double écoute |
| 8. Perte de la machine ou des données | Machine unique | — | Sauvegarde quotidienne sur SSD externe, copie hebdomadaire sur stockage ONF ; labels versionnés (Git) |

---

## 10. Questions aux encadrants en semaine 1 (par priorité)

1. « Quels enregistreurs, quelle fréquence d'échantillonnage, quel format et quelle convention de nommage ? » — Conditionne décodage, rééchantillonnage et métadonnées site/horodatage.
2. « Puis-je avoir les données dès la première semaine, et sur quel support ? » — Le chemin critique commence là.
3. « D'où viennent les cris labellisés : quels sites, combien d'individus, quel enregistreur, sous quelle forme ? » — Décide si une validation par site est possible.
4. « Les deux jeux partagent-ils des sites, des enregistreurs, des saisons ? Lesquels ont une présence confirmée ? » — Fixe le protocole de généralisation et le test de décalage de domaine.
5. « Les enregistrements de l'étude de phénologie de 2025 sont-ils accessibles, avec les instants des détections ? » — Peut multiplier les positifs et lever le risque 1.
6. « Sur les 16 kHz : la fondamentale est à 4,75 kHz, sous le plafond de 8 kHz d'un signal à 16 kHz ; l'objection porte-t-elle sur les harmoniques, sur la résolution temporelle, ou sur les indices haute fréquence utiles au rejet des orthoptères ? » — Décide si BEATs et NatureLM restent dans le benchmark (contrôle passe-bas du §2).
7. « Quelle décision de gestion découle d'une présence, à quelle échelle, et quel taux de sites manqués est acceptable ? » — Fixe le point de fonctionnement et la règle (k, n).
8. « Qui vérifiera les candidats, combien de temps par semaine, avec quelle séance de calibration ? » — Fixe le budget d'annotation.
9. « Quel taux d'émission et quels intervalles entre notes chez *A. blanci*, et existe-t-il des enregistrements des congénères ? » — Conditionne le post-traitement séquentiel et le negative mining.
10. « Quelles machines chez les utilisateurs finaux (OS, mémoire), et une connexion est-elle possible pour un téléchargement initial ? » — Conditionne le packaging.
11. « Qui tranche la question des licences à l'ONF, et à quelle échéance ? » — Filtre les encodeurs avant le packaging (fin S12).
12. « Quels outils sont utilisés aujourd'hui pour dépouiller les enregistrements ? » — Définit la baseline du benchmark.
13. « De nouvelles données de terrain arriveront-elles pendant le stage, et quand ? » — Décide si la boucle continue peut être testée en conditions réelles.

---

## 11. Montée en compétence

**Traitement du signal (S1–S4, ≈ 14 h).** Dans l'ordre : échantillonnage, théorème de Nyquist et rééchantillonnage avec anti-repliement ; transformée de Fourier à court terme et spectrogramme, compromis fenêtre/pas — pour une note de 90 ms, fenêtre d'analyse de 16–25 ms et pas ≤ 10 ms, résolution fréquentielle de l'ordre de 50 Hz ; filtrage passe-bande, enveloppe, détection d'onsets, rapport signal/bruit ; normalisation (log, énergie par canal) ; corrélation croisée et template matching. Ressources : documentation de scikit-maad et librosa ; Smith, *The Scientist and Engineer's Guide to Digital Signal Processing*, libre en ligne `[À VÉRIFIER]` ; Müller, *Fundamentals of Music Processing*, chapitres STFT et onsets `[À VÉRIFIER]`.

**Écologie des anoures (S1–S6, ≈ 18 h).** Vocabulaire : note, pulse, cri, fréquence dominante contre fondamentale, modulation, taux d'émission et sa dépendance à la température — donc variation attendue des notes, à intégrer dans les filtres ; le genre *Anomaloglossus*, sympatrie, habitat ; phénologie et plan d'échantillonnage ; occupation et probabilité de détection. Ressources : Fouquet et al. 2018 lu intégralement en S1 (3 h) ; Courtois et al. 2025 (2 h) ; le document du Plan national d'actions `[À VÉRIFIER]` ; Köhler et al. 2017 sur la bioacoustique en taxonomie des anoures `[À VÉRIFIER]` ; séances d'écoute avec l'expert (3 h) et écoute autonome de références, 2 h/sem. en S2–S4.

**Placement.** Première heure de chaque journée en S1–S4, puis 1 h/sem. : ≈ 40 h sur six semaines. Rien n'est placé sur le chemin critique après S6.

---

## 12. Hypothèses formulées

| Id | Hypothèse | Question qui la lève |
|---|---|---|
| H1 | Enregistreurs autonomes (AudioMoth, Song Meter), WAV PCM mono, fréquence native ≥ 32 kHz | §10 Q1 |
| H2 | Cris labellisés = instants ou extraits provenant de 1 à 3 sites et d'un seul type d'enregistreur | §10 Q3 |
| H3 | Les deux jeux sont disjoints en sites et diffèrent par saison ou enregistreur | §10 Q4 |
| H4 | Positions des encadrants reconstituées comme en tête de document | Relecture des extraits verbatim par les intéressés |
| H5 | Décision gestionnaire = intégration de la présence dans la planification forestière à l'échelle du point ou du bassin de crique | §10 Q7 |
| H6 | Utilisateurs sur portables Windows sans GPU, 8–16 Go, connexion possible au bureau pour un téléchargement initial | §10 Q10 |
| H7 | Les enregistrements de l'étude de phénologie 2025 existent et sont partageables | §10 Q5 |
| H8 | *A. blanci* absent de l'espace de labels de Perch 2.0 | Liste de classes, S1 |
| H9 | Intervalles entre notes d'*A. blanci* de l'ordre de la seconde, plusieurs notes par fenêtre de 5 s | §10 Q9 |
| H10 | Dépouillement actuel par écoute et spectrogrammes ; MacBook Air à 16 Go | §10 Q12 ; vérification locale |
| H11 | Rapport remis vers la fin du stage, soutenance à IMT Atlantique en mars ou avril 2027 | Calendrier école |
| H12 | Un expert disponible 3 h puis 1 h toutes les deux semaines | §10 Q8 |
| H13 | Au moins un rapatriement de données de terrain avant février 2027 | §10 Q13 |
| H14 | Des enregistrements de référence des congénères sont accessibles | §10 Q9 |
| H15 | Saison des pluies principale de décembre à juillet, saison sèche d'août à novembre `[À VÉRIFIER]` ; le stage commence en saison sèche | Calendrier ONF |
| H16 | Volume total des deux jeux de l'ordre de quelques milliers d'heures | §10 Q2 |
| H17 | Deux semaines réduites autour des fêtes de fin d'année | Calendrier ONF et école |

---

## Angles morts

1. **Le plan d'échantillonnage peut être la contrainte dominante, pas le modèle.** À 2 min par heure, la probabilité de capter un cri sur un site à faible densité reste faible quel que soit le rappel ; cette feuille de route optimise le classifieur alors qu'un enregistrement continu sur les heures de pic changerait davantage la donne.
2. **Pseudo-réplication.** Si les positifs viennent d'un ou deux sites, toute affirmation de généralisation repose sur presque rien, et le risque que le modèle apprenne un paysage sonore plutôt qu'une note n'est pas mesurable avant la phase 4.
3. **La vérité terrain est un jugement humain non mesuré.** L'accord entre experts sur *A. blanci* contre congénères n'est estimé que sur 100 fenêtres ; s'il est faible, aucune métrique du §6 n'a de sens.
4. **Biais phénologique induit.** L'échantillonnage guidé par les heures de pic entraîne le modèle sur les conditions de pic ; les 20 % hors strates sont un palliatif, pas une garantie, et la phénologie vient d'une étude dont la portée spatiale est inconnue.
5. **Après le 14 mars.** La boucle continue suppose un effort de vérification hebdomadaire que l'ONF n'a pas promis ; sans lui, l'outil se dégrade silencieusement, et personne n'est désigné pour maintenir le code.
6. **Présence/absence sans modèle d'occupation.** La règle (k, n) est une heuristique ; la formulation statistique correcte de la non-détection — modèles d'occupation avec probabilité de détection `[À VÉRIFIER]` — est hors périmètre alors qu'elle est ce que la décision gestionnaire devrait utiliser.
7. **Transfert oiseaux → anoure non garanti, attentive probing probablement hors de portée.** Les benchmarks de la revue sont majoritairement aviaires, la couverture amphibienne de Perch 2.0 n'est pas quantifiée ici, et le test de la semaine 1, avec 30 positifs, est peu puissant : le document tranche sur un benchmark dont il reconnaît la faiblesse. L'attentive probing requiert un effectif peut-être inatteignable.
8. **Sur-ingénierie.** Index, versionnage, migrations, paquets de modèles : sur six mois, le temps d'infrastructure est retiré à l'annotation et à l'évaluation, seuls leviers réels avec si peu de positifs.
9. **Valeur pour l'ONF non mesurée.** Le benchmark mesure des AP ; la valeur réelle est le temps de naturaliste économisé par site détecté, estimée sur un test à deux personnes.
10. **Variation intraspécifique.** Les paramètres acoustiques proviennent d'une description fondée sur peu d'individus ; la dépendance à la température et à la taille peut déplacer durée et fréquence hors des filtres proposés, que le document fixe pourtant assez étroitement.
