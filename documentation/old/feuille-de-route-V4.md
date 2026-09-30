# Feuille de route — Stage ONF Guyane
## Détection acoustique automatique d'*Anomaloglossus blanci* — 15/09/2026 → 14/03/2027

Version 3 (18/09/2026). `[HYPOTHÈSE Hn]` = information encore manquante (liste en §12) ; `[À VÉRIFIER]` = à contrôler sur les données ou les sources ; « établi » = bibliographie fournie (dont Courtois et al. 2025, *pheno-blanci.pdf*) ou source vérifiée ; « jugement » = arbitrage d'ingénieur ; « tuteur » = confirmé en réunion. Le §13 est écrit pour un agent de code.

**Journal v2 → v3**
- Quinze hypothèses levées ou requalifiées (§12) ; positions des encadrants confirmées par les extraits.
- Données réelles : Song Meter Mini 2 partout ; 345 positifs = 345 enregistrements distincts, 13 micros, tous à Mataroni ; 158 négatifs d'espèces identifiées ; 2026 = Kaw 45 micros, Trésor 40, Mataroni > 70 ; pas de semaine réduite.
- Renversement du protocole de généralisation : Mataroni est le site d'entraînement, Trésor et Kaw les sites tenus à l'écart, 2023 le test temporel (§6).
- Phénologie établie par Courtois et al. 2025 : pics 7–9 h et 15–17 h, activité forte de janvier à avril, quasi nulle de juillet à octobre (§1, §5).
- Baseline = écoute humaine ; détecteur Biophonia 2024 (rappel ≈ 0,69, précision 1) conservé comme point de repère chiffré, modèle non disponible (§6).
- Trois congénères dans les classes de Perch 2.0 → générateur de candidats (§3).
- Machines cibles : portables Windows i5-1145G7, 16 Go ; l'ONF réentraîne après le stage (§7).
- ProtoCLR rétrogradé (son auteur ne le recommande pas) ; NatureLM/BEATs maintenu dans le benchmark avec le contrôle passe-bas (§2).
- §13 ajouté : spécification d'implémentation pour Claude Code.

**Positions des encadrants (établi, extraits reçus)**
- Tuteur (B) : idée initiale = embeddings BirdNET avec entraînement progressif *human in the loop* pour ajouter des sites ; BirdNET et Perch + linear probing lui ont donné satisfaction ; ouvert à d'autres modèles ; suggère bacpipe ; NatureLM « limité à 16 kHz, très limite pour les paysages sonores ».
- Encadrant A (auteur de ProtoCLR) : éviter BirdNET (TensorFlow, dépendance à la base de code de Cornell, extraction d'embeddings peu pratique ; seule l'interface graphique vaut, pour non-codeurs) ; modèles HuggingFace ; d'après la revue d'*Ecological Informatics* (Schwinger et al. 2026), BEATs_NLM et BirdMAE au-dessus du lot ; ProtoCLR pas loin mais pas facile à réutiliser, non recommandé ; Streamlit ou Gradio pour la réannotation.
- Arbitrage : aucun TensorFlow à l'exécution dans le livrable ; benchmark bacpipe sur `birdmae`, `beats`, `naturebeats`, `perch_v2`, `birdnet` (référence du tuteur, non déployable : licence, TensorFlow) ; `protoclr` seulement s'il tourne sans effort ; 16 kHz tranché par le contrôle passe-bas (§2).

---

## 1. Reformulation du problème

**Cadre (tuteur)** : détecteur robuste d'*A. blanci*, fonctionnant sur de nouveaux sites ; métriques AP et rappel ; six mois, un Macbook puce M4 16 giga de RAM portable, logiciel prenable en main par un naturaliste ; comparaison des approches par score, vitesse, prise en main.

**Données (établi et tuteur)**
- Enregistreurs : Wildlife Acoustics Song Meter Mini 2 partout ; fréquence d'échantillonnage et format lus dans les en-têtes à l'inventaire `[À VÉRIFIER]`.
- Jeu 2023 (établi : Courtois et al. 2025) : 6 enregistreurs, 2 par site (Kaw_A/B, Molokoï_E/F, Trésor_C/D), du 25/11/2023 au 24/11/2024, 2 min toutes les 30 min de 5 h à 20 h, 4 705 à 5 490 créneaux horaires par enregistreur ≈ 2 200 h au total ; capteurs de température et d'humidité associés.
- Jeu 2026 : Kaw 45 micros et Trésor 40 micros pendant une semaine, Mataroni > 70 micros pendant 5 jours ; 2 min toutes les 30 min de 5 h à 19 h 30 (29 créneaux/jour) ≈ 27 400 enregistrements ≈ 910 h.
- Total ≈ 3 100 h ≈ 3,7 millions de fenêtres de 3 s (convention Biophonia : 40 par enregistrement) ou 4,5 millions à 5 s, pas 2,5 s.
- Annotations (avec le tuteur, expert naturaliste) : 345 fenêtres positives de 3 s, chacune dans un enregistrement de 2 min distinct, 13 micros, **toutes à Mataroni pour l'instant**, ajout de données annotées d'autres sites dans le futur ; commentaires de qualité (« second plan », « malgré la pluie », « lointain », « avec fourmilier tacheté ») ; 158 fenêtres de faux amis avec espèce identifiée. Origine : détections du modèle Biophonia confirmées à la main, certaines à score médiocre.
- Conséquence 1 : effectif effectif = 345 enregistrements, ce qui autorise le probing linéaire dès S3.
- Conséquence 2 : le jeu étiqueté est **conditionné par le détecteur précédent** — les cris qu'il n'a pas vus n'y sont pas. Le rappel mesuré dessus est un rappel relatif à ce que Biophonia trouvait. Un audit aléatoire indépendant est indispensable (§6).
- Conséquence 3 : aucun positif étiqueté hors Mataroni. La généralisation à de nouveaux sites se mesure sur Trésor et Kaw, après récolte et validation de positifs (§5, §6).

**Phénologie (établi : Courtois et al. 2025)**
- Journalier : pics 7–9 h et 15–17 h à Kaw et Trésor ; à Molokoï, activité haute et constante de 7 h à 17 h en saison.
- Annuel : activité forte de janvier à avril, jusqu'en juin à Trésor et Molokoï ; quasi nulle de juillet à octobre ; reprise en novembre–décembre avec les premières pluies.
- Détection : à Molokoï, probabilité journalière ≈ 1 de fin novembre à mars ; à Trésor, maximale en décembre puis ≈ 0,5 jusqu'en juin ; Kaw_B ne dépasse jamais 0,5 ; ≈ 0 partout de juillet à octobre.
- Stage : septembre–novembre = saison basse ; les enregistrements 2026 de Kaw/Trésor/Mataroni ont été faits entre janvier et avril.
- Chant continu (tuteur) : présente, elle chante du début à la fin des 2 min ; l'indice horaire du rapport 2025 atteint rarement 1, mais ce chiffre mélange silences réels et rappel du détecteur (0,69) : H20 reste ouverte.

**Problème d'apprentissage**
- Espèce à note brève mais à chant continu, dans une bande saturée, avec ≈ 50 sources de fausses alarmes recensées (oiseaux, amphibiens, orthoptères, artefacts).
- Détail d'une étude sur la structure du chant de Blanci et d'autres congénères : Anomaloglossus Blanci : single note call of 0.090– 0.103s length (X¯ =0.094s ) with a slight upward modulation (ca. 0.1 kHz) and dominant frequency at 4.48–5.41 kHz (X¯ = 4.75 kHz) at a regular pace (inter-note interval X¯ = 1.414 s; range 1.200– 1.906 s). The spectral structure of the note has a developed harmonic structure. Males call from stream banks during the day with peak in intensity at dawn (6–7 am) and late afternoon (4–5 pm) during the rainy season, but also during humid days of the dry season.

tonal note X¯ =0.032, range 0.028– 0.037 s in A. surinamensis
intervals X¯ =0.573, range 0.372– 0.825 s in A. surinamensis
Frequency X¯ =4.89 kHZ, range 4.55–5.35 kHz in A. surinamensis vs. 3.60– 3.62 s in A. degranvillei
call with longer notes 0.157– 0.160 s in A. degranvillei
- Classement de fenêtres sous fort déséquilibre → décision par enregistrement → classement des points ; vérification humaine structurelle ; rappel prioritaire.

**Unité de décision (jugement)**

| Niveau | Rôle | Règle |
|---|---|---|
| Fenêtre (3 s ou 5 s selon l'encodeur, pas ≤ moitié) | Apprentissage, score, vérification | Métriques de développement |
| Enregistrement (2 min) | **Unité principale** ; score = proportion de fenêtres où A. Blanci est détectée | Vérifier manuellement les scores faibles, chance de faux positif ; détection sur des fenêtres isolées → file « suspect » |
| Point × période | Décision gestionnaire `[HYPOTHÈSE H5]` (règle du stagiaire) | « À vérifier » dès qu'un enregistrement est positif ; points classés par force (fraction, nombre d'enregistrements, jours) ; « présence confirmée » après validation humaine ; sinon « non détecté », jamais « absent » |

**Écart métrique / décision** : fausse absence irréversible, fausse présence réversible → rappel ; chant continu → un créneau en saison suffit, la contrainte est la **saison**, pas le cycle utile ; la charge de vérification n'est bornée que par la précision plancher et le classement (§6).

**Échelle (note de 0,09 s contre fenêtre de 3–5 s)** : fenêtres pleines par défaut. Contre le détecteur amont : bande saturée (tuteur), rappel plafonné, encodeurs et revue évalués sur fenêtres pleines ; pour : mesure directe de durée et d'intervalles. Le seuillage spectral ne sert qu'aux onsets du module séquentiel. Protocole de vérification (S3–S5, plis par micro) : A grille standard ; B pas resserré ; C fenêtres centrées sur onsets ; E agrégation des tokens (moyenne / max / attention) ; C adopté seulement s'il dépasse A/B de plus que l'incertitude.

---

## 2. Benchmark initial des embeddings (S1–S3)

- Colonnes : licence ; exécution sur M4 ; vitesse sur M4 et **projection sur i5-1145G7** ; prise en main ; tokens accessibles ; AP ; rappel à précision 0,5 et 0,1 ; kNN top-1.
- Pré-benchmark AnuraSet (S2, établi : Cañas et al. 2023) : deux ou trois anoures à note brève en 3–6 kHz, plis par site, milliers de positifs → classement des encodeurs sur anoures en paysage sonore néotropical. Indicateur, pas garantie.
- Jeu annoté v0 : 345 positifs + 158 négatifs d'espèces + négatifs **appariés** (mêmes micros de Mataroni, mêmes heures, sans chant), ratio 1:20 à 1:50.
- Extraction bacpipe : `birdmae`, `beats`, `naturebeats`, `perch_v2`, `birdnet` ; `protoclr`, `perch_bird`, `convnext_birdset` si sans effort. Relevé : débit, mémoire, dimension, tokens.
- Sondes : kNN cosinus ; prototype différentiel ; régression logistique L2. Validation **groupée par micro** dans Mataroni (13 micros positifs + micros sans détection). Comparaisons appariées sur les mêmes plis, bootstrap par enregistrement.
- Générateur de candidats : Perch 2.0 contient *A. baeobatrachus*, *A. stepheni*, *A. surinamensis* (levé H8). Leurs logits, non calibrés (établi), servent de première liste de candidats sur les données non étiquetées et de descripteurs optionnels.
- Trompeuses : exactitude ; AUROC ; F1 au seuil 0,5 ; AMI/ARI. Moins de 0,1 d'AP d'écart = égalité, départagée par licence, vitesse, prise en main.
- `perch_v2` (TensorFlow, GPU) : (1) wrapper CPU `[À VÉRIFIER]` ; (2) `perch_v2_no_dft.onnx` repéré dans un notebook BirdCLEF+ 2026, à valider contre des embeddings de référence ; (3) extraction déportée pour le benchmark seul ; (4) variante CPU officielle si elle sort. `naturebeats` : encodeur seul.
- Désaccords → mesures : AP(birdnet) contre les autres ; AP(beats/naturebeats) à 16 kHz contre AP(birdmae) à 32 kHz **avec contrôle** birdmae sur audio filtré à 8 kHz — si le contrôle égale birdmae natif, l'objection du tuteur ne s'applique pas à ce signal ; AP(perch_v2) sur AnuraSet puis ONF.
- BirdCLEF+ 2026 (vérifié) : working notes après CLEF (21–24/09) ; transférable : pseudo-étiquetage, lissage temporel, focal → paysage, calibration.
- Sortie : benchmark compelt des encodeurs, avec projection de coût sur la machine cible.

---

## 3. Cartographie des approches

| Approche | Principe | Annotations | Décalage de domaine | Verdict |
|---|---|---|---|---|
| Template matching | Corrélation croisée de spectrogrammes (scikit-maad `[À VÉRIFIER]`) | 1–10 | Faible | Baseline obligatoire ; conserve la note ; indépendant de l'encodeur |
| Seuillage spectral | Passe-bande 4,4–5,5 kHz, enveloppe, seuil, durée 0,08–0,11 s | 0 | Faible (bande saturée) | Onsets pour le module séquentiel ; pas de filtre amont |
| Indices acoustiques | Statistiques par enregistrement | 0 | — | Contrôle qualité : pluie, saturation, « micro dans sac » |
| Prototype simple | Cosinus au centroïde des positifs | 5–30 | Sensible au fond partagé | Baseline de similarité |
| Prototype différentiel | $\langle x, \mu_+ - \mu_-\rangle + b$, négatifs appariés | 10–50 | Retire le fond partagé | Baseline permanente ; diagnostic du fond capté |
| Linear probing | Régression logistique L2 sur embeddings gelés de modèle de foundation (via bacpipe)| Dizaines → centaines | Selon l'encodeur | **Principale dès S3** (345 enregistrements positifs) |
| Attentive probing | Tête d'attention sur tokens pris avant agrégation, même couche que l'embedding | ≥ 150–200 | Idem | P2 si tokens accessibles (établi : nécessaire pour les transformers) |
| Logits de congénères (Perch 2.0) | Scores des 3 *Anomaloglossus* connus | 0 | Inconnu | Générateur de candidats ; descripteur optionnel de la fusion |
| Clustering | HDBSCAN sur ACP ; UMAP pour voir | 0 | — | §5 bis ; détecteur seulement si C1 réussit |
| LoRA / fine-tuning | Adaptation partielle ou totale | Centaines à milliers ; AnuraSet | Sur-apprentissage au site | Conditionné ; hors chemin critique |
| Distillation (modèle maison) | Petit CNN bande 3–7 kHz imitant le pipeline gelé | 0 | Hérite de l'enseignant | Livrable léger pour l'i5 ; S13 au plus tôt |

**Prototype différentiel** : $x \approx c_{\text{site}} + c_{\text{espèce}} + \varepsilon$ ; $w = \mu_+ - \mu_- \approx c_{\text{espèce}}$ avec négatifs **appariés** (mêmes micros, heures, jours) ; centroïde le plus proche sous variance commune = analyse discriminante linéaire à covariance identité = solution fermée de la régression logistique ; première instance de l'arithmétique d'embedding (tuteur, §11).

**Module séquentiel** (descripteurs calculés depuis l'audio, hors encodeur)
- Rythme intra-fenêtre : débuts de notes (onsets) en bande ; la structure du chant est connue
- Persistance : fraction de fenêtres positives dans l'enregistrement ; continuité entre fenêtres voisines ; présence dans les créneaux voisins du même micro. Sépare chant continu et chant ponctuel (fourmilier tacheté), et isole les détections uniques.
- Solo contre chœur `[HYPOTHÈSE H21]` : densité d'onsets, chevauchements, étalement spectral.
- Heure et saison : **hors classifieur par défaut** (risque d'apprendre la phénologie de Mataroni) ; utilisées pour l'échantillonnage, le classement des points et un drapeau de plausibilité issu des courbes de Courtois et al. 2025 ; entrée du classifieur seulement si validée sur Trésor et Kaw.

**Tête de fusion (stacking à deux niveaux)** : `head` et `sequential` au niveau 1 ; régression logistique au niveau 2 sur (score de `head` **hors-pli**, 2–4 descripteurs, logits de congénères en option). Hors-pli = score produit par une version de `head` entraînée sans l'exemple ; les versions par pli fabriquent une colonne, elles ne sont pas empilées entre elles. ≈ 10 enregistrements positifs indépendants par coefficient. Le score séquentiel module, jamais de veto.

---

## 4. Architecture

```
audio brut (Song Meter Mini 2 ; f_e et format lus à l'inventaire)
  └─ ingest      inventaire, métadonnées (jeu, site, micro, horodatage), drapeaux QC → SQLite
  └─ decode      forme d'onde float32 mono, f_e native (soundfile)
  └─ grid        fenêtres (recording_id, offset_s, dur_s), indépendantes de l'encodeur
  └─ encoder[k]  rééchantillonnage vers f_e(k) ; spectrogramme interne ; E_k ∈ R^{N×d_k}
  └─ store       Parquet partitionné encoder_id / jeu / site / mois, float16 (audio intact)
  └─ index       cosinus exhaustif par fragments ; requêtes positives et négatives empilées
  └─ head[v]     régression logistique → score par fenêtre
  └─ sequential  rythme, persistance, solo/chœur → fusion
  └─ aggregate   fenêtre → enregistrement (fraction) → point (classement)
  └─ queue       file de vérification → labels append-only → réentraînement de head
```

- `decode` s'arrête à la forme d'onde ; le log-mel est calculé dans chaque encodeur.
- Découplage : labels attachés à (enregistrement, décalage) ; encodeur derrière un `Protocol` ; GUI → couche de service.
- Volume : ≈ 4 millions de fenêtres → 6 Go en 768-d float16, 12 Go en 1536-d ; SSD externe.
- Index : une multiplication $n \times d$ par $d \times N$ ; secondes sur MPS ; ajout seul.
- Changement d'encodeur : ré-encodage en tâche de fond ; tête réentraînée sur les mêmes labels ; rapport de migration sur le jeu gelé ; décisions estampillées (`encoder_id`, `head_version`, `threshold_id`).
- Croissance : encodeur figé ; index en ajout ; tête réentraînée de zéro sur tous les labels (réentraînement périodique, pas apprentissage continu au sens technique) ; seuils recalibrés. Détail d'implémentation en §13.

---

## 5. Annotation et apprentissage actif

**Point de départ** : 345 positifs et 158 négatifs (Mataroni, 13 micros), commentaires libres à convertir en champs : espèce du faux ami, qualité (A/B/C selon Courtois et al. 2025 : A clair sans chevauchement, B chevauchement mais toutes caractéristiques visibles, C fort chevauchement ou lointain), conditions (pluie, fourmilier présent, second plan).

**Annotation par enregistrement (chant continu, sous H20)** : l'expert tranche sur 20 s si l'enregistrement de 2 min contient *A. blanci* ; toutes ses fenêtres héritent du label. Validation préalable : 20 des 345 enregistrements écoutés en entier (fraction de fenêtres réellement positives).

**Schéma de labels** (levé H14) : positifs {blanci-solo, blanci-chœur, blanci-incertain} ; négatifs par famille {oiseau:<espèce>, amphibien:<espèce>, orthoptère, cri-de-contact-amphibien, pluie, artefact:micro-dans-sac, fond, autre} ; espèces recensées : fourmilier tacheté (le plus fréquent), moucherolle, manakin, tangara mordoré, pigeon plombé, évêque de Rothschild, sclérures, myrmidon, psittacidés, pic à cou rouge, martinet ; *Adenomera andreae*, *Allobates femoralis*, *A. hahneli*, *Hyalinobatrachium cappellei / mondolfii / iaspidiense*, *Amazophrynella teko*, *Otophryne* ; grillons. Les noms scientifiques des oiseaux sont `[À VÉRIFIER]` avant publication.

**Apprentissage actif dès S3** : file = 60 % incertains, 20 % scores maximaux, 20 % aléatoire stratifié (micro, heure) — la strate aléatoire mesure les faux négatifs confiants, dont ceux que Biophonia n'avait jamais remontés. Lots de 30–50 enregistrements. Arrêt : gain d'AP sur le jeu gelé < 0,02 sur deux tours.

**Récolte hors Mataroni (S4–S9)** : requêtes de similarité (positifs Mataroni, prototype différentiel, logits des congénères Perch) sur Trésor et Kaw 2026, puis sur Molokoï 2023 en janvier–avril 7–17 h (probabilité de détection ≈ 1, établi) : positifs abondants et faciles à valider ; validation par le tuteur, sur place.

**Negative mining** : les 158 faux amis en point de départ ; négatifs durs par vérification des mieux classés ; négatifs faciles en volume par clustering (§5 bis) ; fichiers « micro dans sac » et pluie battante détectés en contrôle qualité et exclus de l'entraînement.

**Échantillonnage phénologique** (établi) : 7–9 h et 15–17 h en priorité (7–17 h à Molokoï) ; janvier–avril, puis novembre–décembre ; 20 % hors strates.

**Établi d'annotation** : YAPAT essayé en S2 (½ j, conditions : BirdNET contournable, mise en place ≤ 1 j) ; sinon Streamlit minimal (recommandation de l'encadrant A) ; Whombat en alternative. Outils de travail, pas livrable.

**Budget** : annotation « quand on a le temps » (tuteur, expert sur place) ; stagiaire ≈ 66 h ; calibration inter-annotateurs sur 100 fenêtres tout de même, pour chiffrer l'accord.

**Après le stage (levé H13)** : l'ONF réentraîne sur ses portables avec chaque nouvelle campagne de pose de micros, sur plusieurs années ; Exigences : procédure documentée, tête réentraînable sans GPU (une nuit de calcul est acceptable), ré-encodage par lots reprenable, comparaison automatique au jeu gelé avant bascule.

---

## 5 bis. Voie non supervisée : clustering

- Fonctions : négatifs en volume ; exploration ; détecteur seulement si C1 réussit.
- Méthode (jugement) : HDBSCAN sur ACP (≈ 50 composantes) ; UMAP pour visualiser.
- C0 (S3, 1 h) : clustering global d'un échantillon ; AMI entre groupes et micros ; anomalies (micro dans sac, saturation).
- C1 (S3, 1 j) : Mataroni, positifs connus + 5 000 fenêtres aux mêmes heures ; fenêtres pleines contre recadrées. Seuils (jugement) : rappel du meilleur groupe ≥ 0,5 ; enrichissement ≥ 20 ; AMI micro faible.
- C2 (S4–S7) : étiquetage en bloc des groupes homogènes après dix écoutes.
- C3 (S6–S9) : sub-clustering par micro sur les heures de pic ; récolte de positifs sur Trésor et Kaw.
- Limite : 2–3 % de fenêtre pour la note → groupes = paysages sonores ; le chant continu et le recadrage améliorent le rapport.

---

## 6. Évaluation

**Protocole de généralisation (renversé en v3)**
- Niveau 1, intra-site : validation groupée par micro dans Mataroni (5 plis sur ≥ 13 micros) ; AP et rappel par fenêtre et par enregistrement.
- Niveau 2, nouveaux sites : Trésor 2026 et Kaw 2026, positifs récoltés et validés (§5), jamais entraînés avant P4 ; c'est **la** mesure demandée par le tuteur.
- Niveau 3, temporel et matériel : jeu 2023 (autre année, autre saison, capteurs Song Meter mini de génération précédente `[À VÉRIFIER]`) ; validation sans étiquettes par **reproduction des patrons publiés** — notre détecteur, appliqué aux 2 200 h de 2023, doit retrouver les courbes journalières et annuelles de Courtois et al. 2025 (pics 7–9 h et 15–17 h, creux juillet–octobre, Kaw_B faible) ; un désaccord signale un problème de généralisation ou un artefact.
- Audit aléatoire indépendant (S4, 300 enregistrements de Mataroni tirés au hasard, écoutés en entier) : seule mesure du rappel non conditionnée par le détecteur Biophonia.

**Jeu gelé (S6–S9)** : 60 enregistrements de Mataroni écoutés en entier, stratifiés par micro et heure ; tous positifs validés de Trésor et Kaw ; qualité A/B/C, solo/chœur, SNR renseignés. Versionné, jamais entraîné.

**Métriques** : AP et rappel (tuteur) par enregistrement et par fenêtre ; rappel à précision ≥ 0,1 et ≥ 0,5 ; fausses alarmes par heure ; rappel par qualité A/B/C et par SNR estimé (énergie en bande pendant les notes contre 0,5 s voisines) ; accord du classement des points avec l'expert. Rejetées : exactitude, AUROC (annexe), F1 au seuil 0,5, kappa.

**Baseline et repères**
- Baseline = écoute humaine (levé H10) : protocole homme–machine sur un lot commun de 60 enregistrements (30 positifs, 30 négatifs difficiles, mélangés) : rappel, précision et temps d'un naturaliste à l'oreille ; rappel, précision et temps de l'outil suivi d'une vérification humaine des seuls candidats. C'est la comparaison principale du rapport, conçue dès P2, exécutée en P4 avec deux naturalistes.
- Repère chiffré : détecteur Biophonia 2024 (établi : Courtois et al. 2025) — réseau de neurones d'architecture non communiquée, ≈ 1 000 extraits annotés, test sur 450 : 0 faux positif, 24 faux négatifs, 53 vrais positifs, soit rappel ≈ 0,69 à précision 1. Modèle indisponible, jeu de test différent : comparaison indicative seulement, à la même précision.
- Méthodes reproductibles : template matching ; détecteur en bande ; BirdNET + sonde (référence, non déployable).

| Critère (tuteur) | Mesure | Comment |
|---|---|---|
| Score | AP, rappel par enregistrement, par qualité, par site | Plis par micro ; Trésor/Kaw tenus à l'écart ; bootstrap par enregistrement |
| Vitesse | Temps pour 1 h d'audio sur M4, puis sur i5-1145G7 | Même fichier ; encodage + tête ; CPU seul sur la cible |
| Prise en main | Installation, dépendances, taille, documentation | Grille à 5 points, stagiaire puis naturaliste en P4 |

- Rappel prioritaire ; précision plancher : file hebdomadaire ≤ 1 h à 10 s par candidat (jugement) ; cible rappel ≥ 0,9 au seuil le plus élevé gardant précision ≥ 0,1 sur Trésor et Kaw.
- Intervalles : Wilson en enregistrements (n = 30, rappel 0,9 → [0,74 ; 0,97] ; n = 345 → [0,86 ; 0,93]) ; bootstrap par enregistrement pour l'AP ; comparaisons appariées ; « A meilleur que B » seulement si l'intervalle apparié exclut zéro.
- Calibration : seuils sur scores hors-pli ; par site si décalage confirmé ; isotonique possible dès Mataroni (345 positifs).
- Combinaisons (décision du stagiaire) : concaténation d'embeddings et fusion séquentielle d'abord ; le reste si le jeu gelé compte ≥ 200 positifs indépendants ; chaque encodeur ajouté double l'inférence sur l'i5.
- Boucle : courbe d'apprentissage sur le jeu gelé ; ablation contre tirage aléatoire à budget égal.

---

## 7. Outil livrable

**Cible (levé H6)** : portables Windows 10/11 x64, Intel Core i5-1145G7 (4 cœurs / 8 fils, GPU intégré Iris Xe), 16 Go ; pas de CUDA. L'ONF réentraîne après le stage.

**Budget de calcul sur la cible (jugement, à mesurer en S3 sur une machine ONF)**
- Bird-MAE-Base en ONNX Runtime CPU : de l'ordre de 5–15 fenêtres/s → une campagne d'une semaine (≈ 575 h, ≈ 690 000 fenêtres de 3 s) = 13 à 38 h. Une nuit ne suffit pas sans levier.
- Leviers, dans l'ordre : quantification int8 dynamique du modèle ONNX (× 2 environ) ; traitement des heures de pic d'abord (7–9 h, 15–17 h : ÷ 3,6) ; sous-échantillonnage des fenêtres par enregistrement sous H20 (8 fenêtres sur 40 suffisent à estimer la fraction : ÷ 5) ; fournisseur d'exécution OpenVINO pour le GPU intégré `[À VÉRIFIER]` ; encodeur plus léger (Perch 2.0 ≈ 12 M paramètres si ONNX fonctionne, ou modèle distillé).
- Réentraînement de la tête : secondes. Ré-encodage complet après changement d'encodeur : plusieurs nuits, par lots reprenables.

| Option | Coût | Remarques |
|---|---|---|
| A. Bundle Python + ONNX Runtime + GUI | 2–3 sem. + construction Windows | Hors ligne ; ni TensorFlow ni PyTorch à l'exécution ; réentraînement intégré |
| C. PAMGuard comme hôte | 1–2 sem. | Bureau Java sans code, modèle ONNX générique (établi : pamguard.org) ; pas de réentraînement en place ; module séquentiel à porter ; ergonomie d'acousticien |
| D. Dossier Python portable + scripts | 1 sem. | Repli ; réentraînement possible en ligne de commande documentée |

- Décision S13 : A contre C, testées par un naturaliste ; le réentraînement par l'ONF après le stage pèse pour A.
- Écrans (A) : Analyser ; Vérifier (file, spectrogramme 2–8 kHz, boutons du schéma de labels, raccourcis) ; Résultats (points classés, export CSV) ; Modèle (versions, réentraînement, seuil, comparaison au jeu gelé).
- Mise à jour : tête dans l'application ; encodeur par paquet (`manifest.json` : nom, version, SHA-256, licence, f_e, fenêtre).
- Documentation : README ; « Interpréter un score » ; « Vérifier une file » ; **guide de réentraînement pour l'ONF** (pas à pas, une nuit) ; `LICENSES.md`.
- Licences (sans avis juridique) : BirdNET (CC BY-NC-SA) exclu par défaut ; Perch 2.0 (Apache 2.0), AnuraSet (CC0) compatibles ; Bird-MAE, BEATs, NatureLM `[À VÉRIFIER]` ; confirmation ONF avant S12.

---

## 8. Planning (26 semaines, S1 = 14–18/09/2026, aucune semaine réduite)

| Phase | Semaines | Contenu | Livrable | Go / no-go |
|---|---|---|---|---|
| P0 Cadrage, données, benchmark | S1–S3 (15/09–02/10) | Lecture (Courtois 2025 en détail, Fouquet 2018, Kath 2024, fils Kaggle) ; inventaire des jeux 2023 et 2026 (f_e, format, dates 2026) ; import des 345 + 158 annotations avec champs structurés ; négatifs appariés ; protocole d'évaluation figé ; essai YAPAT ; pré-benchmark AnuraSet ; benchmark ONF par micro ; C0–C1 ; mesure de débit sur une machine ONF ; questions §10 | Tableau de benchmark ; inventaire ; verdict C1 ; dépôt initialisé selon §13 | ≥ 1 encodeur local avec AP ≥ 0,5 et rappel ≥ 0,8 à précision ≥ 0,1 en plis par micro (seuil relevé : 345 positifs) ; sinon DSP + template matching en primaire |
| P1 Pipeline et récolte | S4–S7 (05/10–30/10) | Pipeline CLI ; probing linéaire en service ; audit aléatoire ; annotation par enregistrement ; récolte Trésor/Kaw 2026 et Molokoï 2023 ; C2 ; expérience d'échelle | Dépôt fonctionnel ; jeu v1 (≥ 30 enregistrements positifs validés hors Mataroni) | ≥ 30 positifs validés sur Trésor ou Kaw ; sinon risque 1 |
| P2 Optimisation et évaluation | S8–S12 (02/11–04/12) | Tours actifs ; jeu gelé ; module séquentiel et fusion ; reproduction des patrons 2023 ; C3 ; attentive probing si tokens ; choix d'encodeur ; licences ; conception du test homme–machine | Jeu gelé ; courbes ; rappel par qualité et par site ; note d'arbitrage | Rappel ≥ 0,85 à précision ≥ 0,1 sur Trésor et Kaw ; sinon pivot. **Fin S12 = limite de pivot** |
| P3 Outil v1 | S13–S18 (07/12–15/01) | Expérimentation S13 : A contre PAMGuard ; export ONNX, quantification ; GUI ; test sur machine ONF ; distillation si enseignant stable | Application testée sur une machine ONF ; guide de réentraînement v0 | 1 h d'audio en < 10 min sur l'i5 ; réentraînement de la tête sans intervention ; sinon option D |
| P4 Généralisation et transfert | S19–S22 (18/01–12/02) | Test homme–machine avec deux naturalistes ; audit 2026 complet ; combinaisons et LoRA/AnuraSet si effectifs ; documentation ; répétition de réentraînement par un agent ONF | Rapport d'évaluation ; v1.1 ; documentation | Gel fin S22 |
| P5 Rapport | S23–S26 (15/02–12/03) | Rédaction, soutenance en mars 2027 (levé H11), transfert | Rapport ; soutenance ; dépôt transféré | — |

- Rédaction continue dès S17 (½ j/sem.). Réunion hebdomadaire : avancement, chiffre clé (positifs validés hors Mataroni, AP sur jeu gelé), décision demandée.
- Chemin critique : données (S1) → benchmark (S3) → positifs hors Mataroni (S4–S7) → jeu gelé (S9) → encodeur (S12) → packaging (S13–S16) → test homme–machine (S19–S20) → gel (S22). Après S12 : plus de changement de famille ; après S16 : plus de changement d'encodeur.

---

## 9. Risques

| Risque | Signal | Seuil | Repli |
|---|---|---|---|
| 1. Aucun positif validé hors Mataroni | Récolte S4–S7 | < 30 fin S7 | Molokoï 2023 janvier–avril (probabilité ≈ 1) comme site tenu à l'écart ; généralisation spatiale annoncée partielle |
| 2. Biais de sélection du jeu étiqueté | Audit aléatoire S4 | Rappel sur l'audit < 0,8 × rappel en validation | Ré-échantillonnage de l'entraînement depuis l'audit ; rappel rapporté sur l'audit seul |
| 3. Chant ponctuel avéré (H20 fausse) | Écoute entière des 20 enregistrements | > 10 % avec < 50 % de fenêtres positives | Score par enregistrement = max ou top-k ; file « suspect » réintégrée |
| 4. Confusion avec les faux amis | > 30 % des 100 meilleurs candidats = faux amis après le tour 3 | Précision < 0,1 à rappel 0,85 fin S11 | Poids accru de persistance et rythme ; négatifs par espèce ; sortie hiérarchique |
| 5. Cible trop lente | Débit mesuré S3 sur l'i5 | > 15 h pour une campagne d'une semaine après leviers | Perch 2.0 ONNX si valide, sinon distillation ; heures de pic seules ; sous-échantillonnage des fenêtres |
| 6. Décalage 2023 (capteurs, saison) | Patrons non reproduits | Corrélation des courbes journalières < 0,5 | Recalibration par jeu ; négatifs 2023 appariés ; résultat rapporté tel quel |
| 7. Pas de Perch 2.0 sur CPU | Échec des replis | Fin S3 | Référence déportée seulement |
| 8. Bande saturée | Candidats/h en S3 | > 500/h aux heures de pic | Fenêtres pleines ; seuillage limité aux onsets |
| 9. AnuraSet non représentatif | Classement inversé sur ONF | Premier ↔ dernier | Réservé à l'adaptation |
| 10. Perte machine / données | Machine unique | — | SSD externe quotidien, copie ONF hebdomadaire, labels sous Git |

---

## 10. Questions restantes (par priorité)

1. As-tu lu jusqu'ici ?
2. « *A. blanci* chante-t-elle parfois de façon ponctuelle ? Chœur et mâle seul sont-ils distinguables à l'oreille ? » — H20, H21, règle d'agrégation.
3. « Les sorties du détecteur Biophonia sur 2023 (instants, scores) et ses ≈ 1 000 extraits annotés sont-ils récupérables auprès d'ENIA ou de Trésor ? » — Positifs 2023 et repère de comparaison.
4. « Quand un point est remonté « à vérifier », que se passe-t-il : visite de terrain, contrainte d'exploitation, à quelle échelle ? Et quel volume de points l'ONF peut-il vérifier par semaine ? » — H5 et précision plancher.
5. « Taux d'émission et intervalles entre notes d'*A. blanci* ? » — H9.
6. « Les enregistreurs 2023 (Song Meter mini) et 2026 (Mini 2) ont-ils la même réponse et les mêmes réglages ? » — Décalage matériel niveau 3.
7. « Qui tranche les licences à l'ONF, et à quelle échéance ? » — Avant S12.
8. « Peut-on disposer d'une machine ONF dès S3 pour mesurer le débit ? » — Risque 5.
9. « Date du prochain rapatriement de terrain ? » — H13, test réel de la boucle.

---

## 11. Montée en compétence

- Signal (S1–S4, ≈ 12 h) : Nyquist, STFT et compromis fenêtre/pas (note de 90 ms : fenêtre 16–25 ms, pas ≤ 10 ms), log-mel, passe-bande, enveloppe, onsets, SNR en bande, corrélation croisée. Docs scikit-maad, librosa ; Smith `[À VÉRIFIER]` ; Müller `[À VÉRIFIER]`.
- Écologie (S1–S6, ≈ 16 h) : Courtois et al. 2025 en entier (protocole, indice d'activité, patrons, préconisations) ; Fouquet et al. 2018 ; Cañas et al. 2023 ; PNA `[À VÉRIFIER]` ; écoute des 345 positifs et des 158 faux amis avec le tuteur.
- Apprentissage actif (S2, 4 h) : Kath et al. 2024 ; Hamer, Laber, Denton 2023.
- Arithmétique d'embedding (après S16) : directions et décalages ; prototype différentiel ; solo/chœur, distance, pluie comme décalages.

---

## 12. Hypothèses

**Levées** : H1 (Song Meter Mini 2 ; f_e et format restent à lire) · H2 (345 enregistrements, 13 micros, Mataroni, 3 s) · H3 (vraie pour Biophonia ; pour ce projet, Mataroni = entraînement) · H4 (extraits reçus) · H6 (i5-1145G7, 16 Go, Windows) · H7 (rapport accessible) · H8 (trois congénères dans Perch) · H10 (16 Go ; écoute humaine comme existant) · H11 (mars 2027) · H12 (annotation à la disponibilité) · H14 et H18 (liste des faux amis) · H15 (phénologie établie) · H16 (2026 partiellement : micros et durées connus, dates non) · H17 (pas de semaine réduite) · H19 (annotations expertes, classes A/B/C de Biophonia).

**Restantes**

| Id | Hypothèse | Levée par |
|---|---|---|
| H5 | Décision gestionnaire = intégration dans la planification forestière (zones tampons, exploitation) à l'échelle du point ou du bassin de crique | Q4 |
| H9 | IOI de l'ordre de la seconde ; plusieurs notes par fenêtre | Q5 |
| H13 | Un rapatriement de terrain avant la fin du stage | Q9 |
| H16 | Dates des campagnes 2026 ; saison correspondante | Q1 |
| H20 | *A. blanci* chante rarement de façon ponctuelle ; détection isolée = suspicion de faux positif | Q2 |
| H21 | Solo et chœur séparables par densité d'onsets et chevauchements | Q2 |
| H22 | Fréquence d'échantillonnage 2026 ≥ 32 kHz, WAV PCM | Inventaire |
| H23 | Les enregistreurs 2023 et 2026 ont une réponse comparable | Q6 |

---

## 13. Spécification d'implémentation (pour Claude Code)

Ce chapitre est autosuffisant : un agent doit pouvoir démarrer le dépôt sans lire le reste. Les décisions ci-dessous sont prises ; les points marqués « à mesurer » se règlent par expérience, pas par discussion.

### 13.1 Contraintes
- Machine de développement : MacBook Air M4, 16 Go, PyTorch avec backend MPS ; pas de CUDA.
- Machine cible : Windows x64, Intel i5-1145G7, 16 Go, CPU seul ; l'ONF y réentraîne la tête et ré-encode par lots.
- Interdit à l'exécution du livrable : TensorFlow, TFLite, PyTorch. Autorisé : ONNX Runtime, NumPy, scikit-learn.
- En recherche : bacpipe (`pip install bacpipe` ou `uv add bacpipe`) pour l'extraction multi-modèles ; torch/MPS.
- Python 3.11 ; environnement géré par `uv` ; `pyproject.toml` ; tests `pytest` ; formatage `ruff`.
- Pas de code complet dans ce document : signatures et schémas font foi.

### 13.2 Arborescence

```
blanci/
  pyproject.toml
  README.md
  config/
    default.yaml            # encodeur, fenêtre, pas, seuils, heures de pic, chemins
  blanci/
    ingest.py               # scan récursif, en-têtes WAV, nommage Song Meter, table recordings
    audio.py                # load(path) -> (wav float32 mono, sr) ; resample(wav, sr, target_sr)
    grid.py                 # window_grid(duration_s, window_s, hop_s) -> list[tuple[float, float]]
    qc.py                   # indices scikit-maad, drapeaux pluie / saturation / micro-dans-sac
    encoders/
      base.py               # Protocol Encoder
      bacpipe_encoder.py    # recherche
      onnx_encoder.py       # livrable
      export.py             # torch -> onnx, quantification int8, test d'équivalence
    store.py                # écriture/lecture Parquet partitionné
    index.py                # search(queries, filters, k) -> DataFrame
    head.py                 # prototype différentiel, régression logistique, scores hors-pli
    sequential.py           # onsets, IOI, persistance, solo/chœur
    fusion.py               # niveau 2 du stacking
    aggregate.py            # fenêtre -> enregistrement -> point
    evaluate.py             # plis par micro/site, AP, rappel@P, bootstrap, Wilson
    active.py               # construction des files
    labels.py               # import des annotations, schéma, validation
    db.py                   # SQLite : schéma, migrations, accès
    cli.py                  # typer : ingest, embed, benchmark, search, train, score, queue, evaluate, export
  app/
    streamlit_app.py        # prototype de vérification
  tests/
  data/                     # hors git
    raw/{2023,2026}/<site>/<micro>/*.wav
    db/blanci.sqlite
    embeddings/<encoder_id>/<dataset>/<site>/<yyyymm>.parquet
    labels/imports/         # fichiers d'annotation reçus, jamais modifiés
    models/<kind>/<name>-<version>/ (manifest.json + poids)
    frozen_test/            # jeu gelé, lecture seule
```

### 13.3 Schéma SQLite

```
recordings(recording_id TEXT PK, path TEXT UNIQUE, dataset TEXT, site TEXT, mic_id TEXT,
           start_utc TEXT, duration_s REAL, sample_rate INT, channels INT, sha256 TEXT,
           qc_flags TEXT)                       -- JSON : rain, saturation, in_bag, silent
windows(window_id TEXT PK, recording_id TEXT FK, offset_s REAL, dur_s REAL)
labels(label_id INTEGER PK, window_id TEXT FK, label TEXT, quality TEXT, species TEXT,
       conditions TEXT, annotator TEXT, source TEXT, created_at TEXT)
       -- label ∈ {blanci_solo, blanci_chorus, blanci_uncertain, bird, amphibian, orthoptera,
       --          amphibian_contact_call, rain, artefact_in_bag, background, other, uncertain}
       -- quality ∈ {A, B, C, NULL} ; source ∈ {import, similarity, active, random, audit}
models(model_id TEXT PK, kind TEXT, name TEXT, version TEXT, sha256 TEXT, params_json TEXT,
       created_at TEXT)                        -- kind ∈ {encoder, head, fusion, threshold}
scores(window_id TEXT, model_id TEXT, score REAL, PRIMARY KEY(window_id, model_id))
decisions(recording_id TEXT, encoder_id TEXT, head_version TEXT, threshold_id TEXT,
          fraction REAL, status TEXT, created_at TEXT)
          -- status ∈ {positive, suspect, negative, verified_positive, verified_negative}
```

- `window_id = f"{recording_id}:{offset_s:.2f}"` ; `recording_id = sha256(path)[:16]`.
- Les labels sont en ajout seul : une correction est un nouvel enregistrement, jamais une mise à jour.
- Parquet : colonnes `window_id, recording_id, offset_s, emb` avec `emb` en liste de taille fixe `float16[dim]` ; un fichier par (encodeur, jeu, site, mois).

### 13.4 Interfaces

```
class Encoder(Protocol):
    name: str; version: str; sample_rate: int; window_s: float; dim: int; has_tokens: bool
    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray            # (n_windows, dim)
    def embed_tokens(self, wav: np.ndarray, sr: int) -> np.ndarray | None  # (n_windows, n_tokens, dim)

def window_grid(duration_s: float, window_s: float, hop_s: float) -> list[tuple[float, float]]
def load_audio(path: Path) -> tuple[np.ndarray, int]
def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray
def search(queries: np.ndarray, store: EmbeddingStore, k: int, filters: dict) -> pd.DataFrame
def differential_prototype(E_pos: np.ndarray, E_neg_paired: np.ndarray) -> tuple[np.ndarray, float]
def train_head(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C_grid: list[float]) -> Head
def oof_scores(X, y, groups, n_splits: int = 5) -> np.ndarray
def detect_onsets(wav: np.ndarray, sr: int, band: tuple[int, int] = (4400, 5500)) -> np.ndarray
def sequential_features(onsets: np.ndarray, window_scores: np.ndarray) -> dict[str, float]
def aggregate_recording(window_scores: np.ndarray, threshold: float) -> tuple[float, str]
def rank_points(decisions: pd.DataFrame) -> pd.DataFrame
def build_queue(scores: pd.DataFrame, labels: pd.DataFrame, n: int,
                mix: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> pd.DataFrame
def evaluate(scores, labels, groups, level: Literal["window", "recording"]) -> dict
```

### 13.5 Commandes

```
blanci ingest data/raw --dataset 2026            # inventaire + QC + tables
blanci import-labels data/labels/imports/*.csv    # 345 + 158, parse des commentaires
blanci embed --encoder birdmae --filter peak_hours --batch 64
blanci benchmark --encoders birdmae,beats,naturebeats,perch_v2,birdnet --group-by mic
blanci search --positives --paired-negatives --site tresor --k 300
blanci train --head logistic --groups mic
blanci score --encoder birdmae --head v3
blanci queue --n 150 --mix 0.6,0.2,0.2
blanci evaluate --level recording --holdout tresor,kaw
blanci export-onnx --encoder birdmae --quantize int8 --check-equivalence
```

### 13.6 Jalons et critères d'acceptation

| Jalon | Contenu | Accepté quand |
|---|---|---|
| M0 (S1–S2) | Dépôt, config, `ingest`, `import-labels`, tests unitaires de `grid`, `resample`, `search` | `ingest` sur 100 fichiers sans erreur ; 345 + 158 labels importés avec espèce et qualité parsées ; grille sans note coupée vérifiée sur les 345 annotations |
| M1 (S2–S3) | `embed` via bacpipe pour 2 encodeurs au moins ; `store` ; `benchmark` en plis par micro | Tableau produit automatiquement (AP, rappel@P, intervalles, débit) ; résultats reproductibles à graine fixée |
| M2 (S3–S4) | `head` (prototype différentiel + logistique), `search`, `queue`, prototype Streamlit | Un tour actif complet en < 30 min de bout en bout ; scores hors-pli stockés |
| M3 (S5–S7) | `sequential`, `fusion`, `aggregate`, audit aléatoire | Fusion évaluée en plis par micro ; classement des points exporté en CSV |
| M4 (S8–S12) | Jeu gelé, `evaluate --holdout`, reproduction des patrons 2023 | Courbes journalières/annuelles 2023 produites et comparées à Courtois et al. 2025 |
| M5 (S13–S16) | `export-onnx`, quantification, test d'équivalence (cosinus > 0,99 avec torch), bundle Windows | 1 h d'audio en < 10 min sur l'i5 ; réentraînement de la tête sans intervention |
| M6 (S19–S22) | Guide de réentraînement, test homme–machine, transfert | Un agent ONF réentraîne seul sur une nouvelle campagne |

### 13.7 Règles pour l'agent
- Ne jamais écrire dans `data/raw` ni `data/frozen_test`. Les labels ne se modifient pas, ils s'ajoutent.
- Toute évaluation passe par `evaluate.py` avec groupes explicites ; un découpage aléatoire est une erreur.
- Aucun score n'entre dans `fusion` s'il n'est pas hors-pli.
- Un encodeur n'est jamais appelé sans passer par `Encoder` ; le rééchantillonnage se fait dans le wrapper.
- Tout résultat chiffré est produit par une commande reproductible avec graine, jamais dans un notebook non versionné.
- Commencer par M0 ; ne pas toucher à la GUI, à PAMGuard ni à la distillation avant M4.
- En cas de doute sur une décision de conception, se référer au numéro de section de cette feuille de route et ne pas rediscuter les choix marqués « jugement » ; les remettre en question dans un fichier `DECISIONS.md` daté.

---

## Angles morts

- **Le jeu étiqueté hérite du détecteur précédent.** Les 345 positifs sont ceux que Biophonia a remontés ; ce qu'il manquait manque aussi ; l'audit aléatoire est la seule correction, et elle est chère.
- **Mataroni est le seul site étiqueté.** La généralisation à Trésor et Kaw dépend d'une récolte qui n'a pas encore commencé ; si elle échoue, la question centrale du tuteur reste sans réponse chiffrée.
- **Le modèle précédent n'est pas comparable.** Non communiqué, jeu de test différent : le rapport ne pourra pas démontrer une amélioration par rapport à l'existant, seulement par rapport à l'écoute humaine.
- **H20 porte la règle d'agrégation et le sous-échantillonnage des fenêtres.** Si elle est fausse, deux économies disparaissent en même temps.
- **Phénologie comme piège.** Excellente prédictive à Mataroni ; hors classifieur par défaut, mais la tentation reviendra.
- **La machine cible est lente.** Sans quantification ou encodeur léger, une campagne d'une semaine ne se traite pas en une nuit ; ce point n'est mesuré qu'en S3 et peut renverser le choix d'encodeur.
- **Transfert à l'ONF.** Réentraînement sur i5 par des non-développeurs, sur plusieurs années : le guide de réentraînement est un livrable à part entière, sous-estimé par le planning.
- **Informations orales.** Chant continu, solo/chœur, comportement des faux amis : à vérifier sur enregistrement.
- **Sur-ingénierie.** Deux voies, combinaisons, distillation, migrations : chaque heure d'infrastructure est retirée à la récolte de positifs hors Mataroni, qui est le vrai chemin critique.
- **Occupation.** La non-détection devrait passer par un modèle d'occupation `[À VÉRIFIER]` ; hors périmètre.

---

## Sources vérifiées (hors bibliographie fournie)

- Cañas et al. (2023), AnuraSet, *Scientific Data* ; arXiv:2307.06860 ; github.com/soundclim/anuraset ; CC0.
- Kath, Serafini, Campos, Gouvêa & Sonntag (2024), *Ecological Informatics* 82, 102710 ; YAPAT, yapat.readthedocs.io.
- Hamer, Laber & Denton (2023), Agile Modeling for Bioacoustic Monitoring, Climate Change AI ; Zenodo 10.5281/zenodo.11585179.
- Martínez Balvanera et al. (2025), Whombat, *Methods in Ecology and Evolution*.
- PAMGuard : pamguard.org ; JASA 159(1), 2026 `[À VÉRIFIER auteurs]`.
- BirdCLEF+ 2026 : page Kaggle ; notebook mentionnant `perch_v2_no_dft.onnx` `[À VÉRIFIER]`.
- Rauch et al. (2024), arXiv:2406.18621.
- Schwinger et al. (2026), *Ecological Informatics*, article S1574954126001718 (version publiée de la revue citée par l'encadrant A).
