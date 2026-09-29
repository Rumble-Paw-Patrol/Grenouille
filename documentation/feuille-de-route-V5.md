# Feuille de route — Stage ONF Guyane
## Détection acoustique automatique d'*Anomaloglossus blanci* — 15/09/2026 → 14/03/2027

Version 5 (29/09/2026), après l'entretien de Léonard avec Élodie. `[HYPOTHÈSE Hn]` = information encore manquante (liste en §12) ; `[À VÉRIFIER]` = à contrôler sur les données ou les sources ; « établi » = bibliographie fournie (dont Courtois et al. 2025, *pheno-blanci.pdf*) ou source vérifiée ; « jugement » = arbitrage d'ingénieur ; « tuteur » = confirmé en réunion ; « Élodie » = décidé à l'entretien du 29/09. Le §13 est écrit pour un agent de code.

La version 4 est archivée (`old/feuille-de-route-V4.md`). Les numéros de section n'ont pas changé : les renvois de `DECISIONS.md` (« §5 », « §6 »…) restent valables. Les passages remplacés en v5 portent la mention **(v5)**.

**Journal v4 → v5 (entretien du 29/09/2026 avec Élodie)**
- **Encodeur libre d'accès impératif** pour le livrable. Après le benchmark AnuraSet, au plus deux encodeurs non libres restent, et seulement comme objectifs à battre s'ils font mieux que le meilleur libre (§0, §2).
- **Annotations reprises de zéro.** Tirage stratifié par point d'écoute, heure et période ; Léonard annote seul ; Élodie et Benoît vérifient un sous-échantillon, à l'aveugle (§5). Les 345 + 150 labels Blancinet sortent de l'entraînement et de l'évaluation.
- **Aucun benchmark sur les données ONF avant le « go » d'Élodie et Benoît** (§5.7, §6).
- **Benchmarks bornés dans le temps** : AnuraSet clos le 09/10, choix de l'encodeur le 20/11. Le temps libéré va à l'outil livrable, une application Windows qui s'installe par double-clic (§7).
- **Ordre des critères** (Élodie) : performance du modèle, puis facilité d'utilisation, puis durée d'encodage. Un naturaliste n'écrit pas plus d'une ou deux lignes de code, zéro visé (§0, §7).
- Pour aller plus loin : article de phénologie d'A. blanci ; gabarit du projet pour d'autres espèces (§14).
- Planning refait à partir de S3 (§8) ; risques, questions et hypothèses mis à jour (§9, §10, §12).

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
- **Élodie (entretien du 29/09/2026)** : encodeur libre impératif ; annotations reprises de zéro, vérifiées par elle et Benoît (experts naturalistes) avant tout benchmark ONF ; moins de temps sur les benchmarks, plus sur l'interface finale ; critères : performance d'abord, puis facilité d'utilisation (et durée d'encodage) ; pas plus d'une ou deux lignes de code pour un naturaliste. Elle propose, sans certitude et en demandant l'avis de Sylvain, d'écouter en priorité février–mars (pic d'activité de son rapport de phénologie) et les pics journaliers, avec aussi du milieu de journée pour trouver des chants manqués (avis en §5.8).

---

## 0. Objectifs et critères (v5)

**Objectif** : un détecteur d'*A. blanci* qui fonctionne sur les **points d'écoute des futures campagnes** (couples micro + site jamais vus). Il est livré à l'ONF sous une forme qu'un naturaliste utilise sans écrire de code, et que l'ONF réentraîne seul après le stage.

**Critères, dans l'ordre (Élodie)**

| Rang | Critère | Mesure (§6) | Rôle |
|---|---|---|---|
| 0 | Encodeur libre d'accès | poids publics, licence qui permet l'usage par l'ONF | **filtre** : un encodeur non libre n'entre jamais dans le livrable |
| 1 | Performance du modèle | AP et rappel par enregistrement sur des points tenus à l'écart | départage d'abord |
| 2 | Facilité d'utilisation | installation et usage sans code ; tâches réussies seul | départage ensuite |
| 3 | Durée d'encodage | heures pour une campagne d'une semaine sur l'i5 de l'ONF | seulement à performance égale |

« Performance égale » : intervalle apparié qui contient zéro (§6). Remplace la règle du §2 v4 (« moins de 0,1 d'AP d'écart = égalité, départagée par licence, vitesse, prise en main »). La licence n'est plus un critère de départage : c'est un filtre posé avant toute comparaison.

**Définition de travail de « libre d'accès »** (jugement, à confirmer par Élodie, Q1 du §10) : poids téléchargeables sans demande, **et** licence qui autorise l'usage par l'ONF sans accord particulier. L'ONF est un établissement public à caractère industriel et commercial : une clause « non commerciale » ne le couvre pas à coup sûr.
- Libres : Apache 2.0, MIT, BSD, CC BY, CC BY-SA. CC BY-SA oblige à redistribuer les dérivés sous la même licence ; une tête entraînée sur les embeddings n'est pas un dérivé des poids `[À VÉRIFIER]`.
- Non libres : clause non commerciale (NC) ; **aucune licence déclarée**, ce qui revient par défaut à « tous droits réservés » ; accès sur demande.

C'est la règle « non commerciale = non déployable » de la v4 (§7). Elle s'applique désormais aussi au benchmark ONF.

**Livrables, par priorité**
1. Jeu annoté v1, vérifié par Élodie et Benoît (§5). Sans lui, aucun chiffre ne vaut.
2. Détecteur (encodeur libre + tête) évalué sur des points tenus à l'écart (§6).
3. Application Windows installable sans code, et guide de réentraînement pour l'ONF (§7).
4. Rapport de stage et soutenance (mars 2027).
5. Pour aller plus loin, si 1 à 4 sont finis en avance : article de phénologie ; gabarit du projet pour d'autres espèces (§14).

---

## 1. Reformulation du problème

**Cadre (tuteur)** : détecteur robuste d'*A. blanci*, fonctionnant sur de nouveaux sites ; métriques AP et rappel ; six mois, un Macbook puce M4 16 giga de RAM portable, logiciel prenable en main par un naturaliste ; comparaison des approches par score, vitesse, prise en main.

**Données (établi et tuteur)**
- Enregistreurs : Wildlife Acoustics Song Meter Mini 2 partout ; fréquence d'échantillonnage et format lus dans les en-têtes à l'inventaire `[À VÉRIFIER]`.
- Jeu 2023 (établi : Courtois et al. 2025) : 6 enregistreurs, 2 par site (Kaw_A/B, Molokoï_E/F, Trésor_C/D), du 25/11/2023 au 24/11/2024, 2 min toutes les 30 min de 5 h à 20 h, 4 705 à 5 490 créneaux horaires par enregistreur ≈ 2 200 h au total ; capteurs de température et d'humidité associés.
- Jeu 2026 (v5, inventaire du 29/09 : README, DECISIONS n° 153) : 5 sites (CDR, Mataroni, Patawa Est, Patawa Ouest, RNRT), **un relevé d'environ une semaine par site**, de décembre 2025 à février 2026 (mois de chaque relevé `[À VÉRIFIER]` par `blanci status`) ; 2 min toutes les 30 min de 5 h à 19 h 30 (29 créneaux par jour) ; 29 513 enregistrements, 979 h. Conséquence : en 2026, le mois est confondu avec le site.
- Total (v5) : 96 292 enregistrements, 3 204 h de 120 s, 2 216 Go ; 94 588 encodables après les drapeaux. 4,5 millions de fenêtres de 5 s au pas de 2,5 s.
- Annotations (v5) : **reprises de zéro** (§5). Les 345 positifs et 150 négatifs importés de Blancinet (51 enregistrements, 13 micros, tous à Mataroni, DECISIONS n° 34–35) restent dans la base, qui est en ajout seul. Ils sortent de l'entraînement et de l'évaluation, pour trois raisons : ils sont conditionnés par un détecteur ; ils ne couvrent qu'un site ; les négatifs sont des faux amis choisis, pas un échantillon du stock.
- Conséquence 1 (v4, caduque) : effectif = 345 enregistrements. En réalité 51 (n° 35).
- Conséquence 2 : un jeu étiqueté **conditionné par le détecteur précédent** ne contient pas les chants que ce détecteur n'a pas vus : le rappel mesuré dessus est relatif. C'est la raison de la reprise : en v5, le premier lot est tiré sans aucun détecteur (§5.1).
- Conséquence 3 : la généralisation à de nouveaux points se mesure sur des points tenus à l'écart **dès le tirage** (§5.2, §6), et non plus sur des positifs récoltés après coup.

**Phénologie (établi : Courtois et al. 2025)**
- Journalier : pics 7–9 h et 15–17 h à Kaw et Trésor ; à Molokoï, activité haute et constante de 7 h à 17 h en saison.
- Annuel : activité forte de janvier à avril, jusqu'en juin à Trésor et Molokoï ; quasi nulle de juillet à octobre ; reprise en novembre–décembre avec les premières pluies.
- Détection : à Molokoï, probabilité journalière ≈ 1 de fin novembre à mars ; à Trésor, maximale en décembre puis ≈ 0,5 jusqu'en juin ; Kaw_B ne dépasse jamais 0,5 ; ≈ 0 partout de juillet à octobre.
- Stage : septembre–novembre = saison basse ; les enregistrements 2026 ont été faits entre décembre et février (v5 ; mois par site `[À VÉRIFIER]`).
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

**(v5) Ce qui change**
- **AnuraSet clôt le choix des candidats.** La vague 2 (`benchmarks/outils_anuraset/VAGUE_ENCODAGE_2.md`) se termine le **09/10/2026**. Les sessions non finies ce jour-là sont abandonnées. Pas de nouvelle tête ni de nouvelle régularisation sur AnuraSet, et des fiches courtes (≤ 30 lignes).
- **Filtre de licence (§0) avant le benchmark ONF.** Tous les encodeurs libres qui passent le témoin BOAFAB (n° 151) y vont. Il y va aussi **au plus deux non libres**, choisis parmi ceux qui font mieux que le meilleur libre sur AnuraSet. Ils servent d'objectifs à battre et ne sont jamais livrés. Au 29/09, aucun non libre ne fait mieux : perch_v2, libre, est en tête du benchmark 07 à égalité avec perch_bird. Si cela tient, BirdNET 2.4 reste le seul non libre, comme repère que les naturalistes connaissent (à valider avec Élodie).
- **Pas de benchmark sur les données ONF avant le go d'Élodie et Benoît** (§5.7). Ensuite, le benchmark ONF est borné : encodeurs retenus × logistique (plus une tête sur jetons pour un transformer libre, n° 151) ; courbe d'amorçage par point ; choix de l'encodeur le **20/11/2026**, puis plus aucun benchmark d'encodeur.
- Les passages ci-dessous sur le « jeu annoté v0 » (345 positifs, négatifs appariés de Mataroni) sont caducs : le benchmark ONF se fait sur le jeu v1 (§5).

**Licences relevées (29/09/2026 ; fiches Hugging Face, dépôts, `encodeurs-bacpipe.md`)**

| Encodeur | Licence des poids | Statut v5 |
|---|---|---|
| perch_v2 (Perch 2.0) | Apache 2.0 (établi, v4 §7) | **libre** |
| perch_bird (Perch 1) | Apache 2.0 `[À VÉRIFIER]` sur la page Kaggle | libre sous réserve |
| birdnet_v3 (BirdNET+ 3.0, préversion) | CC BY-SA 4.0 ; conditions d'usage lues : mention « Powered by BirdNET », braconnage et usage militaire interdits (n° 151) | **libre** ; ONNX officiel |
| beats | dépôt `microsoft/unilm` sous MIT (lu le 29/09) ; poids du même dépôt | libre sous réserve |
| birdnet (2.4) | CC BY-NC-SA 4.0 | non libre : repère des naturalistes |
| esp-aves2 (dix points de contrôle) | CC BY-NC-SA 4.0 (n° 151) | non libre |
| naturebeats | NatureLM-audio : CC BY-NC-SA 4.0 (fiche lue le 29/09) ; mêmes poids `[À VÉRIFIER]` | non libre |
| audioprotopnet | CC BY-NC 4.0 (fiche lue le 29/09) | non libre |
| Bird-MAE (Base, Large, Huge), convnext_birdset, biolingual | aucune licence sur la fiche Hugging Face (lue le 29/09) ; dépôt de code non lu | non libre tant qu'aucune licence n'est publiée ; demander aux auteurs si l'un d'eux gagne |
| protoclr, rcl_fs_bsed, mix2, avesecho_passt, insect66, insect459, MetaPerch | non relevée | à relever avant le benchmark ONF |

**Suite du §2 : texte de la v4, inchangé.**

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

## 5. Annotation et apprentissage actif (v5 : reprise de zéro)

**Décision (Élodie)** : on repart de zéro. Léonard tire et annote seul. Il envoie un sous-échantillon à Élodie et Benoît, experts naturalistes, qui le vérifient. Aucun benchmark sur ces données avant leur go. L'inventaire complet (96 292 enregistrements, 8 sites) permet de choisir quoi écouter : points d'écoute, heures et périodes. Le but est un jeu aussi équilibré que possible pour **généraliser à de nouveaux points**.

### 5.1 Principes (jugement)

1. **Le point d'écoute (couple site + micro, n° 38) est l'unité qui compte**, ni la fenêtre ni l'enregistrement. Les fenêtres d'un même enregistrement partagent le fond sonore et, sous H20, le même chant : 40 fenêtres d'un enregistrement valent à peine plus qu'une. Il vaut mieux **beaucoup de points, peu d'enregistrements par point**. Benchmark 07 (AnuraSet) : sur un site neuf, 5 à 10 enregistrements positifs annotés suffisent à relever l'AP de 0,81 à 0,89. Au-delà, chaque point apporte surtout de la diversité.
2. **Deux jeux, deux règles.**
   - Le jeu d'**entraînement** peut être enrichi là où l'espèce chante : un tirage biaisé ne fausse pas une tête, pourvu que les négatifs couvrent tous les fonds.
   - Le jeu d'**évaluation** est tiré au hasard dans des strates, sur des **points tenus à l'écart** ; la probabilité de tirage de chaque enregistrement est notée. Sans ces probabilités, aucune métrique ne se ramène au stock réel.
3. **Aucun détecteur dans le tirage du premier lot**, ni Blancinet, ni nos têtes. Sinon le jeu hérite à nouveau des angles morts d'un détecteur (§1, conséquence 2). Les modèles reviennent après le go, et pour les lots d'entraînement seulement (§5.9).
4. **Chaque strate qui donne des positifs donne aussi des négatifs**, des mêmes points aux mêmes heures, et inversement. Sinon la tête apprend l'heure, la saison ou le site à la place du chant : c'est un raccourci, qui se paie sur un point nouveau.
5. **Tout tirage est fait par un script à graine fixée, avant écoute**, et sa liste est versionnée. Ce qu'on a écouté ne change pas ce qu'on tire ensuite dans le même lot.

### 5.2 Partition des points, avant toute écoute

| Jeu | Points | Pourquoi |
|---|---|---|
| Évaluation, niveau 2 | **un site 2026 entier**, choisi avec Élodie : présence connue, au moins 10 points | mesure « nouveau site », la situation des futures campagnes |
| Évaluation, niveau 1 | **20 % des points** de chacun des autres sites 2026, tirés au hasard | mesure « nouveau point d'un site connu » |
| Évaluation, niveau 3 | 2023 : **une station sur deux** par site (A ou B, tirée au hasard) | toutes les saisons, sur un enregistreur jamais vu |
| Entraînement | tout le reste | — |

- Un point tenu à l'écart ne donne jamais rien à l'entraînement, ni à la validation croisée, ni au réglage du seuil.
- Les enregistrements qui portent un label Blancinet sont exclus du tirage d'évaluation : ils ont été choisis par un détecteur.
- Pour 2023, le point est la station (Kaw_A, Kaw_B…) : un enregistreur remplacé en cours d'année ne crée pas un nouveau point `[À VÉRIFIER]` sur l'inventaire.

### 5.3 Strates

- **Tranche horaire** (heure locale, 5 tranches) : aube 5–7 h ; pic du matin 7–9 h ; journée 9–15 h ; pic du soir 15–17 h ; soir 17–20 h. Pics : Courtois et al. 2025.
- **Période** (2023 seulement) : haute (décembre–avril, dont **février–mars** comme sous-strate, proposition d'Élodie) ; transition (mai–juin, novembre) ; basse (juillet–octobre).
- **2026** : chaque site n'a qu'un relevé d'une semaine, la période est donc fixée par le site. On stratifie par point et par tranche horaire.
- Dans une strate : jour tiré au hasard, puis enregistrement tiré au hasard ; au plus un enregistrement par point, jour et tranche.
- Les enregistrements écartés par un drapeau (n° 79, 153, 154) sont exclus du tirage. Pluie et saturation restent : ce sont des conditions réelles.

### 5.4 Lot d'entraînement 1 : environ 450 extraits de 30 s

| Jeu | Part du lot | Répartition |
|---|---|---|
| 2026, points d'entraînement | ≈ 300 extraits | **3 par point** : un au pic du matin, un au pic du soir, un dans une autre tranche tirée au hasard |
| 2023, stations d'entraînement (3) | ≈ 150 extraits (50 par station) | période haute 50 % (dont deux tiers en février–mars), transition 20 %, basse 30 % ; pics 60 %, autres tranches 40 % |

- Au total, environ 60 % des extraits tombent aux heures de pic et 40 % hors pic. En 2023, 30 % sont en saison basse.
- Le quota par point (3) se cale sur le décompte réel des points d'entraînement. Il n'est pas estimé ici : `blanci status` donne le nombre de micros par site.
- Ce lot mesure aussi la **prévalence** par strate, qui manque à `decision.prevalence` (R73, DECISIONS n° 128) : c'est un tirage probabiliste, contrairement aux fenêtres choisies pour l'annotation.
- Temps : ≈ 1 min par extrait, manipulation comprise, soit 8 à 10 h. Chronométré sur les 50 premiers, puis recalé.

### 5.5 Jeu d'évaluation v1 : environ 250 enregistrements entiers

- Points tenus à l'écart (§5.2) ; **enregistrements de 2 min écoutés en entier**. L'unité de décision est l'enregistrement (§1), et l'écoute entière teste H20 au passage : quelle fraction de l'enregistrement une grenouille qui chante occupe-t-elle ?
- Tirage stratifié (tranche horaire, et période en 2023). Les strates de pic, et la période haute en 2023, sont **surreprésentées deux fois**. La probabilité de tirage de chaque enregistrement est notée, et les métriques sont repondérées (§6).
- Cible : **au moins 60 enregistrements positifs** (rappel 0,9 → intervalle de Wilson [0,80 ; 0,95]) et au moins 150 négatifs. S'il manque des positifs après 250 enregistrements, on complète dans les strates de pic, probabilités notées.
- Après le go (§5.7), c'est le **jeu gelé v1** (`blanci freeze`), jamais entraîné. Il remplace le jeu gelé de 60 enregistrements et l'audit aléatoire de 300 enregistrements de Mataroni (v4 §6).
- Temps : ≈ 3 min par enregistrement, soit ≈ 12 h.

### 5.6 Que noter, et combien de fenêtres par enregistrement

- **Unité d'écoute** : un extrait de **30 s** à l'entraînement, à position tirée au hasard dans l'enregistrement ; l'enregistrement entier à l'évaluation.
- **Ce qu'on note** : les **intervalles** où *A. blanci* chante (début, fin, à 0,5 s près, sûr ou incertain). Pour l'extrait : familles présentes (schéma de labels ci-dessous), qualité A/B/C, pluie, chœur ou solo si on l'entend, canal écouté.
- **Les labels de fenêtres se déduisent des intervalles, pour n'importe quelle grille** (3, 5 ou 6 s). Une fenêtre est positive si elle tombe dans un intervalle, négative si elle n'en touche aucun, « bord » sinon (exclue de l'entraînement). Les labels ne dépendent plus de l'encodeur, contrairement aux fenêtres de 3 s de Blancinet (`check-grid`, n° 4).
- **Combien de fenêtres** : toutes celles de l'extrait, obtenues d'une seule écoute, soit 6 fenêtres jointives de 5 s ou 10 de 3 s (11 et 19 au pas de 50 %). À l'entraînement, **chaque enregistrement pèse autant** : poids 1 / nombre de fenêtres. Un enregistrement de 10 fenêtres positives ne compte pas dix fois.
- **Pourquoi 30 s** : c'est environ 20 notes (intervalle moyen 1,4 s), de quoi reconnaître le rythme. Cela prend ≈ 1 min par extrait, contre ≈ 3 min pour un enregistrement entier : on écoute trois fois plus d'enregistrements, donc de points et de jours, pour le même temps, alors que les secondes supplémentaires d'un même enregistrement apportent peu. À vérifier sur le jeu d'évaluation : un chant absent des 30 s tirés mais présent ailleurs dans l'enregistrement. S'il y en a plus de 10 %, passer à 60 s (risque 5, §9).

### 5.7 Vérification par Élodie et Benoît, et règle du go

**Paquet de vérification**, envoyé fin S6. Il est **à l'aveugle** : le label de Léonard est caché et l'ordre mélangé, pour que l'expert ne confirme pas par politesse ce qu'on lui montre.

| Contenu | Part vérifiée | Nombre attendu | Pourquoi |
|---|---|---|---|
| Tout ce que Léonard a noté « incertain » | 100 % | ≈ 40 | Léonard ne tranche pas seul |
| Positifs du jeu d'évaluation | 100 % | 60–80 enregistrements (écoute des intervalles notés) | ils définissent le rappel |
| Négatifs du jeu d'évaluation | 20 %, au hasard | ≈ 35–40 enregistrements de 2 min | un chant manqué fausse le rappel |
| Extraits d'entraînement | au hasard, stratifiés par site | 60 positifs + 100 négatifs | mesure l'erreur de Léonard |
| Recouvrement | écouté par les deux experts | 30 éléments pris dans ce qui précède | accord entre experts (`blanci agreement`) |

- Au total, environ 15 à 20 % de ce que Léonard a annoté, et **≈ 2 h 30 d'écoute par expert** (estimation, jugement) : ≈ 4 h 30 à se partager, plus 30 min de recouvrement chacun.
- **Format pour les experts, sans installation** : un dossier d'extraits WAV nommés par identifiant et une feuille Excel (identifiant, *A. blanci* oui / non / incertain, commentaire). Ils écoutent avec leur outil habituel (Raven, Audacity, Kaleidoscope…), ou avec le poste d'annotation s'ils le préfèrent (Q2 du §10).
- **Règle du go** (jugement ; bornes unilatérales à 95 %) :
  - positifs de Léonard : 0 erreur sur 60 → taux d'erreur ≤ 5 % ;
  - négatifs : au plus 1 chant manqué sur 100 → ≤ 4,7 % ;
  - au-delà, la strate fautive (site, tranche, type de fond) est réécoutée par Léonard, puis contrôlée par un nouveau tirage ;
  - toute réponse d'expert s'ajoute comme label (annotateur = l'expert). Elle prime sur celle de Léonard ; la base reste en ajout seul.
- Le go est écrit dans `DECISIONS.md`, avec ses chiffres. Avant, aucune tête n'est jugée sur ces données.

### 5.8 Avis sur les propositions d'Élodie (à discuter avec Sylvain)

**Écouter en priorité février–mars ?** Oui pour récolter des positifs, non comme seule période.
1. En 2026, chaque site n'a été enregistré qu'une semaine, et le mois y est confondu avec le site. « Février–mars » y revient à choisir des sites, donc moins de points : c'est l'inverse de l'objectif. En 2026, on enrichit par l'heure, pas par le mois.
2. Une tête qui n'a vu que des fonds de saison des pluies rencontrera en saison sèche d'autres insectes et d'autres oiseaux : c'est là que naissent les fausses alarmes. Elles dessineraient une fausse activité de juillet à octobre sur les courbes de phénologie (§14), et gêneraient toute campagne posée hors saison.
3. Les chants de début et de fin de saison, peut-être plus rares et plus faibles `[À VÉRIFIER]`, sont ceux qui datent la saison. Il faut que la tête en ait vu.
4. Un jeu d'évaluation tiré en février–mars ne dit rien des autres mois.

Proposition : en 2023, la période haute fait 50 % du tirage, dont deux tiers en février–mars (contre 40 % si le tirage suivait le calendrier de la période haute, soit un facteur 1,7) ; transition 20 %, basse 30 % (§5.4). Les probabilités de tirage sont notées.

**Les pics journaliers en priorité, et du milieu de journée pour les faux négatifs ?** D'accord, avec deux précisions.
1. Aux heures de pic, on tire aussi des négatifs aux mêmes points. Sinon la tête apprend « matin = *A. blanci* ».
2. Les heures creuses servent autant aux négatifs qu'aux chants manqués. Le chœur d'oiseaux de l'aube (5–7 h) et les amphibiens du soir (17–20 h) sont des sources probables de faux amis (oiseaux et amphibiens dominent la liste du schéma de labels ; heures `[À VÉRIFIER]` sur le lot 1). Avant qu'un modèle existe, il n'y a pas de « faux négatifs » à proprement parler : on cherche des chants hors des heures attendues. Ce sont eux que le modèle manquera s'il n'en voit jamais.

Proposition : 60 % aux heures de pic, 40 % ailleurs (§5.4).

**Questions pour Sylvain**
1. Enrichir l'évaluation (×2 aux pics) et repondérer par les probabilités de tirage, ou tirer l'évaluation proportionnellement, sans enrichissement ?
2. 30 s par enregistrement à l'entraînement et l'enregistrement entier à l'évaluation : d'accord ?
3. La partition des points (un site entier, 20 % des points ailleurs, une station 2023 sur deux) : trop, ou pas assez ?
4. Un poids de 1 / nombre de fenêtres par enregistrement dans la tête, ou un sous-échantillonnage des fenêtres ?
5. Après le go, mélanger au lot 1 (probabiliste) des lots d'apprentissage actif (choisis par un modèle) pour l'entraînement : d'accord, si l'évaluation reste purement probabiliste ?

### 5.9 Après le go : lots suivants (entraînement seulement)

- Apprentissage actif : file de 60 % d'incertains, 20 % de scores maximaux, 20 % tirés au hasard et stratifiés (v4). Récolte par similarité sur les points peu servis ; negative mining parmi les mieux classés. Ces lots sont choisis par un modèle : ils ne vont **jamais** dans l'évaluation, et leur source est notée (`active`, `similarity`).
- Vérification experte : 10 % tirés au hasard par lot, plus les incertains.
- Seuil d'un site : 5 à 10 enregistrements positifs vérifiés sur place (benchmark 07 : le seuil ne voyage pas d'un site à l'autre). C'est aussi la procédure de l'ONF pour chaque nouvelle campagne (§7).
- Arrêt : gain d'AP sur le jeu gelé v1 inférieur à 0,02 sur deux tours (v4).

### 5.10 Poste d'annotation

- Poste Streamlit existant (`blanci annotate`, écoute des deux canaux) avec un nouveau mode **« extrait + intervalles »** : curseur double sous le spectrogramme de 2 à 8 kHz, raccourcis clavier, labels du schéma.
- Export du paquet d'experts (WAV + Excel, à l'aveugle) et réimport de leurs réponses comme labels.
- YAPAT et Whombat restent des possibilités (v4). Ce sont des outils de travail, pas le livrable.

**Schéma de labels** (levé H14, inchangé) : positifs {blanci-solo, blanci-chœur, blanci-incertain} ; négatifs par famille {oiseau:<espèce>, amphibien:<espèce>, orthoptère, cri-de-contact-amphibien, pluie, artefact:micro-dans-sac, fond, autre} ; espèces recensées : fourmilier tacheté (le plus fréquent), moucherolle, manakin, tangara mordoré, pigeon plombé, évêque de Rothschild, sclérures, myrmidon, psittacidés, pic à cou rouge, martinet ; *Adenomera andreae*, *Allobates femoralis*, *A. hahneli*, *Hyalinobatrachium cappellei / mondolfii / iaspidiense*, *Amazophrynella teko*, *Otophryne* ; grillons. Les noms scientifiques des oiseaux sont `[À VÉRIFIER]` avant publication.

**Budget (v5)** : Léonard, ≈ 20 h d'écoute (lot 1 : 8–10 h ; évaluation : ≈ 12 h), soit ≈ 25 h avec la manipulation, de S4 à S7. Experts : ≈ 2 h 30 chacun (§5.7). Le budget de la v4 (« quand on a le temps », 66 h pour le stagiaire) ne tient plus : l'annotation est devenue le chemin critique.

**Après le stage (levé H13)** : l'ONF réentraîne sur ses portables avec chaque nouvelle campagne de pose de micros, sur plusieurs années ; Exigences : procédure documentée, tête réentraînable sans GPU (une nuit de calcul est acceptable), ré-encodage par lots reprenable, comparaison automatique au jeu gelé avant bascule.

---

## 5 bis. Voie non supervisée : clustering

(v5) Après le go seulement, sur le jeu v1 ; C1 se refait sur les points d'entraînement, pas sur les seuls positifs de Mataroni.

- Fonctions : négatifs en volume ; exploration ; détecteur seulement si C1 réussit.
- Méthode (jugement) : HDBSCAN sur ACP (≈ 50 composantes) ; UMAP pour visualiser.
- C0 (S3, 1 h) : clustering global d'un échantillon ; AMI entre groupes et micros ; anomalies (micro dans sac, saturation).
- C1 (S3, 1 j) : Mataroni, positifs connus + 5 000 fenêtres aux mêmes heures ; fenêtres pleines contre recadrées. Seuils (jugement) : rappel du meilleur groupe ≥ 0,5 ; enrichissement ≥ 20 ; AMI micro faible.
- C2 (S4–S7) : étiquetage en bloc des groupes homogènes après dix écoutes.
- C3 (S6–S9) : sub-clustering par micro sur les heures de pic ; récolte de positifs sur Trésor et Kaw.
- Limite : 2–3 % de fenêtre pour la note → groupes = paysages sonores ; le chant continu et le recadrage améliorent le rapport.

---

## 6. Évaluation

**Préalable (v5)** : rien n'est jugé sur les données ONF avant le go d'Élodie et Benoît (§5.7).

**Protocole de généralisation (v5, remplace celui de la v3)**
- **Niveau 2, nouveau site** : le site 2026 tenu entier à l'écart (§5.2). C'est **la** mesure principale : les futures campagnes posent des micros sur de nouveaux points, souvent sur de nouveaux sites.
- **Niveau 1, nouveau point d'un site connu** : les 20 % de points tenus à l'écart dans les autres sites 2026.
- **Niveau 3, temporel** : les stations 2023 tenues à l'écart, toutes périodes. Précision et rappel **par période** (haute, transition, basse). S'y ajoute la **reproduction des patrons publiés** : le détecteur, appliqué aux 2 225 h de 2023, doit retrouver les courbes journalières et annuelles de Courtois et al. 2025 (pics 7–9 h et 15–17 h, creux de juillet à octobre, Kaw_B faible). Un désaccord signale un problème de généralisation ou un artefact.
- **Développement** (choix de la tête, de C, du pooling) : plis par point sur les seuls points d'entraînement, jamais sur les points d'évaluation.
- **Repondération** : sur le jeu d'évaluation, chaque enregistrement pèse 1 / sa probabilité de tirage (estimateur de Horvitz–Thompson). L'AP, la précision, les fausses alarmes par heure et la prévalence sont ainsi celles du stock réel, pas du tirage enrichi. Le rappel est rapporté aussi strate par strate.
- L'audit aléatoire de 300 enregistrements de Mataroni (v4) disparaît : le jeu d'évaluation, tiré sans détecteur, joue ce rôle.

**Jeu gelé (v5)** : c'est le jeu d'évaluation v1 (§5.5) après le go, versionné et jamais entraîné. Qualité A/B/C, solo ou chœur, SNR et probabilité de tirage y sont renseignés.

**Métriques** : AP et rappel (tuteur) par enregistrement et par fenêtre ; rappel à précision ≥ 0,1 et ≥ 0,5 ; fausses alarmes par heure ; rappel par qualité A/B/C et par SNR estimé (énergie en bande pendant les notes contre 0,5 s voisines) ; rappel par période et par tranche horaire (v5) ; accord du classement des points avec l'expert. Rejetées : exactitude, AUROC (annexe), F1 au seuil 0,5, kappa (pour les modèles ; l'accord entre annotateurs se rapporte à part, §5.7).

**Baseline et repères**
- Baseline = écoute humaine (levé H10) : protocole homme–machine sur un lot commun de 60 enregistrements (30 positifs, 30 négatifs difficiles, mélangés) : rappel, précision et temps d'un naturaliste à l'oreille ; rappel, précision et temps de l'outil suivi d'une vérification humaine des seuls candidats. C'est la comparaison principale du rapport, conçue dès P2, exécutée en P4 avec deux naturalistes.
- **Objectifs à battre (v5)** : au plus deux encodeurs non libres (§2), jugés sur le même jeu d'évaluation et jamais livrés.
- **Blancinet v0.1.0 (v5)** : ses détections importées (`import-detections`, score ≥ 0,10, jeu 2026) sont jugées au niveau de l'enregistrement sur les enregistrements d'évaluation qu'il a traités. C'est enfin une comparaison à l'existant sur les mêmes enregistrements. Réserve : on ignore sur quoi il a été entraîné, et un recouvrement avec nos points le favoriserait.
- Repère chiffré : détecteur Biophonia 2024 (établi : Courtois et al. 2025) — réseau de neurones d'architecture non communiquée, ≈ 1 000 extraits annotés, test sur 450 : 0 faux positif, 24 faux négatifs, 53 vrais positifs, soit rappel ≈ 0,69 à précision 1. Modèle indisponible, jeu de test différent : comparaison indicative seulement, à la même précision.
- Méthodes reproductibles : template matching ; détecteur en bande ; BirdNET 2.4 + sonde (référence non libre).

| Rang (Élodie, §0) | Critère | Mesure | Comment |
|---|---|---|---|
| 1 | Performance | AP repondérée et rappel par enregistrement : niveau 2, puis 1 et 3 ; par qualité, par période | bootstrap par point ; comparaisons appariées ; Holm (n° 139) |
| 2 | Facilité d'utilisation | installation (clics, minutes) ; lignes de code à écrire (cible 0, plafond 2) ; tâches réussies seul (analyser un dossier, vérifier une file, régler le seuil d'un site, exporter) ; grille à 5 points | Élodie et Benoît en P4, sur une machine ONF |
| 3 | Durée d'encodage | heures pour une campagne d'une semaine (575 h) sur l'i5 | même fichier ; encodage + tête ; CPU seul |

- Rappel prioritaire ; précision plancher : file hebdomadaire ≤ 1 h à 10 s par candidat (jugement). Cible : rappel ≥ 0,9 au seuil le plus élevé qui garde une précision ≥ 0,1 sur le site tenu à l'écart.
- Intervalles : Wilson en enregistrements (n = 60, rappel 0,9 → [0,80 ; 0,95]) ; bootstrap par point pour l'AP (n° 139) ; comparaisons appariées ; « A meilleur que B » seulement si l'intervalle apparié exclut zéro.
- Calibration : seuils sur scores hors-pli. **Seuil propre à chaque site** (v5) : le benchmark 07 montre qu'un seuil appris ailleurs ne voyage pas ; 5 à 10 enregistrements positifs vérifiés sur place le fixent (§5.9, §7).
- Combinaisons (décision du stagiaire) : concaténation d'embeddings et fusion séquentielle d'abord ; le reste si le jeu gelé compte ≥ 200 positifs indépendants ; chaque encodeur ajouté double l'inférence sur l'i5.
- Boucle : courbe d'apprentissage sur le jeu gelé ; ablation contre tirage aléatoire à budget égal.

---

## 7. Outil livrable

**Cible (levé H6)** : portables Windows 10/11 x64, Intel Core i5-1145G7 (4 cœurs / 8 fils, GPU intégré Iris Xe), 16 Go ; pas de CUDA. L'ONF réentraîne après le stage.

**Recommandation (v5, jugement)** : une **application de bureau Windows**, installée par un fichier `.exe` (double-clic, « Suivant », « Terminer »), ouverte depuis une icône, **sans aucune ligne de code**. Ni Élodie ni Léonard ne connaissent l'empaquetage : c'est une raison de suivre un chemin déjà éprouvé pour ce public. BirdNET-Analyzer, que les naturalistes connaissent, est distribué ainsi : une interface Gradio dans une fenêtre pywebview, et une version Windows prête à l'emploi.

| Brique | Choix | Pourquoi |
|---|---|---|
| Moteur | `blanci/service.py` réduit au livrable : ONNX Runtime, NumPy, scikit-learn | déjà la couche que la CLI appelle (§4) ; ni PyTorch ni TensorFlow (§13.1) |
| Interface | pages locales dans une fenêtre (pywebview) ou dans le navigateur ; l'écran « Vérifier » reprend le poste Streamlit existant ; Gradio si l'empaquetage de Streamlit résiste à l'essai | rien ne quitte la machine ; fonctionne hors ligne |
| Empaquetage | PyInstaller (Python, bibliothèques et modèle ONNX dans un dossier), puis Inno Setup (installateur `.exe`, raccourci, désinstallation) | outils standard et gratuits |
| Construction | GitHub Actions sur une machine Windows, à chaque version | l'ONF télécharge un `.exe` ; personne ne compile |
| Taille | quelques centaines de Mo `[À MESURER]` | runtime, modèle, bibliothèques |

- **Étape intermédiaire, disponible tout de suite** : avec `uv`, une ligne PowerShell installe l'outil, une autre lance l'application. C'est déjà la règle « une ou deux lignes » d'Élodie ; cela sert aux premiers essais et reste le repli si l'installateur échoue.
- **Écartés** : serveur web hébergé (2 To d'audio à téléverser, terrain hors ligne, serveur à maintenir par l'ONF) ; PAMGuard comme hôte (pas de réentraînement sur place, ergonomie d'acousticien, v4) ; notebooks et ligne de commande seule (du code) ; application native Qt (même résultat, plus de travail).
- **Essai d'empaquetage en S4** : le poste d'annotation actuel en `.exe`, installé sur une machine Windows propre. Il lève le risque 6 (§9) avant d'écrire les écrans.

**Écrans** : Analyser (choisir un dossier de campagne, lancer, voir l'avancement ; reprise après coupure) ; Vérifier (file, spectrogramme de 2 à 8 kHz, boutons du schéma de labels, raccourcis) ; **Seuil du site** (v5 : 5 à 10 positifs vérifiés, §6) ; Résultats (points classés, export CSV et Excel) ; Modèle (versions, réentraînement, comparaison au jeu gelé avant bascule).

**Budget de calcul sur la cible** (mesuré sur bruit synthétique, i5-1145G7, n° 72 ; remplace l'estimation de la v4)
- perch_v2, fenêtres de 5 s au pas de 2,5 s : 7 fenêtres/s, **33 h pour une campagne d'une semaine** (575 h), 9 h pour les seules heures de pic. BirdNET 2.4 : 9 h ; Bird-MAE-Base et ConvNeXt : 62 h ; BEATs : 81 h.
- Leviers, dans l'ordre : fenêtres jointives au lieu du pas de 50 % (÷ 2) ; quantification int8 du modèle ONNX (× 2 environ `[À MESURER]`) ; heures de pic d'abord (÷ 3,6) ; sous-échantillonnage des fenêtres sous H20 (÷ 5) ; OpenVINO pour le GPU intégré `[À VÉRIFIER]`. Avec les deux premiers, perch_v2 tiendrait en une nuit ou deux `[À MESURER]`.
- Réentraînement de la tête : secondes. Ré-encodage complet après changement d'encodeur : plusieurs nuits, par lots reprenables.

| Option | Coût | Remarques |
|---|---|---|
| A. Application installable (`.exe`), ONNX Runtime + interface | 3–4 sem. réparties de S4 à S16 | **retenue** ; hors ligne ; réentraînement intégré |
| B. `uv` : deux lignes PowerShell | quelques jours | étape intermédiaire et repli |
| C. PAMGuard comme hôte | 1–2 sem. | écartée (v5) : pas de réentraînement sur place |
| D. Dossier Python portable + scripts | 1 sem. | repli ultime ; ligne de commande documentée |

- Mise à jour : tête dans l'application ; encodeur par paquet (`manifest.json` : nom, version, SHA-256, **licence**, f_e, fenêtre). Un paquet dont la licence n'est pas libre est refusé (v5).
- Documentation : README ; « Interpréter un score » ; « Vérifier une file » ; « Régler le seuil d'un nouveau site » ; **guide de réentraînement pour l'ONF** (pas à pas, une nuit) ; `LICENSES.md`.
- Licences (sans avis juridique ; règle du §0, tableau du §2) : Perch 2.0 (Apache 2.0) et BirdNET 3 (CC BY-SA 4.0, avec la mention « Powered by BirdNET ») sont libres ; BirdNET 2.4, esp-aves2, NatureBEATs et AudioProtoPNet sont non libres ; Bird-MAE et ConvNeXt-BirdSet n'ont pas de licence déclarée ; AnuraSet est sous CC0. Confirmation ONF avant S12.

---

## 8. Planning (v5 : refait à partir de S3 ; S1 = 14–18/09/2026, aucune semaine réduite)

| Phase | Semaines | Contenu | Livrable | Go / no-go |
|---|---|---|---|---|
| P0 Cadrage (fait) | S1–S3 | lecture ; inventaire complet (96 292 enregistrements) ; chaîne CLI (M0–M5 écrits) ; benchmarks AnuraSet 01–07 | inventaire ; benchmarks | — |
| P0 bis Clôture et plan | S3–S4 (29/09–09/10) | vague 2 AnuraSet close le 09/10 ; licences relevées (§2) ; partition des points et plan de tirage figés et versionnés (§5.2–5.5) ; mode « extrait + intervalles » du poste ; essai chronométré sur 50 extraits ; avis de Sylvain ; lecture du rapport Biophonia ; essai d'empaquetage `.exe` (§7) | plan d'échantillonnage v1 ; tableau des licences ; `.exe` d'essai | plan validé par Sylvain et Élodie ; au plus deux non libres retenus |
| P1 Annotation v1 | S4–S7 (05/10–30/10) | lot d'entraînement 1 (≈ 450 extraits) ; jeu d'évaluation (≈ 250 enregistrements entiers) ; paquet de vérification envoyé fin S6 ; **en parallèle** : application v0 (Analyser, Résultats) | jeu v1, non validé ; paquet de vérification | ≥ 60 enregistrements positifs dans le jeu d'évaluation, sinon tirage complémentaire |
| Go des experts | S7–S8 (≈ 06/11) | retour d'Élodie et Benoît ; corrections ; règle du go (§5.7) | go écrit dans `DECISIONS.md` ; jeu gelé v1 | sans go : pas de benchmark ONF, l'application continue |
| P2 Benchmark ONF borné | S8–S10 (02/11–20/11) | encodeurs libres + au plus deux non libres × logistique (+ tête sur jetons pour un transformer libre) ; courbe d'amorçage par point ; seuil par site ; lot 2 par apprentissage actif ; module séquentiel seulement s'il reste du temps | tableau ONF ; **choix de l'encodeur le 20/11** | rappel ≥ 0,85 à précision ≥ 0,1 sur le site tenu à l'écart ; sinon pivot. Fin S12 = limite de pivot (v4) |
| P3 Outil v1 | S5–S16 (effort principal S11–S16) | export ONNX + int8 de l'encodeur retenu ; écrans ; seuil par site ; installateur ; test sur une machine ONF ; guide de réentraînement v0 | application v1 installée sur une machine ONF | 1 h d'audio en < 10 min sur l'i5 ; installation et analyse sans code par Élodie ; sinon option B |
| P4 Transfert | S17–S21 (04/01–05/02) | test homme–machine (§6) ; test d'usage par Élodie et Benoît (grille du §6) ; répétition du réentraînement par un agent ONF ; documentation | v1.1 ; rapport d'évaluation | gel fin S21 |
| P5 Rapport | S22–S26 (08/02–12/03) | rédaction, soutenance en mars 2027 (levé H11), transfert | rapport ; soutenance ; dépôt transféré | — |
| Pour aller plus loin (§14) | dès S17, si P1–P3 tiennent | détecteur appliqué aux 2 225 h de 2023 ; courbes de phénologie ; plan d'article ; gabarit multi-espèces | brouillon d'article ; gabarit documenté | seulement si le go et l'application v1 sont acquis |

- Rédaction continue dès S12 (½ j/sem.). Réunion hebdomadaire : avancement ; chiffre clé (extraits annotés, positifs, points couverts ; puis AP repondérée sur le jeu gelé) ; décision demandée.
- Chemin critique (v5) : plan de tirage (S4) → annotation v1 (S7) → go des experts (S8) → choix de l'encodeur (S10) → ONNX + application v1 (S16) → tests d'usage (S19–S20) → gel (S21). Après le 20/11 : plus de benchmark d'encodeur ; après S16 : plus de changement d'encodeur.
- Ce qui n'attend pas le go : l'application, l'export ONNX de perch_v2 (libre, en tête du benchmark 07) et sa validation sur l'i5, le guide, la lecture.

---

## 9. Risques (v5)

| Risque | Signal | Seuil | Repli |
|---|---|---|---|
| 1. Go des experts tardif | paquet non rendu | > 2 semaines après l'envoi | l'application avance (elle ne dépend pas des labels) ; benchmark décalé d'autant ; paquet réduit aux positifs et aux incertains d'abord |
| 2. Trop peu de positifs dans le tirage | lot 1 et jeu d'évaluation | < 40 enregistrements positifs à l'entraînement ; < 60 à l'évaluation | tirage complémentaire dans les strates de pic, probabilités notées ; ensuite, récolte par similarité pour l'entraînement seulement |
| 3. Erreurs d'annotation de Léonard | vérification (§5.7) | au-delà de la règle du go | réécoute de la strate fautive ; séance d'écoute des faux amis avec Élodie ; nouveau contrôle |
| 4. Le meilleur encodeur n'est pas libre | benchmark ONF | écart apparié significatif avec le meilleur libre | le libre est livré quand même, l'écart est rapporté ; distillation seulement si la licence de l'enseignant la permet (un modèle distillé d'un poids NC pourrait en être un dérivé `[À VÉRIFIER]`) |
| 5. Chant ponctuel avéré (H20 fausse) | enregistrements entiers du jeu d'évaluation | > 10 % des positifs chantent < 50 % du temps, ou > 10 % absents des 30 s tirés | extraits de 60 s ; score par enregistrement = max ou top-k ; file « suspect » réintégrée |
| 6. Empaquetage Windows | essai S4 (§7) | `.exe` qui ne tourne pas sur une machine propre | Gradio à la place de Streamlit ; sinon option B (`uv`, deux lignes) |
| 7. Cible trop lente | débit mesuré sur l'i5 (n° 72) | > 15 h pour une campagne d'une semaine après leviers | fenêtres jointives, int8, heures de pic d'abord ; BirdNET 3 si ses performances suivent |
| 8. Pas de Perch 2.0 en ONNX valide | test d'équivalence (cosinus > 0,99, §13.6) | fin S10 | BirdNET 3 (ONNX officiel, libre) ; sinon l'encodeur libre suivant |
| 9. Confusion avec les faux amis | > 30 % des 100 meilleurs candidats = faux amis après le tour 3 | précision < 0,1 à rappel 0,85 fin S11 | poids accru de persistance et rythme ; négatifs par espèce ; sortie hiérarchique |
| 10. Décalage 2023 (capteurs, saison) | patrons non reproduits | corrélation des courbes journalières < 0,5 | recalibration par jeu ; négatifs 2023 appariés ; résultat rapporté tel quel |
| 11. Les benchmarks débordent | une semaine sans extrait annoté ni écran écrit | — | dates fermes : AnuraSet le 09/10, ONF le 20/11 |
| 12. Bande saturée | candidats par heure aux heures de pic | > 500/h | fenêtres pleines ; seuillage limité aux onsets |
| 13. Perte machine ou données | machine unique | — | SSD externe quotidien, copie ONF hebdomadaire, labels sous Git |

Sortis en v5 : « aucun positif validé hors Mataroni » et « biais de sélection » (le tirage du §5 les traite à la source) ; « AnuraSet non représentatif » (AnuraSet ne fait plus que présélectionner, le benchmark ONF tranche).

---

## 10. Questions restantes (par priorité)

**Nouvelles (v5)**
1. Élodie : « libre d'accès » veut-il dire « licence qui permet l'usage par l'ONF, sans clause non commerciale » (définition du §0) ou seulement « téléchargeable gratuitement » ? Dans le second cas, BirdNET 2.4 et esp-aves2 redeviennent candidats au livrable.
2. Élodie et Benoît : combien de temps pour la vérification (≈ 2 h 30 chacun, §5.7), et quel délai de retour ? Avec quel outil écoutez-vous d'habitude (Raven, Audacity, Kaleidoscope, notre poste) ?
3. Sylvain : les cinq questions du §5.8 (enrichissement et repondération, 30 s, partition des points, poids par enregistrement, apprentissage actif après le go).
4. Élodie : quel site 2026 tenir entier à l'écart ? Existe-t-il une carte des points (distance à la crique, type de crique) pour stratifier le tirage ?
5. Élodie : à quels mois poseront les futures campagnes ? C'est la distribution que l'outil devra servir, et elle pèse sur la part de saison basse dans le tirage (§5.4).
6. Élodie : qui, à l'ONF, lancera l'application et réentraînera la tête, et sur quelle machine ? C'est pour elle que l'on teste la facilité d'utilisation (§6).

**Reprises de la v4**

7. As-tu lu jusqu'ici ?
8. « *A. blanci* chante-t-elle parfois de façon ponctuelle ? Chœur et mâle seul sont-ils distinguables à l'oreille ? » — H20, H21, règle d'agrégation. Le jeu d'évaluation y répond en partie (§5.5).
9. « Les sorties du détecteur Biophonia sur 2023 (instants, scores) et ses ≈ 1 000 extraits annotés sont-ils récupérables auprès d'ENIA ou de Trésor ? » — repère de comparaison (le rapport Biophonia est à lire, §8).
10. « Quand un point est remonté « à vérifier », que se passe-t-il : visite de terrain, contrainte d'exploitation, à quelle échelle ? Et quel volume de points l'ONF peut-il vérifier par semaine ? » — H5 et précision plancher.
11. « Taux d'émission et intervalles entre notes d'*A. blanci* ? » — H9.
12. « Les enregistreurs 2023 (Song Meter mini) et 2026 (Mini 2) ont-ils la même réponse et les mêmes réglages ? » — Décalage matériel niveau 3.
13. « Qui tranche les licences à l'ONF, et à quelle échéance ? » — Avant S12.
14. « Peut-on disposer d'une machine ONF pour mesurer le débit et tester l'installateur ? » — Risques 6 et 7.
15. « Date du prochain rapatriement de terrain ? » — H13, test réel de la boucle.

---

## 11. Montée en compétence

- Signal (S1–S4, ≈ 12 h) : Nyquist, STFT et compromis fenêtre/pas (note de 90 ms : fenêtre 16–25 ms, pas ≤ 10 ms), log-mel, passe-bande, enveloppe, onsets, SNR en bande, corrélation croisée. Docs scikit-maad, librosa ; Smith `[À VÉRIFIER]` ; Müller `[À VÉRIFIER]`.
- Écologie (S1–S6, ≈ 16 h) : Courtois et al. 2025 en entier (protocole, indice d'activité, patrons, préconisations) ; Fouquet et al. 2018 ; Cañas et al. 2023 ; PNA `[À VÉRIFIER]` ; écoute des 345 positifs et des 158 faux amis avec le tuteur.
- Apprentissage actif (S2, 4 h) : Kath et al. 2024 ; Hamer, Laber, Denton 2023.
- Arithmétique d'embedding (après S16) : directions et décalages ; prototype différentiel ; solo/chœur, distance, pluie comme décalages.

---

## 12. Hypothèses

**Levées** : H1 (Song Meter Mini 2 ; v5 : f_e et format lus à l'inventaire) · H2 (v5 : 51 enregistrements et non 345, n° 35 ; caduque avec la reprise des annotations) · H3 (vraie pour Biophonia ; pour ce projet, Mataroni = entraînement ; caduque en v5, §5.2) · H4 (extraits reçus) · H6 (i5-1145G7, 16 Go, Windows) · H7 (rapport accessible) · H8 (trois congénères dans Perch) · H10 (16 Go ; écoute humaine comme existant) · H11 (mars 2027) · H12 (annotation à la disponibilité ; caduque en v5 : l'annotation est au chemin critique) · H14 et H18 (liste des faux amis) · H15 (phénologie établie) · H17 (pas de semaine réduite) · H19 (annotations Blancinet expertes, classes A/B/C de Biophonia ; ne vaut pas pour le jeu v1, voir H25) · **H22 (v5 : 96 176 enregistrements sur 96 292 en 48 kHz stéréo, WAV et FLAC)**.

**Restantes**

| Id | Hypothèse | Levée par |
|---|---|---|
| H5 | Décision gestionnaire = intégration dans la planification forestière (zones tampons, exploitation) à l'échelle du point ou du bassin de crique | Q10 |
| H9 | IOI de l'ordre de la seconde ; plusieurs notes par fenêtre | Q11 |
| H13 | Un rapatriement de terrain avant la fin du stage | Q15 |
| H16 | Dates des relevés 2026 : décembre 2025 à février 2026, un relevé par site (v5) | `blanci status` |
| H20 | *A. blanci* chante rarement de façon ponctuelle ; détection isolée = suspicion de faux positif | Q8 ; jeu d'évaluation (§5.5) |
| H21 | Solo et chœur séparables par densité d'onsets et chevauchements | Q8 |
| H23 | Les enregistreurs 2023 et 2026 ont une réponse comparable | Q12 |
| H24 (v5) | « Libre d'accès » = licence qui permet l'usage par l'ONF, sans clause non commerciale (§0) | Q1 |
| H25 (v5) | L'annotation de Léonard seul atteint la règle du go (≤ 5 % d'erreur sur chaque classe) | vérification (§5.7) |
| H26 (v5) | 30 s suffisent à dire si un enregistrement contient le chant (§5.6) | jeu d'évaluation (§5.5) |
| H27 (v5) | Les futures campagnes se posent en saison d'activité, comme en 2026 | Q5 |

---

## 13. Spécification d'implémentation (pour Claude Code)

Ce chapitre est autosuffisant : un agent doit pouvoir démarrer le dépôt sans lire le reste. Les décisions ci-dessous sont prises ; les points marqués « à mesurer » se règlent par expérience, pas par discussion.

**(v5)** Le dépôt a dépassé cette spécification : ses écarts sont dans `DECISIONS.md`, qui fait foi. Nouveautés de la v5 à implémenter, dans l'ordre :
1. **Plan de tirage** (§5.2–5.5) : partition des points versionnée ; `candidates --plan` (strates, quotas, graine) ; probabilité de tirage enregistrée pour chaque enregistrement tiré.
2. **Annotation par intervalles** (§5.6) : mode « extrait + intervalles » du poste ; labels de fenêtres déduits pour chaque grille (positive, négative, bord) ; poids 1 / nombre de fenêtres par enregistrement.
3. **Sources exclues par défaut** : les labels `import` (Blancinet) hors de l'entraînement et de l'évaluation ; les sources `active` et `similarity` hors de l'évaluation.
4. **Vérification à l'aveugle** (§5.7) : export WAV + Excel sans le label de Léonard ; réimport des réponses des experts comme labels ; `agreement` par classe, avec bornes.
5. **Métriques repondérées** (§6) : AP, précision et fausses alarmes par heure pondérées par 1 / probabilité de tirage ; rappel par strate.
6. **Licence dans le manifeste** (§7) : paquet d'encodeur refusé si sa licence n'est pas dans la liste libre du §0.
7. **Application installable** (§7) : essai PyInstaller + Inno Setup du poste actuel ; construction Windows par GitHub Actions.

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
- ~~Commencer par M0 ; ne pas toucher à la GUI, à PAMGuard ni à la distillation avant M4.~~ (v5) L'application commence en S4, en parallèle de l'annotation (§8). PAMGuard est écarté ; la distillation reste après le choix de l'encodeur.
- (v5) Aucun résultat sur les données ONF avant le go d'Élodie et Benoît, écrit dans `DECISIONS.md` (§5.7).
- (v5) Un encodeur non libre (§0) n'entre jamais dans le livrable ; au benchmark ONF, au plus deux, comme objectifs à battre.
- (v5) Aucun label choisi par un modèle (sources `active`, `similarity`, `import`) dans le jeu d'évaluation.
- En cas de doute sur une décision de conception, se référer au numéro de section de cette feuille de route et ne pas rediscuter les choix marqués « jugement » ; les remettre en question dans un fichier `DECISIONS.md` daté.

---

## 14. Pour aller plus loin (v5, si le cœur du stage est fini en avance)

**Article de phénologie d'*A. blanci*.** Il s'agit d'aller au-delà du rapport de phénologie (*pheno-blanci.pdf*, Courtois et al. 2025) : le détecteur validé appliqué aux 2 225 h de 2023 (trois sites, une année entière), plus les points 2026.
- Questions possibles (à choisir avec Élodie) : courbes journalières et saisonnières par site ; lien avec la pluie, la température et l'humidité (capteurs associés au jeu 2023) ; différences entre criques rocheuses (Kaw, Molokoï) et lit évasé et marécageux (Mataroni, notes de Léonard) ; probabilité de détection et occupation.
- Prérequis : un rappel et une précision **connus par période**, y compris en saison basse. C'est une raison de plus pour le tirage toutes saisons du §5.8 : une fausse alarme constante dessine une fausse activité hors saison.
- Auteurs, revue et calendrier : à décider avec Élodie, Benoît et Sylvain. Rien n'est écrit avant l'application v1 (S16).

**Gabarit du projet pour d'autres espèces.**
- Ce qui est propre à *A. blanci* : bande de fréquence (4,4–5,5 kHz), durée de note et intervalle (module séquentiel), heures et mois d'activité (tirage), liste des faux amis et schéma de labels, congénères dans les classes de Perch et de BirdNET 3.
- Le reste est générique : inventaire, drapeaux, plan de tirage, poste d'annotation, encodeur, tête, évaluation, application.
- Travail : regrouper ce qui est propre à l'espèce dans une section `species` de la configuration, sans rien d'autre de codé en dur ; un guide « adapter à une nouvelle espèce » ; un essai sur une deuxième espèce. Une espèce d'AnuraSet déjà benchmarkée, ou un congénère, ne coûte presque rien.
- À faire au fil de l'eau : chaque nouvelle fonction évite de coder *A. blanci* en dur. C'est peu coûteux tant que c'est fait tout de suite.

---

## Angles morts (v5)

- **Un seul annotateur.** Le jeu v1 repose sur l'oreille de Léonard ; la vérification à l'aveugle (§5.7) mesure son erreur sur un échantillon, pas sur tout. Une confusion systématique (un faux ami précis, une condition de pluie) peut passer entre les mailles si la strate n'est pas tirée.
- **Le go des experts est au chemin critique.** Leur temps n'est pas celui du stage ; le paquet doit être court, clair, sans installation, et envoyé tôt (fin S6).
- **Sans probabilités de tirage, pas de métrique honnête.** L'enrichissement (pics, février–mars) n'est sans danger que si chaque tirage est noté. Un enregistrement ajouté « à la main » au jeu d'évaluation le casse.
- **En 2026, mois et site sont confondus.** Chaque site n'a qu'une semaine : l'effet saison ne se mesure que sur les six stations de 2023.
- **Le modèle précédent reste mal comparable.** Les détections Blancinet se jugent enfin sur nos enregistrements (§6), mais on ignore sur quoi il a été entraîné.
- **H20 porte la règle d'agrégation, le sous-échantillonnage des fenêtres et, en v5, les extraits de 30 s.** Si elle est fausse, trois économies disparaissent ensemble.
- **Phénologie comme piège.** Très prédictive d'un site à l'autre ; hors classifieur par défaut, mais la tentation reviendra, surtout avec l'article du §14.
- **La machine cible est lente.** perch_v2 : 33 h par campagne sans levier (n° 72) ; l'export ONNX et l'int8 ne sont pas encore validés sur l'i5.
- **« Libre » est une question juridique.** La définition du §0 est un jugement d'ingénieur, sans avis juridique ; l'ONF tranche (Q13).
- **Transfert à l'ONF.** Réentraînement sur i5 par des non-développeurs, sur plusieurs années : le guide de réentraînement et le réglage du seuil d'un nouveau site sont des livrables à part entière.
- **Informations orales.** Chant continu, solo/chœur, comportement des faux amis : à vérifier sur enregistrement.
- **Sur-ingénierie.** Chaque heure passée sur un benchmark est retirée à l'annotation et à l'application, qui sont désormais le chemin critique (Élodie).
- **Occupation.** La non-détection devrait passer par un modèle d'occupation `[À VÉRIFIER]` ; hors périmètre du stage, dans celui de l'article (§14).

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
- (v5) BirdNET-Analyzer : github.com/birdnet-team/BirdNET-Analyzer ; interface Gradio + pywebview, version Windows prête à l'emploi (vérifié le 29/09/2026).
- (v5) Licences lues le 29/09/2026 sur Hugging Face : `DBD-research-group/AudioProtoPNet-20-BirdSet-XCL` (CC BY-NC 4.0), `EarthSpeciesProject/NatureLM-audio` et `esp-aves2-*` (CC BY-NC-SA 4.0) ; aucune licence sur `DBD-research-group/Bird-MAE-{Base,Large,Huge}`, `ConvNeXT-Base-BirdSet-XCL`, `davidrrobinson/BioLingual`. `microsoft/unilm` (BEATs) : MIT.
