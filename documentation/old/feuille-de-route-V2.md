# Feuille de route — Stage de fin d'études ONF Guyane
## Détection acoustique automatique d'*Anomaloglossus blanci* — 15/09/2026 → 14/03/2027

Version 1 (17/09/2026). Conventions : `[HYPOTHÈSE Hn]` = information manquante remplacée par une hypothèse explicite (liste en §12) ; `[À VÉRIFIER]` = fait ou source hors bibliographie fournie ; « établi » = appuyé sur la bibliographie fournie ou sur une source vérifiée le 16–17/09/2026 (liste en fin de document) ; « jugement » = arbitrage d'ingénieur ; « tuteur » = point confirmé en réunion de mi-septembre.

**Journal des révisions v0 → v1.**
- Cadre confirmé par le tuteur (objectif, métriques, contraintes, critères de comparaison) intégré aux §1, §2 et §6.
- Prototype différentiel ajouté au §3 (forme fermée, négatifs appariés).
- Voie non supervisée ajoutée en §5 bis, avec test de pureté décisionnel en S3.
- Apprentissage actif avancé en début de phase 1, en parallèle de tout le reste.
- Classes de qualité du prestataire précédent adoptées comme schéma de labels (§5, §6).
- Negative mining ciblé étendu au fourmilier tacheté et aux autres espèces déclencheuses (§5).
- Règle d'agrégation remplacée : tout point ayant une fenêtre au-dessus du seuil est remonté pour vérification, les points étant classés par force de détection (§1, §6).
- AnuraSet intégré comme pré-benchmark des encodeurs et corpus d'adaptation (§2, §3, §6).
- YAPAT intégré comme établi d'annotation candidat et référence d'architecture (§4, §5).
- PAMGuard ajouté comme option d'hébergement du livrable (§7).
- BirdCLEF+ 2026 vérifié ; export ONNX communautaire de Perch 2.0 repéré (§2).
- Modèle maison reformulé en distillation (§3).
- Attentive probing reformulé (tokens avant agrégation, même couche) ; tête de fusion formalisée comme stacking à deux niveaux (§3, §6).
- Combinaisons de modèles conservées à la demande du stagiaire, conditionnées à la taille du jeu gelé (§6).
- Arithmétique d'embedding notée en perspective (§3, §11).
- Planning recalé sur la feuille de route manuscrite en six étapes (§8).

**Lecture des positions** (`[HYPOTHÈSE H4]` : extraits verbatim non fournis). Encadrant A : déconseille les embeddings BirdNET (dépendance TensorFlow, base de code BirdNET-Analyzer), oriente vers des modèles PyTorch sur HuggingFace — encodeur BEATs de NatureLM-audio, Bird-MAE — en citant la revue de Schwinger et al., vraisemblablement sa v1, et propose Streamlit ou Gradio pour la réannotation. Encadrant B : embeddings Perch 2.0 en première intention, extraits via bacpipe, probing linéaire et apprentissage actif ; objecte que NatureLM-audio travaille à 16 kHz. **Arbitrage** : (i) aucun composant TensorFlow à l'exécution dans le livrable ; (ii) l'encodeur est choisi par benchmark sur les données du projet, sous filtre de licence et de déployabilité ; (iii) l'objection des 16 kHz devient un test empirique (§2, §10).

---

## 1. Reformulation du problème

**Cadre confirmé (tuteur).** Objectif : un détecteur robuste d'*A. blanci* qui fonctionne sur de nouveaux sites. Métriques : average precision et rappel. Contraintes : six mois, un portable, un logiciel facile à prendre en main par un naturaliste. Critères de comparaison entre approches : score, vitesse de calcul, prise en main. « Nouveaux sites » impose la validation groupée par site (§6) ; « prise en main » devient une colonne du benchmark (§2).

**Objectif écologique.** Établir, pour chacun des 86 points d'écoute, si *A. blanci* (En danger, Plan national d'actions) y chante, pour que la gestion forestière — exploitation, dessertes, zones tampons autour des criques — en tienne compte `[HYPOTHÈSE H5]`.

**Problème d'apprentissage.** Détection d'un événement rare et très bref (note de 0,09–0,10 s, énergie dominante entre 4,5 et 5,4 kHz) dans un flux à faible cycle utile (2 min par heure, 3,3 % du temps), avec quelques dizaines de positifs, une classe négative immense et structurée (congénères, fourmilier tacheté, orthoptères, autres espèces déclencheuses, pluie, cours d'eau) et un décalage de domaine attendu entre sites et entre jeux. Posé proprement : un problème de **classement** sous très fort déséquilibre. Le modèle ordonne des fenêtres par vraisemblance de présence, un humain vérifie le haut de la liste, la décision gestionnaire agrège les vérifications ; à prévalence très faible, 99 % de spécificité donnent une précision de l'ordre du pourcent, donc la vérification humaine est structurelle et le rappel prime.

**Rapport signal sur bruit (SNR, tuteur).** Rapport entre l'énergie du cri et celle de ce qui l'entoure dans la même bande, en décibels. Il structure tout le projet : un cri net et un cri lointain chevauché par un chœur sont deux problèmes différents pour le même modèle. Les trois classes de qualité du prestataire précédent `[HYPOTHÈSE H19]` en sont une échelle ordinale ; le SNR estimé sur l'enveloppe en bande en est une version continue (§6).

**Unité de décision.** Cinq niveaux : note, fenêtre d'analyse (3 ou 5 s, imposée par l'encodeur), enregistrement (2 min), point d'écoute, session. Choix :

| Niveau | Rôle | Conséquence |
|---|---|---|
| Fenêtre, grille glissante à pas ≤ 2,5 s (aucune note coupée) | Apprentissage, score, vérification humaine | Métriques de développement : AP, rappel à précision fixée |
| Enregistrement de 2 min | Restitution : présence confirmée / candidat non vérifié / rien | Score = max ou top-k des fenêtres |
| Point × période | Décision gestionnaire (règle du stagiaire, priorité au rappel) | « À vérifier » dès qu'une fenêtre dépasse le seuil d'exploitation ; points classés par force de détection (score maximal, nombre de fenêtres, jours distincts) ; « présence confirmée » seulement après validation humaine ; sinon « non détecté », jamais « absent » |

**Écart entre métrique optimisée et décision.** (1) L'AP mesure un classement ; la décision est un statut par point avec asymétrie de coût : une fausse absence peut détruire un habitat (irréversible), une fausse présence coûte du temps de vérification (réversible). (2) Une absence ne se démontre pas. Deux pics d'une heure à 2 min par heure : 4 min par jour dans la fenêtre favorable. Si p est la probabilité qu'un échantillon de 2 min contienne un cri détecté (probabilité de chant × rappel), la probabilité d'au moins une détection sur D jours vaut 1 − (1 − p)^(2D) : pour p = 0,05, 76 % à 14 jours, 95 % à 30 jours. Le déploiement permanent rend la non-détection crédible ; le rappel entre multiplicativement dans p. (3) La règle d'agrégation ne fixe plus de seuil de présence : elle remonte tout ce qui dépasse le seuil et déplace la charge sur la vérification humaine. Le classement des points par force de détection est ce qui rend cette charge gérable, et la précision plancher du §6 ce qui la borne.

**Problème d'échelle.** Une note de 0,09 s occupe 1,8 % d'une fenêtre de 5 s (Perch 2.0, Bird-MAE) et 3 % d'une fenêtre de 3 s (BirdNET). L'embedding — vecteur résumant la fenêtre par agrégation de représentations locales — est dominé par tout ce qui n'est pas la grenouille.

*Pour un détecteur d'événements en amont* : il évite la dilution de la note dans l'agrégation temporelle et la variance due à sa position dans la fenêtre ; il mesure directement durée, fréquence et intervalles ; il réduit le nombre de fenêtres à encoder.

*Contre* : le rappel du détecteur plafonne celui du système ; la bande 4,4–5,5 kHz est saturée de chants d'autres espèces (tuteur), donc un détecteur en bande déclenche presque en permanence et n'isole rien ; les encodeurs et les résultats de la revue (établi : Schwinger et al. 2026) portent sur des fenêtres pleines de 3–5 s, pas sur 0,1 s complétée de zéros ; surtout, le caractère diagnostique d'*A. blanci* est la **répétition de notes isolées** pendant des heures (tuteur) : si plusieurs notes tombent dans 5 s `[HYPOTHÈSE H9]`, la fenêtre pleine contient le rythme qui la distingue des trains soudés des congénères et du chant ponctuel du fourmilier. Enfin, l'*attentive probing* — petite tête d'attention entraînée sur les représentations locales (tokens) d'un encodeur gelé, prises avant l'agrégation temporelle, le gradient ne traversant que la tête — est nécessaire pour exploiter les transformers (établi) et répond à la parcimonie de l'événement sans détecteur amont (jugement).

Conséquence de la saturation de la bande (tuteur) : le pouvoir discriminant ne viendra pas du filtrage fréquentiel mais de la structure temporelle, de la persistance du chant et de l'heure (§3, module séquentiel).

*Protocole pour trancher (S3–S5, même jeu annoté, validation par site).* A = grille 5 s, pas 2,5 s ; B = pas 1 s ; C = fenêtres centrées sur les événements du détecteur DSP ; D = fenêtres de 1 s complétées de zéros ; E = pour Bird-MAE et BEATs, agrégation des tokens par moyenne, maximum ou attention. Mesures : AP, rappel à précision ≥ 0,1, rappel propre du détecteur sur les notes annotées (≥ 0,95 exigé, jugement). Règle : C n'est adopté que s'il dépasse A/B de plus que l'incertitude (≈ 0,1 d'AP avec 30 positifs, §6) ; sinon le détecteur ne sert qu'au post-traitement séquentiel.

---

## 2. Benchmark initial des embeddings (S1–S3)

**Critères (tuteur).** Trois colonnes dans le tableau final : score (AP, rappel à précision fixée), vitesse (fenêtres/s sur le M4, temps pour 1 h d'audio), prise en main (installation, dépendances, documentation, exécutable par un non-développeur). Une quatrième, propre au projet : accès aux tokens avant agrégation, condition de l'attentive probing.

**Pré-benchmark sur AnuraSet (S2).** Avant les données ONF, faire tourner la même comparaison sur AnuraSet (établi : Cañas et al. 2023) : 93 000 fenêtres de 3 s, 42 anoures, quatre sites, labels faibles et forts, données CC0. Choisir deux ou trois espèces à note brève dans la bande 3–6 kHz, validation groupée par site, milliers de positifs. Résultat : un classement des encodeurs sur des grenouilles en paysage sonore néotropical, avec une puissance statistique que les 30 cris d'*A. blanci* n'auront jamais. Le benchmark ONF ne fait ensuite que vérifier la cohérence. Limite : biomes et enregistreurs différents, aucun *Anomaloglossus* attendu ; c'est un indicateur, pas une garantie.

**J1–J2 — inventaire et jeu annoté v0.** Inventaire (format, fréquence d'échantillonnage, enregistreur, nommage, sites, calendrier) ; conversion en WAV PCM mono à fréquence native (ffmpeg ou sox si format propriétaire `[À VÉRIFIER]`). Positifs = fenêtres contenant intégralement au moins une note labellisée, avec leur classe de qualité. Négatifs de trois origines tracées et **appariés** aux positifs (mêmes sites, mêmes heures, mêmes saisons, condition du prototype différentiel du §3) : (a) tirage aléatoire ; (b) fenêtres à forte énergie en bande sans note labellisée, prises à des sites sans présence connue `[HYPOTHÈSE H2]` ; (c) espèces déclencheuses si des enregistrements existent `[HYPOTHÈSE H14, H18]`. Ratio 1:20 à 1:50 (jugement). Contrôle annexe : un *Anomaloglossus* dans les classes de Perch 2.0 `[HYPOTHÈSE H8]` ?

**J2–J3 — extraction.** `bacpipe` pour `birdnet`, `perch_v2`, `birdmae`, `beats`, `naturebeats`, `protoclr`, plus `perch_bird` et `convnext_birdset` s'ils tournent sans effort. Relevé par modèle : exécutable ou non, débit, mémoire, dimension, tokens accessibles — il alimente le risque 4.

**J3–J4 — évaluation.** Trois sondes : k plus proches voisins (cosinus, k ∈ {1, 3, 5}), prototype différentiel (§3), régression logistique L2 (C par validation interne, classes rééquilibrées). bacpipe fournit probing linéaire et kNN (établi) ; son découpage est `[À VÉRIFIER]` — s'il est aléatoire, un script maison impose une validation **groupée par site** : leave-one-site-out si les positifs couvrent au moins trois sites, sinon leave-one-recording-out en signalant que la généralisation n'est pas mesurée. Vingt sous-échantillonnages des négatifs ; comparaison **par paires sur les mêmes plis**.

**J5 — tableau de décision** : modèle × (licence, exécution sur M4 : native / ONNX / déportée / impossible, débit, prise en main, tokens, AP avec intervalle interquartile, rappel à précision 0,5 et 0,1, kNN top-1). Une UMAP peut être montrée ; elle ne décide de rien.

**Métriques qui concluent** : AP ; rappel à précision fixée ; différences appariées avec bootstrap par enregistrement ; stabilité du classement entre plis. **Trompeuses à cet effectif** : exactitude ; AUROC — avec 30 positifs contre 1 500 négatifs, 0,95 d'AUROC est compatible avec une précision catastrophique ; F1 au seuil 0,5 ; AMI/ARI sur deux classes minuscules. Un positif vaut 3,3 points de rappel : moins de 0,1 d'AP d'écart = **égalité**, départagée par licence, déployabilité, vitesse et prise en main.

**Modèles qui ne tourneront pas.** `perch_v2` : TensorFlow 2.20 et GPU requis (établi), sans accélération sur Apple Silicon. Replis : (1) wrapper bacpipe sur CPU `[À VÉRIFIER]` ; (2) export ONNX — un fichier `perch_v2_no_dft.onnx` circule parmi les ressources d'un notebook de BirdCLEF+ 2026 (vérifié le 16/09/2026), ce qui suggère qu'un export communautaire existe déjà : à récupérer et à valider contre les embeddings de référence avant usage ; (3) extraction déportée sur une session GPU tierce, limitée au jeu de benchmark ; (4) variante CPU officielle si elle sort. Le repli (3) suffit au benchmark, pas au déploiement (risque 6). `birdnet` : TFLite sur CPU, débit acceptable. `naturebeats` : encodeur seul, jamais le LLM de 8 milliards de paramètres `[À VÉRIFIER que bacpipe le sépare]`. Bird-MAE-Base (92,9 M paramètres) tient en mémoire unifiée `[HYPOTHÈSE H10]`.

**Du désaccord à la question empirique.** « BirdNET plafonne » → AP(birdnet) contre les autres. « 16 kHz est trop bas » → AP(beats, naturebeats) à 16 kHz contre AP(birdmae) à 32 kHz, **avec un contrôle** : birdmae alimenté par le même audio filtré passe-bas à 8 kHz puis rééchantillonné à 32 kHz, pour isoler l'effet de la bande passante. « Perch 2.0 est le meilleur » : établi pour BirdSet et BEANS en v2 de la revue, non établi pour une note d'anoure → AP(perch_v2) sur AnuraSet puis sur nos données, sous réserve d'exécutabilité. Sortie : deux encodeurs au plus pour la phase 1, après filtre de licence (§7).

**BirdCLEF+ 2026 (vérifié le 16/09/2026).** Édition multi-taxons sur le Pantanal (oiseaux, amphibiens, insectes, reptiles), 234 espèces, score en ROC-AUC macro sur fenêtres de 5 s, inférence limitée à un notebook CPU de 90 min, terminée le 3 juin 2026. Les working notes CEUR-WS paraissent après CLEF 2026 (21–24 septembre) ; les fils « 1st place solution » de Kaggle sont déjà disponibles. Transférable : pseudo-étiquetage des paysages sonores, lissage temporel des scores entre fenêtres voisines, gestion du décalage focal → paysage sonore, calibration des seuils. Non transférable : ensembles lourds et entraînement sur des dizaines de milliers de clips.

---

## 3. Cartographie des approches candidates

| Approche | Principe | Coût | Annotations | Robustesse au décalage de domaine | Faisabilité M4 | Licence | Verdict |
|---|---|---|---|---|---|---|---|
| Template matching | Corrélation croisée d'un spectrogramme de référence de la note avec celui de l'enregistrement (scikit-maad `[À VÉRIFIER]`) | 2–3 j | 1–10 références | Faible (SNR, réverbération, variation individuelle) | Triviale | Libre | Baseline obligatoire (§6) ; conserve la résolution de la note et ne dépend d'aucun encodeur ; insuffisant seul dans une bande saturée |
| Seuillage spectral 4,4–5,5 kHz | Passe-bande, enveloppe d'énergie, seuil adaptatif, filtre de durée 0,08–0,11 s | 2–3 j | 0 (Fouquet et al. 2018) | Faible : la bande est saturée (tuteur) | Triviale | Libre | Générateur d'instants de notes pour le module séquentiel ; jamais décideur, et probablement pas filtre amont (§1) |
| Indices acoustiques | Statistiques résumant un enregistrement | 1–2 j | 0 | Sans objet | Triviale | Libre | Contrôle qualité seulement (pluie, panne, saturation) |
| Prototype simple (few-shot) | Similarité cosinus au centroïde des positifs | 1 j | 5–30 | Sensible au fond sonore partagé | Oui | Par encodeur | Amorçage (`index`, §4) |
| **Prototype différentiel** (few-shot, forme fermée) | Score $\langle x, \mu_+ - \mu_-\rangle + b$ avec négatifs appariés | 1 j | 10–50 | Retire le fond sonore partagé si l'appariement est respecté | Oui | Par encodeur | Classifieur de la phase d'amorçage ; baseline permanente |
| Linear probing sur embeddings gelés | Régression logistique L2 sur l'embedding d'un encodeur fixé | 1 sem. avec bacpipe | Dizaines → centaines | Dépend de l'encodeur ; Perch 2.0 couvre > 14 500 espèces dont des amphibiens (établi) | PyTorch : oui ; Perch 2.0 : conditionnel (§2) | Par encodeur | **Approche principale** dès que l'effectif dépasse le prototype |
| Attentive probing | Tête d'attention sur les tokens de la dernière couche, pris avant l'agrégation temporelle ; même profondeur que l'embedding, granularité différente | 1–2 sem. (accès aux tokens) | ≥ 150–200 positifs (jugement) | Idem | Oui | Idem | Phase 2 si l'effectif le permet (établi : nécessaire pour les transformers) |
| Few-shot contrastif | Prototypes de classe dans l'espace d'embedding d'un encodeur pré-entraîné par contraste | 1–2 sem. | Dizaines | ProtoCLR conçu pour le transfert focal → paysage sonore (établi : Moummad et al. 2024) | Oui, petit modèle | `[À VÉRIFIER]` | Second candidat |
| Clustering non supervisé | UMAP pour voir, HDBSCAN sur ACP pour calculer | 2–3 j | 0 | — | Oui | Libre | Voie parallèle, §5 bis ; statut de détecteur tranché par le test C1 |
| PEFT LoRA / fine-tuning | Adaptation d'une partie ou de la totalité des poids de l'encodeur | 2–4 sem. | Centaines à milliers — AnuraSet comme corpus d'adaptation aux anoures | Risque de sur-apprentissage au site | Partiel possible sur M4 | Par encodeur | Conditionné aux positifs validés ou à AnuraSet ; hors chemin critique |
| Modèle maison = distillation | Petit CNN sur la bande 3–7 kHz entraîné à reproduire les scores du pipeline gelé sur des millions de fenêtres non annotées | 2–3 sem. | 0 (apprend de l'enseignant) | Hérite de l'enseignant | Oui | Libre | Candidat livrable léger (quelques Mo, CPU rapide) ; exige un pipeline enseignant stabilisé, donc S13 au plus tôt ; hors chemin critique |

**Prototype différentiel (ajout v1).** Soit $\mu_+$ le centroïde des embeddings positifs et $\mu_-$ celui de négatifs **appariés** (mêmes sites, mêmes heures, mêmes saisons). En décomposant approximativement un embedding en $x \approx c_{\text{site}} + c_{\text{espèce}} + \varepsilon$, la composante de fond sonore ne s'annule pas dans $\mu_+$, puisque les positifs partagent leurs sites. Elle s'annule dans la différence :

$$w = \mu_+ - \mu_- \approx c_{\text{espèce}}, \qquad s(x) = \langle x, w\rangle + b$$

Le score cesse de récompenser la ressemblance au paysage sonore et ne récompense plus que ce qui distingue l'espèce de son fond. Cet estimateur est le classifieur du centroïde le plus proche sous variance commune ; il coïncide avec l'analyse discriminante linéaire lorsqu'on suppose la covariance identité. Sa forme $\langle x, w\rangle + b$ est celle d'une régression logistique, dont il constitue la solution fermée : aucune optimisation, aucun hyperparamètre, utilisable dès 10 positifs là où la régression en demande plusieurs dizaines. Il est donc le classifieur de l'amorçage, remplacé par la régression logistique quand l'effectif le permet, et conservé ensuite comme baseline. **Condition critique** : l'appariement des négatifs. Des négatifs prélevés ailleurs ou à d'autres heures feraient disparaître dans la différence autre chose que le fond, et $w$ pointerait vers un artefact de site. C'est une contrainte de construction du jeu de négatifs dès le §2. **Protocole (S3–S4)** : sur les mêmes plis, comparer $\cos(x, \mu_+)$, $\langle x, \mu_+ - \mu_-\rangle$ et la régression logistique L2 ; l'écart entre les deux premiers mesure la part de fond sonore captée par le prototype simple. C'est aussi la première instance concrète de l'**arithmétique d'embedding** évoquée par le tuteur pour le long terme : manipuler des directions de l'espace de représentation (différences, décalages) comme des concepts. À reprendre après S16 seulement (§11).

**Module séquentiel : post-traitement explicite, oui, et plus central qu'en v0.** La bande étant saturée, la structure temporelle devient le principal discriminant, et les informations du tuteur la précisent : *A. blanci* chante pendant des heures par notes isolées répétées ; le fourmilier tacheté `[À VÉRIFIER : nom scientifique]` a un chant ponctuel ; les autres *Anomaloglossus* ont des motifs différents et chantent à d'autres heures. Trois familles de descripteurs, calculés depuis l'audio sans passer par l'encodeur :

1. **Rythme intra-fenêtre** : débuts de notes (onsets) sur l'enveloppe de la bande ; intervalles entre débuts (IOI) ; fraction d'IOI < 0,1 s (signature de train), distribution des durées, régularité. IOI d'*A. blanci* attendus de l'ordre de la seconde `[HYPOTHÈSE H9]`.
2. **Persistance** : nombre de fenêtres consécutives au-dessus du seuil dans l'enregistrement de 2 min, et présence dans les échantillons horaires voisins du même point. Un chant de plusieurs heures et un chant ponctuel se séparent ici, pas dans l'embedding.
3. **Heure et saison** : covariables issues de la phénologie (Courtois et al. 2025) et des heures de chant des congénères `[HYPOTHÈSE H18]`.

**Tête de fusion.** Régression logistique à deux niveaux (stacking : Wolpert 1992 `[À VÉRIFIER]`) : entrées = score de `head` **hors-pli** (prédiction obtenue par validation croisée, par une version de `head` n'ayant jamais vu l'exemple, sans quoi la fusion apprend sur des scores mémorisés et surestime `head`), plus 2 à 3 descripteurs séquentiels. Contrainte d'effectif : environ dix positifs par coefficient (règle des événements par variable, Peduzzi et al. 1996 `[À VÉRIFIER]`), donc au plus 4–5 entrées avec 80 positifs, et moins si les positifs viennent des mêmes enregistrements. Le score séquentiel module ; il n'oppose jamais de veto à un score d'embedding élevé.

---

## 4. Architecture cible

```
audio brut (format et f_e inconnus, H1)
  └─ ingest      inventaire, métadonnées (site, horodatage, enregistreur) → SQLite
  └─ decode      forme d'onde float32 mono, f_e native conservée (soundfile ; ffmpeg si exotique)
  └─ grid        fenêtres (recording_id, offset_s, dur_s), INDÉPENDANTES de l'encodeur
  └─ encoder[k]  rééchantillonnage vers f_e(k) ; spectrogramme interne au modèle ; E_k ∈ R^{N×d_k}
  └─ store       Parquet partitionné encoder_id / site / mois, float16
  └─ index       similarité cosinus, exhaustive par fragments ; positifs et négatifs appariés ; FAISS optionnel
  └─ head[v]     prototype différentiel puis régression logistique → score par fenêtre
  └─ sequential  descripteurs de rythme, persistance, heure → tête de fusion avec head
  └─ aggregate   fenêtre → enregistrement → point × période, classement par force de détection
  └─ queue       file de vérification (GUI) → labels append-only → ré-entraînement de head
```

`decode` s'arrête à la forme d'onde ; le spectrogramme (log-mel : axe des fréquences compressé selon l'échelle mel, amplitudes en logarithme) est calculé à l'intérieur de chaque encodeur avec ses propres paramètres, pour ne pas s'écarter de son pré-entraînement. Le spectrogramme apparaît aussi dans l'interface de vérification et dans les approches DSP (template matching, seuillage), pas dans le flux principal.

**Interfaces (signatures, pas de code).**
- `class Encoder(Protocol)` : `name`, `version`, `sample_rate`, `window_s`, `dim` ; `embed(wav: np.ndarray, sr: int) -> np.ndarray` (n_fenêtres, dim) ; `embed_tokens(...) -> np.ndarray | None` (n_fenêtres, n_tokens, dim) pour l'attentive probing. Implémentations : `BacpipeEncoder` (recherche), `OnnxEncoder` (livrable).
- `train_head(X, y, groups, C_grid) -> Head` (scikit-learn) ; `Head.score(E) -> np.ndarray`
- `detect_onsets(wav, sr, band=(4400, 5500)) -> np.ndarray` (scikit-maad, scipy)

**Points de découplage.** (1) La grille est définie sur (enregistrement, décalage) : les labels s'y rattachent, pas aux embeddings, et survivent à tout changement d'encodeur. (2) L'encodeur est derrière le protocole, sélectionné par un fichier de configuration. (3) La GUI parle à une couche de service, jamais aux modèles.

**Stockage et volume.** 86 points × 48 min/jour ≈ 69 h d'audio par jour `[HYPOTHÈSE H16]` ; à pas 2,5 s, ≈ 3 millions de fenêtres par mois, soit ≈ 4,5 Go/mois en 768-d float16 (Bird-MAE) ou ≈ 9 Go/mois en 1536-d (Perch 2.0). Le float16 ne touche jamais l'audio, archivé intact ; il ne concerne que les embeddings, recalculables. Parquet, colonne de taille fixe, un fichier par (encodeur, site, mois) ; métadonnées, labels, versions et décisions dans SQLite. Inspiration YAPAT : les embeddings vivent dans la base (pgvector chez eux ; `sqlite-vec` ou Parquet + calcul matriciel ici, sans serveur), et la recherche est une requête.

**Index de similarité.** Les requêtes (positifs et négatifs étiquetés) sont empilées en une matrice $n \times d$ et multipliées par la base $d \times N$ : une seule passe, quelques dizaines de secondes de calcul sur MPS, dominées par la lecture disque. FAISS seulement si la GUI exige moins d'une seconde. *Append-only* : chaque partition est indexée à l'arrivée, aucune suppression n'est nécessaire. C'est le mécanisme qui absorbe la croissance du corpus à coût constant (retrieval, sans génération).

**Changement d'encodeur.** Embeddings clés par `encoder_id`. Un changement déclenche une tâche de fond qui ré-encode le corpus, ré-entraîne la tête sur les mêmes labels et produit un rapport de migration (AP ancienne contre nouvelle sur le jeu gelé) avant basculement. Chaque décision porte (`encoder_id`, `head_version`, `threshold_id`) : une analyse passée reste reproductible ; seules les fenêtres non vérifiées sont rescorées, avec journal des changements de statut.

**Ce que le système fait et ne fait pas quand les données grossissent.** Encodeur figé ; index en croissance (ajout seul) ; tête ré-entraînée de zéro sur tous les labels cumulés, ce qui n'est pas de l'apprentissage continu au sens technique (mise à jour sans accès au passé) mais un réentraînement périodique, sans oubli possible parce que le modèle est minuscule ; seuils recalibrés périodiquement, car c'est là que vit la dérive.

---

## 5. Amorçage et apprentissage continu

**Ordre.** L'apprentissage actif démarre dès la fin du benchmark initial (S3) et tourne en parallèle de tout le reste : c'est lui qui fabrique les positifs sans lesquels attentive probing, LoRA et combinaisons restent inaccessibles.

**Schéma de labels.** {blanci-net, blanci-lointain-chevauché, blanci-faible, congénère, fourmilier, orthoptère, autre, incertain}. Les trois niveaux de qualité reprennent les classes du prestataire précédent `[HYPOTHÈSE H19]` ; ils servent l'évaluation stratifiée (§6) et, en entraînement, une pondération ou un ordre de présentation (nets d'abord). Les labels d'espèces négatives sont conservés pour une évaluation hiérarchique.

**Tour 0 (S3–S4) — recherche par similarité.** Embeddings des cris de référence ; requêtes par exemple, par prototype simple et par prototype différentiel, sur les fenêtres des heures de pic (6–7 h, 16–17 h) des mois humides d'abord, tous sites du jeu 1. File : 300 meilleures similarités + 100 fenêtres aléatoires des heures de pic + 100 négatifs durs en bande. Rendement attendu : 20 à 80 nouveaux positifs `[HYPOTHÈSE H2, H9]`.

**Negative mining ciblé (tuteur).** Trois sources d'espèces déclencheuses, chacune avec son label : les congénères (*A. degranvillei*, *A. dewynteri*, *A. surinamensis*, *A. baeobatrachus*, *A. saramaka*), aux motifs et aux heures différents ; le fourmilier tacheté, au chant ponctuel ; les autres espèces déclencheuses signalées par le tuteur `[HYPOTHÈSE H18]`. Si des enregistrements de référence existent `[HYPOTHÈSE H14]`, ils servent de requêtes de similarité. Sinon, ils se récoltent en vérifiant les fenêtres les mieux classées : ce sont les **négatifs durs**, ceux qui déplacent la frontière. Les orthoptères et le fond sonore se récoltent en volume par clustering (§5 bis, C2) : négatifs **faciles**, utiles à l'équilibre des classes et au calibrage, inutiles à la frontière. Les deux mécanismes sont nécessaires et non substituables.

**Phénologie.** Priorité aux heures de pic et aux mois humides `[HYPOTHÈSE H15]`, stratification par site ; 20 % de chaque file tirée hors de ces strates pour ne pas enfermer le modèle dans les conditions de pic.

**Apprentissage actif (tours 1–5, S5–S12).** Après chaque ré-entraînement de la tête : file = 60 % de fenêtres incertaines, 20 % de scores maximaux, 20 % d'échantillon aléatoire stratifié. La strate aléatoire n'entraîne rien : elle mesure ce que la pré-annotation ne voit pas, en particulier les faux négatifs confiants qu'une file d'incertains ne remonte jamais. Lots de 100–150 fenêtres, sélection par grappes contre les quasi-doublons. Arrêt : gain d'AP sur le jeu gelé < 0,02 sur deux tours consécutifs (jugement).

**Établi d'annotation.** Essai de YAPAT en S2, une demi-journée : installation, 200 fenêtres chargées, ergonomie. Conditions de poursuite : dépendance BirdNET contournable, et pas plus d'une journée de mise en place. Sinon, prototype Streamlit minimal, ou Whombat (établi : Martínez Balvanera et al. 2025) pour l'interface de vérification. YAPAT et Whombat sont des outils de travail, pas le livrable.

**Budget d'annotation.** 250–350 fenêtres/h avec raccourcis et spectrogramme (jugement). Stagiaire ≈ 66 h sur le stage, soit 16 000 à 20 000 fenêtres. Expert (naturaliste ONF ou auteurs du rapport de phénologie `[HYPOTHÈSE H12]`) : 3 h de calibration (100 fenêtres en commun, accord inter-annotateurs mesuré), puis 1 h toutes les deux semaines sur les files `incertain`, `congénère` et `fourmilier` ≈ 14 h. Un label `blanci` n'entre en entraînement qu'après validation experte.

**Boucle après déploiement.** Les naturalistes ONF vérifient une file hebdomadaire de ≈ 150 fenêtres en saison de chant ; l'application journalise auteur et horodatage ; ré-entraînement local de la tête, comparaison affichée avec la version en service sur le jeu gelé embarqué, nouvelle version seulement si l'AP n'a pas baissé. Invariants : labels validés immuables ; analyses passées gardant leur triplet (encodeur, tête, seuils) ; encodeur changé uniquement par migration. Le stage commence en saison sèche `[HYPOTHÈSE H15]` : la boucle est démontrée en simulation (§6), et en conditions réelles seulement si un rapatriement a lieu avant février `[HYPOTHÈSE H13]`.

---

## 5 bis. Voie non supervisée : clustering

Voie parallèle à la voie supervisée, avec deux fonctions distinctes : produire des étiquettes négatives en volume à bas coût, et explorer la structure du corpus. Son statut de détecteur est une question ouverte, tranchée par le test C1.

**Choix méthodologique (jugement).** Clustériser sur les embeddings réduits par ACP (≈ 50 composantes), pas sur la projection UMAP, qui déforme les densités et produit des groupes artéfactuels ; UMAP reste l'outil de visualisation. HDBSCAN plutôt que k-moyennes : pas de nombre de groupes à fixer, et une classe de rejet adaptée à un corpus majoritairement fait de fond sonore.

**C0 — exploration globale (S3, 1 h).** Clustering sur un échantillon de tout le corpus. Diagnostic imposé : information mutuelle ajustée (AMI) entre l'affectation aux groupes et l'étiquette de site. Une AMI élevée confirme que le premier étage de structure est le site ; le clustering global ne sert alors qu'à repérer des anomalies (enregistreur dérivant, saturation, espèce inattendue).

**C1 — test de pureté (S3, 1 j).** Sur un site riche en positifs : les positifs connus plus 5 000 fenêtres tirées aux mêmes heures. Deux variantes : fenêtres pleines de 5 s contre fenêtres recadrées par le seuillage spectral. Trois mesures :

| Mesure | Définition | Seuil (jugement) |
|---|---|---|
| Rappel du meilleur groupe | Part des positifs tombant dans un seul groupe | ≥ 0,5 |
| Enrichissement | Prévalence des positifs dans ce groupe rapportée à la prévalence globale | ≥ 20 |
| AMI avec le site | Degré auquel le clustering capte le site plutôt que l'espèce | Faible |

Seuils atteints : le clustering est un détecteur crédible et entre au benchmark du §6. Non atteints : outil d'annotation et d'exploration, sans rôle décisionnel. L'écart entre variantes mesure l'effet du recadrage et alimente l'arbitrage d'échelle du §1.

**C2 — negative mining en volume (S4–S7).** Étiquetage en bloc des groupes homogènes après contrôle de dix exemples chacun, label de groupe conservé. Produit des négatifs faciles en grande quantité ; les négatifs durs viennent du §5.

**C3 — sub-clustering par site (S6–S9).** Le premier étage étant le site (C0), clustériser directement site par site sur les heures de pic ; inspection du voisinage du prototype positif pour récolter des positifs sur des sites sans présence connue.

**Limite structurelle.** Une note de 0,09 s pèse 2 % d'une fenêtre de 5 s : la structure dominante des embeddings est le paysage sonore, non l'espèce. Ce n'est pas une question de couverture taxonomique des modèles — un encodeur qui ignore *A. blanci* peut représenter une note brève à 4,75 kHz — mais de rapport entre signal cible et fond. Le recadrage est la seule voie qui modifie ce rapport, d'où l'importance de la variante recadrée de C1.

---

## 6. Protocole d'évaluation

**Métriques confirmées (tuteur) : AP et rappel.** S'y ajoutent le rappel à précision ≥ 0,1 (plancher) et ≥ 0,5 ; les fausses alarmes par heure d'audio au seuil d'exploitation ; le rappel par enregistrement ; et l'accord du classement des points avec le jugement expert. **Rejetées** : exactitude, AUROC (annexe seulement), F1 au seuil 0,5, kappa.

**Stratification par qualité.** Rappel calculé séparément par classe de qualité (net / lointain-chevauché / faible) et, en continu, par tranche de SNR estimé (énergie en bande pendant la note contre énergie en bande dans les 0,5 s voisines). Un rappel global de 0,9 peut cacher un rappel de 0,3 sur les cris faibles, qui sont précisément ceux des sites à faible densité, ceux que la gestion doit protéger.

**Séparation par site.** Chaque point a son paysage sonore et souvent les mêmes individus ; un découpage aléatoire met des fenêtres d'un même enregistrement des deux côtés, le modèle apprend le site et le score ne mesure pas ce que le tuteur demande : de nouveaux sites. Donc validation groupée par site (5 plis) sur le jeu 1, et **jeu 2 réservé à la généralisation** : hors entraînement avant la phase 4, et seulement par ses labels d'audit aléatoire `[HYPOTHÈSE H3]`.

**Jeu de test gelé.** Construit en S6–S9 : au moins 10 sites du jeu 1, 60 enregistrements de 2 min écoutés intégralement (≈ 2 h d'audio, 4–6 h d'annotation), plus tous les enregistrements positifs connus de sites tenus à l'écart, avec classes de qualité. Versionné, jamais utilisé pour l'entraînement.

**Trois critères de comparaison (tuteur).**

| Critère | Mesure | Comment |
|---|---|---|
| Score | AP, rappel à précision fixée, par classe de qualité | Plis groupés par site, bootstrap par enregistrement |
| Vitesse | Temps pour 1 h d'audio sur le M4 (CPU seul, puis MPS) | Chronométré sur le même fichier, encodage + tête |
| Prise en main | Installation, dépendances, taille, documentation, exécutable sans code | Grille à 5 points, remplie par le stagiaire puis par un naturaliste en phase 4 |

**Priorité au rappel et plancher de précision.** Le rappel parce qu'une vérification corrige une fausse alarme et que rien ne corrige un site manqué. Le plancher parce que vérifier doit coûter moins qu'écouter directement les heures de pic : à 10 s par candidat et précision p, un vrai positif coûte 10/p secondes, et la file hebdomadaire doit tenir en une heure (jugement). Avec la règle d'agrégation « tout point dépassant le seuil est remonté », ce plancher devient la seule borne de la charge de vérification. Cible : rappel ≥ 0,9 au seuil le plus élevé maintenant une précision ≥ 0,1 sur sites tenus à l'écart ; sinon, présenter la frontière précision–rappel et laisser l'ONF choisir.

**Calibration.** Les scores ne sont pas des probabilités : seuils choisis sur les scores hors-pli pour le rappel cible ; recalibration par jeu si le décalage de domaine est confirmé ; régression isotonique ou de Platt seulement au-delà de ≈ 200 positifs (jugement).

**Intervalles de confiance.** Wilson sur le rappel : n = 30 et rappel 0,9 → [0,74 ; 0,97] ; n = 100 → [0,83 ; 0,94]. Bootstrap par enregistrement (1 000 tirages) pour l'AP ; bootstrap apparié entre modèles ; McNemar sur les détections par positif. « A meilleur que B » n'est écrit que si l'intervalle apparié exclut zéro.

**Benchmark contre l'existant.** Méthodes actuelles `[HYPOTHÈSE H10]` : écoute et spectrogrammes, éventuellement Kaleidoscope ou BirdNET-Analyzer, et le travail du prestataire précédent `[HYPOTHÈSE H19]`. Sur le jeu gelé : template matching ; détecteur en bande seul ; embeddings BirdNET + sonde (référence non déployable) ; pipeline retenu — AP, rappel par qualité, fausses alarmes/h, vitesse, prise en main. Phase 4 : test de temps de détection avec deux naturalistes, avec et sans l'outil ; indicatif.

**Combinaisons de modèles (décision du stagiaire, conditionnée).** Concaténation d'embeddings sous une seule régression et fusion avec le score séquentiel d'abord ; soft voting, pondéré, stacking d'encodeurs, sélection gloutonne avec remise seulement lorsque le jeu gelé compte ≥ 200 positifs, faute de quoi un gain de 0,01–0,03 d'AP est indétectable dans un intervalle de ± 0,1. Axes de diversité réellement disponibles avec des encodeurs gelés : architecture, pré-entraînement, découpage temporel ; résolution du spectrogramme et augmentations ne redeviennent des leviers qu'avec LoRA, fine-tuning ou distillation. Chaque encodeur ajouté double le temps d'inférence et ajoute 200–400 Mo au livrable : le critère « vitesse » du tuteur tranche.

**Démontrer que la boucle sert.** Courbe d'apprentissage : tête ré-entraînée avec les labels cumulés après chaque tour, AP sur le jeu gelé en fonction du nombre de labels ; ablation à budget égal en rejouant hors ligne un tirage aléatoire de même taille. La boucle fonctionne si la courbe active domine la courbe aléatoire et si l'AP croît au-delà du bruit de l'intervalle.

---

## 7. Outil livrable

| Option | Taille indicative | Coût | Remarques |
|---|---|---|---|
| A. Bundle Python (PyInstaller ou équivalent) + ONNX Runtime + GUI | 300–600 Mo | 2–3 sem. + une construction par OS | Hors ligne, ni TensorFlow ni PyTorch à l'exécution ; conversion ONNX à valider par encodeur |
| B. Idem avec PyTorch CPU | +300–800 Mo | 2–3 sem. | Conversion évitée, taille et démarrage pénalisés |
| C. PAMGuard comme hôte (ajout v1) | Logiciel existant + modèle ONNX + configuration | 1–2 sem. | Voir ci-dessous |
| D. Dossier Python portable + script de lancement | 300–600 Mo | 1 sem. | Repli minimal |

**PAMGuard (établi : documentation PAMGuard, JASA 2026 `[À VÉRIFIER auteurs]`).** Logiciel de bureau Java multiplateforme, sans code pour l'utilisateur, avec un module d'apprentissage profond qui charge un modèle générique (PyTorch JIT, TensorFlow, ONNX via DJL), segmente la forme d'onde, applique des transformations configurées, et intègre les résultats à ses affichages, sa gestion de données et ses exports ; support terrestre récent (chauves-souris, gibbons dans le tutoriel officiel). En fusionnant encodeur et tête en un seul graphe ONNX (la tête est linéaire), le livrable devient un fichier de modèle et une configuration, et la phase 3 se réduit au guide. Trois limites : pas de ré-entraînement dans l'application, donc la boucle du §5 se fait hors PAMGuard avec livraison de nouveaux fichiers de modèle ; le module séquentiel n'entre pas dans un graphe ONNX simple, à porter ou à sacrifier ; ergonomie d'acousticien, plus lourde que le critère « prise en main » ne le souhaite. Décision par l'expérimentation de packaging de S13 : option A contre option C, testées chacune par un utilisateur ONF.

Poids embarqués : Bird-MAE-Base ≈ 370 Mo en float32, ≈ 185 Mo en float16 ; BEATs comparable `[À VÉRIFIER]` ; ProtoCLR plus petit `[À VÉRIFIER]` ; Perch 2.0 ≈ 50 Mo si une voie CPU existe ; un modèle distillé, quelques Mo. Pour l'option A, GUI choisie en S13 entre Gradio, NiceGUI et PySide6 ; Streamlit convient au prototype de recherche, son packaging est plus incertain `[À VÉRIFIER]`. Test sur une machine Windows de l'ONF `[HYPOTHÈSE H6]` en S14 et S20. YAPAT n'est pas le livrable : il exige PostgreSQL (serveur de base de données) et Celery (files de tâches avec courtier de messages), trois services en arrière-plan incompatibles avec un exécutable autonome hors ligne.

**Interface (option A).** *Analyser* (dossier, reprise idempotente, progression) ; *Vérifier* (file triée, spectrogramme 2–8 kHz avec repère de durée de note, boutons du schéma de labels, raccourcis clavier) ; *Résultats* (points classés par force de détection, statut, notes validées, export CSV) ; *Modèle* (versions, mise à jour de la tête, seuil).

**Mise à jour chez l'utilisateur.** Tête : dans l'application. Encodeur : paquet de modèle (`manifest.json` : nom, version, empreinte SHA-256, licence, fréquence, durée de fenêtre) déposé dans `models/`.

**Documentation minimale viable.** README ; « Interpréter un score » (2 pages) ; « Vérifier une file » (1 page) ; guide du mainteneur ; CHANGELOG ; `LICENSES.md` ; cartes des modèles.

**Licences.** Règle du projet, sans avis juridique : tout composant redistribué doit pouvoir l'être par l'ONF dans son contexte réel, confirmé par sa voie administrative avant la fin de S12. BirdNET (CC BY-NC-SA 4.0) exclu du bundle par défaut ; Perch 2.0 (Apache 2.0) compatible ; AnuraSet CC0 sans contrainte ; Bird-MAE, BEATs, NatureLM-audio, ProtoCLR à relever `[À VÉRIFIER]` ; statut d'une tête entraînée sur les embeddings d'un modèle non commercial : à poser.

---

## 8. Planning (26 semaines, S1 = 14–18/09/2026)

Correspondance avec la feuille de route manuscrite : (1) lecture bibliographique → P0 ; (2) base de données → P0–P1 ; (3) pipeline Python → P1 ; (4) tester et comparer → P0 puis P2 ; (5) optimiser → P1–P3 ; (6) combiner → P4, conditionnel.

| Phase | Semaines | Objectif | Livrable vérifiable | Go / no-go à la réunion hebdomadaire de fin de phase |
|---|---|---|---|---|
| P0 Cadrage et benchmark | S1–S3 (15/09–02/10) | Lecture (doc ONF, Fouquet 2018, Courtois 2025, fils Kaggle BirdCLEF+ 2026, PAMGuard, AnuraSet, Kath et al. 2024) ; inventaire des données ; essai YAPAT (½ j) ; pré-benchmark AnuraSet (S2) ; jeu annoté v0 avec négatifs appariés ; protocole d'évaluation figé (plis par site) ; benchmark ONF (S3) ; tests C0–C1 ; questions du §10 | Tableau de benchmark AnuraSet + ONF avec les trois critères ; inventaire ; verdict C1 ; hypothèses mises à jour | ≥ 1 encodeur exécutable localement (natif ou ONNX) avec AP ≥ 0,3 et rappel ≥ 0,8 à précision ≥ 0,1 en validation par site ; sinon DSP + template matching en primaire |
| P1 Pipeline v0 et amorçage | S4–S7 (05/10–30/10) | Pipeline CLI bout en bout (heures de pic du jeu 1) ; prototype différentiel ; tour 0 et tour 1 d'apprentissage actif ; C2 ; expérience d'échelle (§1) ; calibration expert | Dépôt Git ; jeu annoté v1 (≥ 100 positifs validés avec classes de qualité, ≥ 1 000 négatifs dont ≥ 300 durs étiquetés par espèce) ; note sur l'échelle | ≥ 60 positifs validés sur ≥ 3 sites ; sinon risque 1 |
| P2 Apprentissage actif et évaluation | S8–S12 (02/11–04/12) | Tours 2–5 ; jeu gelé ; C3 ; module séquentiel et tête de fusion ; attentive probing si tokens et effectif ; choix d'encodeur ; position ONF sur les licences | Jeu gelé documenté ; courbes d'apprentissage ; rappel par classe de qualité ; note d'arbitrage encodeur | Rappel ≥ 0,85 à précision ≥ 0,1 sur sites tenus à l'écart ; sinon pivot (§9). **Fin S12 = date limite de pivot méthodologique** |
| P3 Outil v1 | S13–S18 (07/12–15/01 ; S15–S16 réduites `[HYPOTHÈSE H17]`) | Expérimentation de packaging S13 : option A contre PAMGuard ; export ONNX ; GUI ; test sur machine ONF ; distillation exploratoire si le pipeline enseignant est stable (hors chemin critique) | Application installable ou configuration PAMGuard testée sur deux machines ; guide utilisateur v0 | Tourne hors ligne sur une machine ONF et analyse 1 h d'audio en moins de 15 min (jugement) ; sinon repli option D |
| P4 Validation et généralisation | S19–S22 (18/01–12/02) | Jeu 2 ; test utilisateurs (score, vitesse, prise en main) ; démonstration de la boucle ; combinaisons et LoRA/AnuraSet si les effectifs le justifient ; documentation | Rapport d'évaluation ; application v1.1 ; documentation complète | Gel des résultats fin S22 |
| P5 Rapport et soutenance | S23–S26 (15/02–12/03) | Rédaction, soutenance, transfert | Rapport de stage `[HYPOTHÈSE H11]` ; support de soutenance ; dépôt transféré | — |

Rédaction continue dès S17 (une demi-journée par semaine). Chaque réunion hebdomadaire reçoit avancement, chiffre clé de la semaine (positifs validés par classe, AP sur jeu gelé) et décision demandée.

**Chemin critique** : accès aux données (S1) → benchmark (S3) → positifs validés (S4–S7) → jeu gelé (S9) → choix d'encodeur (S12) → packaging (S13–S16) → test utilisateurs (S19–S20) → gel (S22). Marge : une semaine en P2, une en P4. Un retard de plus d'une semaine sur les données comprime P3–P4 et ferait sauter le test utilisateurs. Après S12, changer de famille de méthode n'est plus possible ; après S16, changer d'encodeur non plus.

---

## 9. Risques et plans B

| Risque | Signal d'alerte précoce | Seuil de déclenchement | Repli concret |
|---|---|---|---|
| 1. Trop peu de positifs | Rendement faible du tour 0 ; positifs concentrés sur un site | < 60 positifs validés sur ≥ 3 sites fin S7 | (a) Enregistrements de l'étude de phénologie `[HYPOTHÈSE H7]` et du prestataire `[HYPOTHÈSE H19]` ; (b) prototype différentiel comme classifieur final ; (c) augmentation par mixage des notes validées dans des fonds sonores de sites, entraînement seulement ; (d) livrable = outil de tri par similarité |
| 2. Confusion non résolue avec les espèces déclencheuses | > 30 % des 100 meilleurs candidats sont des congénères ou du fourmilier après le tour 3 | Précision < 0,1 à rappel 0,85 sur sites riches en déclencheurs fin S11 | Poids accru du module séquentiel (rythme, persistance, heure) ; jeu de négatifs par espèce ; sortie hiérarchique « *Anomaloglossus* sp. » avec avertissement |
| 3. Bande saturée : le détecteur amont déclenche en continu (tuteur) | Taux de candidats par heure d'audio mesuré en S3 | > 500 événements/h sur les heures de pic (jugement) | Abandon du filtre amont ; fenêtres pleines ; le seuillage ne sert qu'aux onsets du module séquentiel |
| 4. Décalage acoustique entre les deux jeux | AP sur un audit du jeu 2 très inférieure | AP jeu 2 < 50 % de l'AP en validation par site, mesurée S19 | Recalibration des seuils par jeu ; négatifs durs du jeu 2 en gardant une partie à l'écart ; normalisation par enregistreur |
| 5. Volume ingérable | Débit mesuré en S1 | Encodage complet projeté > 10 jours-machine | Priorité heures de pic et mois humides (÷ 12), pas plus grossier, encodeur plus léger, lots nocturnes ; extraction déportée `[À VÉRIFIER]` |
| 6. Fréquence d'échantillonnage ou format incompatibles | Inventaire S1 | f_e native < 11 kHz ou format propriétaire | Rééchantillonnage dans les wrappers ; conversion ffmpeg/sox |
| 7. Pas de variante CPU de Perch 2.0 | Échec des replis (1)–(2) du §2 | Non exécutable localement fin S3 | Perch 2.0 en référence déportée seulement ; déploiement sur Bird-MAE, BEATs ou ProtoCLR |
| 8. Annotations du prestataire inaccessibles ou incompatibles | Réponse à la question 1 du §10 | Non reçues fin S3 | Reconstituer les classes de qualité par SNR estimé ; perte d'un point de comparaison avec l'existant |
| 9. AnuraSet non représentatif | Classement des encodeurs sur AnuraSet contredit par le benchmark ONF | Inversion du premier et du dernier | Ne conserver AnuraSet que pour l'adaptation de domaine, pas pour le choix d'encodeur |
| 10. Expert indisponible | Calibration non planifiée fin S4 | Aucun label validé fin S6 | Politique stricte « incertain » ; validation asynchrone par lots ; double écoute |
| 11. Perte de la machine ou des données | Machine unique | — | Sauvegarde quotidienne sur SSD externe, copie hebdomadaire sur stockage ONF ; labels versionnés (Git) |

---

## 10. Questions aux encadrants (par priorité ; celles de v0 déjà tranchées par le tuteur sont retirées)

1. « Les annotations du prestataire précédent — ses trois classes de qualité — existent-elles, sous quel format, sur quels sites ? » — Peut multiplier les positifs, fournir la stratification par qualité et un point de comparaison avec l'existant.
2. « Quels enregistreurs, quelle fréquence d'échantillonnage, quel format, quelle convention de nommage, et puis-je avoir les données cette semaine ? » — Le chemin critique commence là.
3. « D'où viennent les cris labellisés : sites, individus, enregistreur, forme (horodatages ou extraits) ? » — Décide si une validation par site est possible.
4. « Quelle est la liste des espèces déclencheuses, avec leurs heures et leurs motifs de chant, et existe-t-il des enregistrements du fourmilier tacheté et des congénères ? » — Conditionne le negative mining et le module séquentiel.
5. « Les deux jeux partagent-ils des sites, des enregistreurs, des saisons ? Lesquels ont une présence confirmée ? » — Fixe le protocole de généralisation.
6. « Les enregistrements de l'étude de phénologie de 2025 sont-ils accessibles avec les instants des détections ? » — Peut lever le risque 1.
7. « Quel taux d'émission et quels intervalles entre notes chez *A. blanci* ? » — Conditionne le module séquentiel.
8. « Sur les 16 kHz : la fondamentale est à 4,75 kHz, sous le plafond de 8 kHz ; l'objection porte-t-elle sur les harmoniques, la résolution temporelle, ou les indices haute fréquence ? » — Décide si BEATs et NatureLM restent dans le benchmark (contrôle passe-bas du §2).
9. « Qui vérifiera les candidats, combien de temps par semaine, avec quelle séance de calibration ? » — Fixe le budget d'annotation.
10. « Quelles machines chez les utilisateurs finaux, et une connexion est-elle possible pour un téléchargement initial ? » — Conditionne le packaging et l'option PAMGuard.
11. « Qui tranche la question des licences à l'ONF, et à quelle échéance ? » — Filtre les encodeurs avant S12.
12. « Quels outils sont utilisés aujourd'hui pour dépouiller les enregistrements ? » — Baseline du benchmark.
13. « De nouvelles données de terrain arriveront-elles pendant le stage, et quand ? » — Test réel de la boucle.
14. « Quelle décision de gestion découle d'un point remonté, et quel volume de vérification l'ONF peut-il absorber par semaine ? » — Fixe la précision plancher.

---

## 11. Montée en compétence

**Traitement du signal (S1–S4, ≈ 14 h).** Échantillonnage, Nyquist, rééchantillonnage avec anti-repliement ; transformée de Fourier à court terme et spectrogramme, compromis fenêtre/pas (note de 90 ms : fenêtre d'analyse de 16–25 ms, pas ≤ 10 ms) ; log-mel ; filtrage passe-bande, enveloppe, détection d'onsets ; SNR et son estimation sur enveloppe en bande ; corrélation croisée. Ressources : documentation de scikit-maad et librosa ; Smith, *The Scientist and Engineer's Guide to DSP* `[À VÉRIFIER]` ; Müller, *Fundamentals of Music Processing* `[À VÉRIFIER]`.

**Écologie des anoures (S1–S6, ≈ 18 h).** Vocabulaire du chant, dépendance à la température ; le genre *Anomaloglossus*, sympatrie, habitat ; phénologie ; occupation et probabilité de détection ; les espèces déclencheuses signalées par le tuteur. Ressources : Fouquet et al. 2018 (S1, 3 h) ; Courtois et al. 2025 (2 h) ; Cañas et al. 2023 (AnuraSet, 2 h) ; le Plan national d'actions `[À VÉRIFIER]` ; Köhler et al. 2017 `[À VÉRIFIER]` ; séances d'écoute avec l'expert.

**Apprentissage actif en bioacoustique (S2, 4 h).** Kath et al. 2024 (Ecological Informatics) ; tutoriel « agile modeling » de Hamer, Laber et Denton 2023 ; Rauch et al. 2024 `[À VÉRIFIER]`.

**Arithmétique d'embedding (après S16, 4 h, tuteur).** Directions et décalages dans l'espace de représentation : le prototype différentiel en est le premier cas ; à explorer ensuite pour représenter les classes de qualité comme un décalage (« lointain » ≈ « net » + direction « distance ») et pour interpréter ce que l'encodeur code réellement.

**Placement.** Première heure de chaque journée en S1–S4, puis 1 h/sem. ; ≈ 45 h sur six semaines. Rien sur le chemin critique après S6.

---

## 12. Hypothèses formulées

| Id | Hypothèse | Question qui la lève |
|---|---|---|
| H1 | Enregistreurs autonomes (AudioMoth, Song Meter), WAV PCM mono, fréquence native ≥ 32 kHz | §10 Q2 |
| H2 | Cris labellisés = instants ou extraits provenant de 1 à 3 sites et d'un seul type d'enregistreur | §10 Q3 |
| H3 | Les deux jeux sont disjoints en sites et diffèrent par saison ou enregistreur | §10 Q5 |
| H4 | Positions des encadrants reconstituées comme en tête de document | Relecture par les intéressés |
| H5 | Décision gestionnaire = intégration de la présence dans la planification forestière à l'échelle du point ou du bassin de crique | §10 Q14 |
| H6 | Utilisateurs sur portables Windows sans GPU, 8–16 Go, connexion possible au bureau | §10 Q10 |
| H7 | Les enregistrements de l'étude de phénologie 2025 existent et sont partageables | §10 Q6 |
| H8 | *A. blanci* absent de l'espace de labels de Perch 2.0 | Liste de classes, S2 |
| H9 | Intervalles entre notes d'*A. blanci* de l'ordre de la seconde, plusieurs notes par fenêtre de 5 s | §10 Q7 |
| H10 | Dépouillement actuel par écoute et spectrogrammes ; MacBook Air à 16 Go | §10 Q12 ; vérification locale |
| H11 | Rapport remis vers la fin du stage, soutenance à IMT Atlantique en mars ou avril 2027 | Calendrier école |
| H12 | Un expert disponible 3 h puis 1 h toutes les deux semaines | §10 Q9 |
| H13 | Au moins un rapatriement de données de terrain avant février 2027 | §10 Q13 |
| H14 | Des enregistrements de référence des congénères et du fourmilier sont accessibles | §10 Q4 |
| H15 | Saison des pluies principale de décembre à juillet, saison sèche d'août à novembre `[À VÉRIFIER]` | Calendrier ONF |
| H16 | Volume total des deux jeux de l'ordre de quelques milliers d'heures | §10 Q2 |
| H17 | Deux semaines réduites autour des fêtes de fin d'année | Calendrier ONF et école |
| H18 | Les « autres espèces déclencheuses » sont en nombre limité (< 10), avec des heures de chant partiellement disjointes de celles d'*A. blanci* | §10 Q4 |
| H19 | Les annotations du prestataire précédent existent, sont récupérables, et ses trois classes correspondent à des niveaux de SNR décroissants | §10 Q1 |

---

## Angles morts

1. **Le plan d'échantillonnage peut être la contrainte dominante, pas le modèle.** À 2 min par heure, la probabilité de capter un cri sur un site à faible densité reste faible quel que soit le rappel ; un enregistrement continu sur les heures de pic changerait davantage la donne que tout le §3.
2. **Charge de vérification induite par la règle d'agrégation.** Remonter tout point dépassant le seuil, sur 86 points et un an d'enregistrement, fera remonter presque tous les points ; le classement par force de détection et la précision plancher sont les seuls garde-fous, et la capacité réelle de vérification de l'ONF n'est pas connue (§10 Q14).
3. **Pseudo-réplication.** Si les positifs viennent d'un ou deux sites, toute affirmation de généralisation repose sur presque rien.
4. **La vérité terrain est un jugement humain non mesuré.** L'accord entre experts sur *A. blanci* contre déclencheurs n'est estimé que sur 100 fenêtres ; et les informations sur le fourmilier et les congénères sont orales, non vérifiées sur enregistrement.
5. **Biais phénologique induit.** L'échantillonnage guidé par les heures de pic entraîne le modèle sur les conditions de pic ; les 20 % hors strates sont un palliatif.
6. **AnuraSet est un indicateur, pas une preuve.** Le classement des encodeurs qu'il produit vaut pour d'autres grenouilles, d'autres biomes, d'autres enregistreurs ; le risque 9 le dit, mais le document s'appuie néanmoins dessus pour un choix structurant.
7. **Le test C1 est lui-même peu puissant.** Avec 30 positifs, un rappel du meilleur groupe de 0,5 a un intervalle de confiance qui couvre 0,3 à 0,7 ; la décision qu'il fonde est plus tranchée que ce que les chiffres autorisent.
8. **Après le 14 mars.** La boucle continue suppose un effort de vérification hebdomadaire que l'ONF n'a pas promis, et personne n'est désigné pour maintenir le code.
9. **Présence/absence sans modèle d'occupation.** La formulation statistique correcte de la non-détection — modèles d'occupation avec probabilité de détection `[À VÉRIFIER]` — est hors périmètre.
10. **Transfert oiseaux → anoure non garanti, attentive probing probablement hors de portée.** Les benchmarks de la revue sont majoritairement aviaires ; l'attentive probing requiert un effectif peut-être inatteignable.
11. **Sur-ingénierie.** Index, versionnage, migrations, paquets de modèles, deux voies parallèles, combinaisons : sur six mois, chaque heure d'infrastructure est retirée à l'annotation et à l'évaluation. Les combinaisons du §6 sont conservées à la demande du stagiaire ; le document maintient qu'elles ne seront probablement pas mesurables.
12. **Valeur pour l'ONF non mesurée.** Le benchmark mesure des AP ; la valeur réelle est le temps de naturaliste économisé par site détecté, estimée sur un test à deux personnes.
13. **Variation intraspécifique.** Les paramètres acoustiques proviennent d'une description fondée sur peu d'individus ; température et taille peuvent déplacer durée et fréquence hors des filtres proposés.

---

## Sources vérifiées le 16–17/09/2026 (hors bibliographie fournie)

- Cañas, J. S. et al. (2023). A dataset for benchmarking Neotropical anuran calls identification in passive acoustic monitoring (AnuraSet). *Scientific Data* ; arXiv:2307.06860 ; dépôt github.com/soundclim/anuraset ; données Zenodo, CC0.
- Kath, H., Serafini, P. P., Campos, I. B., Gouvêa, T. S. & Sonntag, D. (2024). Leveraging transfer learning and active learning for data annotation in passive acoustic monitoring of wildlife. *Ecological Informatics* 82, 102710. Documentation YAPAT : yapat.readthedocs.io ; dépôts yapat-app/yapat-backend et yapat-frontend.
- Hamer, J., Laber, R. & Denton, T. (2023). Agile Modeling for Bioacoustic Monitoring. Tutoriel Climate Change AI, NeurIPS ; Zenodo 10.5281/zenodo.11585179.
- Martínez Balvanera, S. et al. (2025). Whombat: an open-source audio annotation tool for machine learning assisted bioacoustics. *Methods in Ecology and Evolution*.
- PAMGuard : pamguard.org (module « Deep Learning », tutoriel, liste de modèles compatibles) ; article JASA 159(1), 2026 `[À VÉRIFIER auteurs]` ; Macaulay & Gillespie (2022), JASA 151(4) supplément.
- BirdCLEF+ 2026 : page Kaggle de la compétition (Pantanal, 234 espèces, ROC-AUC macro, CPU 90 min, fin le 03/06/2026) ; notebook communautaire mentionnant `perch_v2_no_dft.onnx` `[À VÉRIFIER : provenance et fidélité de l'export]`.
- Rauch, L. et al. (2024). Towards deep active learning in avian bioacoustics. arXiv:2406.18621.
- Anthropic, centre d'aide : dates de coupure des modèles Claude (Fable 5.1 : juin 2026).
