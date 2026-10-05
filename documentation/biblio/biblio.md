# Bibliographie du projet

Tenue à jour au fil du stage. Chaque référence : lien, et une ligne « ce qu'on en retient pour
A. blanci ». Mise à jour du 29/09/2026 pour la présentation de suivi n° 2
(`documentation/prez/presentation-suivi-2/`). Le détail des encodeurs (chiffres, licences, ce qui n'a
pas été vérifié) reste dans `documentation/encodeurs-bacpipe.md`.

> Ce fichier a été recréé dans le dépôt : la version locale (quasi vide) n'avait pas été poussée.
> À fusionner avec elle si elle contient des entrées.

## 1. Modèles de fondation en bioacoustique

- **Perch 2.0 : The Bittern Lesson for Bioacoustics** — van Merriënboer et al., 2025.
  <https://arxiv.org/abs/2508.04665>
  EfficientNet-B3 (≈ 12 M paramètres), 32 kHz, 5 s, 1 536 dimensions, 14 795 classes multi-taxons,
  1,5 M d'enregistrements ; prototypes, prédiction de source, auto-distillation. Meilleur en
  sondage linéaire. → notre encodeur de référence (`perch_v2`).
- **Foundation Models for Bioacoustics — a Comparative Review** — Schwinger et al., 2025 ;
  Ecological Informatics 2026 (article conseillé par un tuteur).
  <https://arxiv.org/abs/2508.01277> · <https://www.sciencedirect.com/science/article/pii/S1574954126001718>
  La tête doit suivre la sortie de l'encodeur (BEATs 94,10 → 97,98 AUROC sur BEANS en passant
  à l'attentive) ; plus gros n'est pas meilleur. → règle n° 151.
- **What Matters for Bioacoustic Encoding (ESP-AVES2)** — Earth Species Project, ICLR 2026.
  <https://arxiv.org/abs/2508.11845>
  11 points de contrôle ; mélanger AudioSet et bioacoustique aide. → confirmé sur AnuraSet
  (effnetb0 bio/all 0,59 contre audioset 0,50). Licence non commerciale.
- **MetaPerch** — Chasmai, Dumoulin, Hamer, ICML 2026. <https://arxiv.org/abs/2607.14072>
  Métadonnées (lieu, date) en pertes auxiliaires d'entraînement ; gain net en Amérique du Sud.
  → successeur possible de perch_v2, à tester.
- **NatureLM-audio** — Robinson et al., ICLR 2025. <https://arxiv.org/abs/2411.07186>
  Modèle audio-langage ; son encodeur NatureBEATs est 1er sur les tâches de classification de
  BEANS (qui ne contiennent aucun anoure). 16 kHz.
- **Bird-MAE : Can Masked Autoencoders Also Listen to Birds?** — Rauch et al., 2025.
  <https://huggingface.co/collections/DBD-research-group/bird-mae>
  ViT auto-supervisé ; ses auteurs le sondent par prototypes sur les jetons. → en retrait en
  linéaire sur AnuraSet, verdict suspendu.
- **BirdSet** — Rauch et al., ICLR 2025. <https://arxiv.org/abs/2403.10380>
  Banc d'essai sur enregistrements continus d'oiseaux ; ConvNeXt-BirdSet, AudioProtoPNet.
- **BirdNET** — Kahl et al., 2021, Ecological Informatics.
  <https://doi.org/10.1016/j.ecoinf.2021.101236>
  Le standard des études de terrain ; BirdNET+ V3.0 en préversion (2026) :
  <https://zenodo.org/records/20703646>.
- **Global birdsong embeddings enable superior transfer learning for bioacoustic
  classification** — Ghani et al., 2023, Scientific Reports.
  <https://doi.org/10.1038/s41598-023-49989-z>
  Les embeddings appris sur les oiseaux transfèrent à d'autres taxons : point de départ de
  l'approche « encodeur gelé + tête ».
- **bacpipe** — Kather, Haupert, Ghani, Stowell, 2026. <https://arxiv.org/abs/2604.11560>
  26 encodeurs derrière une même interface. → notre banc d'essai d'encodeurs.
- **ProtoCLR** et **mix2** — Moummad et al., 2024. → protoclr en retrait sur AnuraSet ; mix2
  (appris sur AnuraSet) à tester sur les données ONF seulement.

## 2. BirdCLEF+ 2026 (Kaggle, Pantanal)

- Page du concours : <https://www.kaggle.com/competitions/birdclef-2026> ·
  LifeCLEF 2026 : <https://www.imageclef.org/LifeCLEF2026>
  Cornell Lab of Ornithology, 11/03 → 03/06/2026, 4 094 équipes ; working notes à CLEF 2026
  (Iéna, 21–24/09/2026, actes CEUR-WS). 234 classes : 162 oiseaux, 35 amphibiens, 8
  mammifères, 1 reptile, 28 « sonotypes » d'insectes (types de sons sans espèce nommée).
  Entraînement : 344 h d'enregistrements ciblés (Xeno-canto, iNaturalist) et 178 h de paysages
  sonores de 23 sites, dont 1 h étiquetée ; test caché ≈ 600 enregistrements d'1 min. Soumission :
  notebook Kaggle sur CPU seul (4 cœurs), sans GPU ni internet, 90 min pour prédire tout le test.
  Métrique : ROC-AUC macro par fenêtre de 5 s ; classement final (« privé ») sur la partie
  cachée du test.
- Classement final : 1. Nikita Babych 0,966 · 2. tennogh 0,960 · 3. kapenon 0,960 · 4. Tony Li,
  Yiheng Wang, Starry 0,959 · 5. Jiacheng Ma 0,958.
  <https://www.kaggle.com/competitions/birdclef-2026/leaderboard>
- 1er, « Noisy Student Meets Distillation » : 9 CNN distillés de Perch v2 (un d'AudioProtoPNet),
  puis affinés et entraînés en Noisy Student (0,935 → 0,950 en deux tours) ; spécialiste
  amphibiens/insectes (1 903 espèces de Xeno-canto et iNaturalist), modèle au niveau du genre,
  Perch v2 tel quel ; a priori de site, lissages temporel et taxonomique.
  <https://www.kaggle.com/competitions/birdclef-2026/writeups/1st-place-solution-noisy-student-meets-distillati>
- 2e, « Diverse Ensemble with Pseudo-Labeling and a Taxon Specialist » : 5 tours de
  pseudo-labels, CNN pré-entraînés sur Xeno-canto (2e place 2025), perte AUC + 0,25 BCE,
  spécialiste insectes, chaînes publiques Perch et SED distillé ; pas de distillation dans ses
  propres modèles, pour garder la diversité.
  <https://www.kaggle.com/competitions/birdclef-2026/writeups/2nd-place-diverse-ensemble-with-pseudo-labeling-a>
- 3e : 3 modèles (2 EfficientNetV2 repris de la 2e place 2025, 1 SEResNeXt distillé de Perch v2),
  un tour de pseudo-labels, a priori site/heure, inférence OpenVINO.
  <https://www.kaggle.com/competitions/birdclef-2026/writeups/3rd-place-solution>
- 4e : Perch v2 tel quel (40 % de l'ensemble) + 2 SED + 2 SED distillés de Perch ; seuls les
  pseudo-labels issus de Perch ont aidé ; **AnuraSet en données externes a fait baisser le score**
  (écart de domaine). <https://www.kaggle.com/competitions/birdclef-2026/writeups/4-th-place-solution>
- 5e, « Diversity and Bug — Both Are All You Need » : 4 CNN (HGNetV2, EfficientNetV2 5 s et 10 s,
  B3) + ProtoSSM + SED distillé ; distillation Perch, auto-distillation, pseudo-labels,
  FilterAugment. Code : <https://github.com/jak-ma/BirdCLEF2026-5th-solution>
- 13e — Lu & Tsai, « What Moves, and What Misleads, the BirdCLEF+ 2026 Leaderboard ».
  <https://zenodo.org/records/21545329>
  La validation hors ligne ne prédit plus le classement au-dessus de 0,92 ; la qualité des
  pseudo-labels est le premier levier (0,881 → 0,934 pour un modèle seul).
- DS@GT — Miyaguchi, Gustineli, Cheung, « Can Tokens Compete? ».
  <https://arxiv.org/abs/2607.14474>
  Amphibiens : 76,6 % des fenêtres de soundscapes étiquetés, 1,3 % des enregistrements
  focaux. Tête à prototypes sur Perch v2 gelé, score = cos(e, μ₊) − cos(e, μ₋) (= notre
  prototype différentiel) : AP macro 0,895 sur les non-oiseaux contre 0,074 pour leur SED seul,
  mais mesurée sur les fichiers qui construisent les prototypes (optimiste). Annexe, validation
  site par site : sans enregistrement ciblé de l'espèce, prototypes et ridge tombent au hasard.
  Une tête à attention leur a coûté 0,013 à 0,035. → Perch v2 + prototypes marchent pour des
  espèces absentes de ses classes, comme A. blanci.
- Stratégie générale (non officielle) : E. Benhamou, « BirdCLEF+ 2026 Kaggle strategy
  playbook » (Dauphine). Validation groupée par site, pseudo-labels, distillation, CPU.

## 2 bis. Anoures : transfert depuis les modèles de fondation

- *Transfer learning outperforms other methods of detecting vocalizations of a critically
  endangered tropical anuran*, Ecological Informatics, 2025.
  <https://www.sciencedirect.com/science/article/pii/S1574954125004364>
  Gabarits, CNN, recherche par similarité, transfert (embeddings d'un modèle de fondation + tête
  simple) et zéro-shot comparés : le transfert trouve le plus de vocalisations.
- Sims et al., *Cross-continental zero-shot anuran call classification with CLAP*, Methods in
  Ecology and Evolution. <https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70384>

## 3. Phénologie et écologie d'A. blanci

- **Courtois E.A., Villette B., Decalf G. 2025.** *Phénologie de l'activité de chant
  d'Anomaloglossus blanci (amphibien) par l'utilisation de la bioacoustique pour l'amélioration
  des connaissances sur une espèce endémique en danger.* Association Trésor, ENIA, DGTM, 21 p.
  <https://www.reserve-tresor.fr/wp-content/uploads/2025/05/20250324_Rapport_etude_Pheno_Blanci.pdf>
  (le PDF n'est plus versionné dans le dépôt ; il est cité dans l'offre de stage)
  6 Song Meter mini (Kaw A/B, Trésor C/D, Molokoï E/F), 25/11/2023 → 24/11/2024, 2 min / 30 min
  de 5 h à 20 h ; détecteur automatique. Pics 7–9 h et 15–17 h ; saison haute janvier–avril ;
  ≈ 0 de juillet à octobre ; probabilité de détection journalière ≈ 1 à Molokoï de fin
  novembre à mars ; 1–2 jours d'enregistrement suffisent en forte densité, 3–5 à Kaw.
  → jeu de test temporel (niveau 3), échantillonnage de l'annotation, heure et saison hors du
  classifieur.
- **Fouquet et al. 2018** : description d'A. blanci (citée par Courtois et al. 2025).
- **IUCN SSC Amphibian Specialist Group. 2019.** *Anomaloglossus blanci.* The IUCN Red List of
  Threatened Species 2019 : e.T125200267A125200500.
  <https://dx.doi.org/10.2305/IUCN.UK.2019-1.RLTS.T125200267A125200500.en>
  Statut de conservation de l'espèce (référence de l'offre de stage).
- **Plan National d'Actions des Harttiella et des Anomaloglossus de Guyane.** Coordination :
  Société Herpétologique de France, Fondation Biotope, 98 p. (plan 2022-2031 ; version mise en
  consultation publique).
  <https://www.consultations-publiques.developpement-durable.gouv.fr/IMG/pdf/pna_anomalo_et_harttiela_de_guyane_light.pdf>
  Cadre réglementaire et actions de connaissance de l'espèce (référence de l'offre de stage).

## 4. Jeux de données et bancs d'essai

- **AnuraSet** — Cañas et al., 2023, Scientific Data. Zenodo 8342596, CC BY.
  → pré-benchmark des encodeurs, un site tenu à l'écart à la fois.
- **BEANS** — Hagiwara et al., ICASSP 2023. <https://arxiv.org/abs/2210.12300>
  Seul jeu avec anoures : RFCX, en détection (omis par la revue de Schwinger et al.).

## 5. Apprentissage : attention, régularisation, domaine

- **Attention Is All You Need** — Vaswani et al., NeurIPS 2017. <https://arxiv.org/abs/1706.03762>
- **ViT** — Dosovitskiy et al., ICLR 2021. <https://arxiv.org/abs/2010.11929>
- **SpecAugment** — Park et al., Interspeech 2019. <https://arxiv.org/abs/1904.08779>
- **mixup** — Zhang et al., ICLR 2018. <https://arxiv.org/abs/1710.09412>
- **DANN** — Ganin et al., JMLR 2016. <https://arxiv.org/abs/1505.07818>
- **AdaBN** — Li et al., 2016. <https://arxiv.org/abs/1603.04779>
- **INLP** — Ravfogel et al., ACL 2020. <https://arxiv.org/abs/2004.07667>
- **LEACE** — Belrose et al., NeurIPS 2023. <https://arxiv.org/abs/2306.03819>
- **L2-SP** — Li, Grandvalet, Davoine, ICML 2018. <https://arxiv.org/abs/1802.01483>
- **Ledoit-Wolf** (covariance rétrécie, LDA R31) — Ledoit & Wolf, 2004, J. Multivariate Analysis.
- **GeM pooling** — Radenović, Tolias, Chum, TPAMI 2019. <https://arxiv.org/abs/1711.02512>

## 6. Annotation, détection et outils

- **YAPAT** — Kath, Serafini, Campos, Gouvêa & Sonntag, 2024, *Ecological Informatics* 82, 102710 ;
  <https://yapat.readthedocs.io>. Apprentissage actif pour l'annotation ; non embarqué (extraits
  pour un outil externe : `clips-export`).
- **Agile Modeling for Bioacoustic Monitoring** — Hamer, Laber & Denton, 2023, Climate Change AI ;
  Zenodo 10.5281/zenodo.11585179. Recherche par similarité puis tête légère : la boucle du projet.
- **Whombat** — Martínez Balvanera et al., 2025, *Methods in Ecology and Evolution*. Outil
  d'annotation, une possibilité parmi d'autres pour le poste d'annotation.
- **PAMGuard** — <https://www.pamguard.org>. Écarté comme hôte du livrable (pas de réentraînement
  sur place, ergonomie d'acousticien, post-traitement séquentiel hors ONNX).
- **BirdNET-Analyzer** — <https://github.com/birdnet-team/BirdNET-Analyzer>. Modèle de
  distribution pour non-codeurs : interface Gradio dans une fenêtre pywebview, version Windows
  prête à l'emploi (vérifié le 29/09/2026). → modèle du livrable (DECISIONS n° 159).
- Rauch et al., 2024, arXiv:2406.18621 (cité par la feuille de route V5, sans résumé).
