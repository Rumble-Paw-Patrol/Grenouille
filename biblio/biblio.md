# Bibliographie du projet

Tenue à jour au fil du stage. Chaque référence : lien, et une ligne « ce qu'on en retient pour
A. blanci ». Mise à jour du 29/09/2026 pour la présentation de suivi n° 2
(`documentation/presentation-suivi-2/`). Le détail des encodeurs (chiffres, licences, ce qui n'a
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
  234 taxons du Pantanal (oiseaux, amphibiens, insectes, reptiles, mammifères), ROC-AUC macro
  sur fenêtres de 5 s, inférence CPU seul en 90 min, 4 094 équipes, fin le 27/05/2026 ;
  working notes présentées à CLEF 2026 (Iéna, 21–24/09/2026, actes CEUR-WS).
- Classement final (score privé) : 1. Nikita Babych 0,966 · 2. tennogh 0,960 · 3. kapenon 0,960
  · 4. équipe « BirdCLEF+ 2026 » 0,959 · 5. Jiacheng Ma 0,958.
  <https://www.kaggle.com/competitions/birdclef-2026/leaderboard>
- 1er : « Noisy Student Meets Distillation » — distillation des embeddings de Perch v2 et
  d'AudioProtoPNet dans un EfficientNetV2, puis apprentissage supervisé et Noisy Student
  itératif sur les soundscapes non étiquetés.
  <https://www.kaggle.com/competitions/birdclef-2026/writeups/1st-place-solution-noisy-student-meets-distillati>
  (reproduction partielle : <https://github.com/GrayBloom/Reproduction-For-BirdCLEF2026>)
- 2e : « Diverse Ensemble with Pseudo-Labeling and a Taxon Specialist » (lu par son titre
  seulement, à relire). 3e et 4e : à relire sur Kaggle. 5e : code public.
- 13e — Lu & Tsai, « What Moves, and What Misleads, the BirdCLEF+ 2026 Leaderboard ».
  <https://zenodo.org/records/21545329>
  La validation hors ligne ne prédit plus le classement au-dessus de 0,92 ; la qualité des
  pseudo-labels est le premier levier (0,881 → 0,934 pour un modèle seul).
- DS@GT — Miyaguchi, Gustineli, Cheung, « Can Tokens Compete? ».
  <https://arxiv.org/abs/2607.14474>
  Amphibiens : 76,6 % des fenêtres de soundscapes étiquetés, 1,3 % des enregistrements
  focaux ; une tête à prototypes sur Perch v2 atteint une AP macro de 0,895 sur les amphibiens
  (0,074 pour leur SED seul). → conforte l'approche « Perch v2 gelé + tête » pour un anoure.
- Stratégie générale (non officielle) : E. Benhamou, « BirdCLEF+ 2026 Kaggle strategy
  playbook » (Dauphine). Validation groupée par site, pseudo-labels, distillation, CPU.

## 3. Phénologie et écologie d'A. blanci

- **Courtois E.A., Villette B., Decalf G. 2025.** *Anomaloglossus blanci : connaissances sur
  une espèce endémique en danger.* Association Trésor, ENIA, DGTM, 21 p.
  (`documentation/pheno-blanci.pdf`)
  6 Song Meter mini (Kaw A/B, Trésor C/D, Molokoï E/F), 25/11/2023 → 24/11/2024, 2 min / 30 min
  de 5 h à 20 h ; détecteur Biophonia. Pics 7–9 h et 15–17 h ; saison haute janvier–avril ;
  ≈ 0 de juillet à octobre ; probabilité de détection journalière ≈ 1 à Molokoï de fin
  novembre à mars ; 1–2 jours d'enregistrement suffisent en forte densité, 3–5 à Kaw.
  → jeu de test temporel (niveau 3), échantillonnage de l'annotation, heure et saison hors du
  classifieur.
- **Fouquet et al. 2018** : description d'A. blanci (citée par Courtois et al. 2025).

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
