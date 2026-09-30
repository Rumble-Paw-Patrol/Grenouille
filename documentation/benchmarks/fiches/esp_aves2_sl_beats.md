# esp-aves2 BEATs sur AnuraSet : benchmark 07, vague 2 (session 14, 29/09-30/09)
**Réglages lus** (avex 1.3.0) : BEATs SSL puis supervisé (`sl_beats_bio`, `sl_beats_all`), 16 kHz
(AnuraSet rééchantillonné), fenêtre 5 s jointive, embedding 768 = moyenne des jetons (AVEX) ;
jetons 31 temps × 8 fréquences × 768, moyennés sur la fréquence. Poids vérifiés : 248 tenseurs
sur 252 du point de contrôle chargés et identiques (les 4 autres hors de l'encodeur). torch
2.14.0+cpu, torchaudio 2.11.0+cpu, transformers 4.57.6, onnxruntime 1.30.0, librosa 0.11.0, sans
TensorFlow. 19 166 fenêtres. **`naturelm_audio_v1_beats` non encodé** : sur 50 fenêtres, cosinus 1,0000 (écart max 2e-6) avec `naturebeats` (bacpipe) : mêmes poids.
**Débit** (4 cœurs) : encodage 6,8 f/s seul ; jetons 2 543 s seuls (all), 5 212 s en parallèle de
l'encodage (bio) ; benchmark `--curve --tokens` (2 proc.) 340 (PITAZU) à 1 020 s (PHYCUV). **Témoin BOAFAB** (logistique, AP moyenne par site, minute) : bio **0,989**, all **0,986**.
**Transfert vers un site neuf, minute, AP moyenne par site / poolée** (jetons : proto_probe, attentive)

| Encodeur, tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| bio logistique | 0,677 / 0,881 | 0,444 / 0,154 | 0,869 / 0,710 | 0,822 / 0,326 | 0,989 / 0,946 |
| bio *jetons* proto_probe | **0,746 / 0,906** | **0,460** / 0,156 | 0,921 / 0,810 | 0,846 / 0,400 | **0,994 / 0,963** |
| bio *jetons* attentive | 0,380 / 0,739 | 0,398 / 0,128 | 0,802 / 0,707 | 0,690 / 0,224 | 0,972 / 0,953 |
| all logistique | 0,529 / 0,557 | 0,428 / 0,156 | 0,892 / 0,844 | 0,804 / 0,281 | 0,986 / 0,920 |
| all *jetons* proto_probe | 0,620 / 0,836 | 0,467 / 0,169 | **0,947 / 0,911** | **0,883** / 0,378 | 0,994 / 0,961 |
| all *jetons* attentive | 0,540 / 0,630 | 0,395 / 0,118 | 0,790 / 0,728 | 0,562 / 0,303 | 0,960 / 0,932 |
Autres têtes (lda_shrunk, logistic:max, R37, simple_prototype) et scores : `resultats/global/esp_aves2_sl_beats_*`
(branche `resultats-anuraset-07`). **Courbe d'amorçage** (logistique, AP minute, k = 0 → 20) : bio DENMIN 0,58→0,82, PITAZU
0,85→0,96, PHYCUV 0,89→0,95, LEPLAT 0,88→0,93, BOAFAB 0,99→0,99 ; all 0,50→0,80, 0,80→0,96,
0,91→0,94, 0,89→0,92, 0,99→0,99.
**Lecture** : **légèrement en retrait en sondage linéaire** (n° 151) : la sonde à prototypes
sur jetons gagne sur les cinq espèces (+0,01 à +0,09 d'AP par site ; BOAFAB au plafond), la sonde attentive fait moins bien que la
logistique. bio devance all en logistique sur DENMIN (+0,15), all devance bio sur
PHYCUV et LEPLAT avec proto_probe.
**Anomalies** : trois redémarrages du conteneur (reprise : 150 enregistrements sautés) ; encodage
d'all ralenti à 2,0 f/s en moyenne par les jetons de bio lancés en parallèle (sursouscription des
threads). Stocks : branches `donnees-anuraset-esp_aves2_sl_beats_{bio,all}`.
