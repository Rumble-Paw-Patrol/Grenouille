# birdmae_large sur AnuraSet : benchmark 07, vague 2 (session 6, 29/09)
**Réglages lus** (bacpipe 1.3.5, `DBD-research-group/Bird-MAE-Large`) : ViT-L auto-supervisé
(MAE), f_e 32 kHz (AnuraSet à 22,05 kHz rééchantillonné), fenêtre 5 s jointive, embedding 1 024
= moyenne des patchs de la dernière couche puis `fc_norm`. Jetons 32 temps × 8 fréquences ×
1 024, moyennés sur la fréquence (`jetons.py`). torch 2.6.0+cpu, transformers 4.57.6,
onnxruntime 1.30.0, tensorflow 2.20.0 (importé, inutilisé), librosa 0.11.0. 19 166 fenêtres.
**Débit** (4 cœurs) : encodage 2,0 f/s (~2 h 45) ; jetons 10 003 s ; benchmark `--curve
--tokens`, 2 processus : 582 (PITAZU) à 1 997 s (PHYCUV) par espèce.
**Témoin BOAFAB** (logistique, AP moyenne par site, minute) : **0,938**, tuyau sain.
**Transfert vers un site neuf, minute, AP moyenne par site / poolée**

| Tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| logistique | 0,338 / 0,695 | 0,298 / 0,113 | 0,552 / 0,442 | 0,359 / 0,085 | 0,938 / 0,758 |
| lda_shrunk | 0,381 / 0,734 | 0,287 / 0,121 | 0,705 / 0,575 | 0,424 / 0,195 | 0,948 / 0,825 |
| simple_prototype | 0,354 / 0,290 | 0,285 / 0,116 | 0,201 / 0,110 | 0,319 / 0,201 | 0,633 / 0,373 |
| *jetons* proto_probe | **0,654 / 0,917** | 0,334 / 0,102 | **0,906 / 0,708** | **0,579** / 0,102 | **0,979 / 0,944** |
| *jetons* logistic:max | 0,330 / 0,647 | 0,286 / 0,112 | 0,454 / 0,302 | 0,365 / 0,090 | 0,941 / 0,759 |
| *jetons* attentive | 0,385 / 0,813 | **0,339** / 0,080 | 0,703 / 0,578 | 0,316 / 0,080 | 0,951 / 0,804 |
| *réf.* perch_v2 logistique | 0,666 / 0,911 | 0,507 / 0,150 | 0,960 / 0,946 | 0,824 / 0,253 | 0,985 / 0,964 |
Autres têtes et scores : `resultats/global/birdmae_large_*` (branche `resultats-anuraset-07`).
**Courbe d'amorçage** (logistique, AP minute, moyenne des sites, k = 0 → 20 positifs du site) :
DENMIN 0,50→0,80, PITAZU 0,54→0,90, PHYCUV 0,57→0,79, LEPLAT 0,52→0,82, BOAFAB 0,95→0,96.
**Lecture** : transformer auto-supervisé **en retrait en sondage linéaire** (n° 151) : en
logistique, Large ne fait pas mieux que Base ni Huge (benchmark 07). La sonde à prototypes sur
jetons, tête de ses auteurs, change la donne : +0,04 (PITAZU, BOAFAB) à +0,35 (PHYCUV) d'AP par site sur
la logistique, DENMIN au niveau de perch_v2 (0,654 contre 0,666) ; reste sous perch_v2 +
logistique sur PITAZU, PHYCUV et LEPLAT. La sonde attentive n'apporte presque rien.
**Anomalies** : redémarrages du conteneur (encodage repris à 1 350/1 599), sans effet.
Stock : branche `donnees-anuraset-birdmae_large`.
