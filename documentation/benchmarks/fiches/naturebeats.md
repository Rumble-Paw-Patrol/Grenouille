# naturebeats sur AnuraSet : benchmark 07, vague 2 (session 7, 29/09)
**Réglages lus** (bacpipe 1.3.5) : BEATs aux poids de NatureLM-audio, f_e 16 kHz (AnuraSet à
22,05 kHz rééchantillonné), fenêtre 5 s jointive, embedding 768 = moyenne des jetons de la
dernière couche (cosinus embedding / moyenne des jetons : 1,000). Jetons 31 temps × 8 fréquences
× 768, moyennés sur la fréquence (`jetons.py`). Modèle en `eval()` (fait par `beats.py` de
bacpipe ; le message « Skipping model.eval() » vient du wrapper générique), sorties
déterministes. torch 2.14.0+cpu, transformers 4.57.6, onnxruntime 1.30.0, librosa 0.11.0, pas
de TensorFlow. 1 599 enregistrements, 19 166 fenêtres.
**Débit** (4 cœurs) : encodage 6,0 f/s ; jetons 3 358 s ; benchmark `--curve --tokens`,
2 processus : 373 (PITAZU) à 1 088 s (PHYCUV) par espèce.
**Témoin BOAFAB** (logistique, AP moyenne par site, minute) : **0,962**, tuyau sain.

**Transfert vers un site neuf, minute, AP moyenne par site / poolée**

| Tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| logistique | 0,442 / 0,772 | 0,377 / 0,116 | 0,863 / 0,611 | 0,827 / 0,165 | 0,962 / 0,894 |
| lda_shrunk | 0,643 / 0,818 | 0,373 / 0,333 | 0,856 / 0,762 | 0,843 / 0,584 | 0,962 / 0,913 |
| simple_prototype | 0,357 / 0,359 | 0,278 / 0,096 | 0,463 / 0,171 | 0,236 / 0,125 | 0,878 / 0,368 |
| *jetons* proto_probe | **0,798 / 0,938** | **0,512** / 0,129 | **0,983 / 0,913** | **0,929** / 0,482 | **0,989 / 0,981** |
| *jetons* logistic:max | 0,443 / 0,853 | 0,406 / 0,123 | 0,846 / 0,540 | 0,866 / 0,165 | 0,973 / 0,930 |
| *jetons* attentive | 0,540 / 0,832 | 0,319 / 0,107 | 0,760 / 0,554 | 0,706 / 0,513 | 0,950 / 0,908 |

Autres têtes : `resultats/global/naturebeats_*` (branche `resultats-anuraset-07`).
**Courbe d'amorçage** (logistique, AP minute, moyenne des sites, k = 0 → 20 positifs du site) :
DENMIN 0,62→0,92, PITAZU 0,60→0,97, PHYCUV 0,87→0,96, LEPLAT 0,82→0,95, BOAFAB 0,96→0,97.
**Lecture** : transformer auto-supervisé **en retrait en sondage linéaire** (n° 151) : la sonde
à prototypes sur jetons gagne sur les cinq espèces (+0,03 à +0,36 d'AP par site sur la
logistique) ; la sonde attentive, elle, ne fait pas mieux que la logistique, sauf sur DENMIN.
**Anomalies** : NatureLM-audio a été entraîné sur de nombreux corpus bioacoustiques ; qu'AnuraSet
en fasse partie n'a pas été vérifié. Premier passage de `jetons.py beats` (4 threads, en
parallèle du benchmark) ~15 fois trop lent : relancé à 2 threads, débit normal.
Stock : branche `donnees-anuraset-naturebeats`.
