# beats sur AnuraSet : benchmark 07, vague 2 (session 7, 29/09)
**Réglages lus** (bacpipe 1.3.5) : BEATs affiné sur AudioSet (527 classes), f_e 16 kHz (AnuraSet
à 22,05 kHz rééchantillonné), fenêtre 5 s jointive, embedding 768 = moyenne des jetons de la
dernière couche (cosinus embedding / moyenne des jetons : 1,000). Jetons 31 temps × 8 fréquences
× 768, moyennés sur la fréquence (`jetons.py`). `eval()` fait par `beats.py` de bacpipe (le
« Skipping model.eval() » vient du wrapper), sorties déterministes. torch 2.14.0+cpu,
transformers 4.57.6, onnxruntime 1.30.0, librosa 0.11.0, sans TensorFlow. 19 166 fenêtres.
**Débit** (4 cœurs) : encodage 8,3 f/s ; jetons ~1 h 30 ; benchmark `--curve --tokens`,
2 processus : 375 (PITAZU) à 1 209 s (PHYCUV) par espèce.
**Témoin BOAFAB** (logistique, AP moyenne par site, minute) : **0,940**, tuyau sain.
**Transfert vers un site neuf, minute, AP moyenne par site / poolée**

| Tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| logistique | 0,333 / 0,754 | 0,253 / 0,075 | 0,376 / 0,197 | 0,491 / 0,097 | 0,940 / 0,808 |
| lda_shrunk | 0,306 / 0,316 | 0,266 / 0,076 | 0,474 / 0,377 | 0,373 / 0,096 | 0,914 / 0,883 |
| simple_prototype | 0,324 / 0,291 | 0,273 / 0,087 | 0,191 / 0,101 | 0,243 / 0,105 | 0,502 / 0,237 |
| *jetons* proto_probe | **0,369 / 0,801** | **0,284** / 0,070 | **0,559** / 0,333 | **0,661** / 0,110 | **0,972** / 0,886 |
| *jetons* logistic:max | 0,335 / 0,759 | 0,260 / 0,076 | 0,385 / 0,205 | 0,515 / 0,098 | 0,943 / 0,828 |
| *jetons* attentive | 0,322 / 0,456 | 0,271 / 0,095 | 0,465 / 0,316 | 0,349 / 0,115 | 0,924 / 0,900 |

Autres têtes et scores : `resultats/global/beats_*` (historique : `git show 6f2ff13^:resultats/global/<fichier>`).
**Courbe d'amorçage** (logistique, AP minute, moyenne des sites, k = 0 → 20 positifs du site) :
DENMIN 0,50→0,75, PITAZU 0,47→0,85, PHYCUV 0,39→0,72, LEPLAT 0,54→0,75, BOAFAB 0,94→0,96.
**Lecture** : transformer auto-supervisé **en retrait en sondage linéaire** (n° 151) : la sonde
à prototypes sur jetons gagne sur les cinq espèces (+0,03 à +0,18 d'AP par site sur la
logistique) ; la sonde attentive ne fait pas mieux que la logistique, sauf sur PHYCUV (et à peine
PITAZU). Poids AudioSet : sous naturebeats (même réseau) partout, toutes têtes comprises.
**Anomalies** : `jetons.py` à 4 threads, en parallèle du benchmark, ~15 fois trop lent :
relancé à 2 threads, débit normal. Stock : branche `donnees-anuraset-beats`.
