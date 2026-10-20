# esp-aves2 EAT sur AnuraSet : benchmark 07, vague 2 (session 13, 29/09-30/09)
**Réglages** (avex 1.3.0) : EAT, 16 kHz, fenêtre 5 s jointive, embedding 768 = jeton de classe
(AVEX) ; jetons 32 × 8 × 768 (complément à 10 s retiré), moyennés sur la fréquence. torch
2.14.0+cpu, transformers 4.57.6, onnxruntime 1.30.0, sans TensorFlow ni librosa. 19 166 fenêtres.
**Tuyau réparé** : AVEX ne chargeait aucun tenseur (0/150) de `eat_bio`/`eat_all` (format
fairseq) : EAT générique d'AudioSet à la place. Clés remappées (`avex_encoder.py`), vérifié.
**Débit** (4 cœurs) : encodage 2,3-2,9 f/s ; jetons ~1 h 30 ; benchmark 6-28 min par espèce.
**Témoin BOAFAB** (logistique) : eat_all 0,920 ; eat_bio 0,899 ; sl_eat_all 0,963 ; sl_eat_bio 0,980.
**Transfert, minute, AP moyenne par site / poolée** (proto_probe sur jetons ; autres têtes :
`resultats/global/esp_aves2_*eat*`, historique : `git show 6f2ff13^:resultats/global/<fichier>`)

| Encodeur, tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| eat_all logistique | 0,311 / 0,192 | 0,259 / 0,111 | 0,489 / 0,319 | 0,269 / 0,088 | 0,920 / 0,747 |
| eat_all proto_probe | 0,326 / 0,230 | 0,247 / 0,095 | 0,800 / 0,569 | 0,593 / 0,102 | 0,974 / 0,938 |
| eat_bio logistique | 0,255 / 0,466 | 0,282 / 0,139 | 0,452 / 0,358 | 0,524 / 0,104 | 0,899 / 0,806 |
| eat_bio proto_probe | 0,286 / 0,318 | 0,305 / 0,120 | 0,751 / 0,481 | 0,599 / 0,096 | 0,977 / 0,933 |
| sl_eat_all logistique | 0,384 / 0,878 | 0,361 / 0,136 | 0,614 / 0,413 | 0,672 / 0,115 | 0,963 / 0,811 |
| sl_eat_all lda_shrunk | 0,390 / 0,761 | 0,380 / 0,325 | 0,807 / 0,683 | 0,469 / 0,095 | 0,964 / 0,855 |
| sl_eat_bio logistique | **0,630** / 0,824 | 0,409 / 0,133 | 0,784 / 0,702 | **0,712** / 0,274 | **0,980** / 0,902 |
| sl_eat_bio lda_shrunk | 0,430 / 0,788 | **0,454 / 0,459** | 0,809 / **0,771** | 0,599 / **0,564** | 0,969 / **0,940** |
| sl_eat_bio proto_probe | 0,540 / 0,777 | 0,417 / 0,152 | **0,851** / 0,680 | 0,711 / 0,246 | 0,974 / 0,914 |

**Amorçage** (logistique, k = 0 → 20, moyenne DENMIN, PITAZU, PHYCUV, LEPLAT, BOAFAB) : eat_all
0,45/0,50/0,54/0,41/0,94 → 0,57/0,90/0,82/0,81/0,95 ; eat_bio 0,39/0,52/0,51/0,77/0,91 →
0,56/0,90/0,75/0,85/0,94 ; sl_eat_bio 0,56/0,76/0,83/0,78/0,98 → 0,69/0,95/0,91/0,90/0,98.
**Lecture** : eat_all et eat_bio (SSL seul) **en retrait en sondage linéaire** (n° 151) : la
sonde à prototypes sur jetons gagne +0,26 à +0,32 sur PHYCUV et LEPLAT. Les SL (SSL puis
supervisé) : la logistique suffit, jetons inutiles ; **sl_eat_bio** est le meilleur, logistique
0,70 en moyenne sur 5 espèces (perch_v2 : 0,79, benchmark 07).
**Anomalies** : quatre redémarrages du conteneur (reprises sans perte). Stocks : branches
`donnees-anuraset-esp_aves2_<encodeur>`.
