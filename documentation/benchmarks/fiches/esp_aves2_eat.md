# esp-aves2 EAT sur AnuraSet : benchmark 07, vague 2 (session 13, 29/09-30/09)
**Réglages lus** (avex 1.3.0) : EAT, 16 kHz (AnuraSet rééchantillonné), fenêtre 5 s jointive,
embedding 768 = jeton de classe (agrégation d'AVEX) ; jetons 32 temps × 8 fréquences × 768
(complément à 10 s retiré), moyennés sur la fréquence. torch 2.14.0+cpu, torchaudio 2.11.0+cpu,
transformers 4.57.6, onnxruntime 1.30.0, sans TensorFlow ni librosa. 19 166 fenêtres.
**Tuyau réparé** : AVEX ne chargeait **aucun** tenseur (0/150) de `eat_bio` et `eat_all` (SSL,
format fairseq) : ils tournaient avec l'EAT générique d'AudioSet, identiques. Clés remappées dans
`avex_encoder.py` (tenseurs égaux au point de contrôle). Les `sl_eat_*` chargeaient bien.
**Débit** (4 cœurs) : encodage 2,6-2,9 f/s ; jetons ~1 h 30 seuls ; benchmark `--curve --tokens`
(2 proc.) 6 à 28 min par espèce. **Témoin BOAFAB** (logistique) : eat_all 0,920 ; sl_eat_all 0,963.
**Transfert vers un site neuf, minute, AP moyenne par site / poolée** (proto_probe : sur jetons)

| Encodeur, tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| eat_all logistique | 0,311 / 0,192 | 0,259 / 0,111 | 0,489 / 0,319 | 0,269 / 0,088 | 0,920 / 0,747 |
| eat_all lda_shrunk | 0,304 / 0,374 | 0,292 / 0,199 | 0,477 / 0,455 | 0,279 / 0,075 | 0,941 / 0,896 |
| eat_all proto_probe | 0,326 / 0,230 | 0,247 / 0,095 | **0,800 / 0,569** | 0,593 / 0,102 | **0,974 / 0,938** |
| sl_eat_all logistique | 0,384 / **0,878** | 0,361 / 0,136 | 0,614 / 0,413 | **0,672 / 0,115** | 0,963 / 0,811 |
| sl_eat_all lda_shrunk | **0,390** / 0,761 | **0,380 / 0,325** | 0,807 / 0,683 | 0,469 / 0,095 | 0,964 / 0,855 |
| sl_eat_all proto_probe | 0,339 / 0,646 | 0,345 / 0,117 | 0,643 / 0,383 | 0,433 / 0,106 | 0,961 / 0,855 |

Autres têtes (attentive, logistic:max, R20, R37, simple_prototype) et scores :
`resultats/global/esp_aves2_*eat*` (branche `resultats-anuraset-07`).
**Courbe d'amorçage** (logistique, AP minute, k = 0 → 20) : eat_all DENMIN 0,45→0,57, PITAZU
0,50→0,90, PHYCUV 0,54→0,82, LEPLAT 0,41→0,81, BOAFAB 0,94→0,95 ; sl_eat_all 0,57→0,73,
0,69→0,90, 0,64→0,82, 0,75→0,83, 0,96→0,97.
**Lecture** : eat_all (SSL seul) **en retrait en sondage linéaire** (n° 151) : la sonde à
prototypes sur jetons gagne +0,31 (PHYCUV) et +0,32 (LEPLAT) ; jeton de classe peu discriminant
(cosinus 0,99 entre fenêtres). sl_eat_all (SSL puis supervisé) : la logistique suffit, les
têtes sur jetons n'apportent rien ; meilleur des deux sur 4 espèces sur 5.
**Anomalies** : deux redémarrages du conteneur (reprises : 417 et 538 sautés) ; jetons ~4 fois
plus lents en parallèle de l'encodage. Stocks : branches `donnees-anuraset-esp_aves2_<encodeur>`.
