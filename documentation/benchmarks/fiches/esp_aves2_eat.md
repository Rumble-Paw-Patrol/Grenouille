# esp-aves2 EAT sur AnuraSet : benchmark 07, vague 2 (session 13, 29-30/09)
**Réglages lus** (avex 1.3.0) : EAT, 16 kHz (AnuraSet à 22,05 kHz rééchantillonné), fenêtre 5 s
jointive, embedding 768 = jeton de classe (agrégation d'AVEX). Jetons 32 temps × 8 fréquences ×
768 (complément à 10 s retiré), moyennés sur la fréquence. torch 2.14.0+cpu, torchaudio
2.11.0+cpu, transformers 4.57.6, onnxruntime 1.30.0, sans TensorFlow ni librosa. 19 166 fenêtres.
**Tuyau réparé** : AVEX ne chargeait **aucun** tenseur (0/150) des points de contrôle SSL
`eat_bio` et `eat_all`, publiés au format fairseq : les deux tournaient avec l'EAT générique
d'AudioSet (`worstchan/EAT-base_epoch30_pretrain`), identiques entre eux. Clés remappées dans
`avex_encoder.py` (vérifié : tenseurs égaux au point de contrôle). Les `sl_eat_*` chargeaient bien.

## eat_all (SSL seul, bio + AudioSet)
**Débit** (4 cœurs) : encodage 2,9 f/s (1 h 50) ; jetons ~1 h 30 seuls ; benchmark
`--curve --tokens`, 2 processus : 389 (PITAZU) à 1 256 s (PHYCUV) par espèce.
**Témoin BOAFAB** (logistique, AP moyenne par site, minute) : **0,920**, tuyau sain.
**Transfert vers un site neuf, minute, AP moyenne par site / poolée**

| Tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| logistique | 0,311 / 0,192 | 0,259 / 0,111 | 0,489 / 0,319 | 0,269 / 0,088 | 0,920 / 0,747 |
| lda_shrunk | 0,304 / 0,374 | 0,292 / 0,199 | 0,477 / 0,455 | 0,279 / 0,075 | 0,941 / 0,896 |
| simple_prototype | 0,339 / 0,268 | 0,330 / 0,219 | 0,262 / 0,124 | 0,227 / 0,091 | 0,703 / 0,284 |
| *jetons* proto_probe | 0,326 / 0,230 | 0,247 / 0,095 | **0,800 / 0,569** | **0,593** / 0,102 | **0,974 / 0,938** |
| *jetons* logistic:max | 0,294 / 0,317 | 0,235 / 0,096 | 0,401 / 0,233 | 0,452 / 0,101 | 0,910 / 0,254 |
| *jetons* attentive | 0,286 / 0,210 | 0,326 / 0,115 | 0,507 / 0,400 | 0,411 / 0,077 | 0,932 / 0,844 |

Autres têtes et scores : `resultats/global/esp_aves2_eat_all_*` (branche `resultats-anuraset-07`).
**Courbe d'amorçage** (logistique, AP minute, moyenne des sites, k = 0 → 20 positifs du site) :
DENMIN 0,45→0,57, PITAZU 0,50→0,90, PHYCUV 0,54→0,82, LEPLAT 0,41→0,81, BOAFAB 0,94→0,95.
**Lecture** : transformer auto-supervisé **en retrait en sondage linéaire** (n° 151) : la sonde à
prototypes sur jetons gagne nettement sur PHYCUV (+0,31) et LEPLAT (+0,32), un peu sur BOAFAB ;
égalité sur DENMIN et PITAZU. Jeton de classe peu discriminant (cosinus 0,99 entre fenêtres).
**Anomalies** : conteneur redémarré pendant l'encodage (reprise, 417 sautés) ; jetons ~4 fois plus
lents en parallèle de l'encodage. Stock : branche `donnees-anuraset-esp_aves2_eat_all`.
