# audioprotopnet : fiche AnuraSet (vague d'encodage 2, session 8)

Encodeur de bacpipe 1.3.5 (ConvNeXt + couche à prototypes, BirdSet), licence CC BY-NC 4.0 (non
libre). Têtes sur l'embedding **et sur ses jetons** (19 × 8 × 1 024, n° 151).

## Réglages lus
f_e 32 kHz, fenêtre 5 s jointive, 1 024 dimensions. Embeddings finis, non constants (écart type
0,19 à l'essai). torch 2.14.0+cpu, transformers 4.57.6, bacpipe 1.3.5, librosa 0.11.0 (ni
onnxruntime, ni tensorflow, ni avex).

## Débit
Encodage 1,8 fenêtre/s (CPU 4 cœurs), ≈ 4 h 50 pour 1 599 enregistrements ; jetons 8 269 s
(714 Mo, hors dépôt). Benchmark : 97 à 545 s par espèce (têtes sur l'embedding), 353 à 1 323 s
(avec jetons, 2 processus).

## Témoin BOAFAB (n° 151)
Logistique : AP moyenne par site (minute) **0,990** ≥ 0,85 : tuyau sain.

## AP moyenne par site (minute, site neuf, sans annotation de ce site)
| Tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB | moyenne |
|---|---|---|---|---|---|---|
| proto_probe (jetons) | 0,896 | 0,598 | 0,987 | 0,802 | 0,991 | **0,855** |
| logistic:max (jetons) | 0,854 | 0,494 | 0,921 | 0,836 | 0,961 | 0,813 |
| attentive (jetons) | 0,533 | 0,451 | 0,946 | 0,801 | 0,978 | 0,742 |
| logistic (embedding) | 0,721 | 0,472 | 0,971 | 0,859 | 0,990 | 0,803 |
| lda_shrunk (embedding) | 0,700 | 0,469 | 0,962 | 0,766 | 0,986 | 0,777 |
| simple_prototype (embedding) | 0,391 | 0,398 | 0,330 | 0,309 | 0,870 | 0,460 |

1er sur 24 encodeurs avec la sonde à prototypes (0,855) ; logistique 0,80 (4e). Contre la
référence perch_v2 + logistique : significatif après Holm sur DENMIN (+0,23), pas sur les quatre
autres. Contre le meilleur libre (perch_v2 + sonde à prototypes) : significatif sur DENMIN (+0,16)
seulement (PHYCUV +0,03, p Holm 0,61).

## Statut
CNN à prototypes : la sonde à prototypes sur jetons (sa tête d'origine) change le classement,
comme le prévoit le n° 151 ; pas « en retrait ». Non libre : à comparer au meilleur libre, jamais
livré (licence à confirmer).

## Anomalies
Sorties `resultats/global/audioprotopnet_*` dans l'historique (`git show 443f889:resultats/global/<fichier>`) ;
stock sur `donnees-anuraset-audioprotopnet`. Courbe d'amorçage des têtes sur jetons : non faite.
Un seul site porte l'écart significatif (DENMIN) ; PITAZU reste bas (0,60) pour toutes les têtes.
