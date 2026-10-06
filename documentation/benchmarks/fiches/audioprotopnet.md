# audioprotopnet : fiche AnuraSet (vague d'encodage 2, session 8)

Encodeur de bacpipe 1.3.5 (ConvNeXt + couche à prototypes, BirdSet), licence CC BY-NC 4.0 (non
libre). **Provisoire : têtes sur l'embedding seulement.** Les jetons (19 × 8) sont en cours
d'extraction ; la sonde à prototypes sur jetons reste à mesurer (n° 151).

## Réglages lus
f_e 32 kHz, fenêtre 5 s jointive, 1 024 dimensions, jetons 19 × 8 × 1 024. Embeddings finis, non
constants (écart type 0,19 à l'essai). torch 2.14.0+cpu, transformers 4.57.6, bacpipe 1.3.5,
librosa 0.11.0 (ni onnxruntime, ni tensorflow, ni avex).

## Débit
1,8 fenêtre/s (CPU 4 cœurs), ≈ 4 h 50 pour 1 599 enregistrements (la couche à prototypes coûte
cher). Benchmark `--curve` : 97 à 545 s par espèce.

## Témoin BOAFAB (n° 151)
Logistique : AP moyenne par site (minute) **0,989** ≥ 0,85 : tuyau sain.

## AP moyenne par site (minute, k = 0, têtes sur l'embedding)
| Tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| logistic | 0,747 | 0,786 | 0,976 | 0,931 | 0,989 |
| lda_shrunk | 0,740 | 0,820 | 0,973 | 0,915 | 0,986 |
| prototype | 0,782 | 0,606 | 0,936 | 0,906 | 0,977 |

Moyenne des cinq espèces (benchmark 08) : logistique **0,80**, 4e sur 24 encodeurs ; aucun écart
significatif avec perch_v2 + logistique après Holm.

## Statut
CNN : « en retrait en sondage linéaire » ne s'applique pas (0,80, au niveau de convnext_birdset,
0,81). Pas de verdict sur sa tête d'origine (prototypes sur les cases de la carte) tant que les
jetons ne sont pas faits.

## Anomalies
Sorties sur la branche `resultats-anuraset-audioprotopnet` (ex-`resultats-anuraset-07`,
supprimée au n° 164) ; stock sur `donnees-anuraset-audioprotopnet`. Redémarrages fréquents du
conteneur : jetons à reprendre par paquets de 50.
