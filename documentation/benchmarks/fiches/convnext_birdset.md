# convnext_birdset : fiche AnuraSet (vague d'encodage 2, session 8)

Encodeur de bacpipe 1.3.5 (ConvNeXt, BirdSet), CNN supervisé : tête adaptée = logistique sur
l'embedding (pas de jetons, n° 151). `audioprotopnet` : **pas encore traité** (encodage repris
mais non fini, benchmark à faire plus tard).

## Réglages lus
f_e 32 kHz, fenêtre 5 s jointive, 1 024 dimensions, `pooler_output` (moyenne de la dernière
carte). Embeddings finis, non constants (écart type 0,61 à l'essai). torch 2.14.0+cpu,
transformers 4.57.6, bacpipe 1.3.5, librosa 0.11.0 (ni onnxruntime, ni tensorflow, ni avex).

## Débit
7,0 fenêtres/s (CPU 4 cœurs, 1 599 enregistrements, ≈ 75 min). Benchmark : 97 à 899 s par
espèce (2 processus).

## Témoin BOAFAB (n° 151)
Logistique : AP moyenne par site (minute) **0,980** ≥ 0,85 : tuyau sain.

## AP moyenne par site (minute, sans enregistrement du site cible, k = 0)
| Tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| logistic | 0,732 | 0,836 | 0,966 | 0,932 | 0,980 |
| logistic+R37=glmm | 0,729 | 0,857 | 0,966 | 0,931 | 0,980 |
| logistic+R20 | 0,767 | 0,868 | 0,968 | 0,927 | 0,980 |
| lda_shrunk | 0,762 | 0,797 | 0,959 | 0,857 | 0,964 |
| prototype | 0,610 | 0,762 | 0,917 | 0,927 | 0,970 |

## AP poolée (niveau fenêtre de 5 s, tous sites)
Pas au niveau minute : les décalages du stock n'ont pas été recalculés ici.
| Tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| logistic | 0,886 | 0,107 | 0,770 | 0,591 | 0,956 |
| logistic+R37=glmm | 0,902 | 0,096 | 0,769 | 0,579 | 0,954 |
| lda_shrunk | 0,922 | 0,249 | 0,773 | 0,143 | 0,931 |
| simple_prototype | 0,108 | 0,059 | 0,208 | 0,242 | 0,848 |

## Anomalies et limites
- Têtes limitées à celles que `global_bench.py --curve` produit (pas de comparaison appariée à
  perch_v2 ni de Holm : `rassembler_global.py` ne connaît pas encore cet encodeur).
- PITAZU : AP poolée très basse (0,11) malgré une bonne AP par site (0,84).
- Transformer : sans objet (CNN), pas de verdict « en retrait en linéaire ».
- Données : branche `donnees-anuraset-convnext_birdset` ; sorties dans `resultats/global/` de
  `resultats-anuraset-07`.
