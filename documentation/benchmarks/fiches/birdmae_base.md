# birdmae_base sur AnuraSet : têtes sur jetons, vague 2 (session 6, 30/09)
**Réglages lus** (bacpipe 1.3.5, `DBD-research-group/Bird-MAE-Base`) : ViT-B auto-supervisé
(MAE), f_e 32 kHz, fenêtre 5 s jointive, embedding 768 (moyenne des patchs puis `fc_norm`).
Stock du benchmark 07 importé (`importer_stock.py`), pas réencodé. Jetons 32 temps × 8
fréquences × 768, moyennés sur la fréquence. torch 2.6.0+cpu, transformers 4.57.6. Courbe : 07.
**Débit** : jetons 4 740 s (2 threads, en parallèle du benchmark de Large) ; benchmark
`--tokens`, 2 processus : 548 (PITAZU) à 1 102 s (PHYCUV) par espèce.
**Témoin BOAFAB** (logistique, AP moyenne par site, minute) : **0,934**, tuyau sain.
**Transfert vers un site neuf, minute, AP moyenne par site / poolée**

| Tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| logistique | 0,323 / 0,724 | 0,285 / 0,091 | 0,536 / 0,381 | 0,445 / 0,104 | 0,934 / 0,811 |
| lda_shrunk | 0,412 / 0,712 | **0,391 / 0,160** | 0,640 / 0,612 | 0,606 / **0,340** | 0,954 / 0,920 |
| simple_prototype | 0,353 / 0,311 | 0,281 / 0,119 | 0,246 / 0,111 | 0,312 / 0,163 | 0,669 / 0,348 |
| *jetons* proto_probe | **0,721** / 0,867 | 0,344 / 0,087 | **0,908 / 0,693** | **0,742** / 0,200 | **0,977 / 0,938** |
| *jetons* logistic:max | 0,333 / 0,602 | 0,269 / 0,113 | 0,503 / 0,267 | 0,427 / 0,097 | 0,935 / 0,791 |
| *jetons* attentive | 0,444 / **0,891** | 0,358 / 0,073 | 0,723 / 0,530 | 0,552 / 0,167 | 0,954 / 0,887 |
| *réf.* birdmae_large proto_probe | 0,654 / 0,917 | 0,334 / 0,102 | 0,906 / 0,708 | 0,579 / 0,102 | 0,979 / 0,944 |
| *réf.* perch_v2 logistique | 0,666 / 0,911 | 0,507 / 0,150 | 0,960 / 0,946 | 0,824 / 0,253 | 0,985 / 0,964 |
Scores : `resultats/global/birdmae_base_*` (historique : `git show 6f2ff13^:resultats/global/<fichier>`, transfert refait).
**Lecture** (verdict suspendu des n° 147 et 149) : Bird-MAE est **en retrait en sondage
linéaire**, pas mauvais. Avec la sonde à prototypes sur jetons (tête de ses auteurs), Base gagne
+0,04 (BOAFAB) à +0,40 (DENMIN) d'AP par site sur la logistique. Sur DENMIN, il **dépasse
perch_v2 + logistique** en AP par site (0,721 contre 0,666), sans le dépasser en AP poolée
(0,867 contre 0,911). Il reste en dessous sur PITAZU, PHYCUV et LEPLAT. Base égale ou dépasse
Large sur jetons (LEPLAT 0,742 contre 0,579) : la taille n'aide pas, donc **Huge n'est pas
lancé** (règle de la ligne 6 de `VAGUE_ENCODAGE_2.md`).
**Anomalies** : redémarrage du conteneur (3 espèces relancées) ; un point de reprise et des
durées écrasées, poussés par erreur, corrigés sur la branche (5c733bc).
