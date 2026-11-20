# Après la présentation n° 4 : ce qui reste à faire sur le contrôle qualité

Établi le 19/11/2026 sur la base du 18/11 (drapeaux `c78479120a36fe7b`), par les fonctions de `blanci/exploration/inventaire.py`, sauf mention contraire. Numéros de diapo : `presentation_4_inventaire_qc_v4_ultimate.pptx`.

## 1. Le « trou » de Mataroni (diapo 4) : il n'existe pas

Les tuteurs ont raison : la pose a été faite d'une traite. Le calendrier dessinait un petit bloc les 30-31/12/2025, puis un trou jusqu'au 06/01/2026. Ce petit bloc, ce sont les essais de réglage des enregistreurs au bureau, pas des enregistrements de terrain.

- Le dossier du relevé s'appelle « RELEVE 3 Mataroni - 06-13 janv 2026 ».
- Les 26 fichiers des 30-31/12 durent de 0,1 s à 6 min (médiane 4,9 s) ; aucun ne dure 120 s, et aucun n'est sur un créneau du programme.
- Ils suivent l'ordre des noms de terrain : le 30/12 de 14 h 29 à 16 h, MGM01 → MGM05, puis RB01 → RB07, puis SYL01 → SYL09 ; le 31/12, SYL15 → SYL26 à 11 h, et trois GI (rentrés de CDR le 28/12) à 15-16 h.
- 9 des 26 portent un nom en heure UTC, 17 en heure locale : le fuseau a été réglé pendant ces essais.
- Les fichiers `*_Summary.txt` des enregistreurs (F:) n'ont aucune ligne entre le 30/12 et le 06/01 : l'enregistreur était éteint.
- Le premier fichier du programme tombe le 06/01 entre 8 h 30 et 14 h selon le micro : l'ordre de la tournée de pose. Le dernier tombe le 13/01 entre 9 h 30 et 13 h.

Pourquoi le contrôle les a laissés dans le relevé : le hors relevé (`off_campaign`) coupe une série à un trou de plus de 7 jours ; les essais sont à 5,8-6,9 jours de la pose. Ils étaient quand même écartés, par leur durée. La figure les traçait parce qu'elle ne prenait que les fichiers non hors relevé, quelle que soit leur durée.

Corrigé : `fig_calendrier` (`documentation/prez/scripts/presentation_4_figures.py`) ne trace que les jours avec un fichier de 120 s ; `inv_calendrier.png` est régénérée (Mataroni du 06 au 13/01 ; CDR perd aussi ses essais des 19-20/12). À reprendre à la main dans la diapo 4. Même cas ailleurs : 49 fichiers d'avant le 29/12/2025, sur 42 micros de Mataroni, sont des restes de carte SD de 2024-2025 (tous écartés).

## 2. Positions GPS et carte des points

`uv run python scripts/verification/carte_points.py` écrit `data/reports/carte_points_ecoute.html` (hors dépôt : les positions d'une espèce menacée ne se publient pas). La fonction `inventaire.gps_places` range chaque position lue dans les en-têtes : un lieu où des micros de deux sites ou plus ont pris leur position est un lieu de réglage, pas une pose.

- 7 points sur 164 ont une position de pose : Kaw 2023 (2 sur 3), Molokoï 2023 (3), Trésor 2023 (2).
- Aucun point de 2026 n'en a : les 156 micros ont gardé la position lue au réglage. Huit lieux de réglage : cinq autour de Cayenne et Rémire (dont 78 points en 4,914 ; −52,326, 54 de Mataroni), un vers Matoury (4,845 ; −52,325), et la maison de la réserve Trésor (4,610 ; −52,279), où ont été réglés des micros de Trésor 2023, de RNRT, de Patawa Est et de CDR.
- Les `*_Summary.txt` donnent la même chose (une position par minute, celle du réglage en 2026).
- **À demander aux tuteurs** : les coordonnées de pose de 2026 (fiches de terrain, GPX), avec le nom de terrain de chaque point (MGM01, SYL12, RB05, GI19…) ; la carte les accueillera. Sans elles, le contrôle du site (diapo 22) reste impossible.

## 3. Ce qui reste à faire sur le contrôle

1. **Dates de relevé déclarées au lieu du seuil de 7 jours : fait le 20/11/2026** (`qc.surveys` dans `config/default.yaml` : 12 relevés, dossier d'origine, jour de pose, jour de relève ; les 26 essais de Mataroni sont hors relevé, les 18 fichiers du relevé 7 n'en sont plus ; zone tampon de 24 h, drapeau `survey_edge`, qui garde et signale 37 fichiers dont 2MA07942 de CDR récupéré le 28/12 ; la règle des 7 jours reste en repli pour `audio/R2_2024-02_fevrier`, et `blanci flag` le dit). Dates 2026 prises dans le nom des dossiers, dates 2023-24 dans les sous-dossiers (relève) et la relève précédente (pose) ; **la pose du relevé 1 de 2023 (Kaw 06/11, Trésor 08/11, Molokoï 14/11) est le premier fichier du programme : à remplacer par celle des tuteurs si elle diffère.** Reste : la complétude (diapo 7 : jours-micros 3 139, jours partiels 11, Molokoï E 36 jours au lieu de 54, à cause d'un fichier de SMA14157 le 28/12/2023, `confirmed_off_campaign` mais plus hors relevé) à relire.
2. **Micro sourd** (diapos 20 et 22) : mesuré, pas dans le contrôle. Un drapeau par piste quand l'écart entre pistes s'éloigne de plus de 6 dB du réglage, sur une suite (même règle que l'étouffé). Cinq points : Kaw D SMA14079 (23/05 – 23/07/2024), SMA14636 (CDR puis Patawa Ouest), Molokoï SMA14157, RNRT 2MA08067, Mataroni 2LA05191.
3. **Bascules en plein enregistrement** (11 cas à CDR) : en cours (`scripts/son_etouffe/bascules_fines.py`, files `bascules_fines` et `piste0_encodee_degradee`, non committés).
4. **Doublons renommés** : lancer le hachage du contenu (57 lignes ont une empreinte).
5. **Trous numériques** : mesurés sur 3 607 enregistrements seulement.
6. **Contrôle du site** : attend les coordonnées de pose (section 2).
7. **Deuxième témoin pour les écartés de 2023-24** (pas de BlanciNet) : gabarit A. blanci (bande 4,4-5,5 kHz, notes de 0,09-0,10 s, intervalle 1,2-1,9 s, `frac_ioi_blanci`), à valider sur les 157 positifs connus.

## 4. Ce qui doit être questionné dans le contrôle

- **« Confirmé » ne veut pas dire « écouté ».** Les files ont été confirmées en bloc : 506 `confirmed_muffled`, dont 40 ont une écoute en base ; 173 `confirmed_duration_off`, dont 29 ; 88 `confirmed_off_campaign`, dont 4. Il faudrait séparer « confirmé à l'écoute » de « confirmé en bloc », et ne plus écrire « réécoutés ».
- **Réglages à la main** : 5 % calé sur 8 enregistrements, « 4 de suite » sans loi (diapo 17 : une carte CUSUM le remplacerait), 7 jours (point 1 ci-dessus), tolérance de durée de 1 s.
- **`duration_off` écarte du vrai terrain** : 34 fichiers coupés de 30 à 119 s, dont 22 sur un créneau du programme (fin de batterie, pose, relève) ; 33 sont écartés, 5 seulement écoutés. Décision n° 216 en attente : garder un fichier du programme coupé s'il dure au moins 30 s.
- **La protection ne vaut que pour l'entendu** : un positif n'est jamais écarté, mais un écarté jamais écouté n'est protégé de rien (biais de vérification).
- **L'indice d'aigus ne voit ni le crépitement ni le micro sourd** : voir la discussion sur un indice spectral (section 7).

## 5. Enregistrements à écarter, à remettre, à écouter

À écouter avant de trancher :
- **10 écartés que BlanciNet note à 0,5 ou plus** (sur 18) sans écoute en base. Par score : CDR 2LA04171 27/09/2024 (0,81, reste de carte), 2LA04134 27/09/2024 (0,77, reste de carte), 2LA05017 25/12/2025 6 h (0,71, étouffé), 2MA10139 26/12 19 h 30 et 19 h (0,69 et 0,64), Mataroni 2LA04448 13/01 13 h (0,63, 28,7 s), CDR 2MA01489 26/12 19 h 30 (0,62), 2LA04513 11/10/2024 (0,57, reste de carte), Mataroni 2LA05209 06/01 11 h 30 (0,57, 88 s), CDR 2LA05017 25/12 5 h (0,55). Le 2LA05017 de 6 h a été écouté le 12/11 (fausse alerte) mais rien n'est en base : à enregistrer.
- **20 fichiers coupés sur un créneau du programme**, écartés et jamais écoutés (dont 2 à BlanciNet ≥ 0,5).

À remettre (`keep`), si la décision n° 216 est prise : les coupés d'au moins 30 s sur un créneau, après écoute. Déjà remis : Mataroni SMA13417 du 06/01 (A. blanci), les 18 du relevé 7.

À écarter : rien de nouveau de sûr. Le tirage au hasard (674 enregistrements) dit environ 1 % d'étouffés passés (0,3-1,9 %), soit de l’ordre de 290 à 1 820 enregistrements sur 95 630, qu'on ne sait pas encore désigner (section 7). Les restes de carte et les essais de réglage sont déjà tous écartés.

## 6. Enregistrements à réencoder sur la piste valable

- **407 enregistrements à piste 0 étouffée et piste 1 valable** (n° 203) : gardés, mais pas encore encodés sur la piste 1 ; 36 ont déjà des fenêtres, d'avant le n° 203, donc sur la piste 0 étouffée. `embed` lit la piste 1 de lui-même (`muffled_0` sans `muffled_1`) : lancer `uv run blanci embed --encoder <modèle>` pour chaque modèle (birdnet, birdnet_v3, audioprotopnet ; perch_v2 n'a pas de stock), et vérifier que les 36 sont refaits et pas sautés.
- **Kaw D SMA14079, 23/05 – 23/07/2024** : 1 860 enregistrements encodés sur la piste 0 sourde (1 871 sur la diapo : bornes de dates à revoir), dont 6 dans le jeu de test. Il faut d'abord le drapeau du micro sourd (section 3, point 2) pour que `embed` prenne la piste 1, puis réencoder et refaire l'évaluation.
- **Les quatre autres points à plus de 6 dB du réglage** : dire quelle piste est sourde et sur quelles dates avant de réencoder.
- **21 points au gain inversé** (piste 0 à 18 dB : 18 enregistreurs 2MA de Patawa Est et RNRT, 3 SMA de RNRT) : décider si on encode leur piste à 6 dB, pour que tout le corpus soit encodé au même gain.
- **68 extraits écoutés sur une piste et encodés sur l'autre** (15 à A. blanci) : les réencodages ci-dessus les règlent en partie ; vérifier la piste écoutée de chacun.

## 7. Autres points

- Chiffres de la diapo 13 : « 25 enregistrements de 19 micros » des 30-31/12 n'est plus vrai depuis les dates déclarées (`near_campaign_orphans` rend 0) ; dire « 26 enregistrements de 20 micros, hors relevé » (écartés par leur durée comme avant).
- Après les drapeaux nouveaux (dates de relevé, micro sourd), relancer `blanci flag`, puis `scripts/verification/chiffres_presentation_4.py` et le tableau de bord.
- Indice spectral pour les micros étouffés : discuté dans la session du 19/11 ; à décider avant d'écrire du code.
