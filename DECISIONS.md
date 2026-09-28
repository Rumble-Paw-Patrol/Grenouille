# Décisions d'implémentation

Écarts à la feuille de route (§13) ou précisions qu'elle ne tranche pas. Une entrée par
décision, datée ; une décision remise en cause reçoit une nouvelle entrée, l'ancienne reste.

## 2026-09-21 — Jalon M0

1. **Python 3.11, numpy < 2.** Conforme au §13.1 (le projet `uv init` était en 3.13).
   bacpipe 1.3.5 exige Python ≥ 3.11 et < 3.13 et fige `numpy==1.26.4`, `soundfile==0.12.1`,
   `pandas<3`, `pyarrow<26` : les dépendances de base sont bornées en conséquence pour que
   l'ajout de bacpipe en M1 ne casse rien. bacpipe n'est pas encore installé (torch +
   TensorFlow, plusieurs Go).

2. **`recording_id` = sha256 du chemin *relatif* à `paths.raw`** (POSIX), et non du chemin
   absolu. Les identifiants survivent au passage sur SSD externe ou sur un portable Windows de
   l'ONF. `recordings.path` stocke ce même chemin relatif.

3. **Label `blanci` ajouté** au vocabulaire du §13.3 : positif dont le solo/chœur n'est pas
   renseigné. Les 345 positifs importés n'ont pas cette information (H21 ouverte) ; les
   ranger en `blanci_solo` serait faux, en `blanci_uncertain` aussi (l'identification est
   sûre, H19). Positifs = {`blanci`, `blanci_solo`, `blanci_chorus`} ; `blanci_uncertain` et
   `uncertain` sont exclus de l'entraînement par défaut.

4. **Fenêtres annotées stockées telles quelles** dans `windows` (décalage et durée de
   l'annotation, 3 s Biophonia), indépendamment des grilles d'encodeur. Le transfert des
   labels vers une grille (3 s / 1,5 s ou 5 s / 2,5 s) se fera par inclusion au moment de
   l'entraînement (M1). `blanci check-grid` vérifie que chaque annotation positive tient
   entière dans une fenêtre de chaque grille.

5. **Qualité A/B/C déduite du commentaire** quand le fichier reçu n'a pas de colonne
   qualité : C si « lointain » ou « second plan », B si pluie, bruit ou autre espèce citée,
   A sinon. Marquée `quality_inferred: true` dans `labels.conditions` ; à relire avec le
   tuteur. Une colonne qualité explicite l'emporte toujours.

6. **Import d'un fichier d'annotation en tout ou rien.** Une ligne dont l'enregistrement est
   introuvable ou ambigu bloque le fichier (sauf `--allow-partial`). Table `imports` ajoutée
   au schéma : un fichier déjà importé (même SHA-256) n'est pas réimporté.

7. **Labels en ajout seul imposés par la base** : triggers SQLite refusant UPDATE et DELETE
   sur `labels` (§13.7).

8. **QC en numpy/scipy, pas scikit-maad, pour M0.** Quatre indices suffisent aux drapeaux
   demandés (RMS, fraction saturée, part d'énergie > 2 kHz, platitude spectrale 1–10 kHz).
   Seuils provisoires dans `config/default.yaml`, à calibrer sur les enregistrements étiquetés
   `rain` et `artefact_in_bag`. scikit-maad reste possible si des indices écoacoustiques
   deviennent utiles.

9. **Horodatage** : bloc GUANO du WAV en priorité (fuseau inclus), sinon nom de fichier Song
   Meter + `recorder.filename_utc_offset_h` (UTC−3). [À VÉRIFIER] sur les premiers fichiers
   réels : réglage horaire des enregistreurs 2023 et 2026.

10. **« A. hahneli » interprété comme *Ameerega hahneli*** (Dendrobatidae, présente en
    Guyane). La feuille de route écrit *A. hahneli* à la suite d'*Allobates femoralis*, ce qui
    suggère *Allobates*. [À VÉRIFIER] avec le tuteur. « piau » interprété comme Piauhau hurleur
    (*Lipaugus vociferans*), [À VÉRIFIER].

## 2026-09-21 — Environnement, tests des modules M1

11. **Modules hors de l'arborescence du §13.2** : `dataset.py` (transfert des annotations vers
    une grille d'encodeur, négatifs appariés, assemblage du jeu d'apprentissage) et `embed.py`
    (extraction reprenable, remplissage de `windows`, mesure du débit). Le §13.2 les laisse
    implicites entre `store.py` et `head.py` ; les isoler évite de gonfler `cli.py` et rend
    l'apprentissage testable sans audio. `benchmark.py` et `service.py` suivront de même.

12. **`Encoder.embed(wav, sr)` reçoit un lot de fenêtres**, de forme (n_fenêtres, n_échantillons)
    à la fréquence d'échantillonnage native de l'enregistrement, et non un signal continu. Le
    §13.4 écrit `embed(self, wav, sr) -> (n_windows, dim)` sans dire qui découpe. Découper dans
    `grid.py` puis passer le lot garde une grille unique, indépendante de l'encodeur (§4), et
    permet le traitement par lots. Le rééchantillonnage et le complément de zéros sont faits
    dans `BaseEncoder._prepare`, jamais chez l'appelant (§13.7).

13. **Grille d'un encodeur = sa fenêtre × `grid_hop_ratio` (0,5)**. Un encodeur à fenêtre de 3 s
    est donc échantillonné tous les 1,5 s, un encodeur à 5 s tous les 2,5 s. Le §1 impose un pas
    ≤ la moitié de la fenêtre pour qu'aucune note de 0,09 s ne soit coupée ; un ratio unique
    évite d'avoir à régler le pas encodeur par encodeur.

14. **Score d'enregistrement pour l'AP = maximum des fenêtres** (`to_recordings(how="max")`),
    avec `top3` en variante. Le §1 définit le score d'un enregistrement comme la *proportion* de
    fenêtres positives, mais une proportion suppose un seuil déjà fixé : circulaire pour une
    métrique de classement. Le maximum s'en passe. La proportion reste la sortie de
    `aggregate_recording`, pour la décision, pas pour l'AP.

15. **Négatifs appariés présumés, jamais écrits dans `labels`.** `paired_negatives` marque ses
    fenêtres `background_presumed` avec une colonne `presumed`. Personne ne les a écoutées : en
    saison et à l'heure de pic, une partie contient sans doute *A. blanci*. Les inscrire dans
    `labels` polluerait le jeu annoté et fausserait tout comptage de positifs. Le bruit
    d'étiquette est identique pour tous les encodeurs, donc sans effet sur leur comparaison (§2).

16. **`recall_at_precision` renvoie le seuil le plus élevé à rappel égal.** Il renvoyait le plus
    bas : même rappel, mais davantage de faux positifs dans la file de vérification. Le §6
    demande « le seuil le plus élevé gardant précision ≥ 0,1 ». Corrigé, avec test.

17. **Ruff ne formate pas le Markdown** (`extend-exclude = ["*.md"]`). `ruff format` reformatait
    les blocs de code Python du glossaire, qui est un document manuscrit.

## 2026-09-21 — Benchmark (§2)

18. **`knn_top1` exclut les fenêtres du même enregistrement** du voisinage. Deux fenêtres
    voisines d'un même fichier de 2 min se chevauchent (pas = moitié de la fenêtre) et
    partagent le fond sonore : le plus proche voisin d'un positif serait presque toujours la
    fenêtre d'à côté, et la mesure vaudrait 1 pour tous les encodeurs. Le §2 demande un
    « kNN top-1 » sans préciser le voisinage.

19. **Comparaisons appariées au niveau enregistrement seulement.** Le §2 demande des
    comparaisons appariées « sur les mêmes plis ». Deux encodeurs de fenêtres différentes
    (3 s et 5 s) n'ont pas la même grille, donc pas les mêmes fenêtres ; ils voient en
    revanche les mêmes enregistrements. `compare_encoders` apparie donc sur l'intersection
    des enregistrements, avec la sonde logistique.

20. **`paired_bootstrap` renvoie NaN plutôt que de planter** quand la métrique est indéfinie
    sur tous les rééchantillonnages (une seule classe). Il appelait `np.quantile` sur un
    tableau vide ; `bootstrap_ci` se protégeait déjà de ce cas.

## 2026-09-21 — Couche de service et CLI (§13.5)

21. **Format des annotations confirmé** (Léonard) : ce sont des **fenêtres de 3 s**, ce que
    `labels.import.window_s` suppose déjà. Reste inconnu : si ce sont des lignes de tableau
    (fichier + début) ou des extraits WAV découpés. Dans le second cas, il faudra un
    importeur qui retrouve l'enregistrement d'origine et le décalage.

22. **Registre des modèles dans `db.py`** (`model_params`, `encoder_params`, `register_model`,
    `next_version`), puisque c'est lui qui possède la table `models`. Les versions de tête
    s'incrémentent par encodeur : `v1`, `v2`… et les précédentes restent en base, pour
    comparer une décision ancienne au modèle qui l'a produite (§4).

23. **Identifiants estampillés** : `<encodeur>:head:<version>` pour une tête,
    `<encodeur>:head:<version>:p<précision>` pour un seuil. Le §13.3 impose les colonnes
    `encoder_id`, `head_version`, `threshold_id` dans `decisions` sans fixer leur forme.

24. **Le seuil est calibré sur les scores hors-pli**, pas sur les scores d'entraînement. Un
    seuil calibré en-pli serait optimiste, la tête ayant vu chacun de ses exemples (§6).

25. **Les décisions se remplacent, les labels s'ajoutent.** `score_and_decide` efface les
    décisions du même triplet (encodeur, tête, seuil) avant d'écrire : elles sont
    reproductibles à partir du modèle. Les labels, eux, sont en ajout seul (§13.7).

26. **`evaluate --holdout` sans site** retombe sur la validation groupée par micro (§6,
    niveau 1) au lieu d'échouer : c'est le protocole par défaut tant qu'il n'y a pas de
    positifs hors Mataroni.

27. **`embedded_training_set` dans `dataset.py`** : point d'entrée commun du benchmark et de
    l'entraînement. Les deux doivent voir exactement les mêmes labels et les mêmes négatifs
    appariés, sinon le benchmark ne prédit pas ce que fera la tête.

## 2026-09-21 — Format réel des données (Léonard)

28. **Format confirmé.** Enregistrements de 2 min sur disque externe, nommés
    `2la04530_20260106_103000` : micro `2LA04530`, 6 janvier 2026 à 10 h 30 locales. Les
    fichiers sont des **`.wav`**, mais l'Excel les cite en **`.flac`**. L'inventaire accepte
    désormais les deux (`audio.suffixes`), et l'appariement Excel ↔ disque se fait sur le nom
    **sans extension** : une citation `.flac` retrouve un `.wav`. Si les deux formats
    existent pour un même nom, l'extension citée départage ; sans extension citée, la ligne
    est signalée comme ambiguë plutôt que devinée.

29. **Le micro vient du préfixe du nom de fichier** quand l'arborescence n'a pas de dossier
    micro. Il est conservé tel quel (`2la04530`, minuscules) ; les comparaisons avec l'Excel
    passent par `normalize`, la casse n'a donc pas d'importance.

30. **Colonne « vérif manuelle » = verdict.** `parse_verdict` lit les oui/non (et variantes
    `o`, `x`, `ok`, `vrai`, `1`, `True`, `confirmé`…) et les réponses rédigées mentionnant
    *blanci*. Si la cellule nomme un faux ami reconnu (« fourmilier tacheté »), l'import en
    tire un négatif **et** l'espèce. Si elle ne dit rien de reconnaissable (« à revoir »,
    « ? »), la ligne est signalée : l'ancien code l'aurait rangée en négatif sans un mot, ce
    qui aurait transformé 345 positifs en 345 négatifs silencieusement.

31. **Score de l'ancien prestataire conservé** dans `labels.conditions.previous_model_score`,
    et résumé (min / médiane / max) par `--dry-run`. C'est le repère chiffré du §6 : il
    permettra plus tard de mesurer le rappel relatif au détecteur Biophonia, et de voir quels
    positifs il ne remontait qu'à score médiocre (§5, conséquence 2).

32. **Détection des colonnes en deux passes** : égalité exacte d'abord, puis libellé composé
    par mots entiers (« Nom de l'enregistrement » → `fichier`). Une colonne ne sert qu'à un
    seul champ. La comparaison par mots évite qu'un candidat `time` capture `timecode`.

33. **Timecode robuste aux formats Excel** : nombre, « mm:ss », « hh:mm:ss », `datetime.time`,
    `timedelta` et `Timestamp`. Excel rend une cellule horaire différemment selon son format
    d'affichage, et l'erreur serait silencieuse.

## 2026-09-22 — Acceptation M0 sur le disque réel

34. **Le fichier d'annotations est l'export complet de Blancinet v0.1.0**
    (`All_detections_blancinet_v0.1.0_dataset1BV.xlsx`) : 74 785 détections (score ≥ 0,10),
    dont **345 vérifiées `True`, 150 `False`**, 4 en attente (« à vérif », « à conf ») et
    74 286 jamais écoutées. Seules les vérifiées deviennent des labels. Une cellule de
    vérification vide est comptée « non vérifiée » et ignorée ; « à vérif » / « à conf » sont
    listées et mises de côté, sans label inventé et sans bloquer l'import. Un second fichier
    vérifié existe sur le disque (`..._dataset2_verifBV.xlsx`), non importé à ce jour.

35. **Les 345 positifs viennent de 51 enregistrements, pas de 345.** 13 micros, tous à
    Mataroni ; le point RB04 porte à lui seul 146 fenêtres (42 %). Le §1 (« 345 positifs =
    345 enregistrements distincts ») et H2 sont à corriger : l'effectif indépendant pour les
    intervalles de confiance est de l'ordre de 51 enregistrements, et bien moins de points.
    Conséquence directe sur le §6 : l'intervalle de Wilson « n = 345 → [0,86 ; 0,93] » ne
    s'applique pas ; avec n = 51, un rappel de 0,9 donne environ [0,79 ; 0,96].

36. **Clé S3 de l'ancien prestataire.** Les noms sont préfixés d'un identifiant numérique
    (`2353462-2la03550_20260108_143000.flac`). `file_key` retire ce préfixe, mais seulement
    devant un nom Song Meter, pour ne jamais tronquer un vrai nom de fichier.

37. **Commentaires dans des colonnes sans nom.** Le fichier range ses commentaires dans
    `Colonne1` (nom par défaut d'un tableau Excel) et `Unnamed: 12` (cellule sans en-tête).
    Toute colonne anonyme non attribuée alimente le commentaire, ainsi que le texte de la
    vérification. Une cellule qui recopie un en-tête (« Colonne1 ») est écartée.

38. **Micro = numéro de série, plus le dossier.** Remplace l'ordre de DECISIONS n° 2 et du
    jalon M0 (dossier d'abord). Sur le disque, le même enregistreur s'appelle « SM4 A »,
    « SM A_SMA13417 », « SMA13417 A » ou « SMA13417 » selon le relevé, et les fichiers sont
    souvent dans un sous-dossier `Data`. Priorité : `Serial` GUANO, préfixe du nom de fichier,
    puis dossier en dernier recours. Un même enregistreur change de site d'un relevé à
    l'autre (2LA03550 : Mataroni en janvier, RNRT en février) : le groupe des plis reste le
    couple site/micro (`point`), jamais le micro seul.

39. **Site donné à l'inventaire** (`blanci ingest --site`), et dossier scanné découplé du
    jeu. L'arborescence réelle (`<projet>/RELEVE n Site - dates/<SERIE>[_POINT]/Data/`) ne
    suit pas `<jeu>/<site>/<micro>/`. Les chemins restent stockés relatifs à `paths.raw`
    (racine du disque), dans `config/local.yaml`, ignoré par git.

40. **Doublons de relevés.** 3 668 noms existent deux fois sur le disque : les cartes SD
    emportées à Mataroni contenaient encore les fichiers de CDR, copies identiques octet pour
    octet (12/12 sur échantillon). Sans précaution, un même audio compterait deux fois, sous
    deux sites — fuite entre plis et entre sites tenus à l'écart. L'inventaire écarte un nom
    déjà vu sous un autre chemin et le signale (`ingest_duplicates_<jeu>_<site>.csv`).
    Le premier inventorié l'emporte : **inventorier les relevés dans l'ordre chronologique**.
    Vérifié : les 34 lignes vérifiées tombant sur un nom en double sont toutes des stations
    CDR, datées de la période du relevé CDR.

41. **Horodatage GUANO : 3 h d'erreur corrigées.** Les Song Meter écrivent le fuseau sans
    zéro (`2025-12-27 15:30:00-3:00`), que `datetime.fromisoformat` refuse. Le code se
    rabattait sans le dire sur « nom de fichier + UTC−3 » : juste par chance pour un
    enregistreur en heure locale, **faux de 3 h pour un enregistreur réglé en UTC** (`+0:00`,
    observé sur le disque). Le décalage est maintenant normalisé avant lecture. Répond à la
    question de DECISIONS n° 9 : le fuseau varie d'un enregistreur à l'autre, seul le GUANO
    fait foi.

42. **Inventaire rapide.** Sans contrôle qualité ni empreinte, seuls les en-têtes sont lus :
    16 ms par fichier au lieu de 0,8 s, soit le disque entier (33 210 fichiers) en quelques
    minutes au lieu de 7 h. Le contrôle qualité se lance ensuite sur ce qui sera encodé.

43. **Fichiers vides.** 8 fichiers de 0 octet (enregistreur SMA14826, relevé Patawa, heures
    irrégulières) : signalés en erreur, sans bloquer l'inventaire.

44. **À trancher avec le tuteur — audio stéréo à deux gains.** Les fichiers sont en 48 kHz,
    2 canaux, gains 6 et 18 dB (`WA|Song Meter|Audio settings`). `load_audio` fait la
    moyenne des deux, dominée par le canal à 18 dB, qui sature parfois (crête 0,987 observée).
    Garder un seul canal (lequel ?) ou la moyenne change les embeddings : à décider avant M1.

45. **Identifiant d'enregistrement = nom de fichier.** Remplace DECISIONS n° 2 (et le
    `sha256(path)` du §13.3). `recording_id = sha256(<série>_<date>_<heure> en minuscules)` :
    il ne dépend plus ni du dossier ni de l'extension. Motif : les enregistrements seront
    copiés sur un disque propre, dans une arborescence réorganisée ; avec l'ancien identifiant,
    chaque label et chaque embedding aurait perdu son enregistrement. Le nom est unique par
    construction (numéro de série + horodatage), et l'inventaire écarte déjà les doublons.

46. **Déplacement plutôt que doublon.** Un nom déjà inventorié dont l'ancien chemin n'existe
    plus sous la racine est un fichier déplacé : même identifiant, ligne mise à jour, labels et
    embeddings conservés. Si l'ancien chemin existe encore, c'est un doublon (écarté). Un
    réinventaire sans contrôle qualité n'efface pas le contrôle qualité déjà calculé.

47. **Qualité inconnue sans commentaire.** Remplace la règle « A sinon » de DECISIONS n° 5 :
    sans commentaire, la qualité reste vide. Sur le fichier réel, 339 positifs sur 345
    auraient reçu A par défaut, ce qui aurait vidé de sens le rappel par qualité (§6). Avec la
    nouvelle règle : A 37, B 4, C 2, inconnue 302.

48. **Ne rien écrire sur les disques externes.** Ils n'appartiennent pas au projet. L'inventaire
    et le contrôle qualité n'y font que des lectures ; la base, les rapports et les modèles
    vont dans `data/`, sur la machine. Vérifié le 22/09 : aucun fichier des deux projets n'a
    été modifié par l'inventaire (seuls l'Excel et son verrou `~$` portent une date du jour,
    08:56, antérieure à l'inventaire). L'export Excel du prestataire est ignoré par git
    (`documentation/*.xlsx`) : ce ne sont pas nos données.

49. **Résultat de l'acceptation M0** (22/09, disque « Projet blanci 2025 ») :
    - inventaire de 33 210 fichiers en 8 min (en-têtes seuls) : 29 513 enregistrements,
      980 h, 3 673 doublons de relevés écartés, 24 fichiers vides en erreur ;
    - 148 fichiers d'une durée autre que 2 min (0,1 s à 14 min : tests, déclenchements
      manuels) et 69 datés à plus de 15 jours de leur relevé (juillet 2024 à décembre 2025) :
      aucun n'est vérifié, mais **à écarter avant tout tirage de négatifs** (à faire) ;
    - quelques fichiers à 24 kHz (enregistrements de test) parmi les 48 kHz ;
    - import : 345 positifs, 150 négatifs, 74 286 détections non vérifiées ignorées,
      4 en attente ; les 495 lignes retrouvent leur enregistrement, et le site de
      l'inventaire concorde avec la station de l'Excel sur les 495 (CDR 41, Mataroni 445,
      PatawaOuest 9) — ce qui confirme au passage que « RELEVE 2 Patawa » est PatawaOuest ;
    - `check-grid` : 0 annotation positive coupée sur 345, grilles 3 s et 5 s ;
    - trois enregistreurs ont un préfixe de nom personnalisé (ALOUATTA, LAGOTHRIX, SAIMIRI) :
      le micro retenu est leur numéro de série GUANO (SMA11407, 2MA02726, SMA11393).

## 2026-09-22 — Canal audio, drapeaux d'inventaire, deux disques

50. **Canal audio : premier micro (gain 6 dB) par défaut** (`audio.channel: 0`), plus la
    moyenne des deux canaux. Mesuré sur 91 enregistrements (les 51 à positifs + 40 d'autres
    sites) :
    - les deux canaux sont **deux micros distincts** qui écoutent la même scène : enveloppes
      d'énergie très corrélées (0,75 à 0,94 le plus souvent), formes d'onde presque pas
      (0,05 à 0,40, sans décalage constant) ; un micro unique enregistré deux fois donnerait
      une corrélation proche de 1 au décalage 0 ;
    - l'écart de niveau vaut bien le réglage : +10 à +12 dB pour le second canal ;
    - **les 12 dB de plus n'améliorent pas la lisibilité des notes d'A. blanci** : rapport
      signal/bruit dans la bande 4,4–5,5 kHz de 8,5 dB (6 dB de gain) contre 9,8 dB (18 dB),
      écart médian par enregistrement +0,2 dB. Le fond de la forêt est 46 dB au-dessus du
      bruit de quantification 16 bits : le gain amplifie le fond autant que les notes ;
    - le micro à 18 dB sature dans 12 % des fichiers, celui à 6 dB dans 4 %.
    Moyenner deux micros ne gagne rien, hérite des saturations du plus amplifié (4 fois plus
    fort, il domine la moyenne) et additionne deux points d'écoute décalés de quelques
    centimètres, ce qui creuse des annulations de phase à des fréquences qui dépendent de la
    direction du son — de l'ordre de la bande d'A. blanci pour un écart de ~3,6 cm. Le canal
    est inscrit dans les paramètres de l'encodeur (`models.params_json`) : des embeddings de
    micros différents ne se comparent pas. Piste pour plus tard : le second micro comme
    augmentation de données (deux « vues » du même instant).

51. **Drapeaux d'inventaire, sans lire l'audio** (`duration_off`, `off_campaign`), recalculés
    à chaque inventaire et par `blanci flag`. Un enregistrement signalé reste dans la base et
    sur le disque ; il n'est jamais encodé, donc jamais tiré comme négatif apparié, proposé à
    la vérification ni scoré.
    - `duration_off` : durée hors de 120 s ± 1 s (tests, déclenchements manuels).
    - `off_campaign` : pour chaque (jeu, site, micro), la série de dates est coupée à chaque
      trou de plus de 7 jours ; le plus gros bloc est la campagne, le reste en sort. Un micro
      posé en continu sur plusieurs relevés d'un même site (phénologie 2023-2024) reste un bloc.
    Résultat sur l'inventaire 2026 : 148 durées anormales, 70 hors relevé, 149 enregistrements
    signalés au total (69 cumulent les deux), **aucun enregistrement annoté**. Un seul hors
    relevé a une durée normale (2LA04429, 17 décembre, carte posée à Mataroni le 6 janvier).

52. **Deux disques, un seul utile en plus.** D: (Samsung T7 Shield, 4 To) et E: (Seagate
    Basic, 1 To). Comparaison des noms de fichiers :
    - E: `enregistrements_blanci_mataroni_01-26` = copie exacte du relevé Mataroni du D:
      (les 9 noms « en plus » sont les fichiers vides) ;
    - E: `Pheno_blanci_Molokoi` = copie de la partie Molokoï de « Projet Phénologie blanci »
      du D: (21 009 noms sur 21 012) ;
    - **E: `Enregistrements blanci Mataroni mai 2026` : 18 fichiers absents du D:** (un micro,
      2LA03595 au point GI07, 19 mai 2026) ;
    - D: « Projet Phénologie blanci » (2023-2024, Trésor, Kaw, Molokoï, 66 824 fichiers, jeu
      de test temporel du §6) **n'est pas encore inventorié** ; « Mares » (16 568 fichiers,
      nov. 2024 – fév. 2025) et « SM_MaraisKaw_sd1/sd2 » (808 fichiers, 2026) : rôle à préciser.
    L'inventaire ne suit qu'une racine (`paths.raw`) : les deux disques ne s'inventorient pas
    ensemble. Inutile de le changer, les enregistrements seront regroupés sur le disque du
    stage (DECISIONS n° 45-46).

53. **Feuille de route corrigée par Léonard** : 345 fenêtres positives pour 51 enregistrements
    (DECISIONS n° 35 confirmé). D'autres annotations suivront, sur plus de sites. Le second
    export (`..._dataset2_verifBV.xlsx`) n'est pas importé : même format, aucune fenêtre
    vérifiée.

## 2026-09-23 — Corrections et précisions

54. **Correction de DECISIONS n° 50 : l'écart entre les deux micros est inconnu.** « Décalés de
    quelques centimètres » était une supposition, pas une mesure. Ce qui est établi : ce sont
    deux entrées sonores distinctes (aucune superposition des canaux, même en cherchant un
    décalage jusqu'à ±250 ms : corrélation maximale 0,07 à 0,42) ; le GUANO ne décrit pas les
    micros. L'argument contre la moyenne tient sans connaître l'écart : additionner deux micros
    distants crée des annulations à des fréquences qui dépendent de l'écart et de la direction
    du son. Écart à vérifier sur un enregistreur (deux ouvertures de micro sur le boîtier ?).
    Extraits d'écoute comparée : `data/ecoute/` (hors git), produits le 23/09.

55. **Le gain est fixé à l'enregistrement.** Il s'applique au signal analogique avant la
    numérisation : impossible de le changer après coup. Multiplier un canal sur ordinateur
    change son volume, pas son rapport signal/bruit, et ne rend pas ce qu'une saturation a
    coupé. Les deux canaux étant enregistrés, le choix se fait à la lecture (`audio.channel`).

56. **« Mares » et « SM_MaraisKaw_sd1/sd2 » (disque D:) n'appartiennent pas au projet** :
    autre étude sur le même disque. À ne jamais inventorier.

57. **La machine de travail actuelle est la machine cible de l'ONF** (§7) : Intel Core
    i5-1145G7, 4 cœurs / 8 fils, 16 Go, Windows. Le débit des encodeurs (risque 5, question 8
    du §10) se mesure donc ici, sans attendre le Mac.

## 2026-09-23 — Les deux micros, baselines, sous-ensemble du benchmark, poste d'annotation

58. **Les deux canaux sont conservés** (décision de Léonard après écoute, complète n° 50 et 55).
    Le micro 2 (18 dB) fait ressortir le chant à l'oreille ; sur un positif faible (chant
    lointain, chevauchement), il peut aider l'annotateur. Donc :
    - l'encodage reste sur `audio.channel` (0 par défaut), en attendant que le benchmark dise
      si le canal 1 change quelque chose pour les modèles ;
    - le poste d'annotation fait écouter **les deux micros côte à côte** et note dans
      `conditions.channel_listened` le canal du spectrogramme affiché ;
    - les baselines comparent déjà les deux canaux (`blanci baselines --channels 0,1`).
    L'écart de quelques centimètres entre les deux micros est confirmé sur un enregistreur
    (corrige la réserve du n° 54). Extrait d'écoute d'un positif faible (seul positif noté C,
    RB04) ajouté dans `data/ecoute/3_faible_*`.

59. **bacpipe sous Windows** : TensorFlow tire `tensorflow-io-gcs-filesystem`, sans roue Windows
    depuis la 0.32. Contrainte `<0.32` sous Windows dans `[tool.uv]` (lecture de fichiers sur
    Google Cloud, jamais utilisée ici). Groupes `research` et `app` installés sur la machine ONF.

60. **Sous-ensemble du benchmark** (`dataset.benchmark_recordings`, `blanci embed --subset
    benchmark`) : les enregistrements annotés + tous les candidats aux négatifs appariés (même
    micro, créneau à ± 30 min, non signalés). Sur la base actuelle : 131 annotés + 716 candidats
    = 847 enregistrements, 28 h d'audio au lieu de 980 h. Aucun encodage lancé : il attend
    l'accord de Léonard, une fois le squelette complet.

61. **Prototype simple ajouté aux sondes du benchmark** (§3 le liste comme baseline de
    similarité ; §6/§13 l'avaient omis). L'écart prototype simple / différentiel mesure le fond
    sonore capté par l'embedding (note de Léonard).

62. **Baselines sans encodeur** (`blanci/baselines.py`, `blanci baselines`) : énergie en bande,
    contraste bande / bandes voisines, onsets, rythme, template matching (gabarit moyen et
    meilleur de 30 exemplaires, appris dans chaque pli sans le micro testé). Même jeu que le
    benchmark : 345 positifs, 150 négatifs vérifiés, 1 020 négatifs appariés présumés (20 par
    enregistrement positif), plis par micro. Premier passage le 23/09 (2 min 48 s, lecture seule
    du disque D:) :
    - **AP fenêtre 0,22 à 0,37** (hasard 0,23) ; niveau enregistrement 0,10 à 0,15 (hasard 0,08).
      Aucune baseline n'approche le seuil du go/no-go ;
    - contre les seuls faux amis vérifiés, le template matching sépare bien (AP 0,93 au canal 1,
      hasard 0,70) ; contre les négatifs appariés, presque pas (0,33, hasard 0,25) ;
    - 28 % des négatifs présumés dépassent le score médian des positifs (gabarit moyen). Soit
      ils ressemblent au chant, soit **ils contiennent A. blanci** : à Mataroni, en saison, au
      même créneau, c'est plausible. Le bruit d'étiquette des négatifs présumés (§2) est donc à
      mesurer avant de juger les encodeurs : file d'audit `candidats_audit_negatifs.csv`
      (20 présumés les mieux notés + 20 au hasard) ;
    - canal 1 légèrement devant le canal 0 sur presque toutes les baselines (écarts dans les
      intervalles de confiance) ;
    - **le critère « rappel ≥ 0,8 à précision ≥ 0,1 » est vide au niveau fenêtre** avec ce jeu :
      23 % de positifs, tout accepter donne déjà une précision de 0,23. Il ne départage qu'au
      niveau enregistrement (8 % de positifs). À trancher avec le protocole figé (S3).

63. **Poste d'annotation** (`blanci/workbench.py`, `blanci/app.py` Streamlit, `blanci annotate`,
    `blanci candidates`). Files CSV dans `data/reports/candidats_*.csv` (aussi `queue_*` et
    `search_*`). `candidates --from <export Blancinet>` tire les détections jamais écoutées,
    `per_site` par site, à parts égales entre tranches de score (0–0,3 ; 0,3–0,7 ; 0,7–1) puis
    entre micros, une fenêtre par enregistrement ; `--random` ajoute des fenêtres au hasard aux
    heures de pic, à parts égales entre sites. Chaque réponse est un label en ajout seul
    (source `active`, `random` ou `audit`) ; la fenêtre est créée si besoin. Lot 1 : 100
    candidats (CDR 31, Mataroni 33, PatawaOuest 32, PatawaEst 2, RNRT 2). YAPAT n'est pas
    essayé : le poste suffit pour les premiers lots, l'essai reste possible (§5).

64. **Adaptateur bacpipe validé contre bacpipe 1.3.5** (torch 2.6, TensorFlow 2.15, Windows,
    i5-1145G7) le 23/09. Trois corrections :
    - les modèles sont dans `bacpipe.model_pipelines.feature_extractors.<nom>` (l'adaptateur
      cherchait `embedding_generation_pipelines`) ;
    - `Model(...)` ne lit ses réglages dans `bacpipe.settings` que si `device` est absent : on
      les passe tous, classifieur désactivé, et `prepare_inference()` est appelé ;
    - certains modèles rendent une séquence de jetons (birdmae) : moyennée pour l'embedding,
      exposée par `embed_tokens`, `has_tokens` mesuré au chargement.
    Poids dans `paths.models/bacpipe` (hors git), pas dans le dossier courant. Sous Windows,
    `tensorflow-intel==2.15.1` doit être déclaré (uv oubliait cette dépendance de tensorflow).
    Mesures sur bruit synthétique, CPU de la machine ONF (risque 5) :

    | encodeur | fenêtre | f_e | dim | fenêtres/s | temps réel (grille pas = ½ fenêtre) |
    |---|---|---|---|---|---|
    | birdnet | 3 s | 48 kHz | 1 024 | 26,0 | ×39 |
    | beats | 5 s | 16 kHz | 768 | 3,4 | ×8,5 |

    Ordre de grandeur : le sous-ensemble du benchmark (28 h) prend ~45 min avec birdnet et
    ~3 h 20 avec beats ; le disque entier (980 h), ~25 h et ~115 h. birdmae (version « Huge »,
    plusieurs Go) et perch_v2 ne sont pas encore téléchargés : à mesurer avant de les retenir.

## 2026-09-23 (après-midi) — Encodeurs, clustering, audit, congénères

65. **L'identifiant d'une fenêtre contient sa durée** (sauf 3 s, forme historique inchangée :
    les 495 fenêtres de la base gardent leur identifiant). Sans la durée, une fenêtre de 5 s
    (grille de beats, naturebeats, perch_v2) et une annotation de 3 s au même décalage, ou un
    label d'enregistrement entier (0 s, 120 s) et la fenêtre de 3 s à 0 s, partageaient une
    seule ligne de `windows` : la seconde héritait de la durée de la première, et le transfert
    des labels vers la grille devenait faux. Corrigé avant tout encodage : `<enr>:<décalage>`
    pour 3 s, `<enr>:<décalage>/<durée>` sinon (`db.window_id_for`).

66. **Relevé des encodeurs** (`blanci throughput`, §2 et §7) : débit, mémoire, dimension,
    jetons, projection sur une campagne (575 h) et sur les heures de pic seules. Bruit
    synthétique à 48 kHz, un processus par encodeur, rien n'est lu sur les disques. Premier
    relevé, i5-1145G7, **provisoire** (téléchargement de birdmae en parallèle : beats y tombe à
    2,1 fenêtres/s contre 3,4 mesurées seul, DECISIONS n° 64) :

    | encodeur | f_e | fenêtre | dim | temps réel | campagne 575 h | mémoire |
    |---|---|---|---|---|---|---|
    | birdnet | 48 kHz | 3 s | 1 024 | ×60 | 10 h | 2,0 Go |
    | perch_v2 | 32 kHz | 5 s | 1 536 | ×14 | 41 h | 4,1 Go |
    | beats | 16 kHz | 5 s | 768 | ×5 | 109 h | 2,2 Go |
    | naturebeats | 16 kHz | 5 s | 768 | ×5 | 106 h | 2,2 Go |

    **perch_v2 tourne en ONNX Runtime, sans TensorFlow** (`perch_v2_no_dft.onnx` fourni par
    bacpipe) : le repli (2) du §2 fonctionne sur la cible ; risque 7 levé pour l'exécution,
    la conformité aux embeddings de référence reste à vérifier. Il expose aussi des jetons
    spatiaux (16 × 4 × 1 536) que l'adaptateur n'utilise pas encore (attentive probing, P2).
    birdmae (Bird-MAE-**Huge** dans bacpipe, pas la version Base du §7) : téléchargement en
    cours ; protoclr, perch_bird, convnext_birdset ensuite.

67. **Contrôle passe-bas du §2** : `birdmae_lp8k` = birdmae sur l'audio filtré à 8 kHz
    (Butterworth d'ordre 8, phase nulle, à la f_e d'origine). Si son AP égale celle de
    birdmae, l'objection « 16 kHz, très limite » ne vaut pas pour la note de 4,4–5,5 kHz.
    Toute entrée de `encoders.models` accepte `lowpass_hz`.

68. **Clustering C0/C1** (`blanci cluster --mode c0|c1`, §5 bis) : normalisation L2, ACP
    (50 composantes), HDBSCAN (scikit-learn). C0 : échantillon du stock (tirage au prorata des
    partitions, une à la fois en mémoire), AMI groupes/micros, part de fenêtres signalées par
    groupe. C1 : positifs de Mataroni + 5 000 fenêtres des mêmes micros aux mêmes heures ;
    verdict sur le meilleur groupe (rappel ≥ 0,5, enrichissement ≥ 20, AMI micro ≤ 0,3 :
    seuils de jugement, dans `cluster` de la config). Les affectations par fenêtre sont
    écrites pour C2 (étiquetage en bloc). Fenêtres recadrées sur les onsets : pas encore.

69. **Écoute d'enregistrements entiers** (`blanci candidates --entiers N --reason …`) : audit
    aléatoire (§6, 300 enregistrements de Mataroni) et jeu gelé (60, stratifiés). Parts
    égales entre micros, puis heures locales les moins servies d'abord ; source `audit` ;
    le label vaut pour toute la grille (annotation par enregistrement, §5). Le jeu gelé n'est
    pas encore **exclu de l'entraînement** : à faire avant d'en écouter un (M4).
    **Calibration entre annotateurs** (§5) : case « ne masquer que mes réponses » dans le
    poste, et `blanci agreement --annotators a,b` : accord brut sur le label, accord
    blanci/non (« A. blanci ? » compte non), accord sur les positifs 2a/(2a+b+c), tableau
    croisé.

70. **Logits des congénères de Perch 2.0** (§2, §3, §5) : pendant `embed` avec perch_v2,
    l'adaptateur garde les logits d'*A. baeobatrachus*, *A. stepheni* et *A. surinamensis*
    (`logit_classes` de la config ; vérifié sur le vrai modèle) et `embed` les range dans
    `scores` (`<encodeur>:logit:<espèce>`), sans seconde inférence. `blanci candidates
    --congeners <perch_v2-…>` en tire une file : meilleure fenêtre par enregistrement, le
    meilleur de chaque micro d'abord. Logits non calibrés : classement seulement.

71. **Métriques du §6 complétées** : `evaluate.recall_by_group` (rappel au seuil de précision
    plancher par qualité A/B/C, site, tranche de RSB, avec Wilson), affiché par
    `blanci evaluate` ; `false_alarms_per_hour` (réservé aux ensembles exhaustifs : audit,
    jeu gelé) ; `sequential.note_snr_db` (énergie en bande pendant les notes contre les 0,5 s
    voisines). La qualité de l'annotation suit désormais la fenêtre jusqu'au jeu d'évaluation.
    La CLI écrit en UTF-8 : la console Windows en cp1252 plantait sur « ≥ ».

72. **Relevé des encodeurs, machine au repos** (i5-1145G7, CPU seul, bruit synthétique à
    48 kHz ; remplace les chiffres provisoires du n° 66). Campagne = une semaine de pose,
    575 h d'audio ; pas de la grille = ½ fenêtre ; `data/reports/debit.md` :

    | encodeur | f_e | fenêtre | dim | fenêtres/s | temps réel | campagne | heures de pic | mémoire |
    |---|---|---|---|---|---|---|---|---|
    | birdnet | 48 kHz | 3 s | 1 024 | 43,3 | ×65 | 9 h | 2,5 h | 2,0 Go |
    | protoclr | 16 kHz | 6 s | 384 | 17,5 | ×52 | 11 h | 3 h | 1,7 Go |
    | perch_v2 | 32 kHz | 5 s | 1 536 | 7,0 | ×17 | 33 h | 9 h | 4,1 Go |
    | birdmae_base | 32 kHz | 5 s | 768 | 3,7 | ×9 | 62 h | 17 h | 1,5 Go |
    | convnext_birdset | 32 kHz | 5 s | 1 024 | 3,7 | ×9 | 62 h | 17 h | 2,7 Go |
    | beats | 16 kHz | 5 s | 768 | 2,9 | ×7 | 81 h | 22 h | 2,2 Go |
    | naturebeats | 16 kHz | 5 s | 768 | 2,8 | ×7 | 82 h | 23 h | 2,3 Go |
    | perch_bird | 32 kHz | 5 s | 1 280 | 1,8 | ×4,5 | 129 h | 36 h | 2,4 Go |
    | birdmae (Huge) | 32 kHz | 5 s | 1 280 | 0,5 | ×1,2 | 494 h | 137 h | 4,2 Go |

    - **birdmae tel que bacpipe le charge (Bird-MAE-Huge) est hors de portée de l'i5**, même
      aux heures de pic seules. Variante `birdmae_base` ajoutée (Bird-MAE-Base, 768
      dimensions, le modèle que le §7 chiffrait) ; le benchmark dira si Base tient l'AP de
      Huge. Encoder le sous-ensemble du benchmark (28 h) avec Huge prendrait ~23 h ici : à
      faire sur le Mac, ou à remplacer par Base.
    - Seuil du risque 5 (> 15 h par campagne après leviers) : seuls birdnet (non déployable,
      licence et TensorFlow) et protoclr passent sans levier ; perch_v2 (33 h, 9 h aux
      heures de pic) passe avec les heures de pic ; les autres demandent aussi la
      quantification int8 et le sous-échantillonnage des fenêtres (§7).
    - perch_bird a échoué au premier téléchargement (délai dépassé chez Kaggle), réussi au
      second. Tous les encodeurs du §2 sont désormais installés (`data/models/bacpipe`, cache
      Hugging Face), 12 à 40 s de chargement chacun.

## 2026-09-23 (fin d'après-midi) — Jeu gelé, fusion, contrôle qualité, attentive, AnuraSet

73. **Fenêtres des encodeurs à 5–6 s : règle actuelle conservée** (décision de Léonard).
    Une fenêtre de la grille de l'encodeur hérite du label d'une annotation de 3 s si elle la
    contient entière (DECISIONS n° 4) ; vérifié : les 495 annotations tiennent entières dans
    une fenêtre de 3, 5 et 6 s. Pas de fenêtres recentrées sur les annotations.
    **Risque mesuré sur le contexte ajouté** : aucun des 150 négatifs n'est dans un
    enregistrement qui contient un positif annoté, mais pour 42 d'entre eux (5 s) et 28
    (6 s), Blancinet a une détection non écoutée de score ≥ 0,5 dans les 2–3 s ajoutées.
    Si du chant s'y trouve, la fenêtre est un négatif faux : la tête apprend à baisser le
    score d'un chant vrai, et le benchmark compte une fausse alarme là où l'encodeur avait
    raison. Le bruit touche les encodeurs à fenêtre longue, pas birdnet (3 s).

74. **Jeu gelé programmé** (`blanci/frozen.py`, `blanci freeze`, `blanci evaluate --frozen`).
    Une version = liste d'enregistrements figée (`paths.frozen_test/jeu_gele_<v>.csv`, en
    lecture seule, jamais réécrite). Ses enregistrements sont exclus de tout ce qui apprend
    ou règle : tête, seuil, benchmark, baselines, recherche de similarité, clustering C1,
    fusion, jetons. Chaque tête enregistre les versions exclues ; `evaluate --frozen` refuse
    une tête entraînée avant le gel. Mesures sur le jeu gelé : AP (enregistrement, fenêtre),
    rappel au seuil de la tête, fausses alarmes par heure (le jeu est écouté en entier), rappel
    par qualité et par site.

75. **Module séquentiel et fusion branchés** (§3). Débuts de notes calculés une fois par
    enregistrement pendant `embed` (l'audio est déjà en mémoire), rangés dans la table
    `onsets` (migration 2 de la base, ajout seul) ; `blanci onsets` pour les autres.
    `blanci fusion` : dans chaque pli par micro, une tête sans le micro score les fenêtres
    étiquetées et toutes les fenêtres de leurs enregistrements (persistance) ; fusion
    évaluée hors-pli contre la tête seule (bootstrap apparié par enregistrement), puis
    enregistrée en JSON avec son seuil. `blanci score --fusion` décide avec elle. Colonnes :
    `fusion.columns` (frac_ioi_blanci, onset_rate_hz, frac_windows, longest_run : 4, pour
    ~10 enregistrements positifs par coefficient). Le seuil de persistance est la frontière
    de la tête logistique (0).

76. **Calibration du contrôle qualité** (`blanci qc-calibrate`, lecture seule, 131
    enregistrements, 2 à 3 min). Indice « micro dans sac » = part d'énergie au-dessus de
    2 kHz : enregistrements avec fenêtres « micro dans sac » 0,001 à 0,30 ; enregistrements
    à A. blanci 0,265 à 1 (médiane 0,98) ; autres 0,30 à 1. **Le seuil actuel (0,02) ne
    signale qu'un enregistrement « dans sac » sur 8.** 0,2 en signalerait 6/8 avec une marge
    sous le positif le plus bas ; non appliqué, décision de Léonard. Pluie : trop peu de
    cibles (3 enregistrements) pour calibrer, seuil gardé. Silence et saturation : aucune
    fenêtre étiquetée. Rappel : l'inventaire a été fait sans contrôle audio (`--no-qc`),
    ces drapeaux ne sont donc calculés sur aucun enregistrement réel aujourd'hui.

77. **Attentive probing** (`blanci/attentive.py`) : une requête apprise pondère les jetons
    d'une fenêtre avant le classement (2d + 1 paramètres). Entraîné avec torch, appliqué en
    numpy, sauvegardé sans pickle. Seul perch_v2 expose des jetons (16 temps × 4 fréquences
    × 1 536, moyennés sur la fréquence). `blanci tokens --encoder perch_v2` calcule les
    jetons des seules fenêtres du benchmark (~1 500) ; `blanci benchmark` ajoute alors la
    sonde « attentive ». Sur données synthétiques (note dans 1 jeton sur 16), AP hors-pli
    ~0,67–0,8 contre ~0,53 pour la moyenne des jetons.

78. **Pré-benchmark AnuraSet préparé** (`blanci/anuraset.py`, `config/anuraset.yaml`,
    commandes `anuraset-prepare`, `anuraset-profile`, `anuraset-benchmark`). Licence
    **CC BY** (la feuille de route disait CC0). Téléchargé : `raw_data.zip` (7,2 Go,
    enregistrements bruts d'une minute) + `strong_labels.zip` (chants datés), pas
    `anuraset.zip` (extraits de 3 s, trop courts pour les encodeurs à 5–6 s). Base, stocks et
    rapports séparés des données ONF. Fenêtre positive = contient un chant entier de l'espèce
    (ou tient dans un chœur annoté d'un seul tenant) ; négative = aucun chant de l'espèce ;
    chant coupé = écartée ; négatifs tirés par site (1:20) ; plis par site (4 sites).
    Espèces à choisir avec `anuraset-profile` (note brève, dominante 3–6 kHz, ≥ 300 chants,
    ≥ 2 sites). Encodage d'AnuraSet (~27 h d'audio) : avec celui des données ONF, ce week-end.

79. **Drapeaux : trois origines, une règle d'exclusion** (23/09/2026, avec Léonard). Un
    drapeau est une remarque sur un enregistrement (`recordings.qc_flags`, le fichier n'est
    jamais touché). Origines : inventaire (durée anormale, hors relevé), audio (silencieux,
    saturation, micro dans sac, pluie), **écoute** (nouvelle clé `annotated` : présente sur
    tout enregistrement annoté à la main, avec les drapeaux que l'annotateur y a posés —
    label `artefact_in_bag` → micro dans sac ; label `rain` ou mention de pluie → pluie).
    **Seuls silencieux, micro dans sac, durée anormale et hors relevé écartent du corpus**
    (jamais encodés) ; pluie et saturation sont des remarques : un micro sous la pluie
    enregistre son milieu, ces enregistrements font partie du jeu de données. **Un
    enregistrement où A. blanci a été entendu n'est jamais écarté** : l'écoute prime sur le
    calcul. Le contrôle audio se fait pendant `embed` sur l'audio déjà lu (option `--no-qc`
    pour s'en passer), une fois par enregistrement ; ses indices sont gardés, `blanci flag`
    réapplique un seuil changé sans relire l'audio et recalcule les drapeaux d'écoute (aussi
    recalculés à chaque label ajouté et après chaque import). Sur la base au 23/09 : 131
    enregistrements annotés, 8 micro dans sac (aucun avec un positif), 5 pluie (dont 2 avec
    un positif). Seuil « micro dans sac » inchangé (0,02) tant que Léonard n'a pas tranché
    (n° 76).

80. **Négatifs suspects** (23/09/2026, règle de Léonard). Un négatif annoté est jugé sur
    3 s ; les encodeurs à fenêtre de 5–6 s y ajoutent 1 à 3 s jamais écoutées. Règle : un
    négatif est **suspect** si A. blanci est détecté dans les fenêtres voisines (± 3 s, la
    fenêtre de 3 s de chaque côté, ce qui couvre tous les encodeurs du §2) ; sinon il reste
    négatif. « Détecté » = détection Blancinet de score ≥ 0,5 que personne n'a écoutée
    (aucune fenêtre annotée ne la contient), ou positif annoté. Jamais les scores de nos
    propres têtes (ils choisiraient les labels qui les jugent). Un suspect ne sert ni à
    l'entraînement ni à l'évaluation, **pour tous les encodeurs** (mêmes négatifs pour tous,
    benchmark comparable). Même règle pour les négatifs appariés présumés : aucun n'est tiré
    à moins de 3 s d'une détection non écoutée. Les détections Blancinet (74 785, dont
    74 286 jamais écoutées) sont rangées comme scores du détecteur « blancinet »
    (`blanci import-detections`), pas comme labels. Au 23/09 : **46 négatifs suspects sur
    149**. `candidates --suspects` tire les 52 détections voisines à écouter : une voisine
    écoutée et non-blanci lève le soupçon, une voisine blanci le confirme (et donne un
    positif). **Effet mesuré sur les baselines** (relancées, lecture seule) : écarter les
    46 suspects annotés change peu l'AP (meilleure baseline, fenêtres : 0,368 → 0,374) ;
    écarter aussi les négatifs présumés voisins d'une détection la fait passer à 0,543
    (enregistrements : 0,155 → 0,263). Deux lectures : bruit d'étiquette retiré (au même
    micro, même créneau, en saison, une détection Blancinet ≥ 0,5 est souvent un vrai
    chant) ou négatifs difficiles retirés (évaluation optimiste). Le jeu gelé, écouté en
    entier, tranchera. Revers constaté : les suspects annotés sont surtout les faux amis
    principaux (*A. andreae* 12 sur 22, Fourmilier tacheté 11 sur 23), qui chantent en
    continu et déclenchent Blancinet sur les fenêtres voisines : sans écoute des voisins,
    la tête perd la moitié de ses exemples de ces faux amis. Écouter les 52 voisins avant
    l'entraînement.

81. **Seuil « micro dans sac » : 0,02 → 0,2** (accord de Léonard, calibration n° 76).
    Signale 6 enregistrements « dans sac » annotés sur 8, aucun enregistrement à A. blanci
    (le plus bas est à 0,265). Contrôle audio pendant `embed` réglable
    (`qc.during_embed`), coupé pour AnuraSet : seuils calibrés sur les Song Meter ONF, et
    positifs AnuraSet absents de la table labels, donc non protégés.

82. **Commentaire accolé à chaque fenêtre annotée** (précision de Léonard sur le n° 79 : le
    « drapeau » voulu est le commentaire de l'annotateur, en plus du label). Il était déjà
    gardé tel quel dans chaque label (`conditions.comment`) ; il suit désormais la fenêtre :
    `current_labels` et le jeu d'apprentissage (fenêtres de grille héritières) portent une
    colonne `comment` ; `blanci export-labels` écrit toutes les fenêtres annotées avec label,
    qualité, espèce, commentaire et suspect. Au poste d'annotation, le commentaire est lu
    comme à l'import (conditions pluie, lointain, second plan ; espèces citées) : « pluie »
    écrit au poste pose le drapeau pluie. Les drapeaux d'écoute par enregistrement (n° 79)
    restent : ce sont eux qui écartent les 8 enregistrements « dans sac ».

83. **Réentraînement sans intervention** (§4, M5, `blanci retrain`). Nouvelle tête sur
    tous les labels hors jeu gelé, seuil hors-pli ; nouvelle tête et tête adoptée jugées
    sur le même jeu gelé, à leur seuil ; adoption si l'AP (enregistrements) et le rappel au
    seuil ne perdent pas plus de 0,02 (`retrain.tolerance`). Sans jeu gelé : pas
    d'adoption, sauf `--force`. Tête adoptée non jugeable (entraînée avant le gel) : la
    nouvelle est adoptée. Historique des adoptions dans `models` (kind `adoption`) ; une
    tête refusée reste en base. `blanci score` prend par défaut la tête adoptée (sinon la
    plus récente) : une tête refusée n'est jamais utilisée par mégarde. La fusion n'est pas
    réentraînée par cette commande.

84. **Courbes d'activité** (§6 niveau 3, M4, `blanci activity`). Depuis les décisions par
    enregistrement : indice horaire (part des enregistrements détectés par heure locale,
    l'effort au dénominateur) et probabilité journalière par mois (part des jours avec au
    moins une détection), intervalles de Wilson, par site ou par micro. Confrontées aux
    patrons de Courtois et al. 2025 (`activity.reference` : pics 7–9 h et 15–17 h, mois
    forts janvier–avril, creux juillet–octobre) : un patron est « retrouvé » si les
    intervalles de Wilson (pics contre autres heures, mois forts contre creux) sont
    disjoints — un rapport > 1 seul se lève au hasard sur une activité plate. Avec des
    courbes numérisées de la publication (`--reference-hours`, `--reference-months`),
    corrélation de Pearson, alerte sous 0,5 (§11, risque 6). Enregistrements « suspect »
    non comptés, sauf `--suspects-detected`. À lancer sur 2023 une fois la phénologie
    inventoriée, encodée et scorée.

85. **Négatifs suspects abandonnés** (23/09/2026, Léonard ; remplace la partie « négatifs
    annotés » du n° 80). Les voisins d'un négatif sont souvent le même faux ami (un
    Fourmilier tacheté qui chante sur une fenêtre chante sur les suivantes) : écouter les
    voisins en ferait écouter d'autres, sans fin. Règle : **un négatif annoté est négatif,
    quel que soit le contexte** ajouté par les encodeurs à fenêtre de 5–6 s. Si A. blanci ne
    chante pas pendant les 3 s écoutées, qu'elle commence juste après est peu probable.
    **Biais possible, gardé en tête** : quelques fenêtres « négatives » de 5–6 s peuvent
    contenir la fin d'un chant ; il pèserait sur les encodeurs à fenêtre longue. Supprimés :
    colonne `suspect`, `candidates --suspects`. **Gardé** (question distincte, non
    tranchée) : aucun négatif *présumé* n'est tiré à ± 3 s d'une détection Blancinet ≥ 0,5
    non écoutée ; les détections restent rangées comme scores. Baselines relancées sous
    cette règle.

86. **AnuraSet extrait et profilé** (23/09/2026, lecture et extraction seules, rien
    d'encodé). 1 612 enregistrements d'une minute, 4 sites, archive vérifiée. Aucune espèce
    ne remplit tous les critères (note ≤ 0,3 s, dominante 3–6 kHz, ≥ 300 chants, ≥ 2
    sites) : les durées sont celles des segments annotés, qui regroupent souvent plusieurs
    notes, et la plupart des espèces n'est présente que sur un site. Seule candidate
    multi-sites dans la bande : **DENMIN** (*Dendropsophus minutus*, 1 724 chants, 3 sites,
    ~5,3 kHz, segment médian 0,61 s). Mono-site dans la bande : LEPPOD (761, ~5,8 kHz,
    0,33 s), PHYDIS (419, ~5,3 kHz, 0,33 s), DENNAN (596, ~4,4 kHz, 0,44 s) ; elles
    imposeraient des plis par enregistrement, qui ne disent rien du transfert entre sites.
    Fréquence dominante mesurée sur l'intervalle annoté : peut être captée par des
    insectes (PHYCUV à 6,3 kHz est suspect). Choix des espèces (`anuraset.species`) :
    Léonard.

87. **Règle des négatifs présumés voisins supprimée** (23/09/2026, Léonard). Plus aucune
    règle liée aux détections Blancinet dans la construction des jeux : négatifs annotés et
    présumés sont tirés comme avant le n° 80. Les détections restent rangées comme scores
    du détecteur « blancinet » (`import-detections`), pour comparer Blancinet et nos têtes
    sur les mêmes fenêtres. Baselines relancées : de nouveau celles d'avant le n° 80.

## 2026-09-24 — Ajouts demandés par Léonard (liste du 24/09, après le point d'étape)

88. **Négatifs appariés : trois stratégies** (`benchmark.pairing`, `dataset.py`). Écart
    constaté : la règle « même créneau, autre jour » n'excluait pas le même jour ; comme les
    micros enregistrent toutes les 30 min, les enregistrements voisins du positif (± 30 min, le
    même jour) étaient tirés aussi. Désormais : `other_day` (défaut, autre jour vraiment),
    `same_day` (même jour, les enregistrements les plus proches à au moins
    `same_day_min_gap_min`, et au plus ce délai + la tolérance), `mixed` (moitié, moitié ; l'un
    complète l'autre). Colonne `pairing` sur chaque négatif présumé. Avis donné à Léonard : le
    même jour est le meilleur témoin du fond (météo, chœur, saison), mais A. blanci chante par
    épisodes : le voisin le plus proche est aussi le plus susceptible de la contenir. À
    trancher en écoutant une vingtaine de négatifs présumés de chaque stratégie. Le
    sous-ensemble à encoder (`embed --subset benchmark`) prend les candidats des deux
    stratégies : changer de stratégie ne demande pas de ré-encoder. **Les baselines et tout
    benchmark antérieur sont à relancer** (négatifs changés).

89. **Chevauchement ajustable des fenêtres** (`encoders.overlap`, `embed --overlap`,
    `grid.hop_for_overlap`). De 0 (fenêtres jointives, grille standard) à 0,99 ; pas =
    fenêtre × (1 − chevauchement), au centième (précision de `window_id`), jamais moins de
    0,01 s. Hors 50 %, stock séparé `<encodeur>@o<%>` : deux grilles ne se mélangent jamais,
    et les têtes, scores et décisions de chaque grille restent distincts. À 99 %, 50 fois plus
    de fenêtres qu'à 50 % : réservé au sous-ensemble du benchmark. `grid_hop_ratio` reste lu.

90. **Seuillage spectral en amont, fonctionnalités activables** (`blanci/prefilter.py`,
    section `prefilter`, option `--prefilter`, `blanci prefilter-bench`). Revient, à la
    demande de Léonard, sur le « pas de filtre amont » du §3 (jugement) : tout reste coupé par
    défaut, et le banc d'essai juge avant tout usage. Transformations (le son donné à
    l'encodeur change → autre encodeur, stock `<nom>+bp3-7k`, jugé par le benchmark) :
    passe-bande, débruitage par soustraction spectrale. Portes (une fenêtre arrêtée n'est pas
    encodée, embedding nul, colonne `gated` du stock, score `GATED_SCORE`, jamais apprise ; un
    positif arrêté compte comme manqué) : énergie en bande, contraste aux bandes voisines,
    débuts de notes, rythme ; combinaison « toutes » ou « une ». Banc d'essai : par porte et
    par seuil, négatifs arrêtés (calcul économisé) contre rappel plafond en enregistrements ;
    avec un encodeur, AP après la porte, et la combinaison configurée jugée fidèlement (tête
    réentraînée sans les fenêtres arrêtées) contre l'absence de porte.

91. **Plis communs et stock de scores hors-pli** (`dataset.folds_for`, `blanci/oof.py`,
    `blanci sources`). Les plis étaient recalculés sur chaque jeu (StratifiedGroupKFold sur
    les fenêtres) : deux encodeurs, avec leurs grilles et leurs négatifs présumés, n'avaient
    pas exactement les mêmes micros tenus à l'écart. Désormais le pli de chaque micro est
    calculé une fois, sur les enregistrements annotés (hors jeu gelé), et partagé par
    encodeurs, têtes, baselines, fusion, ensembles et détecteurs. Chaque benchmark range ses
    scores hors-pli (`paths.reports/oof/`) au même format, avec une empreinte des labels et
    réglages : une source calculée sur d'autres labels est signalée au lieu d'être comparée
    en silence. Noms des sources : `<encodeur>/<tête>`, `baseline/<nom>/c<canal>`,
    `<encodeur>/fusion:<emplacement>/<méthode>`, `ensemble:<méthode>(…)`, `detector/<nom>`,
    `external/blancinet`.

92. **Toutes les têtes à comparer** (`head.py`, `pooling.py`, `head_benchmark.py`,
    `blanci heads`). Ajouts : recherche par l'exemple (plus proche positif ;
    `exemplar_medoid` = un seul exemple de référence, le positif le plus central), linear
    probe sur jetons résumés (`logistic:<pooling>` : moyenne, maximum, les deux, moyenne +
    écart-type, top-k, moyenne en fréquence et maximum en temps, et l'inverse), cascade
    (logistique puis attentive sur les 20 % meilleurs candidats, `head.cascade_fraction`). Les
    jetons de perch_v2 sont gardés en grille 16 temps × 4 fréquences (ils étaient moyennés sur
    la fréquence) ; l'attentive les prend à plat. Seul perch_v2 expose ses jetons : pour un
    transformer, `encoders.models.<nom>.token_grid` (à vérifier sur le code du modèle,
    notes.md) remet les jetons en grille. Le benchmark des têtes donne aussi le diagnostic du
    fond capté (prototype différentiel contre simple, apparié). Mécanisme d'attention propre
    au modèle et wrapper bacpipe : non programmés (rien de concret à brancher aujourd'hui).

93. **Courbe selon le nombre d'annotations du site cible** (`blanci heads-curve`). Hypothèse
    de Léonard : le prototype différentiel généraliserait mieux que le linear probe sur un site
    peu annoté. Protocole : par cible (micro tant que tous les positifs sont à Mataroni, site
    ensuite), moitié de test fixe tirée au hasard (positifs, négatifs annotés, négatifs
    présumés séparément) ; entraînement = autres cibles + négatifs présumés de la réserve (le
    fond du site, gratuit) + k enregistrements positifs de la réserve (k emboîtés, 0 = transfert
    pur) et une part proportionnelle de ses négatifs annotés ; 5 tirages ; C choisi une fois
    par cible sur les autres. Écart de chaque tête à la référence, apparié par (cible, tirage),
    intervalle bootstrap sur les cibles : l'hypothèse tient si différentiel − logistique > 0
    aux petits k.

94. **Fusion à N entrées** (`fusion.py`, `stacking.py`, `blanci fusion-bench`, `fusion.method`,
    `fusion.sources`). Entrées : score de la tête (hors-pli), descripteurs séquentiels, et
    toute autre source — tête d'un autre encodeur (`head:<id>`, apprise pli par pli, ramenée
    sur la grille principale), logits des congénères (`congeners:<id>`) : trois entrées ou plus.
    Méthodes : logistique (poids appris), poids fixés à la main (`fusion.weights` ; les entrées
    sans poids se partagent le reste), poids cherchés sur une grille, moyenne, moyenne des
    rangs, maximum (OU), minimum (ET), entrées orientées sur l'entraînement. La part de chaque
    entrée répond à « quel expert écouter en priorité ? ». `blanci fusion` enregistre la
    méthode, l'emplacement et les sources de la config ; une fusion déjà enregistrée dans
    l'ancien format reste lue.

95. **Module séquentiel ≠ seuillage spectral ; emplacement paramétrable**
    (`sequential.position`). Réponse à la question de Léonard : le seuillage agit sur le son
    avant l'encodeur ; le module séquentiel produit des descripteurs (rythme, sur l'audio ;
    persistance, sur les scores de la tête). Ils partagent leurs briques (enveloppe en bande,
    débuts de notes). Emplacements : `upstream` (le rythme sert de porte : c'est une porte du
    seuillage en amont, calculée sur les débuts de notes rangés, `sequential.gate`),
    `parallel` (rythme fusionné avec la tête), `downstream` (persistance, qui n'existe qu'après
    la tête), ou aucun. Défaut : parallèle + aval (la fusion d'avant). `fusion-bench` compare
    les emplacements × méthodes à la tête seule.

96. **Ensemble de modèles** (`ensemble.py`, `blanci ensemble`). Oui, il faut un outil : les
    grilles diffèrent (3 s, 5 s). Trois voies : combinaison par enregistrement de n'importe
    quelles sources du stock hors-pli (méthodes de la fusion, apprises pli par pli) ;
    combinaison par fenêtre = fusion à N entrées (production possible) ; concaténation des
    embeddings (blocs normalisés) pour des encodeurs de même grille. Un ensemble n'est jugé
    que contre sa meilleure source seule.

97. **Emplacements : distillation, modèle fait maison, fine-tuning / LoRA**
    (`blanci/detectors/`, `blanci/finetune.py`, `blanci detector-bench`). Contrat
    « détecteur » (audio → score, `fit` s'il apprend) et banc d'essai déjà branchés : hors-pli
    sur les plis communs, scores rangés dans le stock commun, donc repris par le benchmark
    complet et les ensembles. `band_contrast` montre que la chaîne marche ; `distilled` et
    `homemade` disent qu'ils sont réservés. Fine-tuning / LoRA : contrat de sortie (un
    encodeur en paquet ONNX) et contrainte d'évaluation (adaptation pli par pli, ou jugement
    sur le seul jeu gelé et les nouveaux sites). Conforme au §13.7 : rien avant M4.

98. **Benchmark complet des modèles** (`full_benchmark.py`, `blanci benchmark-all`). Toutes
    les sources au niveau enregistrement, sur les enregistrements évalués par toutes : AP [IC],
    rappels aux précisions plancher [Wilson], rappel par site, coût des encodeurs (dimension,
    débit), fraîcheur des labels ; comparaison appariée à la référence. Sources externes
    rangées à la demande sur les fenêtres des baselines : Blancinet (peut-être entraîné sur ces
    mêmes enregistrements : optimiste, à confirmer avec Biophonia), logits des congénères.

99. **Outil de sélection unique** (`selection.py`, `blanci select`, `cluster-status`,
    `cluster-label`, `yapat-export`, `yapat-import`). Méthodes : 60-20-20 à proportions
    réglables, similarité, couverture (k-centres : YAPAT fait maison automatique), groupes
    HDBSCAN puis étiquetage en bloc d'un groupe homogène (10 écoutes d'un même label,
    `selection.cluster_min_checked`), audit aléatoire, hasard, negative mining (scores hauts là
    où A. blanci est improbable, ou proches des faux amis annotés), gabarit phénologique
    (`selection.phenology`), détections isolées, congénères, Blancinet. Nouvelles sources de
    labels : mining, phenology, suspect, coverage, cluster, bulk (propagé sans écoute), yapat.
    YAPAT (Docker, PostgreSQL, embeddings BirdNET, licence non commerciale) n'est pas embarqué :
    export d'extraits + manifeste, relecture des réponses (format non documenté : à valider au
    premier essai, `selection.yapat_label_map`).

100. **Poste d'annotation refondu** (`blanci annotate`). Mode de sélection dans le panneau de
     gauche (file existante, ou file tirée sur place par n'importe quelle méthode), carte des
     embeddings où l'on entoure une zone à écouter (YAPAT fait maison, à la main), formulaire
     classe + qualité + espèce + commentaire et bouton « Envoyer ▶ » (label ajouté, fenêtre
     suivante), étiquetage en bloc des groupes homogènes. `app/streamlit_app.py` : voir n° 104.

## 2026-09-24 (soir) — Retours de Léonard sur les ajouts

101. **Négatifs appariés : la plus proche, même enregistrement compris, par défaut**
     (`benchmark.pairing: nearest`, décision de Léonard). Rappel : ce ne sont que des fenêtres
     *présumées* négatives (jamais écoutées, jamais écrites dans la table labels) ; les négatifs
     annotés entrent à part. Désormais, par enregistrement positif, les fenêtres les plus proches
     dans le temps d'une annotation positive, dans l'enregistrement lui-même d'abord — jamais une
     fenêtre qui chevauche une annotation positive, jamais une fenêtre encadrée de positifs
     (n° 102) —, puis le même jour, puis un autre jour (colonne `pairing` : same_recording,
     same_day, other_day). **Risque signalé** : si A. blanci chante tout l'enregistrement quand
     elle chante (H20, notes de Léonard), les voisines d'un positif en contiennent : négatifs
     faux, et la tête apprendrait à baisser le score d'un vrai chant. À mesurer en écoutant une
     vingtaine de ces négatifs ; `other_day` reste disponible. Les fenêtres des baselines
     incluent désormais celles des enregistrements positifs.

102. **Faux négatifs suspects : fenêtres négatives encadrées de positives** (idée de Léonard,
     miroir du « suspect » = positif isolé, faux positif probable). Critère commun
     (`sequential.surrounded_by_positives`) : un positif avant et un après, chacun à moins de
     `sequential.gap_radius_s` (6 s ≈ 4 notes). Usages : (1) jamais tirée comme négatif
     présumé ; (2) négatif annoté encadré de positifs annotés : colonne `suspect_fn` du jeu
     d'apprentissage, **il reste négatif** (n° 85 : l'écoute prime), à réécouter ; (3)
     persistance : descripteur `n_gaps` (trous dans une série positive) ; (4) sélection
     `gaps` : mode `scores` (fenêtres sous le seuil de la tête entre deux au-dessus : si
     A. blanci y chante, les positifs que le modèle rate, les plus utiles à annoter), mode
     `labels` (négatifs annotés entre deux positifs, réécoute). Source de label « gap ».

103. **Seuillage spectral et module séquentiel fusionnés** (demande de Léonard). Un seul
     module (`sequential.py` ; `prefilter.py` supprimé, `encoders/prefiltered.py` devient
     `encoders/upstream.py`), une seule section de config (`sequential` : `position`,
     `columns` — ex-`fusion.columns` —, `gap_radius_s`, `upstream` — ex-section `prefilter` —),
     un seul réglage de place : `position` dit où le module agit, `upstream.*.enabled` dit
     quelles fonctionnalités amont. Une seule détection des notes partout : les portes `notes`
     et `rhythm` se comptent sur les débuts de notes de l'enregistrement (`onset_counts`), à
     l'encodage comme dans la chaîne de décision et au banc d'essai ; `sequential.gate` est
     supprimé (la chaîne prend les portes de rythme activées, sinon `notes` à son seuil) ; les
     portes spectrales agissent à l'encodage (stock `+g-…`). `embed` n'applique l'amont que si
     `position` contient `upstream`, ou sur `--upstream a,b` (ex-`--prefilter`) ; banc
     d'essai : `blanci upstream-bench` (ex-`prefilter-bench`). Les anciennes clés
     (`prefilter`, `fusion.columns`) restent lues.

104. **`app/streamlit_app.py` supprimé** (décision de Léonard : désuet). Il écrivait les labels
     sans passer par la couche de service (ni validation, ni drapeaux d'écoute) ; le poste
     d'annotation est `blanci/app.py` (`blanci annotate`). La feuille de route (§13.2) le cite
     encore dans l'arborescence : document de référence, non modifié.

## 2026-09-25 — Notebooks d'exploration ; ce que contiennent les négatifs présumés

105. **Notebooks d'exploration** (demande de Léonard). `notebooks/01_explorer_une_fenetre`
     (un enregistrement et une fenêtre à travers la chaîne : écoute, grille, portes, module
     séquentiel en amont, en parallèle et en aval, négatif apparié, embeddings stockés ou
     calculés à la volée, prototype différentiel dans le son, sur la paire et sur le stock),
     `02_negatifs_apparies` (ce que tire chaque stratégie, feuille d'écoute et taux de
     contamination avec IC de Wilson), `03_module_sequentiel` (réglage de la détection des
     notes, valeurs et balayage des portes). Calculs dans `blanci/explore.py` (base ouverte en
     `mode=ro`, audio lu), graphiques dans `blanci/explore_plots.py` ; groupe `notebook`
     (ipykernel, matplotlib), installé avec `uv sync --inexact` pour garder research et app.
     Les négatifs d'une fenêtre sont tirés par `paired_negatives` (partie « même
     enregistrement » identique au benchmark, tirages au hasard possiblement différents).
     `detect_onsets` prend le lissage de l'enveloppe en paramètre (`smooth_s`, défaut 0,01 s
     inchangé). Sorties jamais committées : les lecteurs audio embarquent le son des
     enregistrements (`tests/test_notebooks.py`). Les trois notebooks tournent sur les
     données réelles (48 s, 3 min, 2 min).

106. **Mesure : « non annotée » ne veut pas dire « négative »** (25/09/2026, lecture seule,
     canal 0). 51 enregistrements positifs de 120 s : 3 fenêtres annotées en médiane (9 s sur
     120), aucune annotée négative. Les 345 positives sont toutes des détections BlanciNet
     validées ; 87 % des 1 695 tranches de 3 s non annotées de ces enregistrements sont aussi
     détectées (score médian 0,93). L'expert a validé une partie des détections, il n'a pas
     marqué tout le chant. `nearest` tire bien la fenêtre libre la plus proche d'une
     annotation, mais elle n'est pas écoutée : sur 1 020 négatifs `nearest`, 965 sont dans
     l'enregistrement positif et 80 % de ceux-là ont une détection BlanciNet ≥ 0,5 ; contraste
     en bande médian 6,1 dB (annotées 8,9 ; 7,6 à moins de 6 s d'une annotation, 5,4–5,9
     au-delà de 15 s). Les autres stratégies ne sont pas propres non plus : BlanciNet ≥ 0,5 sur
     46 % des négatifs `same_day` et 43 % des `other_day` (janvier à Mataroni : A. blanci
     chante aux mêmes heures d'un jour à l'autre). BlanciNet n'est pas la vérité (faux amis),
     l'écoute tranchera (`02_negatifs_apparies`, section 4). Options soumises à Léonard, non
     tranchées — le n° 87 a retiré toute règle liée à BlanciNet : (a) négatif présumé = ni
     annoté, ni détecté par BlanciNet au-dessus d'un seuil ; (b) jamais dans un enregistrement
     positif (« elle chante du début à la fin », notes de phénologie) ; (c) garder, et mesurer
     la contamination à l'écoute.

107. **Détection des notes trop stricte sur les données réelles** (mesure, réglage inchangé).
     Au réglage actuel (`onset_k_mad` 4, enveloppe lissée sur 10 ms), 0 à 14 notes par
     enregistrement positif de 2 min, 1 seule sur le plus annoté (36 fenêtres, chant tout du
     long) : le seuil médiane + 4 MAD (≈ +6 dB) morcelle les notes en éclats de 2 à 20 ms,
     rarement de 0,07–0,13 s. Sur 60 positives et 60 négatives (`03_module_sequentiel`),
     l'AUC du nombre de notes vaut 0,53 à ce réglage, 0,65–0,68 au mieux (k 1–2). Portes
     `notes` et `rhythm` : au seuil 1, 83 % des négatifs arrêtés mais 31 % des enregistrements
     positifs gardés — inutilisables en l'état ; `band_contrast` à 3 dB : 12 % arrêtés, 92 %
     gardés. Les descripteurs de rythme du module en parallèle sont presque toujours nuls.
     À reprendre avec Léonard (bande, durées, seuil) avant toute porte de rythme.

## 2026-09-25 (après-midi) — Régularisations des têtes

108. **Régularisations programmées, coupées par défaut** (tri de Léonard du 25/09 sur la liste
     R1–R84, `documentation/regularisation.md`). `blanci/regularization.py`, numéros conservés
     partout dans le code et la config. Une tête du benchmark les active dans son nom :
     `blanci heads --methods logistic,logistic+R18=16,logistic+R19` (« =v » remplace le réglage
     principal) ; le nom canonique (R triées) est celui des rapports et des scores hors-pli,
     `regularization.variants` en ajoute à la liste par défaut, `head.reference` les suit.
     Ordre : R19/R20 → R17 → R18 → R21 → tête. Programmées :
     - R13 (chaque micro pèse autant dans sa classe) et R15 (négatifs annotés × `hard_weight`,
       3 par défaut, face aux présumés — d'autant plus utile que les présumés sont contaminés,
       n° 106) : poids des fenêtres, de moyenne 1 par classe, `class_weight` inchangé ;
     - R17 (norme 1), R19 (− moyenne du micro), R20 (AdaBN : centrage et réduction par micro).
       R19/R20 lisent tout le stock du micro, partition par partition, sans labels
       (`store_domain_statistics`) : ce que la chaîne aura sur un nouveau site. Embedding par
       défaut seulement (pas les jetons résumés) ;
     - R18 (ACP) et R21 (retrait des directions du micro, sur les négatifs) ajustées dans
       chaque pli sur l'entraînement ; le C est ensuite choisi sur les fenêtres transformées.
       R21 `means` (défaut) retire les écarts entre moyennes des micros (≤ micros − 1
       directions, principe de LEACE) ; `inlp` itère un classifieur de micros, avec arrêt dès
       qu'il ne fait plus mieux que le hasard. Sur données simulées à micros très séparés, l'INLP
       retire presque toutes les dimensions, chant compris : d'où `means` par défaut ;
     - R27 (L1), R28 (Elastic Net) : `l1_ratio` de scikit-learn (`penalty` est déprécié depuis
       la 1.8 ; `penalty="elasticnet"` ajouté pour les versions antérieures), solveur saga ;
     - R22 : pooling `gem` des jetons (`logistic:gem`, p = 3 ; `gem2`, `gem5`… pour un autre p).
       Calculé sur x − min des jetons puis recentré (jetons non positifs) : p = 1 et p → ∞
       redonnent la moyenne et le maximum. p n'est pas appris : on compare quelques valeurs ;
     - R30 : tête `logistic_to_prototype`, perte ½‖w − w₀‖² + C·Σ perte logistique, w₀ = prototype
       différentiel dans l'espace standardisé, mis à l'échelle par une logistique à une variable
       (scipy L-BFGS). C petit : le prototype ; C grand : la logistique libre. Même grille de C ;
     - R31 : tête `lda_shrunk`, LDA à covariance rétrécie de Ledoit-Wolf (scikit-learn,
       `shrinkage="auto"`, qui rétrécit la matrice de corrélation), sans hyperparamètre.
     R30 et R31 rejoignent la courbe selon le nombre d'annotations (`head.curve.methods`). La
     courbe choisit désormais le C de chaque tête sur ses propres entrées (régularisations
     comprises), et non plus une fois pour toutes sur l'embedding par défaut.
     Combinaisons refusées : R19 + R20, R27 + R28, R21 + R19/R20 (n° 109), R19/R20 sur les
     jetons résumés, poids et pénalités hors des têtes logistiques. Hors du benchmark des têtes
     (`train`, `benchmark`, fusion, empilement), rien ne change : logistique L2 tant qu'aucune
     n'est retenue. Environnement : torch, bacpipe, streamlit, pytest et ruff manquaient le
     25/09 vers 11 h 40 (dates d'installation dans `.venv`) — signature d'un `uv sync` exact
     (le défaut de `uv sync`) lancé sans ces groupes ; `uv run`, lui, n'enlève jamais rien.
     Remis par `uv sync --inexact --group research --group app --group notebook`. Toujours
     `--inexact` et tous les groupes voulus.

109. **Mesure sur données simulées : la validation croisée sur un seul site récompense les
     raccourcis de micro** (`tests/test_regularization.py`). Corpus : micros « riches » (50 %
     de positifs) et « pauvres » (5 %) séparés par un axe de fond commun ; 4 micros d'un
     « nouveau site » inversent la relation. AP sur le nouveau site, 4 tirages : sans
     régularisation 0,25–0,50 ; R21 0,50–0,79 ; R19 0,55–0,73 ; R20 0,51–0,69. En validation
     croisée sur les micros d'entraînement, où le raccourci reste vrai, l'ordre s'inverse :
     sans régularisation 0,68–0,82, R21 0,45–0,69, R19 0,50–0,62. Conséquences :
     (a) R19, R20 et R21 ne se jugent pas sur les plis groupés de Mataroni, mais sur un site
     tenu à l'écart (`blanci evaluate --holdout tresor`, `heads-curve --by site`) dès que Trésor
     ou Kaw auront des positifs ; d'ici là, un recul en validation croisée n'est pas un verdict ;
     (b) R19 + R21 : 0,29–0,41 partout — centrés par micro, les négatifs ne diffèrent plus que
     par le chant qui fuit dans la moyenne des micros riches, et R21 efface l'espèce : refusé ;
     (c) R19 retire aussi du chant là où A. blanci occupe beaucoup de fenêtres (la moyenne du
     micro en contient) : à surveiller sur les micros à chœur.

110. **R85, sonde à portes** (« attentive sur l'embedding », idée de Léonard ; tête `gated`,
     `blanci/gated.py`). Score = w · (x̃ ⊙ g(x̃)) + b, porte g = σ(B·A·x̃ + c) de rang 8 : le
     poids de chaque dimension dépend de la fenêtre. Départ B = 0 → portes à ½, la tête part
     d'une logistique. AdamW (weight decay 1e-2, lr 0,05, 300 époques, lot entier), classes
     équilibrées ; torch (groupe research), écartée de la liste par défaut sans torch. Sur
     données simulées où le chant ne compte que selon le contexte : AP 0,84–0,87 sur un jeu
     d'essai contre 0,80 pour la logistique, mais AP ≈ 1 à l'entraînement et 0,58–0,81 en
     validation croisée pooled (logistique 0,73) : elle sur-apprend et l'échelle de ses scores
     varie d'un pli à l'autre. Ni réglée plus avant sur données jouets, ni arrêt précoce :
     à juger au benchmark réel ; R42 (arrêt précoce) serait le premier remède. Aucune
     régularisation par suffixe n'est refusée sauf les poids (R13/R15) et pénalités (R27/R28).

111. **Tableau des encodeurs de bacpipe** (`documentation/encodeurs-bacpipe.md`, demande de
     Léonard, R23) : pour les 8 encodeurs du projet et les 17 autres de bacpipe 1.3.5,
     framework, f_e, fenêtre, ce que bacpipe rend, accès aux jetons et aux couches
     intermédiaires. Relevé dans le code, sans exécuter les modèles (tailles de grille à
     mesurer). Bird-MAE : son `last_hidden_state` est la sortie *agrégée* (moyenne des
     patchs, jeton de classe exclu, puis `fc_norm`), malgré le nom — lu dans
     `modeling_bird_mae.py` ; une première version de ce numéro concluait à tort que bacpipe
     rendait ses jetons : le n° 77 était juste. Jetons par `output_hidden_states=True`
     (1 + 32 temps × 8 fréquences). BEATs et NatureBEATs : jetons par `avg_pooling = False`, sans hook ; BirdNET (Keras),
     ProtoCLR et ConvNeXt-BirdSet : faciles ; Perch v1 et v2 : couches intermédiaires
     difficiles (modèles exportés). Rien n'est encore branché dans l'adaptateur.

## 2026-09-25 (fin d'après-midi) — Pertes, échantillon versionné, poste d'annotation

112. **Benchmark des pertes** (R34, R35 ; `blanci/losses.py`, têtes `loss:<nom>`,
     `blanci heads --methods losses`). Même tête linéaire (standardisation, classes
     équilibrées, L2, C par validation groupée), sept pertes contre la logistique : hinge et
     squared_hinge (`LinearSVC`), least_squares (`RidgeClassifier`, α = 1/2C), focal (γ = 2),
     gce (q = 0,7), sce (α = 0,1, β = 1, A = −4), sigmoid (1 − p, bornée). Les quatre
     dernières par L-BFGS, départ depuis la logistique de même C (gce, sce, sigmoid ne sont pas
     convexes). Gradients vérifiés par différences finies ; focal à γ = 0 = logistique,
     gce à q = 1 = sigmoid. Poids R13/R15 acceptés ; pénalités R27/R28 non (L2 seulement).
     Pourquoi les pertes robustes : les négatifs présumés sont contaminés (n° 106), un positif
     caché parmi eux pèse sans limite dans la logistique, au plus 1 dans sigmoid.

113. **Banc d'essai sur l'échantillon versionné** (`echantillon/`, 66 clips de Léonard ;
     `blanci/echantillon.py`, `blanci echantillon`). Fenêtre de l'encodeur centrée sur la
     fenêtre étiquetée, canal `audio.channel`, cache `data/echantillon/<encodeur>.npz` ; plis
     groupés par micro (5 : un par micro positif) ; R13/R19/R20/R21 groupés par **site** (1 à 3
     clips par micro) ; diagnostic « le site se lit-il dans l'embedding d'un micro jamais vu ».
     Poids de perch_v2 : `hf_hub_download` restait bloqué à 0 octet (transfert xet) ; archive
     récupérée par `curl --http1.1 -C -` depuis le même dépôt, extraite dans
     `data/models/bacpipe/perch_v2` (bacpipe la reconnaît ensuite).
     **Mesure perch_v2, 25/09** (10 positifs de 5 micros, tous à Mataroni ; 55 négatifs
     annotés, 3 sites ; aucun négatif présumé, donc plus facile que le vrai benchmark). La
     plupart des têtes font AP 0,86–0,96, intervalles de ~0,8 à 1 : **aucun classement**.
     Écarts appariés à la logistique (AP 0,957), bootstrap 2 000 :
     - R19 + R21 : −0,54 [−0,74 ; −0,22] — premier appui réel au mécanisme du n° 109 (b),
       exagéré ici (positifs = 10 des 48 clips de Mataroni, bien plus que dans le stock) ;
     - R20 : −0,23 [−0,49 ; −0,05] — écart-type estimé sur 3 à 14 clips par site : artefact
       de l'échantillon, R20 ne se juge pas ici ; R19 seul : −0,09 [−0,23 ; 0] ;
     - cascade : −0,45 [−0,69 ; −0,14] — ses scores sont remis à l'échelle dans chaque pli
       (plancher + 1 + …) ; l'AP poolée sur les plis la pénalise peut-être : à vérifier avant
       de conclure ;
     - `logistic:max` : −0,25 [−0,42 ; 0,00] (gem −0,23, non significatif) : sur les jetons
       16 × 4 de perch_v2, le maximum fait moins bien que la moyenne ;
     - `loss:hinge` : −0,10 [−0,25 ; −0,01] ;
     - R21 seul : scores changés (corrélation de rang 0,89), ordre positifs/négatifs identique :
       même AP.
     Site : exactitude 0,76 contre 0,69 au hasard (38 Mataroni, 14 CDR, 3 Patawa) : pas de
     conclusion à cet effectif. Ce qui est acquis : toute la chaîne (encodage, jetons, têtes,
     régularisations, pertes) tourne sur le vrai son.

114. **Poste d'annotation et R19 + R21** (demandes de Léonard). Streamlit reste : c'est le
     poste d'annotation (`blanci annotate`, `blanci/app.py`), pas l'ancien
     `app/streamlit_app.py` supprimé au n° 104. Les 4 échecs de `tests/test_app.py` venaient
     de Streamlit 1.64, qui résout un chemin relatif depuis le fichier de test : chemin absolu.
     Le refus de R19/R20 + R21 (n° 108–109) est levé : la combinaison se mesure au lieu d'être
     interdite ; le mécanisme du n° 109 (b) reste l'hypothèse à vérifier.

## 2026-09-26 — Tri de Léonard sur R33–R49 ; tableaux des benchmarks en images

115. **R39, voisins : k et pondération** (`head.nearest_similarity`, têtes `knn:k=…`,
     `exemplar:k=…`, suffixe `:w`). k > 1 : moyenne des cosinus aux k plus proches ; `:w` :
     moyenne pondérée par 1 / distance (distance euclidienne entre vecteurs de norme 1,
     `weights="distance"` de scikit-learn), entre k = 1 et la moyenne. Pour `knn`, les k
     voisins sont pris dans chaque classe séparément (un vote toutes classes confondues serait
     écrasé par les ~20 négatifs par positif). `knn:k=3` et `knn:k=5` rejoignent la liste par
     défaut de `blanci heads` ; `--methods neighbors` lance toutes les têtes par similarité
     (avec logistique et prototype) ; la courbe selon le nombre d'annotations ajoute
     `exemplar:k=3`, `knn`, `knn:k=3` (Léonard : ces baselines serviront à amorcer les
     nouveaux sites). Sur données simulées, avec six négatifs étiquetés positifs, k = 5 classe
     mieux que k = 1 (le voisinage d'un faux positif ne l'hérite plus). R19 se combine :
     `knn:k=5+R19`.

116. **R36 (poids des classes), R37 (biais par micro) ; R33 écartée, R38 en attente.**
     - R33 (norme maximale) : écartée, équivalente à la L2 pour une tête linéaire.
     - R36 : les têtes gardent `class_weight="balanced"` (Léonard : le déséquilibre doit être
       pris en compte). `+R36=power` remplace le poids n / (2·n_classe) de chaque classe par
       sa puissance `power` : 0 = aucun rééquilibrage (défaut de la variante, pour mesurer ce
       que l'équilibrage apporte), 0,5 = entre les deux. Passe par les poids des fenêtres :
       logistic, loss:<nom>, logistic_to_prototype, cascade ; le C est rechoisi avec.
     - R37 : un biais par **point** (site/micro, `regularization.by`), à la demande de
       Léonard (un biais par site ne suffirait pas). Un micro déplacé sur un autre site change
       de point, donc de biais : c'est ce qu'on veut pour l'ambiance du lieu. Colonnes
       indicatrices non standardisées, a priori N(0, σ²) sur chaque biais, σ = `scale` en
       logit, indépendant de C (colonne × σ/√C). Micro absent de l'entraînement : biais commun.
       logistic, loss:<nom>, cascade. **Mesure, données simulées du n° 109** (fond du micro
       qui prédit la présence, nouveau site où il ne tient plus), AP sur le nouveau site,
       3 tirages : sans R37 0,25–0,37 ; σ = 0,3 : 0,24–0,42 ; σ = 1 : 0,42–0,57 ; σ = 3 :
       0,56–0,68 ; σ = 10 : 0,61–0,71 ; R21 : 0,50–0,62. Contrairement à la liste du 25/09
       (« biais très pénalisé »), un biais fortement rétréci n'y change rien : w garde le
       raccourci. Défaut σ = 3, provisoire (n° 119).
     - R38 (écarts de w par micro) : non programmée. ~100 micros × 1 536 dimensions ; à
       reprendre après une ACP (R18), avec des effets croisés micro + site si l'on veut
       séparer le matériel (qui suit le micro) du lieu.

117. **Régularisations de l'attentive (R40–R42, R45–R47) et de la sonde à portes (R40, R42,
     R46)** (`attentive.fit_with_options`, `attentive.optimise`, partagé avec `gated`).
     Comportement par défaut inchangé (sorties identiques au bit près, vérifié). R40 : weight
     decay choisi sur `grid` par validation groupée interne (AP). R41 : AdamW, weight decay
     0,01. R45 : chaque jeton masqué avec la probabilité p avant l'attention (au moins un
     gardé). R46 : dropout de z (attentive) ou de x̃ ⊙ g (gated). R47 : w part de la
     logistique sur la moyenne des jetons (C choisi sur la même grille) et ½λ‖w − w₀‖²
     remplace le weight decay sur w. R42 : **mesure** — avec un seul pli de validation (1 à 2
     micros), la perte de validation est au plus bas à l'époque 6, puis à l'époque 1 en
     moyennant les plis, alors que l'AP de test progresse jusqu'à ~200 époques : la perte
     monte dès que la tête devient trop sûre d'elle, même quand son classement s'améliore.
     R42 relève donc la courbe du critère à chaque époque dans chaque pli groupé interne,
     retient l'époque qui optimise la courbe moyenne, puis réentraîne sur tout le pli
     d'entraînement ; critère `monitor: ap` par défaut (`loss` possible). AP hors-pli sur
     données simulées (note dans 1 jeton sur 16, 3 tirages) : sans régularisation 0,67 /
     0,74 / 0,79 ; R42 (loss) 0,54 / 0,65 / 0,78 ; R42 (ap) 0,66 / 0,74 / 0,79 ; R41, R45,
     R47 à ±0,03 de la tête seule ; R46 (p = 0,2) 0,63 / 0,69 / 0,74. Ces données ne font
     pas sur-apprendre la tête : à juger au benchmark réel. R43, R44, R48, R49 : expliquées à
     Léonard, non programmées.

118. **Tableaux des benchmarks en images** (`documentation/tableaux/`, demande de Léonard :
     les tableaux Markdown s'affichent mal dans Xcode). Un PNG par benchmark (protocole,
     encodeurs, têtes, poolings, pertes, voisins, régularisations, fusion, seuillage en
     amont, négatifs appariés, baselines, ensembles), ~600 Ko en tout, générés par
     `generer.py` (données + rendu Pillow, polices de matplotlib : rien à télécharger).
     Un nouveau benchmark = une entrée de `TABLEAUX`. `encodeurs-bacpipe.md` : tableaux
     remplacés par des fiches lisibles en texte brut. Le diagnostic de la cascade (n° 113)
     est abandonné (Léonard). Tests lancés depuis un conteneur en root :
     `test_freeze_writes_a_read_only_version_and_never_overwrites` y échoue (root écrit dans
     un fichier en lecture seule), sans rapport avec le code.

119. **Aucune conclusion sur les régularisations avant la base complète** (Léonard, 26/09). La
     base complète n'est pas accessible aujourd'hui. Les mesures des n° 109, 113, 116 et 117
     (données simulées, échantillon de 66 clips) vérifient que le code fait ce qu'il doit ;
     elles ne classent aucune régularisation. Les réglages par défaut qui en découlent sont
     provisoires et se choisiront sur la base : σ de R37 (comparer `logistic+R37=0.3`, `=1`,
     `=3`, `=10`) et le critère de R42, désormais réglage principal de son suffixe
     (`attentive+R42=ap` et `attentive+R42=loss` dans un même run, scores hors-pli
     distincts). Même prudence pour les avis théoriques donnés en discussion (R44 redondante
     avec le weight decay sur q, par exemple) : à vérifier, pas à appliquer.

120. **R50 : C de la fusion logistique par validation groupée ; index des régularisations.**
     Méthode de fusion `logistic+R50` (`fusion.choose_fusion_C`) : le C est choisi sur
     `fusion.C_grid` par l'AP moyenne sur des micros tenus à l'écart, dans chaque pli (sur les
     seuls micros d'entraînement du pli) et, pour le modèle de production, sur tout le jeu de
     développement ; le C retenu est enregistré avec le modèle. `logistic` garde son C fixé
     (`fusion.C`, 1) : `fusion-bench` compare les deux (`benchmark_methods`), la production
     reste `fusion.method: logistic` tant que rien n'est tranché (n° 119). Organisation
     (question de Léonard) : `blanci/regularization.py` est l'index de toutes les
     régularisations programmées (tableau « où, comment l'activer ») et contient celles qui
     transforment les entrées, les poids ou la pénalité (R13–R21, R27, R28, R36, R37) ainsi
     que R40/R42, qui entourent l'entraînement des têtes torch (`fit_with_options`, déplacé
     depuis `attentive.py`, sorties identiques). Restent là où elles s'appliquent : les options
     de la boucle d'entraînement (R41, R45–R47 dans `attentive.py`, R46 dans `gated.py`), les
     têtes (R22 `pooling.py`, R30/R31/R39 `head.py`, R34/R35 `losses.py`, R85 `gated.py`) et
     la fusion (R50 `fusion.py`, R57 `stacking.py`). R51–R56 et R58 : expliquées à Léonard,
     en discussion.

121. **La mécanique des régularisations regroupée dans `blanci/regularization.py` ; R59, R62,
     R63 ; tri de la section F** (demande de Léonard : ne pas se perdre dans un projet qui
     grandit). Le module contient désormais la mécanique de toutes les régularisations qui en
     ont une, et les autres modules l'appellent là où elle prend effet :
     - `grouped_search` : un réglage choisi par validation groupée, commun au C des têtes (R26,
       `head.select_C`), au weight decay des têtes torch (R40) et au C de la fusion (R50,
       `choose_fusion_C`, déplacée depuis `fusion.py`) — trois copies du même calcul en une ;
     - la boucle d'entraînement torch `optimise` (R41 ; R42 avec `validation_criterion` ; R59),
       `keep_mask` (R45), `dropout` (R46), `logistic_start` (R47), déplacées depuis
       `attentive.py` et `gated.py` ;
     - `nearest_similarity` (R39, depuis `head.py`), `group_bias_scale` (R37, σ/√C, auparavant
       écrit deux fois dans `head.py` et `losses.py`).
     Sorties identiques au bit près, vérifiées sur 12 têtes et variantes (logistique, R30,
     focal, R36, R37, knn:k=5:w, exemplar:k=3, gated+R40+R42, attentive seule et régularisée,
     fusion logistic+R50). Restent dans leur module, avec une ligne de l'index : les têtes qui
     *sont* la régularisation (R22 `pooling.gem`, R30, R31, R34/R35, R85), la L2 elle-même
     (R26, `fit_logistic`) et le hors-pli de la fusion (R57).
     Section F (tri du 26/09) : R59 programmée pour l'attentive et la sonde à portes
     (`+R59` : warm-up linéaire du pas sur `warmup` époques, norme du gradient écrêtée à
     `clip_norm`) — le reste de R59 (dropout, weight decay, arrêt précoce) l'était déjà ; R60 :
     le rang du LoRA est réglable dans la config ; R62 : `l2_sp_penalty`, utilisée dès
     aujourd'hui par R47 (même calcul, vers la logistique) et prête pour le fine-tuning ;
     R63 : `distillation_loss` (labels souples de l'enseignant, température), prête pour le
     détecteur distillé ; R65 accordée, à faire avec le modèle maison et le LoRA. R61, R64,
     R66 expliquées, en discussion ; R67 à explorer. Les emplacements réservés (`finetune.py`,
     `detectors/distilled.py`, `detectors/homemade.py`) nomment les fonctions à appeler.

122. **Sélection des réglages : R74, R75, R76, R77, R79** (tri de Léonard du 26/09, section H).
     Léonard : ne pas supposer 51 positifs pour toujours — d'autres annotations viendront, mais
     des sites resteront peu ou pas annotés ; tous les cas doivent tourner.
     - R75 (règle du « 1 écart-type ») **par défaut** (`head.selection_rule: one_se`, `best`
       pour l'ancien comportement) : parmi les valeurs dont l'AP moyenne des plis internes est à
       moins d'une erreur type de la meilleure, la plus régularisante (plus petit C ; plus grand
       weight decay). Vaut pour le C des têtes (R26), le weight decay (R40) et le C de la fusion
       (R50) : un seul calcul, `grouped_search`. Les C retenus changent donc par rapport aux
       benchmarks d'avant ce numéro.
     - R76 : `+R76` remplace la grille de C d'une tête par 13 valeurs log (facteur ~2,2 au lieu
       de 10) ; à réserver aux têtes rapides. `blanci train` écrit le chemin de régularisation
       (`chemin_C_<encodeur>_<version>.csv`, et `.png` si matplotlib est là).
     - R77 : `head.n_splits: lomo`, un pli par micro positif, les micros sans positif répartis
       entre ces plis ; vaut aussi pour les plis internes.
     - R74 : `blanci train` rapporte, à côté du rappel au seuil de précision plancher, le rappel
       et la précision d'un seuil choisi pli par pli sur les autres plis ; `blanci heads`
       rapporte l'AP par pli de la procédure « garder la meilleure variante » choisie sans voir
       le pli jugé, contre celle de la gagnante du tableau (R80 : ce que la gagnante doit à la
       chance). Le C, lui, était déjà choisi dans chaque pli sur les seuls micros
       d'entraînement : la note de la liste sur R74 était une inquiétude, pas un constat.
     - R79 : `+R79` (logistic, loss:<nom>, logistic_to_prototype), 20 têtes apprises sur des
       tirages bootstrap des **micros**, scores moyennés.

123. **Réseaux : R60, R61, R62, R63, R64** (section F). R64 : `+R64=ema` ou `+R64=swa` pour
     attentive, gated et dann (`WeightAverage`) ; la validation de R42 juge les poids moyens.
     R61 (`layerwise_lr_groups`, `unfreezing_schedule`, `unfreeze_top`) et R62 complet
     (`snapshot`, `l2_sp_model_penalty` : α vers les poids pré-entraînés, β vers 0 pour les
     poids nouveaux) : outils prêts et testés sur un petit réseau, réglages dans `finetune`,
     appelés par `finetune.py` quand il sera écrit. R60 : LoRA sur les couches hautes seulement
     (`finetune.lora.layers: 4`), spécifié dans `finetune.py`. R63 : la perte existe
     (`distillation_loss`) ; le détecteur distillé lui-même reste prévu après M4 (§13.7). R59 :
     l'écrêtage du gradient était déjà dans `optimise`, avec le warm-up.

124. **Nouvelles têtes : R66 (`dann`), R67 (`multiclass`), R81 (`+R81`).**
     - R66, `blanci/dann.py` : h = tanh(A·x̃ + a), score = w·h + b ; un classifieur de micro
       branché sur h à travers une inversion du gradient (`grad_reverse`), dont la force monte
       de 0 à `strength` (`dann_strength`). Le micro n'est appris que sur les négatifs par
       défaut (`domain_on`), comme R21. R40, R42, R46, R59, R64 par suffixe. Données simulées du
       n° 109 (vérification du mécanisme, pas un verdict) : la part de micros reconnue dans h
       par une logistique neuve passe de 0,49–0,55 sans adversaire à 0,41–0,46 avec (hidden 8,
       600 époques) ; l'AP sur le nouveau site n'y dépasse pas celle de R21.
     - R67, tête `multiclass` (logistique multinomiale, classes équilibrées ; score = logit de
       P(A. blanci)). Choix laissés à Claude par Léonard : négatifs présumés → classe `fond`
       (aucun événement noté) ; classes de moins de 10 fenêtres (`R67.min_count`) → `autre`,
       puis `fond` si `autre` reste sous le seuil. Classes : `CLASS_OF_LABEL` (amphibien —
       congénères compris —, orthoptère, oiseau, pluie, fond, artefact, autre). Données
       simulées avec un faux ami : 0,83 d'AP contre 0,74 pour la logistique binaire.
     - R81, `+R81` : réservoir de 20 000 fenêtres non annotées tirées du stock (hors fenêtres
       du benchmark), des seuls micros d'entraînement par défaut (`pool_from: all` : aussi
       ceux du pli jugé, adaptation sans labels) ; pseudo-positifs = logit ≥ 3, au plus 1 % du
       réservoir, au poids 0,3 ; pas de pseudo-négatifs (même contamination que les négatifs
       présumés, n° 106).

125. **Tri des sections G et I ; R78 sur AnuraSet.** Écartées : R68 et R69 (Léonard compte se
     servir des fenêtres négatives prises entre des positives — les faux négatifs suspects du
     n° 102 — pour du « positive mining » ; lisser les scores effacerait justement ces
     contrastes), R71, R72 (les campagnes de 7 jours suffisent, d'après la phénologie de
     Courtois et al., à distinguer absence et présence non détectée), R82. R70 : le maximum,
     déjà le défaut de l'évaluation ; le travail porte sur les fenêtres. R80 : présélection des
     variantes plutôt que tout tester (et R74 pour mesurer la part de chance). R83 : à faire
     plus tard. R84 : en place (20 % aléatoire stratifié par micro et heure). R73 : en
     discussion (origine du plancher 0,1 et correction de la prévalence, voir la réponse du
     26/09). R78 : `blanci anuraset-heads` (config `config/anuraset.yaml`) — toutes les têtes
     sans jetons, un pli par site, réglages choisis par des plis internes eux aussi par site,
     régularisations groupées par site ; AP de chaque tête sur chaque site tenu à l'écart.
     Section E (R51–R56, R58) : en attente de Léonard.

## 2026-09-27 — Section E, R43, seuil du stock, site contre micro

126. **R43 et jugement par type de positif.** `attentive+R43=β` : entropie moyenne des poids
     d'attention ajoutée à la perte × β (`attention_entropy`) ; β > 0 pousse vers une attention
     piquée (une note brève, un ou deux jetons), β < 0 vers une attention diffuse (chœur).
     Léonard : comparer les approches sur les chœurs et sur les mâles seuls. `blanci heads`
     rapporte donc l'AP de chaque tête par label positif (`blanci_solo`, `blanci_chorus`,
     `blanci`), ses positifs de ce type contre tous les négatifs (`by_positive_type`). Les 345
     positifs importés sont tous « blanci » (H21, n° 3) : la comparaison vivra avec les
     prochaines annotations. R48 non programmée : dans l'attentive, query et weight n'ont que
     2 × d paramètres, rien à factoriser ; le rang faible sert déjà dans R85.

127. **Fusion : R52, R53, R56 ; R51, R54, R55, R58.** Variantes de la fusion logistique,
     combinables avec R50 : `+R52` (coefficients ≥ 0 sur les entrées orientées : pour classer,
     une combinaison convexe à l'échelle près), `+R53` (pénalité L1 : les entrées inutiles
     tombent à 0, exactement avec R52), `+R56` (s = w_tête·z_tête + cap·tanh(Σ autres / cap) :
     les descripteurs déplacent le logit d'au plus `fusion.R56.cap` = 2, aucun veto sur la
     tête). Mécanique : `regularization.fit_constrained_logistic`, `fusion_variant`.
     `fusion-bench` compare `logistic+R50`, `+R52`, `+R53`, `+R56`. R51 : remplacée par R52
     (Léonard) — R52 prend l'orientation de chaque entrée sur les données (signe de sa
     corrélation au label à l'entraînement), R51 l'aurait imposée ; sur un site encore peu
     annoté, un signe imposé est plus sûr qu'un signe appris : à garder en tête. R53 : Léonard
     fera un tri des descripteurs (redondances) ; la sélection se fait dans chaque pli. R54,
     R55 : réexpliquées, en discussion. R58 : écartée. `weighted` (poids fixés à la main,
     `fusion.weights`) reste la voie pour pondérer soi-même les entrées.

128. **R73 : le seuil pour la précision du stock.** La précision plancher (0,1) vient du §6 :
     la file hebdomadaire doit tenir en 1 h à 10 s par candidat, et le rappel prime (une fausse
     absence est irréversible, une fausse présence se corrige à l'écoute). Mais la précision
     dépend de la part de positifs : ~1/21 dans le benchmark, bien moins dans le stock. Démarche
     retenue, en chaîne : (1) `blanci prevalence` mesure la part de fenêtres positives sur les
     seules fenêtres écoutées au hasard (sources `random`, `audit`), avec son intervalle de
     Wilson ; (2) reportée dans `decision.prevalence`, `blanci train` choisit le seuil au plus
     grand rappel dont la précision attendue dans le stock, π·TPR / (π·TPR + (1 − π)·FPR),
     atteint le plancher (`threshold_at_prevalence`) — TPR et FPR ne dépendent pas de la part
     de positifs ; (3) `blanci train` écrit la courbe seuil → rappel, précision du benchmark,
     précision attendue dans le stock, part des fenêtres signalées (`seuils_<encodeur>_<v>`,
     CSV et PNG) et range la calibration de Platt (a, b) et la prévalence du benchmark avec
     la tête, pour des probabilités ramenées au stock (`prior_shift`). Sans prévalence, rien
     ne change. Hypothèse à garder en vue : les négatifs du benchmark (mêmes micros, mêmes
     heures que les positifs) représentent-ils ceux du stock ? La strate aléatoire le dira.

129. **Site contre micro.** Question de Léonard : la différence entre micros d'un même site
     n'est-elle pas déjà aussi forte qu'entre sites ? `blanci cluster --mode c0` rapporte
     désormais la décomposition de la variance des embeddings (normés) entre site, micro dans
     son site et intérieur d'un micro (`cluster.variance_partition`, ANOVA emboîtée, parts de
     somme 1), et l'AMI groupes/sites à côté de l'AMI groupes/micros. À lancer sur un
     échantillon qui couvre plusieurs sites.


130. **Fusion : R54 et R55, des descripteurs courbés.** Accord de Léonard (27/09) ; le tri
     des descripteurs viendra après. Les deux ne touchent que les descripteurs du module
     séquentiel (`regularization.is_descriptor`) : le score de la tête et les autres sources
     restent des droites. Chaque descripteur standardisé devient un bloc de colonnes
     (`DescriptorBasis`), la fusion reste une logistique pénalisée
     (`fit_constrained_logistic`, appelée par `fit_fusion_logistic`).
     - `logistic+R54` : `fusion.R54.bins` = 5 classes de même effectif, soit 4 marches aux
       quantiles de l'entraînement ; la hauteur de chaque marche est un coefficient ≥ 0 sur
       le descripteur orienté (signe de sa corrélation au label, comme R52) : la courbe ne
       fait que monter dans le sens « plus A. blanci ». Robuste aux descripteurs asymétriques
       (une longue queue ne tire plus la droite) ; incapable d'une forme en U, par
       construction.
     - `logistic+R55` : modèle additif (GAM) par P-splines (Eilers et Marx 1996, pas de
       dépendance : scipy suffit, mgcv est en R et pygam serait à télécharger). B-splines
       cubiques sur `fusion.R55.segments` = 8 intervalles égaux entre les quantiles 1 % et
       99 % ; au-delà la courbe reste plate : pas d'extrapolation sur un site aux valeurs
       jamais vues. Pénalité ½ βᵀ(G + `smoothness` · DᵀD)β : G = BᵀB / n mesure la taille de
       la courbe (pour une droite, la même L2 qu'une entrée linéaire : quand C baisse, tout
       rétrécit ensemble, rien ne s'échappe) ; DᵀD, différences secondes des coefficients,
       mesure la courbure (`smoothness` grand → une droite). `smoothness` = 1 est provisoire.
     C par validation groupée avec R50 (`logistic+R50+R54`, `+R50+R55`, ajoutées à
     `fusion.benchmark_methods`) ; « =v » remplace le réglage principal (`R54=3`, `R55=10`,
     et désormais `R56=1` pour le plafond, `FUSION_PARAMETER`). Combinables avec R52
     (entrées linéaires ≥ 0) et R56 (plafond sur la somme des courbes, centrées) ; R54 et R55
     s'excluent, R53 et R55 aussi (la L1 ne lisse pas une courbe). La « part » d'une entrée
     (`FusionModel.weights`) devient l'amplitude de sa courbe de −2 à +2 écarts-types. Plus
     de coefficients par descripteur (5 à 11 au lieu de 1) : c'est la pénalité, pas le nombre
     de colonnes, qui fixe ce que la fusion peut apprendre de ~10 positifs par degré de
     liberté (n° 94). Données simulées (un descripteur à seuil, un en U) : AP hors-pli 0,60
     (logistique), 0,60 (R54), 0,76 (R55) — ce que la mécanique sait faire, pas un verdict.
     Les ajustements contraints limitent BLAS à un fil (`threadpoolctl`, déjà là avec
     scikit-learn) : sur ces petites matrices, le multi-fil coûtait jusqu'à 100×.

131. **R37 en GLMM.** Accord de Léonard (27/09), pour le programme, sans usage immédiat. R37
     est déjà l'effet aléatoire « micro » d'un GLMM (biais b_micro ~ N(0, σ²), n° 116) ; il
     lui manquait ce qui fait un GLMM : σ estimé sur les données. `logistic+R37=glmm` (ou
     `R37.scale: glmm`) choisit σ parmi `R37.glmm_grid` en maximisant la vraisemblance
     marginale approchée (Laplace sur les biais, poids de la tête à leur valeur ajustée) :
     − perte − ½‖w‖²/C − Σ b̂²/2σ² − ½ log det(I + Σ H), H la courbure de la perte le long des
     biais (`glmm_evidence`, `glmm_scales`). σ grand laisse chaque micro coller à ses données
     mais se paie en log det ; σ petit coûte en ajustement si les micros diffèrent vraiment.
     Sur données simulées, σ vrai 0 → 0,3 retenu ; 0,5 → 1 ; 1,5 → 1 ; 3 → 3. Mais sur le
     corpus « raccourci » du n° 116 (le fond du micro prédit la présence, pas sur le nouveau
     site), le GLMM retient σ = 1 et fait AP 0,42–0,58 sur le nouveau site, contre 0,58–0,69
     pour σ = 3 et 0,21–0,37 sans R37 : le GLMM choisit le σ qui **décrit** le mieux les
     micros de l'entraînement, pas celui qui **protège** le mieux un site nouveau (des biais
     plus libres absorbent davantage le niveau du micro, w en garde moins). Les deux se
     comparent dans le benchmark (`logistic+R37=glmm` contre `=3`), jugés sur un site tenu à
     l'écart (n° 109). Niveau site en
     option (`R37.site_scale`, σ fixé ou « glmm ») : biais de site + biais de micro dans son
     site, emboîtés ; un micro nouveau d'un site connu hérite du biais de son site, un site
     nouveau du biais commun. Logistique et cascade seulement (pas `loss:`). Coût : un
     ajustement par valeur essayée (5 par défaut, jusqu'à 20 avec deux niveaux). Ce que le
     GLMM lisse vers la moyenne : les biais de la tête (le niveau de score propre à un micro,
     estimé sur ses fenêtres **annotées**), pas une probabilité de présence par point
     d'écoute — ce dernier usage (occupation, abondance) est celui de R72, écartée : le projet
     cherche des sites, une détection attestée suffit à déclencher la visite.

132. **Graphe de l'ACP : `blanci pca`.** Demande de Léonard : la part d'information perdue
     selon le nombre de dimensions gardées. Sur un échantillon du stock de l'encodeur
     (`cluster.c0_sample`, fenêtres arrêtées exclues), `cluster.pca_information_curve` donne,
     pour k = 1 … d, la part de variance des embeddings que perdent les k premières
     composantes : telles quelles (R18 seule) et après centrage par micro (R19 puis R18),
     chacune en % de sa propre variance totale, avec la part que le centrage retire (les
     différences entre micros). CSV et PNG `rapports/acp_<encodeur>` (abscisse logarithmique,
     repères aux k de R18 — 16, 32, 64, 128 — et du clustering, `cluster.pca_components`) ;
     la commande affiche combien de composantes gardent 80, 90, 95, 99 % et ce que perdent
     16, 32, 64, 128. Variance n'est pas information utile : la note occupe 2–3 % d'une
     fenêtre, une direction de faible variance peut porter le chant. La courbe dit combien de
     dimensions décrivent le paysage sonore ; ce qu'elles valent pour A. blanci se lit dans
     le benchmark (`logistic+R18=16`, `=32`, `=64`).

133. **L'AP poolée pénalise les réglages choisis pli par pli (mécanisme confirmé).** Le n° 113
     le soupçonnait pour la cascade. Sur données simulées, `logistic+R50` (fusion) : AP poolée
     0,48 contre 0,60 pour `logistic` (C fixé), mais AP moyenne par pli 0,59 contre 0,61.
     Cause : les plis qui retiennent un C petit rendent des logits resserrés (écart-type 0,6
     au lieu de 2,2) ; mis bout à bout, les scores de plis différents ne sont plus sur la
     même échelle et le classement commun se dégrade, sans que le classement dans chaque pli
     change. Même effet possible pour toute tête dont C (R26), σ (R37=glmm), les époques
     (R42) ou le nombre de composantes varient d'un pli à l'autre. À garder en tête en
     lisant `heads` et `fusion-bench` : un écart d'AP poolée entre une méthode à réglage fixe
     et une méthode à réglage choisi n'est pas forcément un écart de classement. Remèdes :
     l'AP moyenne par pli, et la recalibration de chaque pli avant la mise bout à bout, tous
     deux programmés au n° 135. Le seuil (R74, sur les scores hors-pli des autres plis) subit
     le même effet.

134. **`regularization.py` devient un paquet.** À ~2 100 lignes, découpé comme Léonard
     l'avait demandé (« si c'est trop long, des sous-modules ») : `blanci/regularization/`
     avec `names` (noms des têtes, validation), `windows` (R13, R15, R17–R21, R36,
     indicatrices R37), `assembly` (`Context`, `Regularizer`, `regularizer_for`), `selection`
     (R26, R40, R74–R76, R79, R81), `torch_training` (R40–R47, R59, R61–R64, R66), `glmm`
     (R37), `fusion_logistic` (R50, R52–R56), `decision` (R73), `classes` (R67), `neighbors`
     (R39). Le code est déplacé tel quel (découpage par sections, imports entre sous-modules
     calculés, aucun cycle) ; `__init__.py` garde l'index et réexporte tout :
     `from blanci.regularization import …` ne change nulle part. Suite de tests identique
     avant et après (793 réussis).

135. **AP moyenne par pli et recalibration par pli (remèdes du n° 133).** Accord de Léonard
     (28/09), sans conclusion avant les grands jeux (base complète, AnuraSet).
     (a) Toujours là : `ap_fold_mean`, l'AP de chaque pli (ses scores entre eux) moyennée
     sur les plis à deux classes (`evaluate.fold_mean_ap`, option `folds` de `evaluate`),
     dans `heads`, `fusion-bench`, `anuraset-heads`, le benchmark des encodeurs et
     AnuraSet. Elle ne met jamais bout à bout deux modèles : elle ne voit que le classement.
     Plus bruitée (peu de positifs par pli) : elle se lit à côté de l'AP poolée. Un grand
     écart entre les deux signale des plis sur des échelles différentes.
     (b) Option `benchmark.fold_calibration: platt` (défaut `none`) : chaque pli choisit son C
     une fois, puis 3 plis internes (par micro, `calibration_splits`) donnent, **avec ce même
     C**, des scores hors-pli de son entraînement ; une calibration de Platt à classes
     équilibrées apprise dessus (`regularization.fold_platt`) ramène les scores du pli sur une
     échelle commune (la cote « A. blanci contre fond », indépendante de la part de positifs
     du pli) avant la mise bout à bout. Les réglages restent choisis pli par pli : seule
     l'échelle change, jamais le classement dans un pli (pente > 0). Une Platt apprise sur
     les scores d'entraînement du pli eux-mêmes ne marche pas ici : avec plus de dimensions
     que de fenêtres, l'entraînement est séparable et la pente part à l'infini. Non
     recalibrées : les têtes par similarité (rien n'y est choisi pli par pli) et les règles
     de fusion fixes. `ap_raw` garde l'AP poolée d'avant ; les scores rangés (stock hors-pli,
     entrées de la fusion) sont les recalibrés. Coût : 1 + 3 entraînements par pli.
     Mesuré sur données simulées : fusion `logistic+R50`, AP poolée 0,48 → 0,58 (par pli
     0,59) ; `logistic` à C fixé 0,601 → 0,601 ; `logistic+R50+R55` 0,756 → 0,761 ; têtes
     logistiques (C par pli, 3 tirages) 0,189 → 0,223, 0,175 → 0,173, 0,139 → 0,129 ; une
     première version qui recalibrait aussi le prototype le faisait perdre (0,243 → 0,218),
     d'où l'exclusion des têtes par similarité. Lecture : la recalibration répare un vrai
     écart d'échelle, et ne coûte, ailleurs, que le bruit de sa propre estimation. `blanci
     train` n'est pas touché (seuil sur les scores hors-pli bruts) : à reprendre si l'option
     est retenue.

136. **Campagne AnuraSet d'un seul tenant : `blanci anuraset-campaign`.** Demande de Léonard
     (28/09) : choisir les espèces, encoder, benchmarker, essayer les régularisations. Dans
     cet environnement cloud, l'accès réseau est restreint : zenodo.org (les données
     AnuraSet), huggingface.co et kaggle.com (les poids des encodeurs) sont refusés par le
     proxy ; seuls PyPI et GitHub passent. Aucune donnée AnuraSet n'a donc pu être
     téléchargée ici, et aucun résultat réel n'existe encore. Tout est prêt pour qu'une seule
     commande fasse la campagne dès que les données sont là (cloud avec zenodo.org et
     huggingface.co autorisés, ou un poste bien connecté) :
         uv run blanci --config config/anuraset.yaml anuraset-campaign --encoders perch_v2
     Étapes, reprenables : extraction et inventaire (`prepare`) ; profil des espèces et
     fréquence dominante ; choix des espèces (`choose_species`, sauf `anuraset.species` ou
     `--species`) par niveaux : d'abord note ≤ 0,3 s en 3–6 kHz avec ≥ 300 chants (proche
     d'A. blanci), puis élargi (≤ 0,5 s, 2–7 kHz, ≥ 150 chants), puis multi-sites (≤ 1 s,
     ≥ 100 chants) ; toujours ≥ 2 sites, classées par nombre de sites puis de chants ;
     embeddings de chaque encodeur ; sondes du §3 par encodeur et espèce, encodeurs classés
     par l'AP moyenne de la logistique (`rank_encoders`) ; sur le meilleur, les têtes de
     `CAMPAIGN_HEADS`, par priorité : logistique (référence) ; prototype et knn:k=5
     (amorcer avec quelques exemples) ; R19, R20 (fond du site retiré par le stock non
     annoté : l'hypothèse de Léonard pour amorcer un site) ; R21 et dann (directions du site
     effacées) ; R37, R37=glmm, R19+R37 (biais par site) ; R13, loss:focal (déséquilibre) ;
     R18=64 (l'ACP jette-t-elle le chant ?) ; lda_shrunk. Toutes jugées un site à la fois ;
     classement par rang moyen sur les espèces (`rank_heads`), AP poolée et AP moyenne par
     site, écarts appariés à la logistique, AP de chaque site tenu à l'écart.
     Rapport `anuraset_campagne.md` (et les rapports détaillés). Répétition générale sur un
     AnuraSet simulé (4 sites × 6 enregistrements d'une minute, 3 espèces, fond propre à
     chaque site, encodeur factice ; 1 à 2 min) : toute la chaîne tourne, les 14 têtes
     comprises. Elle a trouvé un défaut : une espèce qui chante dans chaque enregistrement
     d'une minute rend l'AP par enregistrement indéfinie (aucun enregistrement négatif) ; le
     classement se fait donc au niveau fenêtre, et les écarts appariés à la logistique sont
     aussi calculés par fenêtre (`window_comparisons`, bootstrap par enregistrement). Sur ce
     jeu factice (trop facile : AP par site de 1 partout), l'AP poolée de la logistique
     tombait à 0,94 et celle de R37 à 0,81 alors que chaque site était parfaitement classé,
     et R19 la ramenait à 1 : l'effet d'échelle entre plis du n° 133, ici entre sites. Rien
     sur les vraies espèces.

137. **Campagne AnuraSet du 28/09 : premiers résultats réels (perch_v2).** Réseau ouvert par
     Léonard (zenodo.org, huggingface.co, kaggle.com). Données : `raw_data.zip` (7,2 Go,
     téléchargé par 12 plages en parallèle), 1 612 enregistrements d'une minute, 4 sites.
     Encodés : les 1 206 aux chants datés et les 393 sans aucune espèce (vrais négatifs,
     labels faibles) ; 13 écartés (espèce signalée sans chant daté). perch_v2 via bacpipe
     installé sans ses dépendances lourdes (ONNX, ni TensorFlow ni CUDA) ; fenêtres de 5 s
     jointives (`encoders.overlap: 0` dans `config/anuraset.yaml`) : 19 080 fenêtres en 37 min
     (15,5 fenêtres/s, 4 cœurs).
     Espèces : les 5 présentes sur au moins 2 sites sont DENMIN, LEPLAT, PITAZU, BOAFAB,
     PHYCUV. Retenues : DENMIN (5,3 kHz, 3 sites, l'analogue le plus proche d'A. blanci),
     BOAFAB (2 sites équilibrés), PHYCUV (3 sites), PITAZU (note de 0,3 s ; 1 386 fenêtres
     positives à INCT17 contre 29 à INCT41 : un test d'amorçage). LEPLAT (1 270 contre 79)
     laissée pour une prochaine passe. Aucune n'est dans la bande 3–6 kHz avec une note
     ≤ 0,3 s : `choose_species` aurait pris PITAZU, DENMIN, PHYCUV.
     Protocole : un pli par site, 20 négatifs par positif et par site, C sur la grille
     complète dans chaque pli, 14 têtes (`CAMPAIGN_HEADS`), 4 espèces en parallèle (5 à
     15 min chacune). Rapport complet : `documentation/anuraset/`, tableau
     `documentation/tableaux/anuraset.png`.
     Ce que ça dit (indicateur, pas verdict ; A. blanci n'y est pas) :
     - la logistique domine les têtes par similarité d'un site à l'autre : prototype et
       knn:k=5 s'effondrent sur DENMIN (AP poolée 0,27 et 0,44 contre 0,91) et PHYCUV
       (0,68 et 0,57 contre 0,87) ; amorcer un site par similarité à quelques exemples
       d'ailleurs ne suffit pas ;
     - R37 (et R37=glmm, identique) : seule régularisation significativement meilleure, sur
       PITAZU (+0,045 d'AP poolée), l'espèce au site presque vide ; −0,01 à −0,02 ailleurs ;
     - R18=64 (ACP) : égale ailleurs, et la meilleure sur le site presque vide de PITAZU
       (AP 0,53 contre 0,28 à INCT41) ; l'ACP n'a pas jeté le chant ici ;
     - R19, R20 : meilleures site par site sur DENMIN et PITAZU (AP par site 0,69 et 0,67
       contre 0,67 ; 0,56 et 0,58 contre 0,52) mais l'AP poolée s'effondre (DENMIN 0,59 et
       0,31 contre 0,91). Mécanisme probable : le centrage par site retire une part du chant
       là où l'espèce est très abondante (DENMIN dans ~38 % des fenêtres d'INCT17). A. blanci
       est rare dans le stock : l'hypothèse de Léonard (R19/R20 pour amorcer un site) reste
       à juger sur les données ONF, où ce mécanisme devrait peser bien moins ;
     - R13, R21, DANN, lda_shrunk, loss:focal : aucun gain ; R13 perd (DENMIN −0,04,
       PITAZU −0,09) ;
     - BOAFAB est saturée (≈ 0,96 partout) ; DENMIN sur INCT20955 reste à ≈ 0,14 pour toutes
       les têtes : le cas dur, où rien ne généralise.
     Recalibration par pli (n° 135) sur 5 têtes : elle **dégrade** l'AP poolée (DENMIN :
     logistique 0,91 → 0,86, R37 0,90 → 0,60, R19 0,59 → 0,23). Les plis internes sont ici
     d'autres sites : la calibration apprise sur eux se transpose mal au site testé, et elle
     efface des écarts de niveau entre sites qui étaient justes (un site où l'espèce abonde
     mérite des scores plus hauts). Décision provisoire : `fold_calibration: none` reste le
     défaut ; l'AP par site (`ap_fold_mean`) est le complément utile.

## 2026-09-28 (soir) — Corrections de l'audit : AnuraSet, comparaisons, rappel

138. **AnuraSet : chants coupés aux jonctions, espèces signalées sans chant daté.** Audit du
     28/09 (sous-agents, vérifié dans le code et sur les données). Deux défauts de
     `window_labels` faussaient la campagne du n° 137 :
     - fenêtres jointives : un chant à cheval sur la jonction de deux fenêtres les rendait
       toutes deux NaN, il n'avait plus aucune fenêtre positive (biais vers les chants
       centrés). Désormais, un chant qu'aucune fenêtre ne contient entier rend positive celle
       qui en porte la plus grande part (la première à égalité), l'autre reste écartée. Grille
       jointive de 5 s sur tous les fichiers aux chants datés : fenêtres positives DENMIN
       1 813 → 2 001, PHYCUV 905 → 984, PITAZU 1 415 → 1 475, BOAFAB 1 814 → 1 857. Grille à
       50 % : rien ne change pour les chants de 2,5 s ou moins ;
     - labels faibles : un fichier gardé pour les chants datés d'une espèce, qui en signale une
       autre sans aucun chant daté d'elle, comptait comme négatif pour l'autre (positifs
       cachés). Fichiers gardés concernés : PITAZU 25, PHYCUV 20, LEPLAT 12, DENMIN 9,
       BOAFAB 6. Leurs fenêtres sont désormais écartées pour cette espèce-là, à l'évaluation
       (`weak_only_files`, `window_labels(unsure_files=…)`) ; la docstring de
       `campaign_recordings` le promettait sans le faire.
     `anuraset.weak_labels` donné mais absent : erreur explicite (avant : aucun vrai négatif,
     sans un mot). En-tête de `config/anuraset.yaml` : le stock s'appelle
     `perch_v2-bacpipe1.3.5@o0`. Pas de réencodage : seuls les labels et l'évaluation
     changent. Les chiffres du n° 137 sont à refaire.

139. **Comparaisons appariées : micros tirés entiers, Holm, référence fixée d'avance.**
     - Bootstrap par micro : les enregistrements d'un même micro partagent fond et faune ; les
       tirer un à un rend l'intervalle trop étroit, et « A meilleur que B » trop fréquent,
       quand les micros sont peu nombreux. `evaluate(clusters=…)`, `compare_encoders`,
       `compare_to_reference`, `background_diagnostic` et `benchmark-all` tirent désormais des
       points (site/micro) sur les données ONF. AnuraSet garde l'enregistrement : 2 à 4 sites
       ne se tirent pas ; l'intervalle y reste optimiste, l'AP par site tenu à l'écart dit la
       variabilité.
     - Holm : `paired_bootstrap` rend une p-valeur bilatérale (2 × min(P(Δ ≤ 0), P(Δ ≥ 0))) ;
       `with_holm` ajoute `p_holm` et `significant_holm` à chaque tableau de comparaisons
       (encodeurs, têtes, benchmark complet ; AnuraSet : toutes les têtes et toutes les
       espèces d'un même niveau). 13 têtes × 4 espèces jugées à 5 % donnent 2 à 3 victoires
       par hasard : « R37 significativement meilleure sur PITAZU » (n° 137) se rejuge sur
       `significant_holm`. `significant` (chaque intervalle, seul) reste pour mémoire.
     - Référence du benchmark complet : `benchmark.reference` ou `--reference` ; à défaut la
       meilleure AP, signalée comme choisie après coup (`reference_post_hoc`) : elle doit une
       part de sa place à la chance. Le tableau ajoute `ap_fold_mean`.

140. **Rappel à précision plancher : seuil choisi sur les autres plis.** Le seuil choisi et
     jugé sur les mêmes scores est un oracle : il connaît les labels qu'il mesure. Avec des
     plis (`evaluate(folds=…)` ; benchmark complet par la colonne `fold` du stock hors-pli),
     `recall@p…` est désormais le rappel au seuil choisi pli par pli sur les autres plis
     (`evaluate.cross_fitted_recall`, le calcul de R74 déplacé depuis `regularization`) ;
     l'oracle reste en `recall@p…_oracle`, et `threshold@p…` est toujours le seuil qu'on
     déploierait (choisi sur tous les scores). Le rappel par site du benchmark complet suit
     le même seuil. Sans plis (baselines, sources externes), seul l'oracle se calcule : il
     reste en `recall@p…`. Les rappels publiés avant ce numéro sont optimistes.
