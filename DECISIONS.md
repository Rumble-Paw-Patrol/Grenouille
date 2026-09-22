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
