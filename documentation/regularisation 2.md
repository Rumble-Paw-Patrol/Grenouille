# Prompt WIP sur les régu

Finalement, la meilleure régularisation qui soit reste l'augmentation de la taille de ma base de données étiquetées qui est encore trop faible à ce jour. Nous avons prévu avec mon tuteur une séance d'annotation de données sous peu. Cela dit, le choix des données à annoter est crucial. En effet, en bio-acoustique et en particulier dans mon cas, les encodeur ont beaucoup de mal à associer à la même classe des données étiquetées comme positives mais provenant de sites différents. La proportion de chant de ma grenouille dans une fenêtre de 3 secondes avoisinant les 2%. Ainsi, les encodeurs ont tendance à grouper les sites plutôt que les espèces, la signature acoustique du bruit ambiant spécifique à chaque site prenant le dessus. Notre plan d'action est le suivant pour l'instant : lors de l'ajout d'un nouveau site à ma base de donnée (ma base de donnée évoluant au cours du temps, au fil des relevés effectués en fôret), en priorité on utilise une head de prototype différentiel pour identifier des premiers positifs. Une fois le site amorcé et l'ajout progressif de données dessus, le score de têtes de classifications telles que le linear probing devrait prendre le dessus.

# Notes de Claude sur les régularisations

# 0. Ce qui est déjà en place (et ce qui cloche)
Tête logistique (head.py:165) : pénalité L2, C choisi sur [0.001 … 10] en maximisant l'AP sur des plis groupés par micro, standardisation, class_weight="balanced".
Problème : l'AP maximale est choisie sur 5 plis alors qu'il n'y a que 13 micros positifs. La valeur de C retenue est donc bruitée et un peu optimiste (voir R60–R62).
Tête attentive (attentive.py:73) : weight_decay=1e-3 passé à Adam, ce qui revient à une L2 couplée (et non à l'AdamW découplé). Elle tourne 300 époques sur le lot entier, sans arrêt précoce, et le weight decay n'est jamais réglé.
Fusion (fusion.py:79) : C est fixé à 1.0 et jamais réglé. Les 4 descripteurs au plus, limités par la règle d'environ 10 positifs par coefficient, sont déjà une régularisation par le nombre de variables.
Prototypes : ce sont des modèles très régularisés par construction (voir R17).
Modèle maison, distillation, LoRA : rien encore, seulement des emplacements réservés. dropout: 0.1 figure dans la config LoRA.
# A. Données et labels (avant l'encodeur)
Augmentation par mélange de fond : on colle une note ou une fenêtre positive sur le fond d'autres micros ou sites (Trésor, Kaw) avec un rapport signal/bruit tiré au hasard. C'est la régularisation la plus directe contre « apprendre Mataroni ».
Mixup / CutMix audio : on mélange deux fenêtres et leurs labels (label souple), ou on mélange positif + négatif en gardant le label positif (« la grenouille est présente malgré le bruit »).
Variation de gain / normalisation aléatoire du volume, pour simuler l'éloignement, le « second plan » ou « lointain ».
Décalage temporel de la note dans la fenêtre (roll, crop aléatoire), pour ne pas dépendre de sa position sur la grille.
Décalage de hauteur léger (±2–3 %) et étirement temporel léger. À manier avec prudence : l'IOI (1,2–1,9 s) et la bande (4,4–5,5 kHz) sont diagnostiques, et un décalage trop fort crée des A. surinamensis.
SpecAugment (masques en temps et en fréquence) : seulement sur le spectrogramme d'un modèle entraîné (modèle maison, LoRA). Aucun effet sur un encodeur gelé, sauf si on réencode.
Réverbération / réponse impulsionnelle / filtre passe-bas, pour simuler la distance et la végétation.
Ajout de pluie ou de bruit réel : on prend des enregistrements marqués rain et on les superpose. Cela rend le modèle robuste à la pluie au lieu de la filtrer.
Augmentation par canal : le micro 18 dB comme vue alternative du même événement (DECISIONS n° 50). C'est un « deuxième capteur » gratuit, qui double les vues de chaque positif.
Test-time augmentation : on moyenne les scores sur plusieurs décalages ou canaux au moment de l'inférence. C'est une régularisation de la prédiction, pas de l'apprentissage.
Lissage des labels : on remplace 0/1 par 0.05/0.95, ce qui absorbe les erreurs d'annotation, dont les faux négatifs dans les négatifs présumés (voir le risque H20 de nearest).
Labels souples par confiance : un poids ou un label réduit pour les commentaires « second plan », « lointain », « malgré la pluie ».
Pondération des exemples : les poids ne sont plus seulement équilibrés par classe, mais aussi par micro et par enregistrement. Ainsi un micro avec 40 positifs ne domine pas un micro avec 2 positifs (régularisation par la structure des groupes).
Plafonnement par enregistrement : au plus k fenêtres positives par enregistrement à l'entraînement, contre la pseudo-réplication.
Négatifs difficiles (les 158 faux amis, les logits des congénères élevés) : ils contraignent la frontière. Leur proportion est un hyperparamètre de régularisation.
Ratio de négatifs (1:20 à 1:50) : c'est aussi un levier implicite sur la variance et le biais.
# B. Représentation (embedding de l'encodeur)
Normalisation L2 des embeddings : déjà faite pour les prototypes, pas pour la logistique, qui ne fait que standardiser. Il faut tester L2 puis standardiser, ou L2 seule.
Blanchiment PCA / réduction de dimension avant la tête (d = 1024 ou 1280 contre environ 51 enregistrements positifs) : garder 16, 32, 64 composantes. C'est une régularisation par projection, très efficace à faible effectif.
Centrage par site ou par micro : on soustrait la moyenne des embeddings du micro. On retire ainsi le fond sonore, dans l'esprit du prototype différentiel. C'est de la normalisation de domaine, peut-être la plus prometteuse pour la généralisation à d'autres sites.
Normalisation par lot au niveau du site (moyenne et variance par site à l'inférence, façon AdaBN) : c'est l'équivalent de R19 à l'échelle du site.
Retrait des directions de nuisance : on projette pour enlever les axes qui prédisent le micro ou le site (voir l'AMI micro du clustering), à la manière de l'INLP (« null-space projection »). On force la tête à ignorer l'identité du micro.
Choix du pooling (moyenne, max, moyenne+max, GeM avec p appris) : c'est un a priori structurel. La moyenne régularise le plus, le max est le plus sensible à une note isolée.
Choix de la couche de l'encodeur : les couches intermédiaires sont souvent plus génériques et moins spécialisées oiseaux.
Concaténation de plusieurs encodeurs (ensemble.py) : elle augmente d dans un problème à petit n. Il faut alors une régularisation plus forte, ou une PCA par bloc.
Quantification ou binarisation des embeddings : un effet régularisant, marginal.
# C. Tête linéaire (logistique, prototypes)
L2 (ridge) : ✅ en place.
L1 (lasso) : une sélection de dimensions creuse, lisible (« quelles dimensions portent la grenouille »). Elle est instable quand d ≫ n et que les dimensions sont corrélées.
Elastic Net (L1 + L2) : règle le problème de corrélation du lasso. Deux hyperparamètres : C et l1_ratio (solveur saga).
Group lasso : on pénalise par blocs, un bloc par encodeur en concaténation, ou par groupe de descripteurs. Cela sélectionne des encodeurs entiers.
Pénalité de rétrécissement vers le prototype : on pénalise ‖w − w_proto‖² au lieu de ‖w‖². C'est une L2 centrée sur le prototype différentiel : à peu d'annotations, la tête reste proche du prototype, puis s'en écarte avec les données. Un bon candidat vu H(prototype vs logistique sur peu d'annotations).
Analyse discriminante linéaire à covariance rétrécie (Ledoit-Wolf / OAS) : le continuum entre prototype différentiel (covariance identité) et logistique (covariance apprise). Le rétrécissement joue le rôle de régularisation, avec une solution fermée et stable.
Logistique bayésienne / a priori gaussien (Laplace ou variationnelle) : c'est la L2, avec en plus une incertitude par fenêtre, utile pour l'apprentissage actif (file des « incertains »).
Contrainte de norme maximale (‖w‖ ≤ c) au lieu d'une pénalité.
SVM linéaire à marge large (hinge + L2) : une autre fonction de perte, parfois plus robuste aux positifs aberrants.
Perte robuste (focale, logistique tronquée, perte symétrique) : réduit l'influence des labels probablement faux.
Choix de class_weight : « balanced » multiplie en pratique la pénalité effective des négatifs. Il faut donc régler C avec cette pondération. On peut aussi tester sans pondération puis recalibrer le seuil.
Régularisation de l'intercept / du biais par site : un biais par site très pénalisé (modèle à effets aléatoires), pour absorber les différences de niveau de fond sans toucher w.
Modèle mixte / effets aléatoires par micro : on généralise R37. Une pente commune plus des écarts par micro rétrécis vers 0 (régularisation hiérarchique).
kNN : choix de k et pondération pour la recherche par l'exemple. Un k plus grand lisse davantage.
# D. Tête attentive
Réglage du weight decay : à faire par validation groupée, comme C. Il est aujourd'hui fixé à 1e-3.
AdamW (weight decay découplé) au lieu de la L2 d'Adam : le comportement est plus prévisible.
Arrêt précoce sur un pli de validation groupé : c'est aujourd'hui 300 époques fixes. C'est probablement le levier le plus important de cette tête.
Pénalité d'entropie de l'attention : pousser vers une attention diffuse (proche de la moyenne) ou piquée (proche du max), selon l'a priori « la note est brève ».
Température de l'attention (fixe ou apprise, avec pénalité).
Dropout sur les jetons : on supprime au hasard des jetons avant l'attention, pour ne pas s'appuyer sur un seul jeton.
Dropout sur les dimensions de l'embedding agrégé.
Rétrécissement vers la tête logistique : initialiser et pénaliser vers les poids de la logistique, sur le même principe que R30.
Rang faible : query et weight dans un sous-espace de dimension k (projection partagée), ce qui réduit fortement le nombre de paramètres.
Attention multi-têtes avec partage de paramètres : la contrainte opposée, qui ajoute de la capacité mais de façon structurée.
# E. Fusion, stacking et module séquentiel
Régler C de la fusion : il est aujourd'hui fixé à 1.0.
Contraintes de signe / monotonie : le poids du score de la tête doit être ≥ 0, et frac_ioi_blanci ≥ 0. C'est un a priori fort, qui empêche un coefficient absurde appris sur 51 positifs.
Contrainte de somme des poids = 1 (combinaison convexe) : c'est déjà le cas pour weighted et weight_grid. On pourrait l'étendre à la logistique.
Réduction du nombre de descripteurs : ≤ 4 aujourd'hui. On peut aller vers une sélection L1 parmi les 11 candidats.
Discrétisation ou binning des descripteurs (courbes en marches monotones), plus robustes que le linéaire brut sur des variables asymétriques.
Un modèle additif généralisé (GAM) avec pénalité de lissage au lieu d'une logistique, si une relation non linéaire est suspectée (par exemple sur longest_run).
Garde-fou « pas de veto » : ce principe de la feuille de route est une régularisation structurelle. On peut le formaliser par un plafond sur le poids des descripteurs séquentiels.
Hors-pli strict pour le score de niveau 1 : ✅ en place. C'est la régularisation clé du stacking, qui évite qu'il apprenne à faire confiance à une tête sur-apprise.
Modèle phénologique comme a priori bayésien (score × prior(heure, mois)) plutôt que comme variable apprise : cela évite d'apprendre la phénologie de Mataroni (voir §3 de la feuille de route).
# F. Réseaux entraînés (modèle maison, distillation, LoRA, fine-tuning)
Les classiques, en bloc : dropout, weight decay, arrêt précoce, taux d'apprentissage bas et warm-up, gradient clipping, normalisation par lot ou de couche, petite architecture (peu de filtres, bande 3–7 kHz seulement).
Rang LoRA (8 dans la config) : un rang plus bas régularise davantage. Nombre de couches adaptées (seulement les dernières).
Gel progressif / taux d'apprentissage par couche (plus faible dans les couches basses).
L2-SP : on pénalise l'écart aux poids pré-entraînés plutôt qu'à zéro. C'est la régularisation de référence du fine-tuning à petit n.
Distillation : l'enseignant (la chaîne gelée) sert lui-même de régularisation (labels souples, température). On peut distiller sur des données non étiquetées d'autres sites, ce qui apporte de la diversité de domaine gratuitement.
Moyenne des poids (SWA / EMA) : des poids plus lisses, sans coût d'annotation.
Pré-entraînement sur AnuraSet puis ajustement : une régularisation par transfert (les deux anoures multi-sites de tes notes).
Apprentissage multi-tâches : prédire aussi le micro, le site, la pluie avec inversion du gradient (DANN), pour une représentation invariante au site. C'est la version « réseau » de R21.
Tête multi-classes (blanci / congénères / faux amis / bruit, comme dans tes notes « sigmoïde et modèle de classification ») : les classes auxiliaires régularisent la frontière blanci.
# G. Niveau enregistrement, point et temps (post-traitement)
Lissage temporel des scores entre fenêtres voisines (moyenne glissante, médiane, HMM à deux états « chante / ne chante pas »). C'est le « lissage temporel » de BirdCLEF, et il colle au chant continu.
HMM / CRF sur la séquence de fenêtres avec une probabilité de transition faible : une régularisation par a priori de persistance.
Agrégation robuste par enregistrement : moyenne des k meilleures fenêtres (déjà top-3 dans evaluate.py:189), quantile, noisy-OR tempéré. Le k est un paramètre de lissage.
Lissage entre créneaux voisins du même micro (prior de présence si le créneau précédent est positif).
Modèle d'occupation hiérarchique au niveau du point (site-occupancy à la MacKenzie) : il rétrécit la probabilité de présence d'un point vers la moyenne du site quand il y a peu d'enregistrements.
Calibration (Platt, isotonique, régression bêta) : ce n'est pas une régularisation au sens strict, mais l'isotonique sur-apprend à petit n, donc on préfère Platt. Utile pour le seuil de précision plancher.
# H. Réglage et sélection (méta-régularisation)
Validation croisée imbriquée : choisir C dans une boucle interne et évaluer dans l'externe. Aujourd'hui, select_C et l'évaluation risquent de partager les mêmes plis.
Règle du « 1 écart-type » : prendre le C le plus régularisant dont l'AP reste à moins d'un écart-type du meilleur. C'est simple, et adapté au bruit des 13 micros.
Grille de C plus fine (log sur 10–20 points) et chemin de régularisation (tracer AP en fonction de C, pour voir si l'optimum est plat).
Leave-one-micro-out au lieu de 5 plis : 13 plis positifs, moins de biais dans le choix de C.
Sélection d'hyperparamètres visant la généralisation au site : optimiser sur AnuraSet multi-sites, ou, dès qu'on en aura, sur Trésor/Kaw, et pas seulement intra-Mataroni.
Ensembles et bagging : moyenne des têtes entraînées par pli ou par tirage bootstrap. C'est une réduction de variance quasi gratuite, et ces têtes existent déjà dans les plis.
Limiter le nombre de comparaisons : chaque encodeur × tête × fusion × emplacement testé est un degré de liberté. Il faut un jeu gelé consulté une seule fois, ou une correction du « gagnant chanceux ». C'est de la régularisation du chercheur (le jardin des chemins qui bifurquent).
# I. Apprentissage actif et semi-supervisé
Pseudo-étiquetage avec seuil haut et un poids faible pour les pseudo-labels (BirdCLEF), qui régularise par les données non étiquetées.
Régularisation de cohérence : même prédiction pour deux augmentations d'une même fenêtre non étiquetée (Mean Teacher, FixMatch). Réservé aux réseaux entraînés.
Minimisation d'entropie sur les fenêtres non étiquetées des nouveaux sites (adaptation de domaine), à surveiller car elle peut s'effondrer.
Part d'aléatoire dans la file 60-20-20 : c'est une régularisation du jeu d'annotation contre le biais du détecteur (voir la « conséquence 2 » Biophonia).
Si je devais en retenir quelques-unes pour commencer, à cause des 51 positifs sur 13 micros tous à Mataroni :

R19/R21 : centrage par micro et retrait des directions de micro.
R18 : PCA avant la tête.
R30/R31 : rétrécissement vers le prototype, ou LDA rétrécie.
R75/R77 : règle du 1 écart-type et leave-one-micro-out.
R42 : arrêt précoce de l'attentive.
R51 : contraintes de signe sur la fusion.
R68 : lissage temporel.
R1 : augmentation par mélange de fond, pour le modèle maison.
Autrement dit, des régularisations qui ciblent le site, plutôt que de simplement jouer sur la L1 ou la L2.

On en discute dans l'ordre que tu veux.
