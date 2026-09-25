
# À prompter une fois le squelette terminé

 la feuille de route se trompe sur le seuillage spectral en amont

 ajouter un notebook de checking global : un enregistrement écoutable, son spectrogramme, son spectrogramme après seuillage, son embedding, son négatif apparié, l'embedding de son négatif, comparaison des deux, soustraction etc etc etc

 ajouter un script pour l'encodage, empêchant l'ordi de s'éteindre ou quoi ou qu'est-ce

 Le prototype simple apparaît au §3 comme « baseline de similarité », mais il n'est repris ni au §6 ni dans la spécification du §13. Il est sur la carte, pas dans le plan d'exécution. Ça vaut une ligne dans DECISIONS.md pour trancher.


 considérer la sigmoïde et un modèle de classification entre Blanci, les espèces qui cohabitent avec Blanci et les bruits. Permettrais de classer plus facilement des fenêtres overlapées ?

 tableau de quel encodeur fournit quelle info en sortie, token ? taille des embeddings ? autre ? Adapter head en fonction de la sortie ?

 Idée afin de connaître quel encodeur fournit les tokens en sortie :
Lire la fonction forward du modèle. Cherche .mean(dim=…), .max(…), [:, 0] (extraction du token de classe), AdaptiveAvgPool ou GlobalAveragePooling2D.

Le test empirique, en cinq lignes. Tu récupères à la fois les tokens et l'embedding par défaut, puis tu compares :
(schématique : les noms d'attributs dépendent du modèle)
h = tokens_avant_agregation(x)     # (B, N, d)
e = embedding_par_defaut(x)        # (B, d)
cos = torch.nn.functional.cosine_similarity
print(cos(e, h.mean(1), dim=-1).mean())          # moyenne ?
print(cos(e, h[:, 0], dim=-1).mean())            # token de classe ?
print(cos(e, h.max(1).values, dim=-1).mean())    # maximum ?

La candidate dont la similarité approche 1 est la bonne. Si aucune n'y arrive exactement, c'est souvent qu'une normalisation ou une projection s'applique après l'agrégation — il faut alors remonter dans le code.

  revoir avec Claude la méthode de négatif appariés, j'aimerais qu'on puisse apparier le négatif le plus proche, y compris s'il est dans le même enregistrement

identifier des régularisations : L1, L2, Elastic Net (L1 + L2), autres

# Trucs à faire

télécharger le drive au cas où j'ai pas de co demain

Réunion semaine pro : 
- faire un cours sur le mécanisme d'attention dans ma prochaine présentation
- état de l'art
- suis-je noté sur mon SFE sur la biblio ? biblio foundation model biblio BirdCLEF !!! biblio phéno blanci
- github ouvert aux tuteurs
- CORPUS complet, tester anuraset, dans AS il n'y a que deux de leurs grenouilles qui seraient des bons candidats car elles sont multi-sites (au moins deux sites) avec assez de données pour avoir un entrainement fiable. En testant sur ces grnouilles on peut pré-benchmark la capacité des modèles à généraliser multi site

Inventaire complet des enregistrements et annotation, finir demain avec le collègue de la réserve Trésor

Centrage par site ou par micro : on soustrait la moyenne des embeddings du micro. On retire ainsi le fond sonore, dans l'esprit du prototype différentiel. C'est de la normalisation de domaine, peut-être la plus prometteuse pour la généralisation à d'autres sites.

faire un tableau ou un graph camembert des annotations pos/neg par site, par micros... pour conserver un équilibre
Annoter manuellement un max de données d'un max de points d'écoutes différents.

possibilité dans l'application d'annotation de choisir le canal à gain faible ou à gain fort

possibilité de faire des simualtions /!\ certains modèles sont sensibles aux simulations

regarder les embeddings de sortie de l'encodeur solo vs choeur, on sera fixés

vérifier la fiabilité des tests, si un test donne un mauvais score mais que le test est mal foutu ça peut m'induire en erreur

Pour le module sequentiel : timeserie de pluviométrie, phénologie, un module puissant serait complexe et complet, sequential sera dans un second temps pour ameliorer les capciteés

# Régularisations (tri du 25/09, R1–R32 ; suite R33–R84 à trier)

Programmées (DECISIONS n° 108) : R13, R15, R17, R18, R19, R20, R21, R22 (gem), R27, R28, R30, R31
Gardées pour plus tard : R1 (mélange de fond), R3 + R7 (gain, réverbération), R6 (SpecAugment),
R9 (canal 18 dB), R24 + R29 (ensembles par concaténation), R32 (logistique bayésienne → file des incertains)
Outils de performance : R25 (quantification)
Jetées : R2, R4, R5, R8, R10, R11, R12, R14, R16
Impossible pour l'instant : R23 (couches intermédiaires : hook par encodeur PyTorch, à faire à la main)
Nouvelle : R85 = R21 bis, sonde non linéaire (pondération des dimensions selon l'entrée)

⚠ R19/R20/R21 se jugent sur un site tenu à l'écart, pas en validation croisée Mataroni (DECISIONS n° 109)
⚠ R19 + R21 interdit : efface le chant



# Questions en suspens

revoir le principe du standard scaler, et l'intérêt d'un standard scaler par site ou par micro

GeM avec p appris pour le probing ?? C'est quoi ? Je suis hype

heu c'est quoi finalement une régression logistique ? Fonctionnement du linear probe

blancinet la vraie baseline ultime du projet

attentive sur l'embedding de sortie et pas sur les tokens (attentive mets un poids sur les bonnes dimensions) (réponse pour Sylvain) ?

c'est quoi les 546 tests que Claude lance tout le temps

linear probe sur moyenne, max, combinaison des deux, combinaison intelligente des deux (selon l'axe ou autre), sur **autre chose que moyenne et max** ??

wrapper bacpipe c'est quoi

prototype différentiel arithmétique d'embedding ??

clarifier attentive

clarifier cascade

attentive sur l'embedding de sortie et pas les tokens ? Attentive c'est du linear sur token ???

birdnet et perch (et d'autres ?) font nativement de l'overlap

blanci chante sporadiquement en saison sèche ?? Vérifier et coupler l'info avec la phénologie pour l'ajouter dans le module sequential

Pk on mets pas des micros là où on suspecte qu'elle ait disparue ? De mémoire, les endroits où elle est suggérée disparu c'est parce qu'elle n'est plus observée, mais elle pourrait être entendue ? Quelle est la fiabilité de l'observation par rapport à l'écoute ?

perspective post-stage : Peut-être phénologie différente de blanci à Mataroni par rapport à Kaw et Molokoï ? À Kaw et Molokoï, blanci vit dans des criques bien dessinées, rocheuses alors qu'à Mataroni le lit de la crique est plutôt évasé, marécageux.

# Autres

check pb pluie et coup de feu. La signature sonore du coup de feu, diffus lorsque le micro est loin du point de tir, se fond dans le bruit de pluie omniprésent. Quelle solution ? 

stage M2 exploratoire -> perspectives futures à développer en fin de stage. Important

# Extra pro
location colocation guyane
vivre en guyane


# Old
--------------------------------------------------------------------------------------------------------------------------------------------

### négatifs appariés peuvent autant être :
- un négatif à la même heure, un autre jour
- un négatif le même jour, à une heure le plus proche possible 

## Méthode de sélection des candidats à annoter

Queue 60-20-20
Récolte par similarité
YAPAT
Étiquetage en bloc par cluster

Queue 60-20-20 : baseline, proportions modifiable (plus d'aléatoire en début d'entraînement)
Récolte par similarité : identifie des positifs faciles. Idéal pour amorcer l'annotation d'un nouveau site.
YAPAT
Cluster : quand un groupe s'avère homogène après une dizaine d'évaluation, annotation massive du groupe.

##Benchmark des méthodes pour head - Choix de w

Recherche par l'exemple - w = l'embedding d'une fenêtre de référence
Prototype simple - w = μ₊, moyenne des positives
Prototype différentiel - w = μ₊ − μ₋
Linear probe [moyenne, maximum]- w appris par optimisation
Attentive probe - w appris par optimisation (sur certains modèles seulement)
Cascade linear probe + attentive probe sur candidats sélectionnés
Autre méthode spécifique à la sortie d'un foundation model - dépend du modèle
Wrapper - outil de bacpipe à explorer

Prototype simple : baseline
Prototype différentiel : l'écart de score avec baseline mesure la part de fond sonore captée par l'embedding
Linear probe : dès quelques dizaines de positifs, la régression logistique apprend une direction qui tient compte de la covariance, ce que le prototype ne sait pas faire. Généralise mieux sur un site MAIS risque de moins généraliser d'un site à l'autre par rapport au prototype différentiel. Le prototype différentiel à négatives appariées, qui annule explicitement la moyenne du site, pourrait généraliser mieux tant qu'on a pas de positifs multi-sites. Ce serait un résultat intéressant, hypothèse à tester.
Linear probe : moyenne ou maximum ou combinaison des deux ou moyenne en fréquence et max sur l'axe du temps.
Attentive probe : coûteux en mémoire car il conserve les vecteurs de chaque token, si trop coûteux : disque dur stockant les tokens ou méthode de la cascade.
Autre méthode : par exemple, certains modèles ont leur propre mécanisme d'attention. Dans ce cas là, ne pas construire de head ? Que faire ?

intérêt de head
fine tuning vs entraînement


H1 : les enregistreurs sont tous des Wildlife Acoustics Song Meter Mini 2. D'après mon tuteur c'est le top.
H2 : Les 345 fenêtres proviennent de 345 enregistrements de 2 minutes différents sur 13 micros différents. Ce sont des fenêtres de 3 secondes. Tous à Mataroni mais à des emplacements différents. Certaines annotations confirmées à la main avaient eu un score médiocre d'après l'algorithme de l'ancien prestataire. Je remarque des commentaires tels que "chants audibles en second plan", "chants audibles malgré la pluie", "chant audible ET présence du fourmilier tacheté" ou encore "chant lointain".
H3 : vraie pour le modèle développé par l'ancien prestataire.
H4 : vraie. Voici les extraits de discussions de mes encadrants pour information :
-"Nous avions penser à tester les embeddings de Birdnet, avec un entraînement progressif (de type human in the loop) pour ajouter des nouveaux site au fur et à mesure."
-"Je voulais juste rebondir sur le fait que vous comptiez utiliser les embeddings de Birdnet. Pour être franc, j'éviterais, pour deux raisons : 

- Birdnet c'est du tensorflow et c'est pas très pratique pour extraire les embeddings, on reste très dépendant de la codebase de cornell, qui est pas super bien faite, à l'exception de leur interface graphique (mais qui est vraiment faite pour des personnent qui ne codent pas) . 

- Il y a plein de modèles bien meilleurs aujourd'hui, téléchargeables de manière très standard sur huggingface. pour un étudiant comme Leonard qui maitrise très bien tout ça, ce serait à mon avis beaucoup plus adapté. Si vous voulez un aspect interactif pour de l'apprentissage actif, de la réannotation / interface graphique, il saura faire également. Ca se vibecode très bien avec streamlit / gradio, on a fait quelque chose de très similaire avec Jérémy Froidevaux pour de la réannotation de buzz de bourdons. 

Je te conseille ce papier qui vient tout juste de sortir dans Ecological Informatics 

https://www.sciencedirect.com/science/article/pii/S1574954126001718

Il fait un tour d'horizon des modèles récents et globalement c'est BEATS NLM (le modèle fondations BEATS, finetuné sur des données bioacoustics par Earth Species Project dans le modèle Nature LM Audio) et BirdMAE qui sont au dessus du lot. A noter que "mon" modèle ProtoCLR n'est pas très loin non plus  mais on a pas rendu les choses faciles pour le réutiliser donc je ne recommande pas. 

Bref, je pense vraiment qu'il faut arrêter de s'embeter avec Birdnet car les alternatives faciles d'accès existent, surtout si vous avez une personne comme Léonard pour intégrer ça."
-"Effectivement, l'idée était de partir d'embeddings, en particulier ceux de BirdNET.
Je ne suis pas fermé à des embeddings provenant d'autres modèles.
Pour ma part, jusqu'à présent, les embeddings de BirdNET (ou de Perch) m'ont donné satisfaction avec du linear probing. A voir si on peut faire mieux avec d'autres modèles.
Je pensais aussi suggérer à Léonard d'utiliser le nouveau package bacpipe qui simplifie grandement la comparaison entre les différents embeddings des différents modèles, tels que ceux dont tu parles.
Pour d'autres projets, on a voulu passer à NatureLM, mais il est limité à 16kHz, ce qui est très limite pour les paysages sonores."
H5 : je n'ai pas compris cette hypothèse. Éclaircis moi et je rends une décision.
H6 : j'ai un GPU, tu connais mon Macbook avec puce M4, mais à la fin de mon stage l'ONF va récupérer mon projet et continuera l'entraînement avec les nouvelles données sur leurs propres machines. Ils possèdent des ordinateurs portables avec des intel core i5. Le détail :
Processeur  11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz (1.50 GHz)
Mémoire RAM installée   16,0 Go (15,7 Go utilisable)
ID de périphérique      98E52E39-1EB6-4652-9C3E-240D93A28026
ID de produit     00330-80000-00000-AA603
Type du système   Système d’exploitation 64 bits, processeur x64
Stylet et fonction tactile    La fonctionnalité d’entrée tactile ou avec un stylet n’est pas disponible sur cet écran
H7 : j'ai accès à l'étude de phénologie. Je vais rajouter le document dans le contexte du projet, tu y as bien accès ?
H8 :Anomaloglossus Blanci n'est pas dans les classes de Perch. Pour info il y a 3 congénères de Blanci dans les classes, ce sont les suivants :
Anomaloglossus baeobatrachus
Anomaloglossus stepheni
Anomaloglossus surinamensis
H9 : tu trouveras la réponse dans pheno-blanci.pdf dans le contexte du projet. C'est l'étude de phénologie.
H10 : ma machine a 16 giga de RAM.
Méthode du prestataire : non communiquée. Ce n'est donc pas une hypothèse en attente mais une contrainte définitive. Les 345 + 158 annotations restent utilisables comme données ; le modèle, non. À déplacer en angle mort : je ne pourrai pas démontrer une amélioration par rapport à ce qui existait, seulement par rapport à l'écoute.
H11 : vrai. Rapport et soutenance pour Mars 2027
H12 : ok, c'est pas hyper important de toute manière, on  va annoter quand on a le temps.
H13 : toujours une hypothèse. La date de rapatriement des micros pour écoute des enregistrements de février est encore incertaine, je ne sais pas si je serai encore en stage à ce moment là. De toute manière, il y aura d'autres écoutes réparties sur plusieurs années après la fin de mon stage donc mon travail doit être transmettable aux équipes de l'ONF à terme. Ils se débrouilleront après la fin de mon stage avec leurs machines peu puissantes pour réentrainer les modèles. Si ça doit prendre toute une nuit, ça prendra toute une nuit.
H14 : autres déclencheurs : des espèces en tout genre, voici la liste des déclencheurs qui ne sont pas A. Blanci, trouvés lors d'une séance d'annotation à la main :
chants d'oiseau (Fourmilier tacheté) au moment des détections
"cris" d'amphibien de contact
Oiseau (Fourmilier tacheté)
Oiseau Fourmilier tacheté
oiseau Fourmilier tacheté et autres
oiseau Moucherolle manakin
orthoptère (grillon sur les sessions +
micro dans sac
A. andreae
H. cappellei
oiseaux divers bien audibles
un chant unique suspect mais probablement pas blanci
aucun chant audible
oiseaux bien chanteurs
A andreae
oiseau indét. (Martinet ?)
Hyalinobatrachium
pluie mais aucun chant distinguable à l'oreille
oiseaux
psittacidé
oiseau Tangara mordoré
oiseau (pigeon plombé)
dans le sac
Oiseau indet
chants de fin similaire à l'écoute mais probablement autre chose avec la cadence précédente (Pic à cou rouge)
A. hahneli similaire A. aff baeo
Hyalinobatrachium cappellei
Amazophrynella teko
Amazophrynella teko (Oiseaux pïau et Tangara mordoré)
oiseau Evèque de Rothschild
Hyalinobatrachium (mondolfii)
A femoralis sur la séquence
oiseau myrmidon
a priori rien d'audible mais bcp de bruits parasites
oiseau Sclérure à bec court
oiseau (Sclérue obscure)
Hyalinobatrachium mondolfii (et H. iaspidiense)
Amazophrynella teko (Otophryne)
Allobates femoralis
Amphibien probable, pas un chant et pas A. blanci
oiseaux dont Fourmilier tacheté
oiseau indét ?
Otophryne
oiseau (psittacidés) au moment de la sequence +
Hyalinobatrachium mondolfii
cri d'interaction d'amphibien
Oiseau (Evèque de Rothschild)
J'ai également les fenêtres correspondant à ces faux-amis. On pourra faire du negative mining avec.
H15 : phénologie présente sur le pdf que j'ai rajouté dans le contexte du projet. La saisonnalité des chants y est mentionné, le taux d'activité selon l'heure de la journée. Ce pdf est une vraie mine d'or, si tu as du mal à le lire dis moi.
H16 : En 2026, à Kaw : 45 micros. À Trésor : 40 micros. Sur la même base de 2min par demi-heure de 5h à 19h30 pendant 1 semaine
H17 : pas de semaine réduite. Ici ça bosse
H18 : répondu en H14, il y a beaucoup d'espèces susceptibles de déclencher
H19 : les annotations fournies ont été réalisées en collaboration avec mon tuteur, expert naturaliste donc 100% fiable
H20 : voir le pdf de phénologie. Si pas de réponse dans le pdf, je vois avec mon tuteur
H21 : ça reste une hypothèse, à prendre en compte dans le façonnement du modèle sequential



--------------------------------------------------------------------------------------------------------------------------------------------



phénologie : blanci chante sans s'arrêter. Lorsqu'on l'entend chanter, elle chante du début à la fin des enregistrements de 2 minutes. Une détection isolée est probablement un faux positif ? À vérifier s'il arrive à blanci de chanter ponctuellement. À considérer dans les méthodes d'optimisation. Peut s'inscrire dans la phase sequential
forte saisonnalité de l'activité d'A. Blanci
idem, fort impact de l'heure de la journée.
séparer les chants d'une Blanci seule et ceux d'un groupe dans la phase sequential ? Une Blanci chantant seule ne fait pas le même effet qu'un groupe.

Les micros enregistrent 2 minutes toutes les demi-heures donc 4 minutes par heure et non pas 2 minutes par heure.

Des milliers d'enregistrements de 2 minutes donc des dizaines de milliers de fenêtres
345 fenêtres de chant annotées à la main par un expert

158 fenêtres de negative mining 

