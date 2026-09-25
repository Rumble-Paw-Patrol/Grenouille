


#Le stacking - Structure de la fusion

Tu as deux experts qui regardent la même fenêtre sous deux angles. L'un est le linear probe sur embeddings : il a une vision globale et apprise du son. L'autre est le module séquentiel : il mesure des grandeurs physiques précises comme la durée des notes ou leur espacement. Chacun rend son avis.

La fusion est un troisième modèle, très petit, dont le seul travail est d'apprendre quelle confiance accorder à chaque expert et comment combiner leurs avis. Il ne regarde jamais le son lui-même, seulement ce que les deux experts en disent.

C'est ce qu'on appelle un stacking à deux niveaux.

# Trucs qui ont pas encore leur place

Faire écouter A. Blanci au début de la prez

Sur la question du fine-tuning
« Le fine-tuning total est écarté non par manque de calcul mais par manque de données : 503 exemples ne contraignent pas 12 millions de paramètres, et l'adaptation détruirait la généralisation à de nouveaux sites, qui est l'objectif du stage. »

Sur la question de l'intérêt de l'approche "foundation model + head"
« Le foundation model apporte ce que mon corpus ne peut pas apporter — une représentation construite sur 1,5 million d'enregistrements. Mon corpus apporte ce que le foundation model n'a pas — l'espèce cible et le contexte guyanais. Le probe est l'endroit où les deux se rencontrent. »

Sur seuillage spectral vs prototype différentiel (Cours de ML)
Une formulation plus juste : le seuillage spectral exploite une connaissance a priori de la physique du signal, sans données ; le prototype exploite des exemples, sans connaissance a priori. Ce sont deux sources de savoir différentes

Ma plume de zinzin sur la généralisation multi-sites :
Finalement, la meilleure régularisation qui soit reste l'augmentation de la taille de ma base de données étiquetées qui est encore trop faible à ce jour. Nous avons prévu avec mon tuteur une séance d'annotation de données sous peu. Cela dit, le choix des données à annoter est crucial. En effet, en bio-acoustique et en particulier dans mon cas, les encodeur ont beaucoup de mal à associer à la même classe des données étiquetées comme positives mais provenant de sites différents. La proportion de chant de ma grenouille dans une fenêtre de 3 secondes avoisinant les 2%. Ainsi, les encodeurs ont tendance à grouper les sites plutôt que les espèces, la signature acoustique du bruit ambiant spécifique à chaque site prenant le dessus. Notre plan d'action est le suivant pour l'instant : lors de l'ajout d'un nouveau site à ma base de donnée (ma base de donnée évoluant au cours du temps, au fil des relevés effectués en fôret), en priorité on utilise une head de prototype différentiel pour identifier des premiers positifs. Une fois le site amorcé et l'ajout progressif de données dessus, le score de têtes de classifications telles que le linear probing devrait prendre le dessus.
