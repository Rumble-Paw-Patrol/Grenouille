


##Le stacking - Structure de la fusion

Tu as deux experts qui regardent la même fenêtre sous deux angles. L'un est le linear probe sur embeddings : il a une vision globale et apprise du son. L'autre est le module séquentiel : il mesure des grandeurs physiques précises comme la durée des notes ou leur espacement. Chacun rend son avis.

La fusion est un troisième modèle, très petit, dont le seul travail est d'apprendre quelle confiance accorder à chaque expert et comment combiner leurs avis. Il ne regarde jamais le son lui-même, seulement ce que les deux experts en disent.

C'est ce qu'on appelle un stacking à deux niveaux.

## Trucs qui ont pas encore leur place

Faire écouter A. Blanci au début de la prez

Sur la question du fine-tuning
« Le fine-tuning total est écarté non par manque de calcul mais par manque de données : 503 exemples ne contraignent pas 12 millions de paramètres, et l'adaptation détruirait la généralisation à de nouveaux sites, qui est l'objectif du stage. »

Sur la question de l'intérêt de l'approche "foundation model + head"
« Le foundation model apporte ce que mon corpus ne peut pas apporter — une représentation construite sur 1,5 million d'enregistrements. Mon corpus apporte ce que le foundation model n'a pas — l'espèce cible et le contexte guyanais. Le probe est l'endroit où les deux se rencontrent. »

Sur seuillage spectral vs prototype différentiel (Cours de ML)
Une formulation plus juste : le seuillage spectral exploite une connaissance a priori de la physique du signal, sans données ; le prototype exploite des exemples, sans connaissance a priori. Ce sont deux sources de savoir différentes
