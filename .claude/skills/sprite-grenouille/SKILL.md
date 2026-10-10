---
name: sprite-grenouille
description: Dessine une ou plusieurs grenouilles en sprite 8-bit animé, dans le style d'A. blanci du tableau de bord, en étapes que Léonard valide une à une, une par prompt, sans jamais en sauter (0 recherche morphologique et cadrage, 1 ébauche en formes simples avec le squelette, 2 croquis sans couleur, 3a couleurs à plat et motifs, 3b modelé et détails, 4 animation), puis, sur demande, un décor animé autour d'elles (5 : fiche écologique, calques, lumière, vie). Deux cas. Mode espèce : une espèce d'après ses photos et sa description scientifique. Mode image : une photo reprise telle quelle (pose, cadrage, plusieurs grenouilles, support, reflet). À utiliser pour « fais-moi <espèce> en 8-bit », « anime cette photo », « un sprite comme A. blanci », « fais-lui un décor », « /sprite-grenouille », ou pour poursuivre un sprite en cours (corrections, « go », « étape suivante »).
---

# Sprites 8-bit de grenouilles, par étapes

Références :
- A. blanci, le sprite du tableau de bord : `documentation/tableau-de-bord/grenouille.py`. Son
  histoire est visible pas à pas dans `documentation/tableau-de-bord/evolution-grenouille/`.
- Un exemple en mode image : deux *C. tomopterna* sur une branche, avec leur reflet,
  `documentation/tableau-de-bord/bac-a-sable/tomopterna/` (la version finale, avec la scène,
  est sur la branche `claude/trusting-einstein-35jvgu` tant qu'elle n'est pas fusionnée).
- Le décor de référence : la scène de chasse de `documentation/tableau-de-bord/bac-a-sable/chasse/`
  (`decor.py`, `chasse.py`, `page.html`).
- Les fiches morphologiques déjà faites : `fiches/` (*A. blanci*, *C. tomopterna*).
- Outils du skill : `outils.py`, `planche.html` et `modele.py`, le script modèle à copier.

Tout ce qui suit vient des corrections de Léonard. L'appliquer d'emblée.

## Pourquoi cet ordre

Pour *C. tomopterna*, on a enchaîné la silhouette et les couleurs, puis les détails et
l'animation, sans validation entre les deux. Il a ensuite fallu cinq tours de « corrections de
l'étape 3 » qui portaient en fait sur l'anatomie : quel membre est le bras gauche, ce qui passe
devant ou derrière, la patte en Z, le nombre de doigts, la taille de l'œil. Corriger une forme
sur un rendu coloré et nuancé coûte cher et casse ce qui était juste.

D'où trois principes :
- **On connaît l'animal avant de le dessiner** : une fiche morphologique sourcée (étape 0).
- **La morphologie et la posture sont fixées, et confirmées par Léonard et par moi, avant toute
  couleur** : ébauche (étape 1), puis croquis (étape 2).
- **On ne saute aucune étape.**

## La règle : une étape par réponse, aucune sautée

| Étape | On juge | Interdit à ce stade | Livrable |
|---|---|---|---|
| 0 Recherche et cadrage | la fiche morphologique, le mode, les photos, la taille, la destination | dessiner | `morphologie.md` et un message, questions groupées |
| 1 Ébauche | squelette, articulations, proportions, posture, membres et leurs largeurs en pixels, plans, nombre de doigts, yeux entiers (saillie, paupière, puis globe et pupille), traits du visage, en formes simples | couleur, forme fine, détail | planche avec squelette, lecture de la photo, superposition |
| 2 Croquis | les formes affinées, en dessin sans couleur : contours, galbes, mains et pieds au pixel, traits | couleur, ombre | planche, superposition |
| 3a Couleurs à plat | les couleurs justes et surtout les **motifs** | dégradé, ombre, lumière, reflet, effet mouillé, grain, tramage ; changer une forme | planche |
| 3b Modelé et détails | volume, nuances, peau, œil, contour | changer une forme ou un motif sans qu'il le demande | planche |
| 4 Animation | respiration, clignement, tête, regard | saut, chant | planche animée |
| 5 Environnement (facultatif) | 5a fiche et plan, 5b calques, 5c lumière, 5d vie | animer avant 5d | page de scène |

Les règles du passage d'une étape à l'autre :
- **Aucune étape n'est sautée ni fusionnée avec une autre, même si Léonard le demande.** S'il
  demande d'aller plus vite, lui rappeler en une phrase pourquoi (l'histoire de tomopterna), puis
  montrer l'étape en cours. On peut aller vite *dans* une étape, jamais d'une étape à l'autre.
- **Ne jamais entamer l'étape suivante dans la même réponse**, même pour « montrer ce que ça
  donnerait ». Une ébauche n'a pas de forme fine, un croquis n'a pas de couleur, des aplats
  n'ont pas d'ombre.
- **Fin de chaque réponse, à partir de l'étape 1 :**
  - vérifier la planche au navigateur (Chromium sans interface, largeurs ordinateur et téléphone),
    puis la publier en artefact privé, ou la mettre à jour en republiant le même fichier pour
    garder la même URL ; à l'étape 5, c'est la page de scène ;
  - donner le lien et lister 3 à 6 points à vérifier ;
  - demander « Passe-t-on à l'étape N : <nom> ? », puis **s'arrêter**.
- **Réponse de Léonard :**
  - des corrections : les faire, republier, et rester à l'étape, autant de tours qu'il faut ;
  - un accord explicite (« parfait », « go », « on passe ») : passer à l'étape suivante ;
  - un doute : rester à l'étape.
  - Aux étapes 1 et 2, il faut un accord sur la morphologie **et** sur la posture. Pour le
    croquis (étape 2) et le modelé (étape 3b), il faut qu'il les juge **parfaits**.
- **Une correction de forme demandée après la validation du croquis** ramène à l'étape 2 :
  - la faire dans `SQUELETTE` (posture) ou `formes()` (forme), puis refaire le croquis ;
  - le remontrer, le faire valider de nouveau et le geler de nouveau ;
  - reprendre ensuite à l'étape où l'on était : les rendus suivants se refont seuls.
- **Le journal des décisions**, `decisions.md` dans le dossier du projet :
  - y noter chaque remarque de Léonard, au moment où on la traite, avec l'étape (« 2 : le bras
    gauche est au deuxième plan, derrière le ventre ») ;
  - le relire au début de chaque réponse ;
  - ne jamais défaire une décision notée sans qu'il le demande.
- **Geler chaque étape validée :**
  - copier son rendu en `<code>_<etape>_valide.json` et le commiter ;
  - aux étapes suivantes, `outils.py comparer` contre le croquis gelé doit donner 0 pixel de
    silhouette ajouté ou retiré, sauf décision notée dans le journal.
- **Où travailler :**
  - dans `documentation/tableau-de-bord/bac-a-sable/<projet>/`, sur la branche de travail, avec
    un commit par étape validée (« <espèce> : étape N validée ») ;
  - rien sur le tableau de bord ni sur main avant son go explicite pour la destination.
- **Les photos de référence** restent dans le scratchpad : jamais dans le dépôt, jamais dans une
  page publiée. Les images de lecture et de superposition contiennent la photo. Les montrer
  seulement dans la conversation (`SendUserFile` si l'outil existe), jamais en artefact.

## Étape 0 — Recherche morphologique et cadrage (premier message)

### 0a. La fiche morphologique

Avant tout dessin, dans les deux modes. En mode espèce, elle dit quoi dessiner. En mode image,
elle dit comment lire la photo : par exemple, les phyllomédusines ont le premier doigt opposable
et serrent la branche, ce qui explique ce qu'on voit des mains.

- **Partir de `fiches/<genre>-<espece>.md`** si elle existe : la relire, la compléter si besoin.
- **Sinon, chercher sur le web**, dans cet ordre :
  - la description originale ou une redescription (*Zootaxa*, *Zenodo* et *Plazi*, qui publient
    le traitement de chaque espèce) ;
  - AmphibiaWeb, Amphibian Species of the World, les guides de terrain de la région ;
  - à défaut, les caractères du genre ou de la famille, **signalés comme tels**.
- **Ne rien inventer** : un caractère non trouvé s'écrit « non trouvé, à lire sur les photos ».
  Pour un individu donné, la photo l'emporte sur la fiche ; la fiche sert pour ce que la photo
  cache.
- **Écrire la fiche** dans `bac-a-sable/<projet>/morphologie.md`, sur le modèle de celles de
  `fiches/`, avec ses sources. Une fois validée, la copier aussi dans `fiches/` pour les sprites
  suivants.
- **Dans le message**, en résumer les points qui changent le dessin, avec leurs sources.

Les caractères à décrire, avec les termes techniques. Pour chacun, noter ce qu'il change sur
le sprite.

1. **Identité et mode de vie :** famille, genre ; activité (diurne, nocturne) ; locomotion
   (saute, marche, grimpe, nage) ; micro-habitat (litière, berge, branche, broméliacée). Ils
   donnent la posture.
2. **Taille et proportions :**
   - longueur museau-cloaque (LMC) des mâles et des femelles ;
   - tête : longueur, largeur, part de la LMC ;
   - longueurs du tibia, de la main et du pied en part de la LMC ;
   - longueur relative des membres antérieurs et postérieurs ; talons qui se touchent ou non,
     pattes repliées.
3. **Tête :**
   - museau vu de dessus et de profil : arrondi, tronqué, acuminé, saillant au-delà de la
     mâchoire ;
   - canthus rostralis (net ou arrondi) ; région loréale (droite, concave) ; narines : position ;
   - œil : diamètre relatif à la tête, saillie au-dessus du crâne ;
   - **pupille** : horizontale, verticale (en fente), ronde, en losange ;
   - **iris** : couleur, réticulation, anneau pupillaire ;
   - paupière supérieure : tubercules ; paupière inférieure (membrane palpébrale) :
     transparente, opaque, réticulée, de quelle couleur (on la voit au clignement) ;
   - **tympan** : visible ou non, taille par rapport à l'œil, anneau tympanique ;
     **pli supratympanique** : son trajet ;
   - glandes parotoïdes ; sac vocal : position et couleur (il colore la gorge du mâle).
4. **Corps et peau, face par face :**
   - silhouette : élancée, trapue, aplatie ; ligne du dos ; plis dorsolatéraux ;
   - texture du dos, des flancs, du ventre, des membres : **lisse, granuleuse, tuberculée,
     pustuleuse**, verrues, glandes, spicules ;
   - granules ou pustules d'une autre couleur que la peau (blanches chez *C. tomopterna*).
5. **Membres antérieurs :**
   - longueur ; avant-bras (renflé ou non) ;
   - **doigts : nombre** (4 chez les anoures), **formule des longueurs** (par exemple
     III > IV > I > II) ;
   - disques terminaux : présents ou non, taille par rapport au doigt, forme, scutes dorsales ;
   - franges, palmure ;
   - tubercules subarticulaires, palmaire, thénar ;
   - doigt I opposable ou non ; excroissances nuptiales du mâle.
6. **Membres postérieurs :**
   - cuisse, tibia, talon (tubercule calcaire), tarse (pli ou carène tarsale) ;
   - **orteils : nombre** (5), **formule des longueurs** (en général IV le plus long),
     disques ;
   - **palmure** : étendue (de basale à complète, ou sa formule) et couleur ;
   - tubercules métatarsiens interne et externe ; orteil I opposable ou non.
7. **Couleurs et motifs, face par face :**
   - dos, flancs, ventre, gorge, lèvre ;
   - **faces cachées** : cuisses, aine, aisselle, dessous des membres ;
   - iris, disques, palmure ;
   - motifs : bandes dorsolatérale, latérale oblique, ventrolatérale ; masque ; barres des
     membres ; taches ; sablier ; réticulations ;
   - variation : individuelle, entre mâle et femelle, chez le juvénile, de jour et de nuit
     (changement de couleur).
8. **Dimorphisme sexuel :** taille, gorge, doigts, tubercules.
9. **Postures et comportements :** posture de repos, posture sur le support, façon de cligner,
   de dormir ; ce qu'il ne faut pas montrer.

### 0b. Le cadrage

Reconnaître le mode :
- **Mode espèce**, par exemple « fais-moi <espèce> en 8-bit ».
  - Dessiner l'espèce dans la pose d'A. blanci : assise, de profil, tête à gauche, sauf si son
    mode de vie en appelle une autre (une rainette agrippée à une branche).
  - Il faut 2 ou 3 photos, dont une de profil. S'il n'y en a pas, les demander : ne jamais
    dessiner de mémoire.
- **Mode image**, par exemple « anime cette photo ».
  - La photo est le modèle, **telle quelle** : pose, angle, cadrage, nombre de grenouilles,
    leurs regards, le support (branche, feuille) et, pour l'étape 5, le reste de la scène
    (eau, reflet, fond flou).
  - On décalque, on ne réinvente pas l'anatomie. La fiche sert à lire la photo, à savoir ce
    qu'elle cache et que l'animation pourrait montrer, et à préparer le décor.
  - Exemple : deux rainettes-singes (*Callimedusa*) face à face sur une branche au-dessus de
    l'eau, avec leur reflet.

Puis, dans le même message, une proposition par point (ne demander que ce qu'on ne peut pas
décider) :
- **La destination :** bandeau du tableau de bord (à la place d'A. blanci ou à côté), prez, page,
  scène.
- **La taille :**
  - Mode espèce : 2 pixels par unité et un repère réglé sur la grenouille (A. blanci fait
    57 × 43 unités, soit 114 × 86 pixels).
  - Mode image : la toile reprend le cadrage de la photo, à une échelle où chaque grenouille
    garde au moins 90 pixels de large, sinon le style se perd. Donner le calcul : photo de
    1600 × 1081, grenouille de 340 pixels, échelle 0,28, toile de 448 × 303.
- **Ce que comprend le livrable :**
  - les grenouilles seules ;
  - plus le support ;
  - toute la scène, avec l'eau et le reflet (à l'étape 5).
- **Les crédits** des photos : leurs auteurs vont dans la docstring et sur la planche.

Finir par « Valides-tu la fiche et le cadrage ? Passe-t-on à l'étape 1 : ébauche ? ».

## Étape 1 — Ébauche : le squelette et les formes simples

But : que Léonard et moi soyons d'accord sur **ce qu'on dessine** (quel membre est lequel, à
quel plan, dans quelle position, avec combien de doigts) avant de dessiner bien.

1. **Lire la photo** sur une grille : `outils.py grille PHOTO grille.png x0,y0,x1,y1 [pas]`.
   - Mode espèce : cadrer sur la grenouille. La grille fait 48 unités de large ; retourner la
     photo si l'animal regarde à droite.
   - Mode image : cadrer sur toute la zone du livrable, avec `pas = 2 / échelle` (pixels de
     photo par unité ; échelle 0,28 → pas de 7,1). Chaque grenouille a son propre repère :
     choisir son coin haut gauche sur la toile (en pixels pairs) et soustraire sa position, en
     unités, des relevés.
   - **Identifier chaque membre** avant de relever quoi que ce soit, avec la fiche en main.
     Les côtés sont ceux **de l'animal** (`_g`, `_d`), jamais ceux de l'image ; de profil, tête
     à gauche, on voit son côté gauche. En cas de doute (trois quarts, membres croisés), suivre
     chaque membre de l'épaule ou de la hanche jusqu'aux doigts, et le dire.
2. **Copier `modele.py`** en `<code>.py` à côté du travail, une copie par grenouille, et
   remplir :
   - **`SQUELETTE`** : les repères en unités, relevés sur la grille. Tête : museau, narine, œil
     (le centre du globe), autre œil (le centre de toute sa bosse), commissure de la bouche,
     tympan. Tronc : nuque, sacrum, cloaque. Membres : épaule, coude,
     poignet, main ; hanche, genou, talon, tarse, orteils. Bouts des doigts et des orteils
     visibles (`doigt1_g`…), numérotés comme dans la fiche (I = le plus interne) ; un doigt
     ou un orteil qui ne part pas de la main ou du pied visibles (il sort de sous un poignet,
     par exemple) prend sa propre base (`base_<nom>`) et sa direction relevée sur la photo ;
   - **`MEMBRES`** : pour chaque membre, sa chaîne de repères, son plan (`fond`, `corps`,
     `devant` ; en mode image, d'après la photo) et ses bouts de doigts ; **`AXES`** : l'axe du
     corps et la ligne de la bouche ;
   - **`formes()` en formes simples** :
     - le tronc et la tête en polygones ou en ellipses de quelques points ;
     - **chaque membre à sa largeur réelle, mesurée en pixels** : sur la grille, à l'échelle de
       la toile, mesurer la largeur de chaque segment (cuisse, tibia, tarse, bras, avant-bras,
       doigts, orteils) et le diamètre des disques, en pixels du sprite. Les membres sont des
       `os_()`, segments tirés du squelette, de rayon en unités égal à la largeur en pixels
       divisée par 4 ;
     - **les doigts et les orteils** de même, jamais en traits fins par défaut ; les disques
       sont des **boules** au diamètre mesuré. Chez *C. tomopterna*, à l'échelle 0,28 : doigts
       d'environ 3 pixels, disques d'environ 6. Un doigt par forme cernée, pour que deux disques
       voisins restent séparés. En nombre, direction et longueur justes (la formule de la
       fiche) ; les espaces entre doigts et orteils voisins comme sur la photo ;
     - **le profil de la tête sans les yeux** : relever le contour du crâne (bout du museau,
       canthus, dessus de la tête jusqu'à la nuque) en faisant abstraction des yeux. Un œil
       saillant dépasse du crâne : le crâne ne suit pas sa courbure, et il ne se creuse pas
       entre l'œil et le museau. Des droites là où la tête est anguleuse, sans lissage par
       défaut ; poser des repères sur le profil (la narine cachée, la nuque) et les relier ;
     - **l'œil entier, avant la pupille** : d'abord toute la saillie, c'est-à-dire le globe et la
       paupière qui l'entoure et le couvre, souvent de la couleur du dos (verte chez
       tomopterna) ; puis la part visible du globe, à sa taille ; la pupille en dernier
       (`PUPILLE` : ronde, horizontale ou verticale) ;
     - **l'autre œil** : toute la bosse qu'il fait sur le crâne, de la couleur de la peau. On
       n'en voit au plus qu'un mince croissant de globe (`oeil["autre"]`), jamais un œil entier
       ni une pupille au centre ;
     - **la lèvre et le menton** : relever sur la photo la ligne de la lèvre et l'épaisseur du
       menton en pixels ; elles varient d'une espèce et d'un individu à l'autre ;
     - les traits du visage : bouche, narine, tympan, pli supratympanique s'il se voit ;
   - **les plans** : chaque forme a le sien (`fond` : les membres de l'autre côté, `corps`,
     `devant` : les membres proches) ; on la cerne si un contour doit la séparer de ce qu'elle
     recouvre ;
   - **`MIROIR = True`** pour une grenouille tournée vers la droite : les formes restent
     décrites tête à gauche (relever sur une copie retournée de la photo), les côtés `_g` et
     `_d` s'échangent tout seuls à l'export ;
   - **mode image, le support** est dessiné dès l'ébauche, puisque les doigts l'enserrent. Il
     prend un script à part : une seule forme, sans œil ni squelette, matière « écorce » ou
     équivalente. Ce qui passe devant la grenouille (le bord d'une branche sous les doigts) va
     dans un second sprite, posé après elle.
3. **Rendre et vérifier :**
   - `python3 <code>.py ebauche` ;
   - `outils.py lecture <code>_ebauche.json PHOTO x0,y0,x1,y1 lecture.png` : le squelette nommé
     sur la photo et sur le sprite, côte à côte. Chaque repère doit tomber sur son articulation ;
   - `outils.py superposer <code>_ebauche.json PHOTO x0,y0,x1,y1 sup.png`, avec un cadre aux
     mêmes proportions que la toile : le contour cyan doit suivre la photo à 2 ou 3 pixels près
     (les formes sont encore simples) ;
   - mode image : `outils.py composer scene.json LxH a.json@x,y b.json@x,y …`, puis `lecture` et
     `superposer` sur la composition.
4. **La planche :**
   `outils.py planche <code>_ebauche.json planche.html "<Nom>" "<texte>" "Étape 1 · ébauche"`.
   - Le squelette y est tracé et nommé (bouton « Squelette »), avec des os de couleur selon le
     plan : orange devant, jaune corps, bleu fond.
   - Elle a une grille de pixels numérotée, pour que Léonard désigne un pixel (« rapproche de
     3 pixels »).
   - La publier en artefact privé, toujours le même fichier. Montrer la lecture et la
     superposition dans la conversation.
5. **Dans le message**, un tableau de lecture, une ligne par membre : côté de l'animal, plan,
   articulations visibles, largeurs mesurées en pixels, doigts ou orteils visibles sur combien,
   contact avec le support.
   C'est ce tableau que Léonard valide autant que l'image.

À vérifier avant de montrer :
- [ ] Chaque membre est identifié, du bon côté de l'animal et au bon plan ; rien n'est inventé
      ni oublié.
- [ ] Les proportions suivent la fiche : tête, tibia, main et pied en part de la LMC.
- [ ] La posture : angles du coude, du genou, du talon ; inclinaison du dos et de la tête ;
      regard et écart entre les grenouilles (mode image).
- [ ] Le profil du crâne relevé sans les yeux, qui en dépassent ; ni creux ni arrondi que la
      photo ne montre pas.
- [ ] Le museau de profil, la courbure de la lèvre et l'épaisseur du menton, le tympan et le
      pli s'ils se voient.
- [ ] L'œil entier : la saillie et la paupière d'abord, puis le globe (taille, place) et la
      pupille ; l'autre œil en bosse de peau, avec au plus un croissant de globe.
- [ ] Les largeurs de chaque segment de membre, des doigts et des disques, mesurées en pixels
      sur la photo ; aucun vide de fond là où la photo n'en montre pas, et le fond là où elle
      en montre.
- [ ] Le nombre de doigts et d'orteils et leur formule ; le contact avec le support.
- [ ] Rien ne touche les bords de la toile.

## Étape 2 — Croquis sans couleur

But : un dessin juste et beau, à la forme près, qu'il ne restera plus qu'à colorer. C'est là
que se fait le vrai travail de forme. L'ébauche validée est la charpente : le squelette ne bouge
plus, sauf demande de Léonard.

**La méthode, zone par zone,** dans cet ordre : tête et œil, tronc, bras, patte arrière, puis
mains et pieds. Pour chaque zone :
1. **Remplacer les formes simples par le vrai contour**, relevé sur la photo :
   - courbes de Catmull-Rom pour les galbes (un contour qui se croise se remplit mal : des
     polygones simples) ;
   - lignes droites là où l'espèce est anguleuse : dos, museau et mâchoire de tomopterna en
     polygones sans lissage ;
   - membres à rayon variable : mollet du tibia bombé dessus et presque droit dessous,
     avant-bras un peu renflé, cuisse en fuseau ;
   - articulations : un coude ou un genou n'a pas de frontière ; les formes se lient et la pointe
     de l'articulation se voit.
2. **Tracer les traits d'un pixel** (`traits` de `formes()`) :
   - bouche jusqu'à la commissure, narine, tympan et son anneau, pli supratympanique ;
   - paupière supérieure, bosse de l'autre œil ;
   - limites entre plans : on cerne une forme seulement si un contour doit la séparer de ce
     qu'elle recouvre.
3. **Poser les mains et les pieds au pixel près**, en gabarits, jamais en formes calculées (qui
   donnent des pâtés et des doigts collés) :
   - doigts à la largeur mesurée à l'étape 1, en éventail, avec au moins un pixel vide entre
     deux doigts, selon la formule de la fiche ; un pixel de large seulement chez une espèce
     aux doigts fins comme A. blanci ;
   - diagonales **en escalier** : chaque pixel touche le suivant par un côté, sinon le contour
     le coupe et le doigt paraît détaché ;
   - disque au bout, en boule, au diamètre mesuré (environ 6 pixels chez tomopterna à
     l'échelle 0,28 : jamais un 3 × 3 par défaut) ;
   - un pouce opposable serre ou longe le support ; un doigt caché derrière un avant-bras ne se
     dessine pas.
4. **Se relire seul après chaque zone :**
   - `outils.py superposer` : le contour suit la photo à 1 ou 2 pixels près ;
   - `outils.py zoom` sur la zone ;
   - `outils.py controle` : aucune miette, c'est-à-dire aucun groupe de moins de 12 pixels
     détaché par le contour ;
   - la planche aux petites tailles (1,5 à 3,5 pixels par pixel du sprite) : la silhouette se
     lit, l'œil et les doigts se devinent.

Les règles de forme, toutes espèces :
- **Grille et contour :** 2 pixels par unité, pas d'anticrénelage, un contour d'un pixel autour
  de la silhouette.
- **Allure :** fine et vivante, jamais un crapaud ni « en surpoids » (sauf si l'espèce l'est) ;
  les proportions de la fiche.
- **Œil :** l'œil, c'est toute la saillie : la paupière supérieure est en relief, de la couleur
  de la peau, et le globe a un pli dessous ; la pupille a la forme de la fiche. L'autre œil se
  voit comme une bosse de peau sur le crâne, de la couleur de la peau ; on n'en voit au plus
  qu'un mince croissant de globe, jamais de cercle.

Les règles de forme d'A. blanci (pose assise, de profil) :
- **Allure :** ventre haut, avec du vide entre le bras et la patte ; dos peu incliné ; tête haute.
- **Ordre de superposition**, du fond vers l'avant : bras de l'autre côté, corps, cuisse, tarse et
  orteils, **tibia par-dessus la cuisse**, talon, bras proche, main proche, bosse de l'autre
  œil, œil. Le ventre gonflé passe derrière les pattes.
- **Patte arrière en Z :**
  - la cuisse est arrondie et tournée vers nous ;
  - le tibia est plus gros, long, bombé dessus et presque droit dessous ; il va du genou levé
    au talon en descendant très légèrement et **dépasse derrière la croupe** ;
  - le talon est arrondi, épais (environ 5 pixels) et aligné sur le bout du tibia ;
  - le tarse revient vers l'avant au sol ;
  - rien entre le pied et le tibia.
- **Pattes avant :**
  - le coude est ouvert (environ 130°) ;
  - l'avant-bras est un peu renflé ;
  - le poignet arrive sur le talon de la paume ;
  - **l'épaule s'attache en arrondi, sans aucun trait** ;
  - le bras de l'autre côté se voit, plus fin, sa main plus haut et en retrait.

**Rendre :** `python3 <code>.py croquis`, puis la planche « Étape 2 · croquis » (squelette
masqué, à afficher avec le bouton), et la superposition dans la conversation.

À vérifier :
- [ ] Le contour suit la photo à 1 ou 2 pixels près ; le squelette validé n'a pas bougé.
- [ ] Chaque galbe et chaque articulation se lisent ; aucune forme en « boudin ».
- [ ] Les traits du visage sont à leur place ; l'œil a sa taille et sa saillie, l'autre œil sa
      bosse.
- [ ] Les mains et les pieds : nombre, formule, éventail, disques, contact ; `controle` ne
      trouve aucune miette.
- [ ] Le dessin se lit à la taille du téléphone.

Une fois le croquis jugé **parfait** : `cp <code>_croquis.json <code>_croquis_valide.json`,
commit.

## Étape 3a — Couleurs à plat et motifs

- **Une couleur par matière** : dos, flanc, ventre, gorge, lèvre, membres, faces cachées,
  disques, iris, pupille, paupière, support. La fiche dit quelles faces existent ; la photo
  donne leurs teintes.
  - Couleurs prises sur la photo, dans des zones bien éclairées, sans reflet ni ombre :
    `outils.py pipette PHOTO zone [zone …]` donne la couleur médiane de chaque zone.
- **Les motifs comptent le plus** : bandes, barres, taches, réticulations.
  - Leur forme, leur nombre, leur place et leur orientation viennent de la photo. Compter les
    barres sur chaque segment de membre et placer chaque tache.
  - En mode image, ce sont les motifs de **chaque individu** de la photo, pas ceux de l'espèce
    en général.
  - Les décrire dans `MOTIFS` et `formes()`.
- **Les traits du croquis** prennent la couleur qu'ils traversent, assombrie.
- **Rien d'autre** : ni dégradé, ni ombre, ni lumière, ni reflet, ni effet mouillé, ni grain,
  ni tramage.
- **Aucune forme ne bouge** : `outils.py comparer <code>_croquis_valide.json <code>_aplats.json`
  donne 0 pixel de silhouette ajouté ou retiré.
- **Rendre :** `python3 <code>.py aplats`, puis la planche « Étape 3a · couleurs à plat ».

À vérifier :
- [ ] Les teintes, côte à côte avec la photo (panneaux 1 et 2 de `superposer`).
- [ ] Chaque motif à sa place, en bon nombre, dans le bon sens, des deux côtés s'il se voit
      (membres du fond).
- [ ] L'œil : iris, pupille, cercle ; la lèvre ; les faces cachées qui se voient.

## Étape 3b — Modelé et détails

Partir du rendu `details` du modèle :
- un volume par forme, avec la lumière d'en haut à gauche, ou celle de la photo en mode image
  (`LUMIERE`) ;
- des rampes de 7 tons par matière, en trois variantes de teinte réparties par plaques ;
- un grain léger ;
- un œil brillant ;
- un contour plus clair côté lumière.

**Le contour :** proposer à Léonard un contour sombre uniforme, comme pour A. blanci, ou un
contour teinté, qui prend la couleur qu'il borde, assombrie, sans aucun trait noir, comme pour
tomopterna (`CONTOUR_TEINTE = True`). Avec le contour teinté, les articulations se fondent sur
deux pixels.

Puis l'affiner à la main, avec les règles ci-dessous. L'étape demande plusieurs passes. Se
relire seul avant de montrer :
- `outils.py apercu FICHIER.json a.png CLES 6 PHOTO` ;
- `outils.py zoom` sur l'œil, l'épaule, le genou et le tibia, les mains ;
- `outils.py controle` : aucune miette ;
- `outils.py comparer` contre le croquis gelé : 0 pixel de silhouette ajouté ou retiré.

Les fonctions d'A. blanci se reprennent telles quelles quand il le faut : gabarits des mains et
des pieds et `poser`, `nuancer`, `pustules`, `retoucher`.

### Règles de rendu, toutes espèces

La fiche morphologique l'emporte sur ces règles quand l'espèce le demande.
- **Ombre de contact** (par exemple le flanc derrière un bras) : le pixel de base, même teinte,
  assombri et un peu grisé. Jamais des pixels d'une autre matière.
- **Nuances :** flou limité à chaque matière, puis une rampe fine (7 à 12 niveaux) et trois
  variantes de teinte par plaques. On vise une centaine de couleurs. Léonard a trouvé ce rendu
  « bien bien mieux » que les seuls tons de base.
- **Fondus entre zones d'une même face** (flanc, ventre, gorge) : des limites ondulées et
  tramées, pas des lignes.
- **Grain et pustules :**
  - un grain doux, d'un seul cran de rampe ;
  - les pustules de la couleur de la peau : un pixel de la couleur d'origine, plus clair, avec
    son ombre dessous, posés après les nuances ;
  - les granules d'une autre couleur (blanches chez tomopterna) : semées au hasard, jamais deux
    voisines ;
  - pustules et granules gardent des places fixes d'une image à l'autre ;
  - pas de point blanc vif, sauf si la fiche décrit des granules blanches.
- **Reflets humides :** seulement là où la photo en montre (coude, cuisse, genou, disques).
- **Disques :** plus clairs que le doigt, de la couleur de la peau, jamais un beige à part.
- **Œil :**
  - le globe est bombé, avec un reflet blanc et un petit reflet bleuté ;
  - l'iris est celui de la fiche : fente verticale et iris argenté réticulé chez les
    phyllomédusines, iris cuivré à pupille horizontale chez *Anomaloglossus* ;
  - l'œil fermé est de la peau qui couvre le globe, avec une fente en arc, sans cercle de
    couleur autour (sinon « effet lunettes ») ; ou la paupière inférieure de la fiche ;
  - l'autre œil ne cligne jamais.

### Règles de rendu d'A. blanci (pose assise, de profil)

- **Épaule :** un capuchon dont le pourtour tourne au jaune et se fond dans le flanc en tramage.
- **Bras de l'autre côté :** plus sombre que le bras proche.
- **Bande du dos :** elle passe derrière la cuisse au lieu de s'arrêter d'elle-même.

À vérifier :
- [ ] Le volume se lit et la lumière vient d'un seul côté.
- [ ] Les doigts sont détachés et en éventail ; `controle` ne trouve aucune miette.
- [ ] L'œil est vivant et l'œil fermé crédible ; les motifs de l'étape 3a sont intacts.
- [ ] `comparer` contre le croquis gelé : aucune forme n'a bougé.
- [ ] On compte environ 80 à 120 couleurs ; `ruff check` et `ruff format` passent.

Une fois le modelé jugé **parfait** : `cp <code>_details.json <code>_details_valide.json`,
commit.

## Étape 4 — Animation

- **La base :** `python3 <code>.py animation` donne 24 images (gorge × flanc × œil × tête),
  avec les clés `"{gorge}{flanc}{œil}{tête}"`, tête droite = 1.
- **La pose de repos** est l'image validée, pixel pour pixel :
  `outils.py comparer <code>_details_valide.json <code>_animation.json` doit donner 0 pixel
  différent.
- **Les rythmes**, à 12 images par seconde :
  - flanc gonflé 1,3 s toutes les 3,2 s ;
  - gorge qui palpite (0,3 s sur 0,6 s) pendant 2,4 s, toutes les 7 s ;
  - clignement de 0,13 s, toutes les 3 à 7,5 s, parfois double ;
  - tête baissée ou relevée d'un cran (0,06 rad autour du cou, en s'estompant vers le corps)
    pendant 1,6 à 4 s, toutes les 4 à 9 s.
- **Le clignement** suit la fiche : peau qui couvre le globe, ou paupière inférieure qui remonte
  (membrane lilas au réseau doré chez tomopterna).
- **Jamais de saut, jamais de chant** : pas de sac vocal gonflé, pas de notes.
- **Pour une autre pose**, garder le principe : respiration, clignement, regard. Par exemple, une
  grenouille agrippée à une branche bouge la tête mais pas les mains.
- **Plusieurs grenouilles :** chacune a ses propres rythmes, décalés (la planche le fait). Leurs
  regards suivent l'image : deux grenouilles face à face se regardent, se détournent un instant
  et reviennent.
- **Les poses de comportement** (bouche ouverte, yeux rentrés pour avaler, regard vers une
  proie) s'ajoutent en réglages de `formes()`. Avec tous les réglages à zéro, l'image doit être
  la validée, pixel pour pixel : le vérifier avec `comparer`.
- **Mouvement réduit** (`prefers-reduced-motion`) : image fixe.
- **La planche** d'animation est « Étape 4 · animation ». Autour de la grenouille, sur la planche
  comme sur le bandeau, une mosaïque de tuiles ambrées de 2 × 2 scintille, plus vive près d'elle.

## Étape 5 — Environnement (facultatif, sur demande), elle aussi par sous-étapes

Les mêmes règles valent : une sous-étape par réponse, aucune sautée.

**5a. Fiche écologique et plan**, à valider avant de dessiner. En mode image, le décor est celui
de la photo : la fiche sert surtout à choisir ce qui vit autour en 5d (proies, voisins, heure).
- **La fiche**, sourcée par une recherche sur le web si l'outil existe, jamais inventée :
  - activité : diurne ou nocturne, heures ;
  - phénologie : saison des pluies ou sèche, reproduction, chant, pontes, têtards, soins
    parentaux ;
  - habitat et micro-habitat : litière, berge, canopée, mare temporaire, broméliacée ;
  - régime alimentaire : proies, tailles, chasse à l'affût ;
  - prédateurs ;
  - plantes et animaux qui partagent le lieu ;
  - comportements qu'on peut montrer, et ceux à ne pas montrer (une proie toxique qu'elle évite,
    par exemple).
- **Le plan :**
  - le format (256 × 144 par défaut ; en mode image, le cadrage de la photo) ;
  - la place des grenouilles ;
  - une zone calme pour les proies ;
  - la liste des calques et le déroulé dans le temps (une journée en accéléré, ou seulement la
    nuit pour une espèce nocturne) ;
  - les comportements à animer.

**5b. Calques fixes**, un ou deux à la fois, chacun montré avec la grenouille posée dessus. Ils
sont faits en Python, sur le modèle de `chasse/decor.py` (rampes tramées Bayer 4 × 4, bruit,
courbes, traits) :
- **le fond :** perspective aérienne, avec une seule rampe de brume, et un plan est d'autant
  plus proche de la brume qu'il est loin. Garder un fond calme et clair derrière la tête,
  sombre derrière le dos ;
- **le support :** feuille, branche, berge ;
- **l'avant-plan :** un repoussoir sombre (fougère, bord de feuille) qui ne cache rien
  d'important.

En mode image, ces calques reprennent ceux de la photo : fond flou, eau, branche. La photo n'est
jamais copiée : on en relève une grille de couleurs grossière et la place des éléments, puis on
repeint en pixels tramés. Le reflet est un calque calculé depuis les sprites (miroir autour de
la ligne d'eau relevée, assombri, teinté, étiré pour une grenouille vue d'en dessous), animé
seulement en 5d.

**5c. Lumière et ambiance**, montrées en planche fixe à plusieurs heures ou météos :
- la teinte de l'heure, en multiplication ;
- des lumières additives : rayons, lumière dorée du matin et du soir, rose de l'aube, violet du
  crépuscule, clair de lune ;
- des lumières locales : les lucioles éclairent la peau, et seulement dans la pénombre.

**5d. Vie et animation, en dernier :**
- les comportements tirés de la fiche : chasse (langue, prise, avaler, raté), sommeil, regards ;
- les proies et les voisins : insectes, papillons ;
- la météo : averse, gouttes ;
- une interaction (le pointeur porte une proie) et un carnet de terrain ;
- un son synthétisé, coupé par défaut ;
- le mouvement réduit.
- Tester sans interface : simuler plusieurs journées pas à pas, vérifier qu'aucun état ne reste
  bloqué, faire les captures. Publier la page.

Ce qui a marché :
- toile affichée à une échelle entière, encadrée par une ombre et non une bordure ;
- moucherons en trois pixels, avec des ailes qui battent et qui accrochent la lune la nuit ;
- un morpho de 13 × 7 pixels qui traverse parfois la scène et qu'elle suit des yeux ;
- une fougère qui s'égoutte sur son dos après l'averse ;
- au bord de l'eau (tomopterna) : des gouttes qui tombent de la branche et rejoignent leur
  reflet, des ronds qui troublent le reflet ; toucher l'eau fait un rond, l'air lâche un
  moucheron, une grenouille cligne.

## Intégration

- **Tableau de bord :**
  - Le bloc « A. blanci en 8-bit » de `modele.html` lit `D.grenouille`, que `construire.py`
    remplit depuis `grenouille.json`. Le canvas `#grenouille` est dans `.perchoir`.
  - La toile prend la taille du sprite (32 pixels de plus en largeur, 8 en hauteur) et se décale
    vers la droite d'autant que le sprite dépasse 96 pixels.
  - Pour changer d'espèce, remplacer `grenouille.json`. Pour en ajouter une, ajouter un second
    canvas et une seconde clé de données.
  - Publier selon le skill `tableau-de-bord`. Si la version en ligne a changé depuis la dernière
    lecture, repartir d'elle.
  - Vérifier à 1440, 1280, 1100 et 400 pixels de large, en thèmes clair et sombre.
- **Autre page :** reprendre le JavaScript de `planche.html` (décodage par plages, rythmes,
  composition) ou celui de la scène de chasse.
- **Évolution :** ajouter chaque version validée à
  `documentation/tableau-de-bord/evolution-grenouille/`.
- **Fiches :** copier la fiche morphologique validée dans `fiches/`.

## Outils (`python3 .claude/skills/sprite-grenouille/outils.py …`)

| Commande | Usage |
|---|---|
| `grille PHOTO SORTIE.png [x0,y0,x1,y1] [pas]` | grille en unités sur la photo, pour relever le squelette et les formes |
| `lecture FICHIER.json PHOTO x0,y0,x1,y1 SORTIE.png [zoom]` | le squelette nommé sur la photo et sur le sprite : la lecture de l'étape 1 |
| `superposer FICHIER.json PHOTO x0,y0,x1,y1 SORTIE.png [zoom] [CLE]` | photo, sprite, et contour du sprite sur la photo (cadre aux proportions de la toile) |
| `comparer AVANT.json APRES.json [SORTIE.png] [CLE_AVANT] [CLE_APRES]` | pixels de silhouette ajoutés ou retirés, pixels recolorés : geler une étape validée |
| `pipette PHOTO zone [zone …]` | couleur médiane de chaque zone : les aplats de l'étape 3a |
| `apercu SPRITE.json SORTIE.png CLES [zoom] [PHOTO]` | images côte à côte |
| `zoom SPRITE.json SORTIE.png CLE x0,y0,x1,y1` | un détail pixel par pixel |
| `controle SPRITE.json` | image par image : formes cernées à part, et miettes (moins de 12 pixels) à corriger ; ignore le contour, teinté compris |
| `composer SORTIE.json LxH SPRITE.json@x,y[,m] …` | plusieurs sprites sur une toile (`m` : retourné, lumière et squelette compris) |
| `planche FICHIER.json SORTIE.html NOM "TEXTE" "ÉTAPE"` | planche de relecture (sprite ou composition, statique ou animée, grille de pixels, squelette) |

`modele.py` (`python3 <code>.py ebauche|croquis|aplats|details|animation`) écrit
`<NOM>_<étape>.json`. Sa grenouille de démonstration ne sert qu'à vérifier la chaîne :
remplacer `NOM`, le repère (`UNITES`, `PIVOT`), `SQUELETTE`, `MEMBRES`, `AXES`, `formes()`,
`MATIERES`, `MOTIFS`, `OEIL` et `PUPILLE`, et régler `MIROIR`, `LUMIERE` et `CONTOUR_TEINTE`.
