---
name: sprite-grenouille
description: Dessine une ou plusieurs grenouilles en sprite 8-bit animé, dans le style d'A. blanci du tableau de bord, en étapes que Léonard valide une à une, sur plusieurs prompts (1 silhouette, 2 couleurs à plat et motifs, 3 détails, 4 animation), puis, sur demande, un décor animé autour d'elles (5 : fiche écologique, calques, lumière, vie). Deux cas. Mode espèce : une espèce d'après ses photos. Mode image : une photo reprise telle quelle (pose, cadrage, plusieurs grenouilles, support, reflet). À utiliser pour « fais-moi <espèce> en 8-bit », « anime cette photo », « un sprite comme A. blanci », « fais-lui un décor », « /sprite-grenouille », ou pour poursuivre un sprite en cours (corrections, « go », « étape suivante »).
---

# Sprites 8-bit de grenouilles, par étapes

Références :
- A. blanci, le sprite du tableau de bord : `documentation/tableau-de-bord/grenouille.py`. Son
  histoire est visible pas à pas dans `documentation/sprites/evolution-grenouille/`.
- Le décor de référence : la scène de chasse de `documentation/sprites/bac-a-sable/chasse/`
  (`decor.py`, `chasse.py`, `page.html`).
- Outils du skill : `outils.py`, `planche.html` et `modele.py`, le script modèle à copier.

Tout ce qui suit vient des corrections de Léonard. L'appliquer d'emblée.

## Le principe : une étape par réponse, validée avant la suivante

| Étape | On juge | Interdit à ce stade | Livrable |
|---|---|---|---|
| 0 Cadrage | le mode, les photos, la taille, la destination | dessiner | un message, questions groupées |
| 1 Silhouette | morphologie, posture, proportions, tête, œil et pupille, membres en formes simples | couleur, ombre, détail | planche + superposition sur la photo |
| 2 Couleurs à plat | les couleurs justes et surtout les **motifs** | dégradé, ombre, lumière, reflet, effet mouillé, grain, tramage | planche |
| 3 Détails | volume, nuances, peau, œil, mains et pieds au pixel | changer une forme ou un motif sans qu'il le demande | planche |
| 4 Animation | respiration, clignement, tête, regard | saut, chant | planche animée |
| 5 Décor (facultatif) | 5a fiche et plan, 5b calques, 5c lumière, 5d vie | animer avant 5d | page de scène |

Les règles du passage d'une étape à l'autre :
- **Fin de chaque réponse, à partir de l'étape 1 :**
  - vérifier la planche au navigateur (Chromium sans interface, largeurs ordinateur et téléphone),
    puis la publier en artefact privé, ou la mettre à jour en republiant le même fichier pour
    garder la même URL ; à l'étape 5, c'est la page de scène ;
  - donner le lien et lister 3 à 6 points à vérifier ;
  - demander « Passe-t-on à l'étape N : <nom> ? », puis **s'arrêter**.
- **Ne jamais entamer l'étape suivante dans la même réponse**, même pour « montrer ce que ça
  donnerait ». Une silhouette n'a pas de couleur, des aplats n'ont pas d'ombre.
- **Réponse de Léonard :**
  - des corrections : les faire, republier, et rester à l'étape, autant de tours qu'il faut ;
  - un accord explicite (« parfait », « go », « on passe ») : passer à l'étape suivante ;
  - un doute : rester à l'étape.
  - Pour l'étape 3, il faut qu'il juge la grenouille **parfaite** avant d'animer.
- **Une correction de forme demandée à l'étape 2 ou plus tard** se fait dans `formes()`. Les
  rendus suivent. Le dire à Léonard et lui remontrer la silhouette si elle change beaucoup.
- **Où travailler :**
  - dans `documentation/sprites/bac-a-sable/<projet>/` (le dossier est versionné sur main), avec
    un commit par étape validée (« <espèce> : étape N validée ») ;
  - rien sur le tableau de bord (`documentation/tableau-de-bord/`) ni sur la page publiée avant
    son go explicite pour la destination.
- **Les photos de référence** restent dans le scratchpad : jamais dans le dépôt, jamais dans une
  page publiée. Les images de superposition contiennent la photo. Les montrer seulement dans la
  conversation (`SendUserFile` si l'outil existe), jamais en artefact.

## Étape 0 — Cadrage (premier message)

Reconnaître le mode :
- **Mode espèce**, par exemple « fais-moi <espèce> en 8-bit ».
  - Dessiner l'espèce dans la pose d'A. blanci : assise, de profil, tête à gauche.
  - Il faut 2 ou 3 photos, dont une de profil. S'il n'y en a pas, les demander : ne jamais
    dessiner de mémoire.
  - Relever la morphologie propre à l'espèce : museau, tympan, pupille, disques, palmures,
    longueur des membres, motifs, couleurs des faces cachées (cuisses, ventre).
- **Mode image**, par exemple « anime cette photo ».
  - La photo est le modèle, **telle quelle** : pose, angle, cadrage, nombre de grenouilles,
    leurs regards, le support (branche, feuille) et, pour l'étape 5, le reste de la scène
    (eau, reflet, fond flou).
  - On décalque, on ne réinvente pas l'anatomie. On identifie l'espèce seulement pour la nommer,
    pour savoir ce que la photo cache et que l'animation pourrait montrer, et pour le décor.
  - Exemple : deux rainettes-singes (*Phyllomedusa*) face à face sur une branche au-dessus de
    l'eau, avec leur reflet.

Puis, dans un seul message, une proposition par point (ne demander que ce qu'on ne peut pas
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

## Étape 1 — Silhouette

1. **Relever les formes** sur une grille :
   `outils.py grille PHOTO grille.png x0,y0,x1,y1 [pas]`.
   - Mode espèce : cadrer sur la grenouille. La grille fait 48 unités de large ; retourner la
     photo si l'animal regarde à droite.
   - Mode image : cadrer sur toute la zone du livrable, avec `pas = 2 / échelle` (pixels de
     photo par unité ; échelle 0,28 → pas de 7,1). Chaque grenouille a son propre repère :
     choisir son coin haut gauche sur la toile (en pixels pairs) et soustraire sa position, en
     unités, des relevés.
   - Lire en unités : bout du museau, crâne, **bosse de l'autre œil**, œil (centre, rayon,
     forme de la pupille), ligne de la bouche, tympan, dos, croupe, cuisse, genou, talon,
     tarse, orteils, épaule, coude, poignet, doigts, gorge, ventre.
2. **Copier `modele.py`** en `<code>.py` à côté du travail, une copie par grenouille, et
   remplir `formes()`.
   - **Le corps et la tête** sont des courbes de Catmull-Rom, tracées comme des polygones simples
     (un contour qui se croise se remplit mal).
   - **Les membres** sont des segments de rayon variable, d'articulation en articulation (`membre`).
   - **Les doigts et les orteils** sont présents dès cette étape : des traits d'un pixel, en
     nombre, direction et longueur justes, avec les disques en points. Ils deviendront des
     gabarits au pixel à l'étape 3.
   - **Chaque forme** a un plan :
     - `fond` : les membres de l'autre côté ;
     - `corps` ;
     - `devant` : les membres proches.
     On la cerne si un contour doit la séparer de ce qu'elle recouvre.
   - **La pupille** se règle avec `PUPILLE` : ronde, horizontale ou verticale (en fente).
   - **Une grenouille tournée vers la droite** se règle avec `MIROIR = True`. Les formes restent
     décrites tête à gauche (relever sur une copie retournée de la photo) et la lumière reste
     juste.
   - **Mode image, le support** est dessiné dès l'étape 1, puisque les doigts l'enserrent. Il
     prend un script à part : une seule forme, sans œil, matière « écorce » ou équivalente.
     Ce qui passe devant la grenouille (le bord d'une branche sous les doigts) va dans un
     second sprite, posé après elle.
3. **Rendre et vérifier :**
   - `python3 <code>.py silhouette` ;
   - `outils.py superposer <code>_silhouette.json PHOTO x0,y0,x1,y1 sup.png`, avec un cadre aux
     mêmes proportions que la toile : le contour cyan doit suivre la photo à 1 ou 2 pixels près ;
   - mode image : `outils.py composer scene.json LxH a.json@x,y b.json@x,y …`, puis superposer
     la composition sur le cadre de la photo.
4. **La planche :**
   `outils.py planche FICHIER.json planche.html "<Nom>" "<texte>" "Étape 1 · silhouette"`.
   Elle a une grille de pixels numérotée, pour que Léonard désigne un pixel (« rapproche de 3
   pixels »). La publier en artefact privé, toujours le même fichier, et montrer la
   superposition dans la conversation.

À vérifier avant de montrer :
- [ ] Le contour suit la photo. Proportions tête / corps / membres justes.
- [ ] La posture : angles du coude, du genou, du talon ; regard et écart entre les grenouilles
      (mode image).
- [ ] Le museau, la ligne de la bouche, le tympan s'il se voit. L'œil (taille, place, pupille)
      et la bosse de l'autre œil.
- [ ] Les plans : ce qui est devant ou derrière (tibia sur la cuisse, bras de l'autre côté).
- [ ] Le nombre et l'éventail des doigts et des orteils ; le contact avec le support.
- [ ] Rien ne touche les bords de la toile.

## Étape 2 — Couleurs à plat et motifs

- **Une couleur par matière** : dos, flanc, ventre, gorge, lèvre, membres, faces cachées,
  disques, iris, pupille, paupière, support.
  - Couleurs prises sur la photo, dans des zones bien éclairées, sans reflet ni ombre :
    `outils.py pipette PHOTO zone [zone …]` donne la couleur médiane de chaque zone.
- **Les motifs comptent le plus** : bandes, barres, taches, réticulations.
  - Leur forme, leur nombre, leur place et leur orientation viennent de la photo. Compter les
    barres sur chaque segment de membre et placer chaque tache.
  - En mode image, ce sont les motifs de **chaque individu** de la photo, pas ceux de l'espèce
    en général.
  - Les décrire dans `MOTIFS` et `formes()`.
- **Rien d'autre** : ni dégradé, ni ombre, ni lumière, ni reflet, ni effet mouillé, ni grain,
  ni tramage. Le contour reste sombre et uniforme.
- **Rendre :** `python3 <code>.py aplats`, puis la planche « Étape 2 · couleurs à plat ».

À vérifier :
- [ ] Les teintes, côte à côte avec la photo (panneaux 1 et 2 de `superposer`).
- [ ] Chaque motif à sa place, en bon nombre, dans le bon sens, des deux côtés s'il se voit
      (membres du fond).
- [ ] L'œil : iris, pupille, cercle ; la lèvre ; les faces cachées qui se voient.

## Étape 3 — Détails

Partir du rendu `details` du modèle :
- un volume par forme, avec la lumière d'en haut à gauche, ou celle de la photo en mode image
  (`LUMIERE`) ;
- des rampes de 7 tons par matière, en trois variantes de teinte réparties par plaques ;
- un grain léger ;
- un œil brillant ;
- un contour plus clair côté lumière.

Puis l'affiner à la main, avec les règles générales ci-dessous et, en pose assise de profil,
les règles d'A. blanci (`grenouille.py` en est la référence complète). L'étape demande
plusieurs passes. Se relire seul avant de montrer :
- `outils.py apercu FICHIER.json a.png CLES 6 PHOTO` ;
- `outils.py zoom` sur l'œil, l'épaule, le genou et le tibia, les mains ;
- `outils.py controle` : aucune miette (groupe de moins de 12 pixels détaché par le contour).

Les fonctions d'A. blanci se reprennent telles quelles quand il le faut : gabarits des mains et
des pieds et `poser`, `nuancer`, `pustules`, `retoucher`.

### Règles générales (toutes espèces, toutes poses)

- **Grille et contour :** 2 pixels par unité, pas d'anticrénelage, un contour sombre d'un pixel
  autour de la silhouette.
- **Allure :** fine et vivante, jamais un crapaud ni « en surpoids » (sauf si l'espèce l'est).
- **Mains et pieds : des gabarits posés au pixel près**, jamais des formes calculées, qui donnent
  des pâtés et des doigts collés.
  - Doigts d'un pixel en éventail, avec un pixel vide entre deux doigts.
  - Diagonales **en escalier** : chaque pixel touche le suivant par un côté, sinon le contour
    le coupe et le doigt paraît détaché.
  - Disque au bout, plus clair que le doigt.
  - Couleurs de la peau, jamais un beige à part.
- **Ombre de contact** (par exemple le flanc derrière un bras) : le pixel de base, même teinte,
  assombri et un peu grisé. Jamais des pixels d'une autre matière.
- **Nuances :** flou limité à chaque matière, puis une rampe fine (7 à 12 niveaux) et trois
  variantes de teinte par plaques. On vise une centaine de couleurs. Léonard a trouvé ce rendu
  « bien bien mieux » que les seuls tons de base.
- **Grain et pustules :** un grain doux, d'un seul cran de rampe, sans points blancs vifs.
  Les pustules sont un pixel de la couleur d'origine, plus clair, avec son ombre dessous, posés
  après les nuances, à des places fixes d'une image à l'autre. Jamais de pixel blanc.
- **Reflets humides :** seulement là où la photo en montre (coude, cuisse, genou, disques).
- **Œil :**
  - le globe est bombé, avec un reflet blanc et un petit reflet bleuté ;
  - la paupière supérieure est en relief et le globe a un pli sombre dessous ;
  - l'œil fermé est de la peau qui couvre le globe, avec une fente en arc, sans cercle de couleur
    autour (sinon « effet lunettes ») ;
  - l'iris et la pupille sont ceux de l'espèce : fente verticale et iris argenté réticulé chez
    *Phyllomedusa*, iris doré et pupille horizontale chez beaucoup d'autres.
- **L'autre œil :** il se voit comme une simple bosse de peau sur le crâne. Il n'a ni pupille,
  ni cercle, et il ne cligne jamais.

### Règles d'A. blanci (pose assise, de profil)

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
  - **l'épaule s'attache en arrondi, sans aucun trait** : un capuchon dont le pourtour tourne au
    jaune et se fond dans le flanc en tramage ;
  - le bras de l'autre côté se voit, plus fin et plus sombre, sa main plus haut et en retrait.
- **Bande du dos :** elle passe derrière la cuisse au lieu de s'arrêter d'elle-même.

À vérifier :
- [ ] Le volume se lit et la lumière vient d'un seul côté.
- [ ] Les doigts sont détachés et en éventail ; `controle` ne trouve aucune miette.
- [ ] L'œil est vivant et l'œil fermé crédible ; les motifs de l'étape 2 sont intacts.
- [ ] On compte environ 80 à 120 couleurs ; `ruff check` et `ruff format` passent.

## Étape 4 — Animation

- **La base :** `python3 <code>.py animation` donne 24 images (gorge × flanc × œil × tête),
  avec les clés `"{gorge}{flanc}{œil}{tête}"`, tête droite = 1.
- **Les rythmes**, à 12 images par seconde :
  - flanc gonflé 1,3 s toutes les 3,2 s ;
  - gorge qui palpite (0,3 s sur 0,6 s) pendant 2,4 s, toutes les 7 s ;
  - clignement de 0,13 s, toutes les 3 à 7,5 s, parfois double ;
  - tête baissée ou relevée d'un cran (0,06 rad autour du cou, en s'estompant vers le corps)
    pendant 1,6 à 4 s, toutes les 4 à 9 s.
- **Jamais de saut, jamais de chant** : pas de sac vocal gonflé, pas de notes.
- **Pour une autre pose**, garder le principe : respiration, clignement, regard. Par exemple, une
  grenouille agrippée à une branche bouge la tête mais pas les mains.
  - Vérifier sur les photos comment l'espèce cligne : certaines ont une paupière inférieure
    dorée réticulée.
- **Plusieurs grenouilles :** chacune a ses propres rythmes, décalés (la planche le fait). Leurs
  regards suivent l'image : deux grenouilles face à face se regardent, se détournent un instant
  et reviennent.
- **Les poses de comportement** (bouche ouverte, yeux rentrés pour avaler, regard vers une
  proie) s'ajoutent en réglages de `formes()`. Avec tous les réglages à zéro, l'image doit être
  la validée, pixel pour pixel : le vérifier par différence.
- **Mouvement réduit** (`prefers-reduced-motion`) : image fixe.
- **La planche** d'animation est « Étape 4 · animation ». Autour de la grenouille, sur la planche
  comme sur le bandeau, une mosaïque de tuiles ambrées de 2 × 2 scintille, plus vive près d'elle.

## Étape 5 — Décor (facultatif, sur demande), elle aussi par sous-étapes

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

En mode image, ces calques reprennent ceux de la photo : fond flou, eau, branche. Le reflet est
un calque calculé depuis les sprites (miroir, assombri, teinté), animé seulement en 5d.

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

Ce qui a marché dans la scène de chasse :
- toile affichée à une échelle entière, encadrée par une ombre et non une bordure ;
- moucherons en trois pixels, avec des ailes qui battent et qui accrochent la lune la nuit ;
- un morpho de 13 × 7 pixels qui traverse parfois la scène et qu'elle suit des yeux ;
- une fougère qui s'égoutte sur son dos après l'averse.

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
  `documentation/sprites/evolution-grenouille/`.

## Outils (`python3 .claude/skills/sprite-grenouille/outils.py …`)

| Commande | Usage |
|---|---|
| `grille PHOTO SORTIE.png [x0,y0,x1,y1] [pas]` | grille en unités sur la photo, pour relever les formes |
| `superposer FICHIER.json PHOTO x0,y0,x1,y1 SORTIE.png [zoom] [CLE]` | photo, sprite, et contour du sprite sur la photo (cadre aux proportions de la toile) |
| `pipette PHOTO zone [zone …]` | couleur médiane de chaque zone : les aplats de l'étape 2 |
| `apercu SPRITE.json SORTIE.png CLES [zoom] [PHOTO]` | images côte à côte |
| `zoom SPRITE.json SORTIE.png CLE x0,y0,x1,y1` | un détail pixel par pixel |
| `controle SPRITE.json` | image par image : formes cernées à part, et miettes (moins de 12 pixels) à corriger |
| `composer SORTIE.json LxH SPRITE.json@x,y[,m] …` | plusieurs sprites sur une toile (`m` : retourné, lumière comprise) |
| `planche FICHIER.json SORTIE.html NOM "TEXTE" "ÉTAPE"` | planche de relecture (sprite ou composition, statique ou animée, grille de pixels) |

`modele.py` (`python3 <code>.py silhouette|aplats|details|animation`) écrit
`<NOM>_<étape>.json`. Sa grenouille de démonstration ne sert qu'à vérifier la chaîne :
remplacer `NOM`, le repère (`UNITES`, `PIVOT`), `formes()`, `MATIERES`, `MOTIFS`, `OEIL` et
`PUPILLE`, et régler `MIROIR` et `LUMIERE`.
