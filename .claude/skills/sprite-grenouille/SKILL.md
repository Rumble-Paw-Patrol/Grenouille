---
name: sprite-grenouille
description: Dessine une espèce de grenouille en sprite 8-bit animé, dans le style d'A. blanci du tableau de bord (2 pixels par unité, A. blanci fait 114 × 86 pixels, ~80 couleurs nuancées, vue de profil, assise, respire, cligne, bouge un peu la tête, mosaïque de pixels ambrés), d'après des photos de l'espèce, en une seule passe de relecture. À utiliser quand Léonard demande « fais-moi <espèce> en 8-bit », « un sprite de <espèce> comme A. blanci », « /sprite-grenouille », ou veut une autre grenouille animée pour le tableau de bord, une prez ou une page.
---

# Sprite 8-bit d'une grenouille

Référence : `documentation/tableau-de-bord/grenouille.py` (A. blanci) et son historique,
visible pas à pas dans `documentation/tableau-de-bord/evolution-grenouille/index.html`.
Tout ce qui suit est la liste de ce que Léonard a corrigé sur A. blanci : l'appliquer
**d'emblée** pour qu'il n'ait qu'une planche à juger.

## Entrées

- Le nom de l'espèce, et 2 ou 3 photos d'elle, dont au moins une **de profil**, plus si
  possible une d'une espèce proche. Si aucune photo n'est jointe, la demander : ne jamais
  dessiner de mémoire.
- Les auteurs des photos (crédit dans la docstring et la planche). Les photos restent dans
  le bac à sable : jamais dans le dépôt, jamais dans une page publiée.
- La destination : bandeau du tableau de bord (remplace A. blanci ou s'ajoute ?), prez,
  autre page. Si ce n'est pas dit, faire la planche et demander à la fin.

## Méthode

1. **Relever la silhouette sur la photo de profil.**
   `uv run python .claude/skills/sprite-grenouille/outils.py grille PHOTO grille.png x0,y0,x1,y1`
   (cadrer sur la grenouille ; une case = une unité du repère, tête à gauche — retourner
   la photo si l'animal regarde à droite). Lire sur la grille, en unités : bout du museau, haut
   du crâne et **bosse de l'autre œil**, centre et rayon de l'œil, ligne du dos jusqu'à
   l'arrière-train, cuisse (centre, rayons), genou, talon, pointe des orteils, épaule, coude,
   poignet, doigts, mâchoire, gorge, bas du ventre, trajet de la bande ou des motifs.
2. **Copier `grenouille.py`** en `<code_espece>.py` à côté de sa destination (même structure :
   `catmull`, `dans`, `ellipse`, gabarits et `poser`, `separer`, `miettes`, `image()`,
   `nuancer()`, `plages`) et remplacer les points, les gabarits des mains et des pieds, la
   palette de base et les rampes de `MATIERES` par ceux de l'espèce. Sortie :
   `<code_espece>.json`, même format (palette, largeur, hauteur, images codées par plages).
   Le repère (`UNITES`) se règle à la taille de la grenouille : rien ne doit toucher les bords
   de l'image, ni le talon à droite ni les doigts au sol (une rangée de contour sous eux).
3. **Dessiner en respectant les règles ci-dessous**, puis se relire seul (étape 5) avant de
   montrer quoi que ce soit.
4. **Animer** comme A. blanci (le code de la page est déjà prêt, voir Intégration).
5. **Relecture interne** : `outils.py apercu SPRITE.json a.png 0001,0011,1101,0000,0002 6 PHOTO`
   (repos, œil fermé, gorge + flanc, tête baissée, tête relevée, à côté de la photo), puis
   `outils.py zoom` sur l'œil, l'épaule, le genou et le tibia, la main, et
   `outils.py controle SPRITE.json` (aucun pixel détaché). Passer la liste de contrôle point
   par point ; corriger ; recommencer jusqu'à ce que tout passe.
6. **Une seule planche** : `outils.py planche SPRITE.json planche.html "<Nom>" "<texte>"`,
   vérifier une fois au navigateur (Chromium, téléphone et ordinateur), publier en artefact
   privé, donner le lien et attendre le go. Rien sur le tableau de bord avant le go.

## Règles de dessin (tirées des corrections de Léonard)

Repère tracé à 2 pixels par unité (A. blanci : 57 × 43 unités, 114 × 86 pixels), formes en
courbes de Catmull-Rom tracées comme des polygones simples (un contour qui se croise se
remplit mal), sans anticrénelage, contour sombre d'un pixel autour de la silhouette.

- **Allure** : une grenouille fine et élégante, jamais un crapaud ni « en surpoids » :
  ventre haut (du vide sous lui entre le bras et la patte), dos peu incliné. Profil tourné
  vers le titre (à gauche), posture assise, accroupie, tête haute.
- **Tête** : museau court et arrondi, narine, ligne de lèvre claire. **L'autre œil se voit
  comme une simple bosse de peau sur le crâne**, éclairée par-dessus, ombrée à sa base : ni
  pupille, ni cercle, ni paupière, et elle ne cligne jamais.
- **Œil visible** : grand, noir, cercle doré en haut et gris en bas, bas du globe un peu moins
  noir, un reflet blanc et un petit reflet bleuté, paupière supérieure en relief éclairée,
  pli sombre sous le globe.
- **Paupière fermée** : la peau couvre le globe (haut orangé éclairé), fente sombre en arc,
  paupière basse pâle ; le cercle doré disparaît (sinon effet « lunettes »).
- **Ordre de superposition** : bras de l'autre côté → corps → cuisse → tarse et orteils →
  **tibia par-dessus la cuisse** (pli sombre là où il la recouvre, pas au talon) et talon →
  bras proche → main proche → bosse de l'autre œil → œil. Le ventre, même gonflé, passe
  toujours **derrière** les pattes.
- **Patte arrière, en Z de grenouille assise** : cuisse arrondie, tournée vers nous, de la
  hanche au genou ; on n'en voit que le dessus. **Tibia** plus gros que la cuisse, long,
  bombé surtout sur le dessus (le dessous presque droit), couché par-dessus la cuisse, du
  genou (levé, à l'avant) au talon, en descendant très légèrement, et qui **dépasse derrière la
  croupe**. La croupe s'arrête derrière le tibia : rien ne descend entre le pied et le tibia.
  **Talon** arrondi, épais (~5 pixels), aligné sur le bout du tibia, jamais en saillie ni pincé.
  Tarse qui revient du talon vers l'avant, au sol, séparé du tibia par un liseré de fond.
- **Pattes avant** : bras pliés, coude ouvert (~130°) qui dépasse un peu sous le ventre sans
  pointe exagérée ; avant-bras un peu renflé en son milieu, plus fin au coude et au poignet ;
  le poignet arrive sur le **talon de la paume** (en bas, à l'arrière de la main). **L'épaule
  s'attache en arrondi, sans aucun trait** : un capuchon un peu plus haut que large coiffe le
  haut du bras, son pourtour tourne au jaune et se fond dans le flanc en tramage ordonné
  (Bayer 4 × 4) ; contour seulement là où le bras sort du corps. Lumière d'en haut à gauche : bord
  avant éclairé, ombre du bord arrière. **Le bras de l'autre côté se voit** : même épaule, en
  arrière-plan, même profil en plus fin et plus sombre, cerné d'un contour là où le bras
  proche passe devant (effacer les restes isolés entre la gorge et le bras) ; sa main plus
  haut (plus loin) et en retrait vers l'arrière, sans toucher la main de devant.
- **Mains et pieds : gabarits posés au pixel près**, jamais des formes calculées (elles
  donnent des pâtés et des doigts collés). Paume étroite, pas ronde ; trois doigts d'un pixel
  qui partent de la paume, en éventail vers l'avant (diagonale haute, tout droit, diagonale
  basse) ; orteils longs, en éventail eux aussi. Diagonales **en escalier** : chaque pixel
  touche le suivant par un côté, sinon le contour les coupe et le doigt paraît détaché. Un
  pixel vide entre deux doigts (le contour le noircit). Disque de 2 × 2 au bout, plus clair
  que le doigt, reflet bleuté de peau humide. Couleurs de la peau, jamais de beige à part.
- **Couleurs** : dessiner en tons de base (une vingtaine), puis `nuancer()` : par matière
  (peau, motifs, flanc, gorge et ventre), flou limité à la matière puis rampe fine de 7 à 12
  niveaux, en 3 variantes de teinte réparties par plaques (plus rouge, neutre, plus dorée),
  grain conservé — environ 80 couleurs. Léonard a jugé cette version « bien bien mieux » :
  ne jamais livrer la version à tons de base seule.
- **Ombres de contact** (flanc derrière le bras) : le pixel de base, même teinte, assombri et
  un peu grisé ; jamais des pixels d'une autre matière (brun foncé sur le flanc).
- **Bande du dos** : elle ne s'arrête pas d'elle-même ; le haut de la cuisse remonte juste
  assez vers la croupe pour la cacher, sans monter jusqu'au dos.
- **Grain du ventre** : semé (environ un pixel sur quatre) mais doux, un seul cran de rampe
  plus clair ou plus sombre que le flanc alentour ; pas de points blancs vifs.
- **Pustules** : une quarantaine, sur tout le corps sauf la tête (ni mains, ni pieds, ni bras
  du fond) : un pixel de la couleur d'origine, plus clair, et son ombre juste en dessous,
  posés après les nuances, espacés, à des places tirées une fois (elles ne bougent pas
  d'une image à l'autre). Jamais de pixel blanc.
- **Détails** : grain de peau léger (pas de bruit sale), mouchetures, bords de bande
  effilochés, gorge mouchetée, barres discrètes sur la cuisse si l'espèce en a, reflets de
  peau humide (coude, cuisse, genou). Les motifs propres à l'espèce (bandes, taches,
  couleurs vives, ventre) viennent des photos, pas d'A. blanci.

## Animation

24 images : gorge (repos, gonflée) × flanc (repos, gonflé) × œil (ouvert, fermé) × tête
(baissée, droite, relevée : rotation de ±0,06 rad autour du cou, qui s'estompe vers le corps,
pattes immobiles). Clé `"{gorge}{flanc}{œil}{tête}"`, tête droite = 1. À 12 images/s :
flanc gonflé 1,3 s toutes les 3,2 s ; gorge qui palpite (0,3 s sur 0,6 s) pendant 2,4 s
toutes les 7 s ; clignement de 0,13 s toutes les 3 à 7,5 s (parfois double) ; tête baissée ou
relevée 1,6 à 4 s, toutes les 4 à 9 s. **Jamais de saut, jamais de chant** (pas de sac vocal
gonflé en continu, pas de notes). Autour : mosaïque de tuiles ambrées 2 × 2 qui scintillent,
plus vives près de la grenouille. `prefers-reduced-motion` : image fixe.

## Intégration

- **Tableau de bord** : le bloc « A. blanci en 8-bit » de `modele.html` lit
  `D.grenouille` (rempli par `construire.py` depuis `grenouille.json`), canvas `#grenouille`
  dans `.perchoir`, perché sur le spectrogramme, sous le titre. La toile prend la taille du
  sprite (32 pixels de plus en largeur, 8 en hauteur) et se décale vers la droite d'autant que
  le sprite dépasse 96 pixels, pour que la tête reste au même endroit. Pour changer d'espèce :
  remplacer `grenouille.json` ; pour en ajouter une : second canvas et seconde clé de
  données, sur le même modèle. Publier selon le skill `tableau-de-bord` ; si la version en
  ligne a été republiée par une autre session depuis la dernière lecture, repartir de
  la version en ligne et n'y remplacer que les données du sprite.
- **Autre page** : reprendre le JavaScript de `planche.html` (décodage par plages, boucle,
  mosaïque).
- Ajouter les versions validées à `documentation/tableau-de-bord/evolution-grenouille/`
  (une étape par version, sprite régénéré depuis son commit ; brouillons en PNG légers).
- Vérifier le bandeau à 1440, 1280, 1100 et 400 px, thèmes clair et sombre : la grenouille
  tient dans le bandeau, l'œil n'est pas caché par le texte, rien ne déborde ; sur téléphone
  elle a sa propre ligne.

## Liste de contrôle avant la planche

- [ ] Silhouette fidèle à la photo (superposer mentalement l'aperçu et la photo), fine.
- [ ] Patte arrière en Z : tibia par-dessus la cuisse, qui dépasse derrière la croupe, talon
      aligné, rien entre le pied et le tibia.
- [ ] Bosse de l'autre œil, sans détail d'œil, identique quand l'œil visible cligne.
- [ ] Œil bombé et brillant ; paupière fermée crédible.
- [ ] Tibia devant la cuisse ; ventre gonflé derrière les pattes (vérifier `0101` et `1101`).
- [ ] Épaule : aucun trait horizontal ni vertical (zoom). Coudes pliés, poignet sur le talon
      de la paume, bras de l'autre côté visible, sa main plus haut et en retrait.
- [ ] Doigts et orteils d'un pixel en éventail, disques clairs ; `outils.py controle` : pas
      d'autre groupe que le bras de l'autre côté.
- [ ] Rien contre les bords de l'image.
- [ ] ~80 couleurs après `nuancer()`, plaques de teinte visibles mais douces.
- [ ] Animation : pas de saut, pas de chant, tête qui bouge à peine.
- [ ] `ruff check` et `ruff format` propres ; crédit des photos dans la docstring.
