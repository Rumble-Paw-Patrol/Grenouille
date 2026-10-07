---
name: sprite-grenouille
description: Dessine une espèce de grenouille en sprite 8-bit animé, dans le style d'A. blanci du tableau de bord (96 × 80 pixels, ~80 couleurs nuancées, vue de profil, respire, cligne, bouge un peu la tête, mosaïque de pixels ambrés), d'après des photos de l'espèce, en une seule passe de relecture. À utiliser quand Léonard demande « fais-moi <espèce> en 8-bit », « un sprite de <espèce> comme A. blanci », « /sprite-grenouille », ou veut une autre grenouille animée pour le tableau de bord, une prez ou une page.
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
   (cadrer sur la grenouille ; une case = une unité du repère 48 × 40, tête à gauche — retourner
   la photo si l'animal regarde à droite). Lire sur la grille, en unités : bout du museau, haut
   du crâne et **bosse de l'autre œil**, centre et rayon de l'œil, ligne du dos jusqu'à
   l'arrière-train, cuisse (centre, rayons), genou, talon, pointe des orteils, épaule, coude,
   poignet, doigts, mâchoire, gorge, bas du ventre, trajet de la bande ou des motifs.
2. **Copier `grenouille.py`** en `<code_espece>.py` à côté de sa destination (même structure :
   `catmull`, `dans`, `ellipse`, `segment`, `image()`, `nuancer()`, `plages`) et remplacer
   les points, la palette de base et les rampes de `MATIERES` par ceux de l'espèce. Sortie :
   `<code_espece>.json`, même format (palette, largeur, hauteur, images codées par plages).
3. **Dessiner en respectant les règles ci-dessous**, puis se relire seul (étape 5) avant de
   montrer quoi que ce soit.
4. **Animer** comme A. blanci (le code de la page est déjà prêt, voir Intégration).
5. **Relecture interne** : `outils.py apercu SPRITE.json a.png 0001,0011,1101,0000,0002 6 PHOTO`
   (repos, œil fermé, gorge + flanc, tête baissée, tête relevée, à côté de la photo), puis
   `outils.py zoom` sur l'œil, l'épaule, le genou et le tibia, la main. Passer la liste de
   contrôle point par point ; corriger ; recommencer jusqu'à ce que tout passe.
6. **Une seule planche** : `outils.py planche SPRITE.json planche.html "<Nom>" "<texte>"`,
   vérifier une fois au navigateur (Chromium, téléphone et ordinateur), publier en artefact
   privé, donner le lien et attendre le go. Rien sur le tableau de bord avant le go.

## Règles de dessin (tirées des corrections de Léonard)

Repère 48 × 40 unités tracé à 2 pixels par unité (96 × 80), formes en courbes de
Catmull-Rom, sans anticrénelage, contour sombre d'un pixel autour de la silhouette.

- **Allure** : une grenouille fine et élégante, jamais un crapaud. Profil tourné vers le
  titre (à gauche), posture assise redressée : tête haute, **arrière-train posé presque au
  sol**, genoux bien pliés.
- **Tête** : museau court et arrondi, narine, ligne de lèvre claire. **L'autre œil se voit
  comme une simple bosse de peau sur le crâne**, éclairée par-dessus, ombrée à sa base : ni
  pupille, ni cercle, ni paupière, et elle ne cligne jamais.
- **Œil visible** : grand, noir, cercle doré en haut et gris en bas, bas du globe un peu moins
  noir, un reflet blanc et un petit reflet bleuté, paupière supérieure en relief éclairée,
  pli sombre sous le globe.
- **Paupière fermée** : la peau couvre le globe (haut orangé éclairé), fente sombre en arc,
  paupière basse pâle ; le cercle doré disparaît (sinon effet « lunettes »).
- **Ordre de superposition** : pied → corps → cuisse → **tibia par-dessus la cuisse** (pli
  sombre là où il la recouvre, bord du haut éclairé, reflet au genou) → bras → bosse de
  l'autre œil → œil. Le ventre, même gonflé, passe toujours **derrière** les pattes.
- **Patte arrière** : repliée en Z — cuisse ronde, tibia du genou (avant-bas de la cuisse)
  au talon, pied à plat vers l'avant, orteils fins à pelotes claires.
- **Patte avant** : longue, coude en arrière, doigts fins écartés à pelotes claires. **Elle
  sort du flanc sans aucun trait** : sur ~3 unités sous l'épaule, tramage ordonné (Bayer 4 × 4)
  entre flanc et bras ; contour seulement là où le bras sort du corps. Lumière d'en haut à
  gauche : bord avant éclairé, ombre du bord arrière seulement sous l'épaule.
- **Couleurs** : dessiner en tons de base (une vingtaine), puis `nuancer()` : par matière
  (peau, motifs, flanc, gorge et ventre), flou limité à la matière puis rampe fine de 7 à 12
  niveaux, en 3 variantes de teinte réparties par plaques (plus rouge, neutre, plus dorée),
  grain conservé — environ 80 couleurs. Léonard a jugé cette version « bien bien mieux » :
  ne jamais livrer la version à tons de base seule.
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
  dans `.perchoir`, perché sur le spectrogramme, sous le titre. Pour changer d'espèce :
  remplacer `grenouille.json` ; pour en ajouter une : second canvas et seconde clé de
  données, sur le même modèle. Publier selon le skill `tableau-de-bord` ; si la version en
  ligne a été republiée par une autre session depuis la dernière lecture, repartir de
  la version en ligne et n'y remplacer que les données du sprite.
- **Autre page** : reprendre le JavaScript de `planche.html` (décodage par plages, boucle,
  mosaïque).
- Vérifier le bandeau à 1440, 1280, 1100 et 400 px, thèmes clair et sombre : la grenouille
  tient dans le bandeau, l'œil n'est pas caché par le texte, rien ne déborde ; sur téléphone
  elle a sa propre ligne.

## Liste de contrôle avant la planche

- [ ] Silhouette fidèle à la photo (superposer mentalement l'aperçu et la photo).
- [ ] Arrière-train presque au sol, genoux pliés, patte arrière en Z.
- [ ] Bosse de l'autre œil, sans détail d'œil, identique quand l'œil visible cligne.
- [ ] Œil bombé et brillant ; paupière fermée crédible.
- [ ] Tibia devant la cuisse ; ventre gonflé derrière les pattes (vérifier `0101` et `1101`).
- [ ] Épaule : aucun trait horizontal ni vertical (zoom).
- [ ] Doigts et orteils fins, pelotes claires ; couleurs des pattes nuancées.
- [ ] ~80 couleurs après `nuancer()`, plaques de teinte visibles mais douces.
- [ ] Animation : pas de saut, pas de chant, tête qui bouge à peine.
- [ ] `ruff check` et `ruff format` propres ; crédit des photos dans la docstring.
