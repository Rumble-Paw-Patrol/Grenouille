# *Callimedusa tomopterna*, grenouille de droite — fiche morphologique et cadrage

La fiche de l'espèce est `.claude/skills/sprite-grenouille/fiches/callimedusa-tomopterna.md`,
validée par Léonard le 10/10/2026 et complétée avec les leçons de la grenouille de gauche (mains,
palmure, disques) : elle n'est pas recopiée ici. Ce fichier ajoute ce que la photo d'Olivier
Louguet montre de cet individu, et le cadrage.

Point de départ : la grenouille de droite dessinée par la session du 8/10/2026 (branche
`claude/trusting-einstein-35jvgu`, `bac-a-sable/tomopterna/tomopterna.py`, `formes_droite`) :
ses contours, sa lecture des membres et ses motifs relevés servent de premier jet, repris étape
par étape avec le skill.

## Cet individu, d'après la photo

| Caractère | Sur la photo | Pour le sprite |
|---|---|---|
| Orientation | **de profil, tournée vers la gauche**, vers l'autre grenouille ; on voit son flanc gauche | membres gauches (`_g`) devant, droits (`_d`) de l'autre côté |
| Tête | museau tronqué, plus long qu'on ne croit ; narine nette au bout, en bourrelet ; dos du crâne presque droit jusqu'à la nuque ; tête petite pour l'œil | profil en droites, comme la grenouille de gauche |
| Œil | très gros et globuleux, globe gris argenté, cercle noir, pupille en fente verticale en amande ; paupière supérieure verte en bourrelet ; il dépasse nettement le crâne | à mesurer à l'étape 1 (environ 3 unités de rayon) |
| Autre œil | une petite bosse verte en avant de l'œil, au-dessus du museau | une bosse de peau, sans globe visible (à confirmer) |
| Tympan | un disque vert rond, net, derrière l'œil et en dessous | un trait rond, comme sur la grenouille de gauche |
| Lèvre et mâchoire | un liseré crème, du bout du museau ; la ligne de la mâchoire va jusqu'au-dessus de l'épaule ; des mouchetures sombres dessous | une droite ; mouchetures en 3a |
| Gorge et poitrine | blanc lilas, sous la tête | — |
| Flanc | orangé, granuleux, barré et taché de violet-noir (tigré) ; une rangée de granules blanches au bord du vert du dos | — |
| Dos | vert, lisse, en pente douce de la nuque à la croupe, au-dessus de la patte arrière | — |
| Bras gauche (devant) | le bras, fin, part à l'horizontale de l'épaule, loin à l'avant de la grenouille, jusqu'au coude, bien visible, qui ne touche pas le corps ; l'avant-bras, plus épais, vert dehors, liseré de granules blanches, descend à la main sur le dessus de la branche | 4 doigts : 3 bien visibles, à grands disques, et un discret qui monte sur la branche, derrière l'avant-bras |
| Bras droit (de l'autre côté) | de la main, sur la branche à gauche, l'avant-bras orangé et barré part en arrière-plan vers la droite et un peu vers le haut jusqu'au coude, sous la poitrine ; le bras remonte vers le haut et la gauche jusqu'à l'épaule droite, une très légère bosse sur le corps | 4 doigts : 3 visibles, et un qui part vers la droite le long de la branche, jusqu'à presque toucher le doigt du haut de la main gauche |
| Patte arrière gauche (devant) | repliée à l'arrière : le tibia, vertical, barré, du genou, en haut, au pied sur la branche ; la cuisse, qu'on voit à peine, monte à la verticale depuis derrière la branche, derrière le tibia mais devant le postérieur : sa face interne, jaune à rayures foncées horizontales, à gauche du tibia, et un peu de vert à sa droite ; 3 orteils en éventail sur la branche | 3 orteils sur 5 (à compter à l'étape 1) |
| Patte arrière droite (de l'autre côté) | **la masse orangée sous la branche est le pied droit** (sous la grenouille de gauche, c'était une feuille) : il part en diagonale vers le bas et la gauche, à l'ombre ; la plante, claire, est tournée vers le haut ; un orteil monte derrière la branche, un part vers le haut et la droite derrière elle, un à l'horizontale vers la droite, dans le vide ; un tibia, en arrière-plan, le relie à la branche | 3 orteils ; le tibia derrière le pied |
| Coudes et avant-bras | coudes bien visibles ; l'avant-bras plus épais que le bras, qui est fin | — |
| Doigts et orteils | épais, à grands disques ; ceux qui s'enroulent sous la branche montrent leur disque par la tranche | doigts d'environ 3 pixels, disques de 6 à 7, chacun de sa forme |

## Cadrage

- **Mode :** image. La même photo d'Olivier Louguet ; seulement la grenouille de droite, à la
  demande de Léonard, après la grenouille de gauche (`../tomopterna-gauche/`).
- **Livrable :** la grenouille et le bout de branche sous elle (la branche est dessinée dès
  l'ébauche, les doigts l'enserrent). Destination : « pas dans le projet, je fais ça pour le
  plaisir » (Léonard, 8/10) ; rien hors du bac à sable.
- **Taille :** la même échelle que la grenouille de gauche, 0,28 : une unité vaut 7,14 pixels de
  photo. Le sprite couvre le rectangle de 55 × 45 unités qui commence en (118, 23) dans le
  repère de la scène, soit 110 × 90 pixels, comme la grenouille de gauche ; la grenouille y fait
  environ 92 pixels de large et 67 de haut (avec ce qui pend sous la branche). Dans le même
  repère que la grenouille de gauche, les deux se posent ensemble sur une toile, à leur place.
- **Crédit :** photo d'Olivier Louguet, dans les docstrings et sur la planche.
