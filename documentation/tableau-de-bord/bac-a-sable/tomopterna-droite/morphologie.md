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
| Tête | museau court, tronqué ; narine nette au bout, en bourrelet ; dos du crâne presque droit jusqu'à la nuque | profil en droites, comme la grenouille de gauche |
| Œil | gros, globe gris argenté, cercle noir, pupille en fente verticale en amande ; paupière supérieure verte en bourrelet ; il dépasse nettement le crâne | à mesurer à l'étape 1 (environ 3 unités de rayon) |
| Autre œil | une petite bosse verte en avant de l'œil, au-dessus du museau | une bosse de peau, sans globe visible (à confirmer) |
| Tympan | un disque vert rond, net, derrière l'œil et en dessous | un trait rond, comme sur la grenouille de gauche |
| Lèvre | un liseré crème, du bout du museau jusque sous le tympan ; des mouchetures sombres dessous | une droite ; mouchetures en 3a |
| Gorge et poitrine | blanc lilas, sous la tête | — |
| Flanc | orangé, granuleux, barré et taché de violet-noir (tigré) ; une rangée de granules blanches au bord du vert du dos | — |
| Dos | vert, lisse, en pente douce de la nuque à la croupe, au-dessus de la patte arrière | — |
| Bras gauche (devant) | vert dehors, de l'épaule, sous le bord du dos, à la main sur le dessus de la branche ; liseré de granules blanches ; 3 doigts visibles, à grands disques, deux qui s'enroulent sous la branche | 3 doigts sur 4 (le pouce caché ?) |
| Bras droit (de l'autre côté) | sort de sous la gorge, orangé et barré, granuleux dessous ; descend en diagonale vers la gauche jusqu'à la main sur la branche ; 3 doigts visibles, dont un vers la gauche sur la branche | 3 doigts sur 4 |
| Patte arrière gauche (devant) | repliée à l'arrière : la cuisse, orangée et barrée (« tiger-leg »), entre le bras et le pied ; le talon levé, granuleux, en haut ; le tarse redescend à la verticale jusqu'à la branche, bordé de vert dehors ; 3 orteils en éventail sur la branche | 3 orteils sur 5 |
| Patte arrière droite (de l'autre côté) | cachée ; sous la branche pend une forme orangée en éventail, terminée par un disque : son pied, ou une feuille ? (sous la grenouille de gauche, c'était une feuille) | question à Léonard |
| Doigts et orteils | épais, à grands disques ; ceux qui s'enroulent sous la branche montrent leur disque par la tranche | doigts d'environ 3 pixels, disques de 6 à 7, chacun de sa forme |

## Cadrage

- **Mode :** image. La même photo d'Olivier Louguet ; seulement la grenouille de droite, à la
  demande de Léonard, après la grenouille de gauche (`../tomopterna-gauche/`).
- **Livrable :** la grenouille et le bout de branche sous elle (la branche est dessinée dès
  l'ébauche, les doigts l'enserrent). Destination à décider ; rien hors du bac à sable.
- **Taille :** la même échelle que la grenouille de gauche, 0,28 : une unité vaut 7,14 pixels de
  photo. Le sprite couvre le rectangle de 55 × 45 unités qui commence en (118, 23) dans le
  repère de la scène, soit 110 × 90 pixels, comme la grenouille de gauche ; la grenouille y fait
  environ 92 pixels de large et 67 de haut (avec ce qui pend sous la branche). Dans le même
  repère que la grenouille de gauche, les deux se posent ensemble sur une toile, à leur place.
- **Crédit :** photo d'Olivier Louguet, dans les docstrings et sur la planche.
