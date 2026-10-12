# Scène « Tomopterna au-dessus de l'eau » : décisions de Léonard

La scène animée de la photo d'Olivier Louguet (`scene.py`, `scene.html`, `decor.py`,
`branche.py`), où se posent les deux grenouilles dessinées pas à pas (`../tomopterna-gauche/`,
`../tomopterna-droite/`, chacune avec son `decisions.md`).

- 11/10 : les deux nouvelles grenouilles remplacent les anciennes ; le fond était trop
  « vertical », l'herbe de la photo part dans des directions variées ; à gauche, dessiner l'eau
  qui se colle à la branche (tension superficielle), qu'on voie que la branche touche l'eau.
  Fait : brins d'herbe relevés à la main, grain qui suit leur direction ; branche entière ;
  ménisque.
- 12/10 : le ménisque doit être plus discret (on aurait dit du verre bleuté), sa courbure est à
  revoir ; lui et son reflet doivent se toucher (la branche touche l'eau, son reflet est collé à
  elle) ; il manque deux rangées de pixels au reflet de la branche au bord gauche. Le reflet et
  les petites vagues sont « top ». L'eau doit être plus transparente, qu'on voie le fond comme
  sur la photo (« on dirait de la gadoue »).
- 12/10 : le ménisque doit être plus court, avec aussi une légère courbure de l'eau au niveau de
  la branche, pas seulement à la surface.
- 12/10 : le ménisque n'est pas un filtre gris qui laisse voir le fond : il reflète ce qu'il y a
  devant ; on discerne à peine la branche de son reflet, ils fusionnent presque ; s'inspirer des
  pixels de la photo. Puis : « fusionne les deux ménisques, 3 pixels les séparent ».
- 12/10 : le reflet de la branche se perd un peu dans le fond, l'eau n'étant pas un miroir
  parfait, mais seulement à gauche, près de la surface : il est net au centre et à droite.
- 12/10 : la branche mérite plus de détails : en plus du lichen blanc, des bouts d'écorce noire,
  des endroits où l'écorce s'est déchirée et montre un bois plus clair ; toute la branche est
  mouillée, elle brille un peu.

Fait le 12/10 :
- ménisque : couleurs relevées sur la photo colonne par colonne (miroir courbe : sombre contre
  l'écorce, bleu ardoise, liseré clair), sans la ligne sombre de l'eau plate ; il descend jusqu'à
  la ligne d'eau, son liseré contre celui de son reflet ; 40 pixels, bout en coin mouillé (l'eau
  monte le long de la branche et de la surface, l'air entre en arrondi entre les deux) ; de
  petits ronds partent de sa pointe ; reflet sans décalage à gauche, calculé par colonne
  d'arrivée (les deux colonnes du bord ne manquent plus) ;
- eau : le fond visible (sable, feuilles, galets en fines taches couchées par la perspective),
  qui ondule un peu ; une palette pour l'herbe, une pour l'eau ;
- reflet de la branche : net, sauf sur les 110 premiers pixels à gauche, où il s'adoucit, perd
  son contour et laisse voir le fond ;
- branche : repeinte d'un bord à l'autre d'après un relevé de sa matière sur la photo (écorce,
  écorce noire luisante, lichen, bois mis à nu veiné, éclats), lissé en plaques ; le dessus
  mouillé prend un reflet bleuté ; des éclats s'allument et s'éteignent dans la page. Elle
  remplace, dans la scène, les morceaux validés sous chaque grenouille (mêmes bords, même bosse).
