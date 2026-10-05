# Tableau de bord

Page claude.ai privée, pour Léonard et ses encadrants :
<https://claude.ai/artifact/DLfBZopGjpLyJPBi9ViLi8>

Sections : vue d'ensemble et calendrier, travail en cours, chaîne de traitement, données,
encodeurs, explorateur des têtes (comparaison à la carte), rapports de benchmark, tableaux,
bibliographie, glossaire. Chaque carte a un bouton « Commenter » ; les commentaires sont
partagés entre tous ceux qui ont accès à la page. La page n'est visible qu'une fois partagée
depuis son menu « Partager » sur claude.ai.

## Fichiers

| Fichier | Rôle |
|---|---|
| `modele.html` | la page (mise en forme, graphiques, interactions), sans données |
| `contenu.yaml` | ce que le dépôt ne dit pas de lui-même : chaîne, planning, travail en cours, débit, cibles d'annotation. **À tenir à jour à la main** |
| `construire.py` | lit les rapports, CSV, tableaux, bibliographie, glossaire, inventaire, tests et `git log`, et écrit `index.html` et `fichiers.json` (ignorés par git) |

## Mettre à jour

```sh
uv run python documentation/tableau-de-bord/construire.py
```

Puis, dans une session Claude : « republie le tableau de bord ». Claude publie `index.html` à
l'URL ci-dessus avec l'outil Artifact (`url` = cette URL, `files` = le contenu de
`fichiers.json`, chemins des sources préfixés par la racine du dépôt). Le skill `benchmark` le
fait à la fin de chaque nouveau benchmark.
