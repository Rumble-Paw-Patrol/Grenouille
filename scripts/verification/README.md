# Vérification par les encadrants

Tirage au hasard d'extraits déjà annotés, à envoyer pour vérification.

- `tirage_verification.py` : tire `--n` extraits sans remise parmi ceux que j'ai écoutés dans le poste, à leur dernière version, et écrit une file d'écoute dans `paths.queues/<nom>/` (`candidats.csv`, `LISEZMOI.md` : graine, décompte des extraits écartés). La base est lue en lecture seule.

```
uv run python scripts/verification/tirage_verification.py verif_1 --n 50 --seed 1
uv run blanci paquet verif_1 --file verif_1 --message "..."
```

**Jamais tirés** :
- les annotations remplacées par une séance d'annotation plus récente (`remplace_par` renseigné, voir `annotations/README.md`) : seule la dernière version d'un extrait réécouté peut être tirée ;
- ceux d'un tirage déjà envoyé : `--exclure verif_1` au tirage suivant (sans cela, deux tirages peuvent retomber sur les mêmes extraits) ;
- les extraits de contrôle rapide (`source` = `flag`), sauf `--source plan flag` ;
- ceux d'un autre annotateur (`--annotateur`, par défaut Léonard et L).

Un tirage est reproductible : mêmes graine, même base, mêmes exclusions, mêmes extraits. Ne pas réexporter ni réannoter entre le tirage et l'envoi.

Le tirage vit dans `blanci/annotation/verification.py`.
