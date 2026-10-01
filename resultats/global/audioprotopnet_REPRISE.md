# audioprotopnet : état au 01/10 (session « AnuraSet encodage 8 »), à reprendre

- Encodage : **fini** (1 599 enregistrements), branche `donnees-anuraset-audioprotopnet`.
- Benchmark `--curve` (sans jetons) : **fini** pour BOAFAB, DENMIN, PITAZU (fichiers `*_courbe.csv`,
  `*_transfert.npz`, `*_duree_s.txt`). **PHYCUV et LEPLAT : seulement `*_partiel.npz`** (reprise
  automatique en relançant la même commande).
- Jetons (`jetons.py audioprotopnet`) : **pas commencés** (dossier vide), puis `--curve --tokens`.
  Non fait : fiche `documentation/benchmarks/fiches/audioprotopnet.md`.
- Débit d'encodage 1,8 fenêtre/s à 4 threads ; benchmark ≈ 316 s (BOAFAB) à 15 min par espèce.
- Reprise : mêmes étapes que la procédure (`VAGUE_ENCODAGE_2.md`), étapes 2, 3, puis
  `git archive`/checkout de `donnees-anuraset-audioprotopnet` pour récupérer stock et base
  (sans réencoder), `global_bench.py audioprotopnet <ESPECE> sorties --curve` pour PHYCUV et
  LEPLAT (recopier d'abord les `*_partiel.npz` dans `sorties/`), `jetons.py`, puis `--curve --tokens`.
