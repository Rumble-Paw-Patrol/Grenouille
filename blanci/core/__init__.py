"""Socle, utilisé par toutes les étapes.

- `config` : chargement de `config/default.yaml` et des surcharges ; racine du projet.
- `db` : base SQLite, schéma et migrations.
- `audio` : décodage (mono float32) et rééchantillonnage.
- `ahead` : préparation en avance dans des fils à part (lecture, calculs sur le signal).
"""
