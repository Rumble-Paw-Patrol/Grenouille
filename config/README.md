# Config

Les réglages et les chemins d'accès aux données. Les chemins relatifs sont lus depuis la racine du dépôt.

- `default.yaml` : la configuration par défaut. Se surcharge avec `blanci --config mon_fichier.yaml <commande>` (fusion clé par clé).
- `local.example.yaml` : modèle de configuration propre à une machine (racine du disque externe, par exemple). À copier en `config/local.yaml`, ignoré par git et pris d'office s'il existe.
- `plan/` : les plans de tirage.
  - `partition_v1.csv` : la partition des sites et des micros en entraînement et niveau 3, pour le tirage du lot 1.
