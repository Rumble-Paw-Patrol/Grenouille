# Grenouille

## Audits (`/audit-n-from`)

- Lint et tests : `uv run ruff check .` et `uv run pytest -m "not slow" -q` (ou seulement les
  tests des dossiers touchés si N ≤ 2).
- Branche auditée : `origin/main`. Les numéros de commit sont ceux du tableau de bord
  (« Derniers changements du dépôt »).
- Rapport : `documentation/audit-<jour>-<mois en lettres>.md`, sur le modèle de
  `documentation/audit-28-septembre.md`. Pas de renvoi à `DECISIONS.md`.
- En fin d'audit, dire si un rapport de benchmark ou le tableau de bord (`/tableau-de-bord`)
  est à régénérer.

### Axes d'audit

| Axe | Où | Ce qu'on cherche |
|---|---|---|
| Entrées | `blanci/inputs/`, `blanci/core/` | inventaire, drapeaux, import des labels, chemins, config |
| Encodage | `blanci/embedding/` | stocks mélangés, reprise, rééchantillonnage, ONNX / OpenVINO |
| Têtes | `blanci/heads/` | fuites entre plis, régularisations mal branchées, numérique |
| Évaluation | `blanci/evaluation/` | métriques optimistes : plis, bootstrap, seuil choisi sur les scores évalués, Holm |
| Fusion, résultats | `blanci/combination/`, `blanci/results/` | descripteurs manquants, empilement non imbriqué, agrégation |
| Annotation, interfaces | `blanci/annotation/`, `blanci/cli.py`, `blanci/service.py` | labels écrasés, commandes incohérentes |
| Benchmarks | `anuraset/`, `documentation/benchmarks/` | chiffres du rapport ≠ `donnees/`, conclusions plus fortes que les données |
| Documentation | `README.md`, `documentation/`, `notebooks/` | doc qui contredit le code, commandes disparues, chiffres périmés |

Les tests (`tests/`) vont avec l'axe du code qu'ils testent.
