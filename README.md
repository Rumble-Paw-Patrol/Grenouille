# Grenouille — détecter *Anomaloglossus blanci* dans les enregistrements acoustiques de Guyane

Stage à l'ONF Guyane (15/09/2026 → 14/03/2027). *Anomaloglossus blanci* est une petite grenouille
endémique et menacée, qui chante une note de 0,09 s toutes les 1,4 s dans une bande de fréquences
saturée par d'autres espèces. L'ONF et ses partenaires posent des enregistreurs en forêt (96 292
enregistrements de 2 min, 3 204 h, huit sites) ; le projet cherche un **détecteur qui fonctionne
sur de nouveaux points d'écoute**, qu'un naturaliste utilise **sans écrire de code** et que l'ONF
peut réentraîner seul.

Le paquet Python s'appelle `blanci`, la commande aussi : `uv run blanci --help`.

## Où en est le projet (01/10/2026)

| | État |
|---|---|
| Chaîne de traitement | écrite et testée : inventaire des enregistrements, embeddings, têtes de classification, module de traitement du signal (rythme des notes), fusion, évaluation, files de vérification, poste d'annotation, réentraînement |
| Choix de l'encodeur | pré-benchmark sur **AnuraSet** (jeu public d'anoures néotropicaux) : 23 encodeurs comparés, chacun avec la tête adaptée à sa sortie. Indicateur, pas verdict : le choix final se fera sur les données de l'ONF le 20/11/2026 |
| Contrainte d'accès | l'encodeur livré doit être **libre d'accès** (poids publics, licence compatible avec l'usage par l'ONF). Meilleur libre à ce jour : `perch_v2` + sonde à prototypes (AP moyenne par site 0,81) |
| Annotations | reprises de zéro, par plan de tirage stratifié (point d'écoute, heure, période), vérifiées à l'aveugle par des experts. **Aucun benchmark sur les données de l'ONF avant leur accord** |
| Livrable | application Windows installable sans code : à venir (essai d'empaquetage prévu) |

Prochaines étapes : partition des points d'écoute et plan de tirage, première série d'annotations
(par intervalles, DECISIONS n° 182), essai d'empaquetage `.exe`.

## Pour les encadrants : par où commencer

1. **[DECISIONS.md](DECISIONS.md)** : le *cadre du projet* en tête (objectifs et critères §0,
   protocole d'annotation §5, évaluation §6, livrable §7, planning §8, risques §9), puis le
   journal des décisions numérotées (n° 1 à 165) qui dit pourquoi le code est comme il est.
2. **[documentation/benchmarks/](documentation/benchmarks/LISEZMOI.md)** : les huit rapports de
   benchmark AnuraSet (têtes, régularisations, encodeurs), chacun avec ses données, ses figures et
   le script qui les produit. Le plus récent, `2026-09-30_anuraset_encodeurs`, compare 23 encodeurs.
3. **[documentation/tableaux/](documentation/tableaux/LISEZMOI.md)** : un tableau par benchmark, en
   image (protocole, encodeurs, têtes, régularisations…).
4. **[documentation/biblio/biblio.md](documentation/biblio/biblio.md)** : bibliographie commentée.
5. **[notebooks/](notebooks/)** : quatre notebooks pour voir la chaîne fonctionner sur un
   enregistrement (voir plus bas).

## Structure du dépôt

```
Grenouille/
├── blanci/               le paquet Python, un dossier par étape de la chaîne (voir ci-dessous)
├── tests/                suite pytest, rangée comme blanci/ (tests/heads/ teste blanci/heads/…)
├── config/               default.yaml, local.example.yaml
├── notebooks/            4 notebooks d'exploration, committés sans sorties
├── anuraset/             pré-benchmark AnuraSet, terminé : config, scripts, notes (reproductibilité)
├── documentation/
│   ├── benchmarks/       rapports 01 à 08, fiches des encodeurs, modèle de rapport
│   ├── tableaux/         un tableau PNG par benchmark et le script qui les dessine
│   ├── regularizations/  catalogue des régularisations des têtes (R1 à R84)
│   ├── biblio/           bibliographie, offre de stage
│   ├── prez/             présentations de suivi (.pptx) et leurs sources
│   ├── rapport-stage/    notes pour le rapport
│   ├── old/              anciennes feuilles de route (V1 à V4)
│   └── commandes.md, encodeurs-bacpipe.md, glossaire-bioacoustique.md, audit-28-septembre.md
├── DECISIONS.md          cadre du projet et journal des décisions
└── pyproject.toml, uv.lock, .python-version
```

### Le paquet `blanci/`

Chaque dossier est une étape de la chaîne, dans l'ordre où les données la traversent. Le
`__init__.py` de chaque dossier donne en une ligne le rôle de chacun de ses modules.

```
inputs ─► embedding ─► heads ─► combination ─► results
                         └──────────┴─► evaluation       mesure les têtes et leurs combinaisons
annotation ─► inputs                                      les labels écoutés repartent en entrée
```

| Dossier | Étape | Modules |
|---|---|---|
| `core/` | socle | `config`, `db` (SQLite, labels en ajout seul), `audio` |
| `inputs/` | 1. enregistrements et annotations | `ingest` (inventaire), `qc` (drapeaux : fichiers cassés, silencieux, micro dans un sac…), `labels` (import des annotations), `dataset`, `frozen` (jeu gelé) |
| `embedding/` | 2. fenêtres → embeddings | `grid` (fenêtres), `encoders/` (bacpipe, esp-aves2, ONNX du livrable, passe-bas, export), `embed`, `store` et `index` (Parquet, similarité) |
| `heads/` | 3. têtes de détection | `head` (prototypes, logistique…), `attentive`, `proto_probe`, `gated`, `dann`, `pooling`, `losses`, `regularization/` (R1 à R84), `cluster` ; sans encodeur : `signal_processing` (rythme des notes), `baselines`, `detectors/` |
| `combination/` | 4. combiner les modèles | `stacking`, `fusion` (fusion à deux niveaux), `ensemble` |
| `evaluation/` | 5. mesurer | `evaluate` (plis par micro, AP, rappel, bootstrap, Wilson), `oof` (scores hors-pli), `benchmark`, `head_benchmark`, `full_benchmark`, `throughput`, `qc_calibration`, `anuraset` (archivé) |
| `annotation/` | 6. boucle d'écoute | `selection`, `workbench`, `app` (poste d'annotation Streamlit) et `viewer` (spectrogramme zoomable, tracé des intervalles) |
| `results/` | 7. sorties écologiques | `aggregate` (fenêtre → enregistrement → point), `activity` (courbes d'activité) |
| `exploration/` | notebooks | `explore`, `explore_plots`, `attention_map` |
| racine | interfaces | `cli` (commandes Typer ; AnuraSet regroupé sous `blanci anuraset …`), `service` (couche appelée par la CLI et par la future application) |

Réservés : `heads/finetune`, `heads/detectors/distilled`, `heads/detectors/homemade` sont des
emplacements documentés, à programmer plus tard (fine-tuning, distillation, modèle maison).

`service.py` regroupe les opérations de la chaîne de décision (entraîner et enregistrer une
tête, scorer, file de vérification, labels, jeu gelé, réentraînement, courbes d'activité) : c'est
la couche que la future application appellera comme la CLI. Les commandes d'inventaire,
d'encodage et de benchmark de `cli.py` appellent directement leurs modules.

## Installation

Python 3.11 et [uv](https://docs.astral.sh/uv/).

```sh
uv sync                       # dépendances de base et outils de développement
uv sync --group research      # + bacpipe, torch, onnx : encodeurs, têtes apprises, export
uv sync --group app           # + streamlit : poste d'annotation
uv sync --inexact --group notebook   # + matplotlib, ipykernel : notebooks (--inexact garde le reste)
uv run pytest                 # ≈ 850 tests, ~2 min (torch et onnx sautés sans le groupe research)
uv run blanci --help
```

Les enregistrements se déclarent dans `config/local.yaml` (copie de `config/local.example.yaml`,
ignorée par git) : racine du disque externe, jamais modifié.

## Données

**Le dépôt ne contient aucun enregistrement.** L'audio de l'ONF et de Biophonia, la base SQLite,
les embeddings et les modèles vivent sous `data/`, ignoré par git. Les exports d'annotations du
prestataire (`documentation/*.xlsx`) restent eux aussi en local.

Seules données publiques versionnées : celles d'**AnuraSet** (Cañas et al. 2023, CC BY), dont les
embeddings et la base servent aux benchmarks. Elles sont rangées dans les branches
`donnees-anuraset-*`, hors de `main` ; `anuraset/importer_stock.py` les importe.

Inventaire (base au 29/09/2026) : 2023 = phénologie sur trois sites (Kaw, Molokoï, Trésor), un an,
66 779 enregistrements ; 2026 = cinq sites (CDR, Mataroni, Patawa Est et Ouest, RNRT), un relevé
d'une semaine chacun, 29 513 enregistrements. Détail dans
[documentation/commandes.md](documentation/commandes.md#inventaire-base-au-29092026).

## Utilisation

La référence complète des commandes (inventaire, annotation, encodage, benchmarks, détection,
évaluation) est dans **[documentation/commandes.md](documentation/commandes.md)**. Aperçu :

```sh
B="uv run blanci --config config/local.yaml"
$B ingest --dataset 2026 --site Mataroni --no-qc --no-hash "<dossier du relevé>"   # inventaire
$B status                                                  # ce que contient la base
uv run blanci embed --encoder birdmae --subset benchmark   # embeddings des fenêtres annotées
uv run blanci heads --encoder perch_v2-bacpipe1.3.5        # toutes les têtes, mêmes plis
$B annotate                                                # poste d'annotation (navigateur)
```

Les benchmarks AnuraSet utilisent leur propre configuration (`anuraset/anuraset.yaml`) et leur
propre base, séparées des données de l'ONF :

```sh
A="uv run blanci --config anuraset/anuraset.yaml"
$A anuraset prepare && $A anuraset profile
```

## Notebooks d'exploration

| Notebook | Pour |
|---|---|
| `01_explorer_une_fenetre` | un enregistrement et une fenêtre à travers la chaîne : écoute, grille, portes, module de traitement du signal, négatif apparié, embeddings, prototype différentiel |
| `02_negatifs_apparies` | ce que contiennent les négatifs appariés de chaque stratégie ; feuille d'écoute, taux de contamination |
| `03_traitement_du_signal` | régler la détection des notes et les seuils des portes, sans encodeur |
| `04_attention_jetons` | voir sur le spectrogramme les jetons que pèse la tête d'attention |

Ils lisent la base et l'audio en local : il faut les données de l'ONF pour les exécuter. **Ne
jamais committer leurs sorties** (les lecteurs audio embarquent le son) : *Clear All Outputs*
avant un commit ; `tests/exploration/test_notebooks.py` le vérifie.

## Conventions

- **Labels en ajout seul** : une correction est un nouveau label, jamais une mise à jour
  (la base refuse `UPDATE` et `DELETE`).
- **Toute évaluation passe par `evaluate.py` avec des groupes explicites** (plis par micro) :
  un découpage aléatoire des fenêtres serait une erreur, les fenêtres d'un même micro partageant
  le fond sonore.
- **Aucun score n'entre dans la fusion s'il n'est pas hors-pli.**
- Code : `ruff` (lignes de 100 caractères), tests `pytest` (`-m "not slow"` pour la suite rapide).
