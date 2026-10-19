# Grenouille — détecter *Anomaloglossus blanci* dans les enregistrements acoustiques de Guyane

Stage à l'ONF Guyane (15/09/2026 → 14/03/2027). *Anomaloglossus blanci* est une petite grenouille
endémique et menacée, qui chante une note de 0,09 s toutes les 1,4 s dans une bande de fréquences
saturée par d'autres espèces. L'ONF et ses partenaires posent des enregistreurs en forêt (96 292
enregistrements de 2 min, 3 204 h, huit sites) ; le projet cherche un **détecteur qui fonctionne
sur de nouveaux points d'écoute**, qu'un naturaliste utilise **sans écrire de code** et que l'ONF
peut réentraîner seul.

Le paquet Python s'appelle `blanci`, la commande aussi : `uv run blanci --help`.

## Où en est le projet (19/10/2026)

| | État |
|---|---|
| Chaîne de traitement | écrite et testée : inventaire des enregistrements, embeddings, têtes de classification, module de traitement du signal (rythme des notes), fusion, évaluation, files d'écoute, poste d'annotation, réentraînement |
| Choix de l'encodeur | pré-benchmark sur **AnuraSet** (jeu public d'anoures néotropicaux) : 24 encodeurs comparés, chacun avec la tête adaptée à sa sortie. Indicateur, pas verdict : le choix final se fera sur les données de l'ONF le 20/11/2026 |
| Contrainte d'accès | l'encodeur livré doit être **libre d'accès** (poids publics, licence compatible avec l'usage par l'ONF). Meilleur libre à ce jour : `perch_v2` + sonde à prototypes (AP moyenne par site 0,81) |
| Encodage du corpus ONF | Perch 2.0 terminé (61 h sur un processeur Intel Core i5), BirdNET 2.4 terminé, BirdNET 3.0 en cours |
| Annotations | reprises de zéro, par plan de tirage stratifié (point d'écoute, heure, période), les points d'écoute étant répartis entre entraînement et test avant toute écoute. Poste d'annotation terminé ; 40 extraits écoutés au 16/10. Les labels seront vérifiés par des experts. **Aucun benchmark sur les données de l'ONF avant leur accord** |
| Livrable | application Windows installable sans code : à venir (essai d'empaquetage prévu) |

Prochaines étapes : annoter le lot 1 (424 extraits de 30 s), puis le jeu de test (250
enregistrements entiers) ; finir l'encodage BirdNET 3.0 ; essai d'empaquetage `.exe`.

## Pour les encadrants : par où commencer

**[Tableau de bord](https://claude.ai/artifact/DLfBZopGjpLyJPBi9ViLi8)** (page claude.ai, accès sur
invitation) : chaîne de traitement, données, benchmarks à explorer, rapports et travail en cours,
avec un bouton « Commenter » sur chaque carte. Sources : `documentation/tableau-de-bord/`.

## Structure du dépôt

```
Grenouille/
├── blanci/               le code : un dossier par étape de la chaîne (voir ci-dessous)
├── tests/                tests automatiques, rangés comme blanci/
├── config/               réglages (default.yaml) et modèle de réglages locaux
├── notebooks/            4 notebooks pour voir la chaîne fonctionner sur un enregistrement
├── annotations/          annotations exportées (CSV) et files d'écoute
├── anuraset/             pré-benchmark AnuraSet, terminé : réglages et scripts
└── documentation/
    ├── benchmarks/       les 8 rapports de benchmark AnuraSet, avec données, figures et scripts
    ├── tableaux/         un tableau récapitulatif (PNG) par benchmark
    ├── tableau-de-bord/  sources du tableau de bord
    ├── regularizations/  catalogue des régularisations des têtes (R1 à R85)
    ├── biblio/           bibliographie commentée, offre de stage
    ├── prez/             présentations de suivi (.pptx)
    ├── audits/           relectures complètes du dépôt
    ├── sprites/          la grenouille en 8-bit du tableau de bord
    ├── old/              anciennes feuilles de route
    └── commandes.md, glossaire-bioacoustique.md, encodeurs-bacpipe.md
```

Documents utiles pour entrer dans le projet : les
[rapports de benchmark](documentation/benchmarks/LISEZMOI.md), leurs
[tableaux](documentation/tableaux/LISEZMOI.md), la
[bibliographie](documentation/biblio/biblio.md) et le
[glossaire](documentation/glossaire-bioacoustique.md).

### Le code : le paquet `blanci/`

Chaque dossier est une étape de la chaîne, dans l'ordre où les données la traversent.

```
inputs ─> embedding ─> heads ─> combination ─> results
                         └──────────┴─> evaluation       mesure les têtes et leurs combinaisons
annotation ─> inputs                                     les labels écoutés repartent en entrée
```

| Dossier | Étape | Ce qu'il fait |
|---|---|---|
| `inputs/` | 1. enregistrements et annotations | inventaire des enregistrements, contrôle qualité (fichiers cassés, silencieux, micro dans un sac…), import des annotations, jeu de test gelé |
| `embedding/` | 2. encodage | découpe en fenêtres, passage dans un modèle pré-entraîné (Perch, BirdNET…) qui résume chaque fenêtre en un vecteur (embedding) |
| `heads/` | 3. détection | têtes de classification entraînées sur les embeddings (prototypes, logistique, attention…), et un détecteur sans encodeur fondé sur le rythme des notes |
| `combination/` | 4. combinaison | fusion des scores de plusieurs têtes |
| `evaluation/` | 5. mesure | validation croisée par micro, précision moyenne (AP), rappel, intervalles de confiance |
| `annotation/` | 6. écoute | tirage des extraits à écouter, poste d'annotation dans le navigateur |
| `results/` | 7. sorties écologiques | agrégation fenêtre → enregistrement → point d'écoute, courbes d'activité |
| `core/` | socle | réglages, base de données, lecture de l'audio |
| `exploration/` | notebooks | fonctions et figures des notebooks |

La commande `blanci` (`cli.py`) donne accès à toutes les étapes. Quelques emplacements sont
réservés pour plus tard : réglage fin des encodeurs, distillation, modèle maison, export ONNX.

## Données

**Le dépôt ne contient aucun enregistrement de l'ONF.** L'audio, la base de données, les
embeddings et les modèles restent en local, sous `data/`. Seuls quelques extraits courts sont
versionnés, pour les présentations et le tableau de bord.

Seules données publiques versionnées : celles d'**AnuraSet** (Cañas et al. 2023, CC BY), qui
servent aux benchmarks. Elles sont rangées dans les branches `donnees-anuraset-*`, hors de
`main`.

Inventaire (base au 29/09/2026) : 2023 = phénologie sur trois sites (Kaw, Molokoï, Trésor), un an,
66 779 enregistrements ; 2026 = cinq sites (CDR, Mataroni, Patawa Est et Ouest, RNRT), un relevé
d'une semaine chacun, 29 513 enregistrements. Détail dans
[documentation/commandes.md](documentation/commandes.md#inventaire-base-au-29092026).

## Installation et utilisation

Python 3.11 et [uv](https://docs.astral.sh/uv/).

```sh
uv sync --inexact             # installe tout
uv run pytest                 # ≈ 860 tests, ~2 min
uv run blanci --help
```

Sous Windows, lancer une fois `setx UV_NO_SYNC 1` : sinon chaque `uv run` peut réinstaller torch
au milieu d'un encodage.

Les enregistrements se déclarent dans `config/local.yaml` (copie de `config/local.example.yaml`) :
racine du disque externe, qui n'est jamais modifié. Pour écouter, ce n'est pas nécessaire :
`blanci annotate` cherche lui-même les enregistrements sur les disques branchés.

Toutes les commandes sont décrites dans **[documentation/commandes.md](documentation/commandes.md)**.
Les principales :

```sh
B="uv run blanci --config config/local.yaml"
$B ingest --dataset 2026 --site Mataroni --no-qc --no-hash "<dossier du relevé>"   # inventaire
$B status                                                  # ce que contient la base
$B embed --encoder birdmae --subset benchmark              # embeddings des fenêtres annotées
$B heads --encoder perch_v2-bacpipe1.3.5                   # toutes les têtes, mêmes plis
$B annotate                                                # poste d'annotation (navigateur)
$B export-labels                                           # annotations en CSV
$B import-labels                                           # reprendre l'écoute sur une autre machine
```

Les benchmarks AnuraSet ont leurs propres réglages et leur propre base, séparés des données de
l'ONF :

```sh
A="uv run blanci --config anuraset/anuraset.yaml"
$A anuraset prepare && $A anuraset profile
```

## Notebooks

| Notebook | Pour |
|---|---|
| `01_explorer_une_fenetre` | suivre un enregistrement à travers la chaîne : écoute, fenêtres, traitement du signal, embeddings |
| `02_negatifs_apparies` | écouter les exemples négatifs choisis pour l'entraînement et mesurer combien contiennent en fait A. blanci |
| `03_traitement_du_signal` | régler la détection des notes, sans encodeur |
| `04_attention_jetons` | voir sur le spectrogramme les zones que regarde la tête d'attention |

Il faut les données de l'ONF pour les exécuter. Ils sont versionnés sans leurs sorties, qui
contiendraient de l'audio.

## Garanties méthodologiques

- **Pas de fuite entre entraînement et test** : l'évaluation se fait par validation croisée
  *par micro*. Couper au hasard entre fenêtres serait optimiste, car les fenêtres d'un même micro
  partagent le même fond sonore.
- **Fusion honnête** : les scores qu'on combine sont toujours calculés sur des données que la
  tête n'a pas vues.
- **Annotations jamais écrasées** : une correction ajoute un nouveau label, l'historique reste
  intact.
