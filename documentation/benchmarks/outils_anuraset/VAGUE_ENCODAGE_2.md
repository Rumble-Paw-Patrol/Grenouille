# Vague d'encodage 2 sur AnuraSet : sessions « AnuraSet encodage 5 » à « 15 »

**État : en préparation, aucun go donné.** Léonard écrit « go » dans une session quand sa ligne
du tableau dit « prête ». Ce fichier prime sur le message de création de la session ; les
précisions de Léonard au moment du go priment sur ce fichier. Décisions : n° 141, 150, 151.

## Sessions

| Session | Encodeurs | Pourquoi ensemble | Têtes | Prérequis | Prête |
|---|---|---|---|---|---|
| 5 | birdnet_v3 | seul : gros plan | protocole 07 + son classifieur sans entraînement (4 espèces) | P3 | non |
| 6 | birdmae_large, puis birdmae_base (jetons) | lourd | protocole 07 + têtes sur jetons | P1, P2 | non |
| 7 | naturebeats, beats | même architecture, 16 kHz | protocole 07 + têtes sur jetons | P1, P2 | non |
| 8 | convnext_birdset, audioprotopnet | DBD, CNN 32 kHz, 5 s | protocole 07 (prototypes d'audioprotopnet : plus tard, P1) | aucun | **oui** |
| 9 | avesecho_passt, biolingual | transformers supervisés | protocole 07 ; jetons si en retrait | aucun | **oui** |
| 10 | insect66, insect459 | même modèle, deux jeux d'étiquettes | protocole 07 | aucun | **oui** |
| 11 | rcl_fs_bsed | trames de 0,2 s | protocole à définir | P6 | non |
| 12 | esp-aves2 effnetb0-bio, -all, -audioset | CNN légers | protocole 07 | P4 | non |
| 13 | esp-aves2 eat-bio, eat-all, sl-eat-bio-ssl-all, sl-eat-all-ssl-all | EAT | protocole 07 + têtes sur jetons | P4, P1, P2 | non |
| 14 | esp-aves2 sl-beats-bio, sl-beats-all, naturelm-audio-v1-beats | BEATs | protocole 07 + têtes sur jetons ; naturelm = naturebeats ? | P4, P1, P2 | non |
| 15 | MetaPerch | trouver les poids | protocole 07 | P5 | non |

## Prérequis (code sur `main`, faits par la session principale)

| | Prérequis | État |
|---|---|---|
| P1 | jetons des transformers (Bird-MAE, BEATs, NatureBEATs, EAT, audioprotopnet), moyennés sur l'axe fréquence, gardés sur la machine | à faire |
| P2 | têtes sur jetons dans `global_bench.py` (attentive, prototypes), règle du n° 151 | à faire |
| P3 | logits du classifieur de birdnet_v3 (DENMIN, LEPLAT, PHYCUV, BOAFAB, *A. baeobatrachus*) et « tête d'origine » dans `global_bench.py` | à faire |
| P4 | adaptateur esp-aves2 (bibliothèque AVEX) | à faire |
| P5 | poids et adaptateur MetaPerch | à faire (poids non vérifiés) |
| P6 | protocole à trames de 0,2 s pour rcl_fs_bsed | à définir |

## Procédure commune (au go)

1. `git pull origin main`. Lire : `documentation/encodeurs-bacpipe.md` (section « Règle » et
   les fiches de ses encodeurs), `documentation/benchmarks/outils_anuraset/LISEZMOI.md`,
   DECISIONS n° 150 et 151. Outil `read_documentation` (session.resources) : disque et délai
   d'inactivité de la machine.
2. Environnement : `uv sync`, puis `uv pip install --no-deps bacpipe==1.3.5` et seulement les
   modules que bacpipe réclame à l'import du modèle ; torch, torchvision, torchaudio en roues
   **CPU** (`--index-url https://download.pytorch.org/whl/cpu` : les roues CUDA remplissent le
   disque) ; `transformers<5` pour les modèles Hugging Face (n° 147) ; TensorFlow CPU seulement
   si le modèle l'exige. Noter les versions installées (torch, onnxruntime, tensorflow,
   transformers, librosa) dans la fiche.
3. Données : `bash documentation/benchmarks/outils_anuraset/telecharger_anuraset.sh` (7,2 Go,
   ~6 min), `git fetch origin resultats-anuraset-07 && git archive
   origin/resultats-anuraset-07 data | tar -x` (étiquettes), puis `uv run blanci --config
   config/anuraset.yaml anuraset-prepare`, puis supprimer `data/external/anuraset/raw_data.zip`.
4. Essai de 5 min d'abord, sur 20 enregistrements : f_e, durée de fenêtre et dimension lues,
   embeddings ni constants ni NaN, débit (fenêtres/s). Un écart avec `encodeurs-bacpipe.md` :
   s'arrêter et le dire.
5. Encodage : `uv run python documentation/benchmarks/outils_anuraset/encoder.py <encodeurs…>`
   en arrière-plan (sélection du n° 141, fenêtres jointives). Vérifier toutes les 10 à 15 min,
   jamais de boucle serrée. Après un redémarrage du conteneur, relancer : l'encodage reprend où
   il en était.
6. Pousser chaque stock fini sur la branche `donnees-anuraset-<encodeur>` (branche orpheline,
   dans un worktree) : `data/embeddings_anuraset/<encodeur>-bacpipe1.3.5@o0/` et
   `data/db/anuraset.sqlite`, `git add -f`. Jamais de données sur `main`.
7. Benchmark, protocole 07 : pour chaque encodeur et chaque espèce (DENMIN, PITAZU, PHYCUV,
   LEPLAT, BOAFAB), `global_bench.py <encodeur> <ESPECE> sorties --curve`, 4 processus en
   parallèle, `OMP_NUM_THREADS=1`. **Témoin** (n° 151) : si la logistique donne sur BOAFAB une
   AP moyenne par site (minute) nettement sous 0,85, chercher d'abord un tuyau cassé (f_e,
   fenêtre, couche lue) avant de conclure.
8. Pousser `sorties/<encodeur>_*` sous `resultats/global/` de la branche
   `resultats-anuraset-07` (`git pull --rebase` juste avant : d'autres sessions y poussent).
9. Fiche courte sur `main` : `documentation/benchmarks/fiches/<encodeur>.md`, 30 lignes au plus.
   Contenu : réglages lus (f_e, fenêtre, dimension, couche, versions), débit, témoin BOAFAB, AP
   moyenne par site et poolée (minute) par espèce pour les trois têtes, et, si l'encodeur est un
   transformer, « en retrait en sondage linéaire » ou non (n° 151). Anomalies. `git pull
   --rebase` avant le push ; ne rien modifier d'autre (ni DECISIONS, ni les rapports : la
   session principale les rédige dans le rapport global).
10. Finir par 5 lignes : ce qui est poussé, durées, anomalies.

## Précisions par session

- **5 (birdnet_v3)** : gros plan. Deux lectures : la logistique sur l'embedding (protocole 07)
  et le classifieur de BirdNET lui-même, sans entraînement, sur DENMIN, LEPLAT, PHYCUV et BOAFAB
  (PITAZU n'est pas dans ses 11 560 classes). AnuraSet est peut-être dans ses données
  d'entraînement (sources non publiées) : un score spectaculaire est à dire comme tel.
- **6 (Bird-MAE)** : Large d'abord ; puis réencoder Base avec ses jetons (45 min) pour trancher
  le verdict suspendu des n° 147 et 149. Huge : seulement si Large gagne sur les jetons.
- **14 (esp-aves2 BEATs)** : avant l'encodage complet, comparer `naturelm-audio-v1-beats` à
  `naturebeats` (bacpipe) sur 50 fenêtres (cosinus) ; identiques : ne pas l'encoder deux fois.
- **15 (MetaPerch)** : d'abord trouver les poids (article : `github:google-research/perch`,
  dossier `metaperch`), leur licence et leur format ; rendre compte avant d'écrire du code.
