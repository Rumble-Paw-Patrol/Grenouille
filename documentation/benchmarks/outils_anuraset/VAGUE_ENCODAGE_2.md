# Vague d'encodage 2 sur AnuraSet : sessions « AnuraSet encodage 5 » à « 15 »

**État (29/09 soir) : prérequis faits (n° 152), sessions 5 à 14 prêtes ; 15 (MetaPerch)
bloquée, poids non publiés.** Léonard écrit « go » dans une session quand sa ligne dit « oui ».
Ce fichier prime sur le message de création de la session ; les précisions de Léonard au
moment du go priment sur ce fichier. Décisions : n° 141, 150, 151, 152.

**Clôture (feuille de route V5, DECISIONS n° 156)** : la vague se termine le **09/10/2026** ;
les sessions non finies ce jour-là sont abandonnées. Licence de chaque encodeur : V5 §2 ; les
non libres ne servent plus que d'objectifs à battre.

## Sessions

| Session | Encodeurs | Têtes (options de `global_bench.py`) | Prête |
|---|---|---|---|
| 5 | birdnet_v3 | `--curve --native` (son classifieur, sans entraînement) | oui |
| 6 | birdmae_large ; puis jetons de birdmae_base | `--curve --tokens` ; Base : `--tokens` | oui |
| 7 | naturebeats, beats | `--curve --tokens` | oui |
| 8 | convnext_birdset, audioprotopnet | `--curve` ; audioprotopnet : `--curve --tokens` | oui |
| 9 | avesecho_passt, biolingual | `--curve` | oui |
| 10 | insect66, insect459 | `--curve` | oui |
| 11 | rcl_fs_bsed | transfert seul (trames de 0,2 s : pas de `--curve`) | oui |
| 12 | esp_aves2_effnetb0_bio, _all, _audioset | `--curve` | oui |
| 13 | esp_aves2_eat_bio, _eat_all, _sl_eat_bio_ssl_all, _sl_eat_all_ssl_all | `--curve --tokens` | oui |
| 14 | esp_aves2_sl_beats_bio, _sl_beats_all, _naturelm_audio_v1_beats | `--curve --tokens` | oui |
| 15 | MetaPerch | — | **non** : poids non publiés (README du dépôt Perch, 29/09) |

## Prérequis (faits, n° 152)

| | Prérequis | État |
|---|---|---|
| P1 | jetons : Bird-MAE (32 × 8), BEATs et NatureBEATs (31 × 8), AudioProtoPNet (19 × 8), esp-aves2 (EAT 32 × 8, BEATs 31 × 8, EfficientNet 16 × 4) ; `jetons.py` les extrait pour tout le stock, moyennés sur la fréquence | fait, testé sur de vrais enregistrements |
| P2 | têtes sur jetons dans `global_bench.py --tokens` : `attentive`, `logistic:max`, `proto_probe` (sonde à prototypes, `blanci/proto_probe.py`) ; `simple_prototype` ajouté aux têtes de transfert | fait |
| P3 | birdnet_v3 : probabilités de son classifieur rangées à l'encodage (DENMIN, LEPLAT, PHYCUV, BOAFAB, *A. baeobatrachus*) ; `global_bench.py --native` | fait, testé |
| P4 | adaptateur esp-aves2 (`blanci/encoders/avex_encoder.py`, `backend: avex`), fenêtres de 5 s | fait, testé sur trois familles |
| P5 | MetaPerch | bloqué : « Instructions on accessing the MetaPerch model checkpoint will be provided here soon » |
| P6 | rcl_fs_bsed : même protocole sur des trames de 0,2 s, transfert seul, 2 processus | défini |

## Procédure commune (au go)

1. `git pull origin main`. Lire : `documentation/encodeurs-bacpipe.md` (section « Règle » et
   les fiches de ses encodeurs), `documentation/benchmarks/outils_anuraset/LISEZMOI.md`,
   DECISIONS n° 150 à 152. Outil `read_documentation` (session.resources) : disque et délai
   d'inactivité de la machine.
2. Environnement : `uv sync`, puis torch, torchvision, torchaudio en roues **CPU** (`uv pip
   install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio` : les
   roues CUDA remplissent le disque). Selon les encodeurs :
   - bacpipe : `uv pip install --no-deps bacpipe==1.3.5`, puis seulement les modules que
     bacpipe réclame à l'import du modèle ; `transformers<5` (n° 147) ; TensorFlow CPU
     seulement si le modèle l'exige ;
   - esp-aves2 : `uv pip install avex "transformers<5"`.
   Noter les versions (torch, transformers, onnxruntime, tensorflow, avex, librosa) dans la
   fiche.
3. Données : `bash documentation/benchmarks/outils_anuraset/telecharger_anuraset.sh` (7,2 Go,
   ~6 min), `git fetch origin resultats-anuraset-07 && git archive
   origin/resultats-anuraset-07 data | tar -x` (étiquettes), `uv run blanci --config
   config/anuraset.yaml anuraset-prepare`, puis supprimer `data/external/anuraset/raw_data.zip`.
4. Essai de 5 min d'abord, avec le script de la fin de ce fichier : f_e, fenêtre, dimension,
   forme des jetons, embeddings ni constants ni NaN. Un écart avec ce fichier ou
   `encodeurs-bacpipe.md` : s'arrêter et le dire.
5. Encodage : `uv run python documentation/benchmarks/outils_anuraset/encoder.py <encodeurs…>`
   en arrière-plan (sélection du n° 141, fenêtres jointives), `OMP_NUM_THREADS=4`. Vérifier
   toutes les 10 à 15 min, jamais de boucle serrée. Après un redémarrage du conteneur,
   relancer : l'encodage reprend où il en était.
6. Pousser chaque stock fini sur la branche `donnees-anuraset-<encodeur>` (branche orpheline,
   dans un worktree) : `data/embeddings_anuraset/<stock>/` et `data/db/anuraset.sqlite`,
   `git add -f`. Jamais de données sur `main`.
7. Si la ligne dit `--tokens` : `uv run python documentation/benchmarks/outils_anuraset/jetons.py
   <encodeur>` (second passage dans l'encodeur ; reprise par paquets de 50 enregistrements).
   Les jetons restent sur la machine (≈ 1 Go par encodeur de 768 dimensions).
8. Benchmark : pour chaque encodeur et chaque espèce (DENMIN, PITAZU, PHYCUV, LEPLAT, BOAFAB),
   `uv run python documentation/benchmarks/outils_anuraset/global_bench.py <encodeur> <ESPECE>
   sorties <options de la ligne>`, `OMP_NUM_THREADS=1` ; 4 processus en parallèle, **2 avec
   `--tokens`** (mémoire). **Témoin** (n° 151) : si la logistique donne sur BOAFAB une AP
   moyenne par site (minute) nettement sous 0,85, chercher d'abord un tuyau cassé.
9. Pousser `sorties/<encodeur>_*` sous `resultats/global/` de la branche
   `resultats-anuraset-07` (`git pull --rebase` juste avant : d'autres sessions y poussent).
10. Fiche courte sur `main` : `documentation/benchmarks/fiches/<encodeur>.md`, 30 lignes au plus.
    Contenu : réglages lus (f_e, fenêtre, dimension, couche, versions), débit, témoin BOAFAB, AP
    moyenne par site et poolée (minute) par espèce pour chaque tête (les têtes sur jetons à
    part), et, pour un transformer, « en retrait en sondage linéaire » ou non (n° 151).
    Anomalies. `git pull --rebase` avant le push ; ne rien modifier d'autre (ni DECISIONS, ni
    les rapports : la session principale les rédige dans le rapport global).
11. Finir par 5 lignes : ce qui est poussé, durées, anomalies.

## Précisions par session

- **5 (birdnet_v3)** : gros plan. Deux lectures : la logistique sur l'embedding et son
  classifieur sans entraînement (`--native`, colonne `classifieur_origine`) sur DENMIN, LEPLAT,
  PHYCUV et BOAFAB (PITAZU n'est pas dans ses 11 560 classes). AnuraSet est peut-être dans ses
  données d'entraînement (sources non publiées) : un score spectaculaire est à dire comme tel.
- **6 (Bird-MAE)** : Large d'abord (encodage, jetons, benchmark). Puis Base :
  `uv run python documentation/benchmarks/outils_anuraset/importer_stock.py birdmae_base`,
  `jetons.py birdmae_base`, `global_bench.py birdmae_base <ESPECE> sorties --tokens` (sa courbe
  existe déjà, benchmark 07). But : trancher le verdict suspendu des n° 147 et 149. Huge :
  seulement si les jetons de Large changent la donne.
- **11 (rcl_fs_bsed)** : ~480 000 trames ; `global_bench.py` sans `--curve`, 2 processus. Tête
  d'origine : `simple_prototype` (déjà dans le transfert).
- **14 (esp-aves2 BEATs)** : avant l'encodage complet, comparer
  `esp_aves2_naturelm_audio_v1_beats` à `naturebeats` (bacpipe) sur 50 fenêtres (cosinus) ;
  identiques : ne pas l'encoder deux fois, le noter dans la fiche.
- **15 (MetaPerch)** : relire `chirp/projects/metaperch/README.md` du dépôt
  `google-research/perch` ; poids toujours absents : le dire en une ligne et s'arrêter.

## Script de l'essai (étape 4)

```
uv run python - <<'EOF'
from pathlib import Path
import numpy as np
from blanci.audio import load_audio
from blanci.config import load_config
from blanci.encoders import get_encoder
cfg = load_config(Path("config/anuraset.yaml"))
wav, sr = load_audio(next(Path("data/external/anuraset/anuraset/INCT17").rglob("*.wav")), 0)
for name in ["<encodeur>"]:
    enc = get_encoder(name, cfg)
    n = int(enc.window_s * sr)
    w = np.stack([wav[i * n:(i + 1) * n] for i in range(3)])
    e = enc.embed(w, sr)
    t = enc.embed_tokens(w, sr) if enc.has_tokens else None
    print(name, enc.sample_rate, enc.window_s, e.shape, None if t is None else t.shape,
          float(e.std()), bool(np.isfinite(e).all()))
EOF
```
