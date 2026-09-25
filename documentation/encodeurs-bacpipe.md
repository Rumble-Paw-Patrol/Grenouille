# Encodeurs de bacpipe : ce qu'on peut récupérer

Relevé le 25/09/2026 dans le code de bacpipe 1.3.5 (`bacpipe/model_pipelines/feature_extractors/`)
et dans notre adaptateur (`blanci/encoders/bacpipe_encoder.py`). Aucun modèle n'a été exécuté
pour ce relevé : les tailles de grille marquées « à mesurer » se vérifient au premier chargement
(méthode dans `notes.md` : comparer l'embedding à la moyenne, au maximum et au jeton de classe).

## Vocabulaire

- **Embedding** : le vecteur unique que bacpipe rend par fenêtre (ce qu'on stocke aujourd'hui).
- **Jetons** : les vecteurs *avant* l'agrégation en embedding. Pour un transformer, un par
  morceau (patch) du spectrogramme ; pour un CNN, un par case de la dernière carte de
  caractéristiques (fréquence × temps). C'est l'entrée de l'attentive probing et des poolings.
- **Couches intermédiaires** : les sorties des couches plus basses du réseau, plus génériques.
- **Hook** : fonction PyTorch branchée sur une couche, qui en copie la sortie pendant le
  calcul, sans modifier le modèle ni ralentir l'inférence. Pour TensorFlow/Keras, l'équivalent
  est de reconstruire un modèle qui s'arrête à la couche voulue.

## Les encodeurs du projet (`encoders.models`)

| Nom bacpipe | Framework | f_e | Fenêtre | Dim. | Ce que bacpipe rend | Jetons | Couches intermédiaires |
|---|---|---|---|---|---|---|---|
| `birdmae` (Huge ; `birdmae_base` : Base) | PyTorch, Hugging Face (code du modèle fourni par DBD-research-group) | 32 kHz | 5 s | 1 280 (Base : 768) | l'embedding agrégé : malgré son nom, `last_hidden_state` est ici la moyenne des patchs (jeton de classe exclu) puis `fc_norm` | **Faciles** : `output_hidden_states=True` rend, par couche, 257 jetons = 1 jeton de classe + 256 patchs en grille **32 temps × 8 fréquences** (ordre temps d'abord : patch t·8 + f), avant `fc_norm` | Faciles : même option, toutes les couches (13 sorties pour le modèle Base) |
| `beats` | PyTorch | 16 kHz | 5 s | 768 | la moyenne des jetons, faite *dans* bacpipe (`avg_pooling = True`) | **Faciles** : `avg_pooling = False` sur le modèle chargé, une ligne, sans hook. Grille ≈ 8 fréquences × 31 temps (patchs 16 × 16), à mesurer | Moyennes : hook sur les couches de l'encodeur transformer |
| `naturebeats` | PyTorch (BEATs, poids NatureLM-audio) | 16 kHz | 5 s | 768 | idem BEATs | **Faciles**, même réglage que BEATs | Moyennes, idem BEATs |
| `perch_v2` | ONNX (onnxruntime), sans TensorFlow | 32 kHz | 5 s | 1 536 | embedding, jetons spatiaux, spectrogramme, logits des 14 795 classes | **Déjà utilisés** : 16 temps × 4 fréquences × 1 536 (`spatial_embedding`) | Difficiles : le graphe ONNX ne sort que ces quatre tenseurs ; il faudrait le modifier pour exposer des nœuds internes (faisable avec la bibliothèque `onnx`, à valider) |
| `perch_bird` (Perch v1) | TensorFlow SavedModel (via perch-hoplite) | 32 kHz | 5 s | 1 280 | embedding, logits, spectrogramme | **Difficiles** : la signature du SavedModel n'expose pas la carte avant agrégation | Difficiles, même raison |
| `birdnet` (v2.4) | TensorFlow / Keras | 48 kHz | 3 s | 1 024 | la sortie de l'avant-dernière couche (`layers[-3]`) | **Faciles** : la carte de caractéristiques avant l'agrégation globale s'obtient comme bacpipe obtient l'embedding (`tf.keras.Model(entrée, couche.output)`). Taille à mesurer | Faciles : n'importe quelle couche, même méthode. Licence non commerciale : référence seulement, non déployable |
| `protoclr` | PyTorch (CvT-13) | 16 kHz | 6 s | 384 | moyenne des jetons du dernier étage (ou jeton de classe selon la config) | **Faciles** : hook sur le dernier étage | Faciles : 3 étages, un hook chacun |
| `convnext_birdset` | PyTorch, Hugging Face (ConvNeXt) | 32 kHz | 5 s | 1 024 | `pooler_output` (moyenne de la dernière carte) | **Faciles** : `output_hidden_states=True` rend la dernière carte fréquence × temps | Faciles : les 4 étages, même option |

« Facile » : une option ou un hook, puis la vérification du `notes.md`. « Moyen » : lire le code
du modèle pour choisir la couche. « Difficile » : il faut modifier ou reconstruire le modèle
exporté.

**Correction** (n° 111) : une première version de ce tableau disait que bacpipe rendait déjà
les jetons de Bird-MAE. Faux : le code du modèle (`modeling_bird_mae.py` de Bird-MAE-Base, lu le 25/09 ; Huge à confirmer) nomme
`last_hidden_state` la sortie agrégée. Le n° 77 avait raison : seul perch_v2 rend ses jetons
tels quels aujourd'hui.

## Les autres modèles de bacpipe (hors projet)

| Nom bacpipe | Framework | f_e | Fenêtre | Domaine | Jetons / couches | Intérêt ici |
|---|---|---|---|---|---|---|
| `audioprotopnet` | PyTorch, Hugging Face | 32 kHz | 5 s | oiseaux (BirdSet) | jetons **déjà gardés** par bacpipe (`results.last_hidden_state`) | à considérer : oiseaux, 32 kHz |
| `birdnet_v3` | PyTorch | 32 kHz | 3 s | oiseaux | à lire | à considérer si la licence le permet |
| `surfperch` | TensorFlow (Perch) | 32 kHz | 5 s | récifs coralliens | comme Perch v1 | faible |
| `biolingual` | PyTorch (CLAP) | 48 kHz | 10 s | bioacoustique + texte | couches du modèle audio via Hugging Face | faible (fenêtre de 10 s) |
| `aves_especies`, `birdaves_especies` | PyTorch (wav2vec2) | 16 kHz | 1 s | animaux | **toutes les couches** rendues par `extract_features` | moyen (fenêtre courte, adaptée à une note) |
| `avesecho_passt` | PyTorch (PaSST) | 32 kHz | 3 s | oiseaux | hook | moyen |
| `audiomae` | PyTorch (ViT) | 16 kHz | 10 s | audio général | hook | faible |
| `vggish` | TensorFlow | 16 kHz | 1 s | audio général | — | faible (référence ancienne) |
| `mix2` | PyTorch | 16 kHz | 3 s | amphibiens / insectes (à vérifier) | hook | à lire |
| `rcl_fs_bsed` | PyTorch | 22,05 kHz | 0,2 s | détection d'événements, peu d'exemples | hook | à lire (fenêtre de la taille d'une note) |
| `insect66`, `insect459` | PyTorch | 44,1 kHz | 5,5 s | insectes | `embeddings` du modèle | faible (orthoptères = faux amis) |
| `bat`, `batdetect2_clip_avg`, `batdetect2_dets_avg` | PyTorch | 256 kHz | 1 s (batdetect2) | chauves-souris | — | aucun (ultrasons) |
| `google_whale`, `hbdet` | TensorFlow | 24 kHz / 2 kHz | 2–4 s | cétacés | — | aucun |

## Ce que ça change pour la suite

- **Bird-MAE, BEATs, NatureBEATs, ProtoCLR et ConvNeXt-BirdSet** : une option ou un hook par
  modèle pour les jetons, à ajouter à l'adaptateur (non fait). Bird-MAE est le plus simple
  (option Hugging Face, grille connue : 32 × 8).
- **Couches intermédiaires** (R23) : faciles pour les modèles PyTorch et BirdNET, difficiles
  pour Perch v1 et v2. Le coût est ailleurs : stocker plusieurs couches multiplie le volume des
  embeddings, sur des centaines de milliers de fenêtres.
