# Encodeurs de bacpipe

Les modèles disponibles dans bacpipe 1.3.5, plus deux encodeurs hors bacpipe. Pour chacun : fréquence d'échantillonnage (f_e), durée de fenêtre, dimension de l'embedding quand elle est connue, et ce qui le distingue.

## Utilisables dans le projet

| Encodeur | Cadre | f_e | Fenêtre | Dim. | Spécificité |
|---|---|---|---|---|---|
| perch_v2 | ONNX | 32 kHz | 5 s | 1 536 | Perch 2.0, EfficientNet-B3, 14 795 classes, oiseaux et autres taxons ; rend aussi des jetons spatiaux (16 × 4) |
| perch_bird | TensorFlow | 32 kHz | 5 s | 1 280 | Perch v1, oiseaux |
| birdnet | TensorFlow | 48 kHz | 3 s | 1 024 | BirdNET v2.4, oiseaux ; licence non commerciale |
| birdnet_v3 | ONNX | 32 kHz | 3 s | — | BirdNET+ V3.0 (préversion), 11 560 classes dont 647 amphibiens ; CC BY-SA 4.0 |
| birdmae_base, birdmae_large, birdmae_huge | PyTorch | 32 kHz | 5 s | 768 (Base), 1 280 (Huge) | Bird-MAE, ViT auto-supervisé (MAE) appris sur BirdSet ; demande une tête sur les jetons (grille 32 × 8) ; pas de licence déclarée |
| convnext_birdset | PyTorch | 32 kHz | 5 s | 1 024 | ConvNeXt supervisé sur BirdSet |
| audioprotopnet | PyTorch | 32 kHz | 5 s | — | ConvNeXt à prototypes (ProtoPNet) sur BirdSet ; CC BY-NC |
| beats | PyTorch | 16 kHz | 5 s | 768 | BEATs de Microsoft, transformer auto-supervisé, audio général |
| naturebeats | PyTorch | 16 kHz | 5 s | 768 | BEATs affiné dans NatureLM-audio ; licence probablement non commerciale |
| protoclr | PyTorch | 16 kHz | 6 s | 384 | CvT-13, contrastif, peu d'exemples |
| avesecho_passt | PyTorch | 32 kHz | 3 s | — | PaSST, oiseaux |
| biolingual | PyTorch | 48 kHz | 10 s | — | CLAP audio-texte, bioacoustique |
| insect66, insect459 | PyTorch | 44,1 kHz | 5,5 s | — | Insectes ; deux jeux d'étiquettes |
| rcl_fs_bsed | PyTorch | 22,05 kHz | 0,2 s | — | Contrastif, détection à peu d'exemples, trames de 0,2 s |
| mix2 | PyTorch | 16 kHz | 3 s | — | MobileNetV3-Large entraîné sur AnuraSet (42 espèces d'anoures) : à ne pas évaluer sur AnuraSet |

## Hors bacpipe

| Encodeur | Spécificité |
|---|---|
| esp-aves2 | Famille d'Earth Species Project (EfficientNet-B0, EAT, BEATs ; entraînée sur bioacoustique seule, avec ou sans AudioSet), 16 kHz, via la bibliothèque AVEX ; CC-BY-NC-SA-4.0 |
| MetaPerch | Perch entraîné avec les métadonnées d'enregistrement (lieu, date) ; poids pas encore publiés |

## Les autres modèles de bacpipe

| Encodeur | f_e | Fenêtre | Spécificité |
|---|---|---|---|
| aves_especies, birdaves_especies | 16 kHz | 1 s | wav2vec2, animaux ; toutes les couches rendues |
| audiomae | 16 kHz | 10 s | ViT, audio général |
| vggish | 16 kHz | 1 s | TensorFlow, audio général, ancien |
| surfperch | 32 kHz | 5 s | Perch pour les récifs coralliens |
| google_whale | 24 kHz | 2–4 s | Cétacés |
| hbdet | 2 kHz | 2–4 s | Cétacés |
| bat, batdetect2_clip_avg, batdetect2_dets_avg | 256 kHz | 1 s | Chauves-souris (ultrasons) |
