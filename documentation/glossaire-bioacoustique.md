# Glossaire — Stage ONF Guyane / détection acoustique *Anomaloglossus blanci*

Mis à jour le 17/09/2026. Fusion de deux sources : les termes rencontrés dans les échanges
de préparation avec les encadrants, et les notions de machine learning reprises au fil des
révisions.

Organisé par thème plutôt qu'alphabétiquement, pour que les notions liées se lisent
ensemble. Le terme anglais figure en italique quand il diffère. Les sections 13 à 16
rassemblent ce qui est propre au stage : modèles, outils, acteurs et paramètres du signal
cible.

---

## 1. Matériel et calcul

**CPU** — Processeur généraliste. Sur le M4 : 10 cœurs (4 performance, 6 efficacité).
Exécute n'importe quel programme, peu parallèle. C'est lui qui fait tourner scikit-learn
et XGBoost, qui n'ont pas de version GPU sur Mac.

**GPU** — Processeur graphique, ici 8 ou 10 cœurs soit ~1000 à 1300 unités arithmétiques
travaillant en parallèle, de l'ordre de 3 à 4 TFLOPS en FP32. Programmable : on y écrit
des *kernels*, petits programmes exécutés par des milliers de fils simultanés.
C'est le seul des trois processeurs sur lequel on peut entraîner.

**NPU / Neural Engine (ANE)** — ASIC dédié aux réseaux de neurones, 16 cœurs, annoncé à
38 TOPS sur le M4. Jeu d'opérations figé (produits matriciels, convolutions, quelques
activations), en INT8 et FP16. **Inférence uniquement** : il ne calcule pas de gradients
et Apple n'expose aucune interface bas niveau. Accessible seulement via Core ML.
Son intérêt n'est pas le débit brut mais la consommation, donc l'absence de bridage
thermique sur une machine sans ventilateur.

**ASIC** — Circuit intégré conçu pour une tâche unique. Beaucoup plus rapide et sobre sur
sa tâche, incapable de faire autre chose.

**Mémoire unifiée** — Un seul bloc de RAM accessible par CPU, GPU et ANE, sans copie ni
bus PCIe. Toute la RAM (16, 24 ou 32 Go) est disponible pour le GPU, contrairement à une
carte graphique dédiée limitée à sa VRAM.

**Bande passante mémoire** — Volume de données transférable par seconde entre RAM et
unité de calcul. 120 Go/s sur le MacBook Air M4 (273 sur un M4 Pro, > 1 To/s sur un GPU
de centre de calcul). C'est très souvent le vrai facteur limitant, pas la puissance brute.

**Bridage thermique** *(thermal throttling)* — Réduction automatique des fréquences quand
la puce chauffe. Le MacBook Air est sans ventilateur : la charge soutenue se dégrade au
bout de quelques minutes.

**FP32 / FP16 / INT8** — Formats numériques. FP32 : 32 bits, précision relative ~6·10⁻⁸.
FP16 : 16 bits, précision ~5·10⁻⁴, maximum représentable 65504. INT8 : entier sur 8 bits,
256 valeurs. Moins de précision = moins de mémoire et calcul plus rapide, au prix
d'erreurs d'arrondi et de risques de débordement.

**Quantification** — Conversion d'un modèle vers une précision plus basse (typiquement
FP32 → INT8).

**FLOPS / TOPS** — Opérations par seconde, en virgule flottante (FLOPS) ou entières (TOPS).
Comparables uniquement à précision égale : 38 TOPS INT8 et 4 TFLOPS FP32 ne se comparent
pas directement.

**Metal** — Interface de programmation graphique et de calcul d'Apple, équivalent de CUDA
chez NVIDIA.

**MPS** — Deux sens à distinguer. (1) *Metal Performance Shaders*, bibliothèque de
primitives GPU optimisées. (2) Le backend `mps` de PyTorch, qui traduit les opérations
vers le GPU Apple : `torch.device("mps")`. Aucun rapport avec l'ANE. À ne pas confondre
avec le *Multi-Process Service* de NVIDIA, qui porte le même sigle.
Conséquence pratique : rend le M4 utilisable pour tout modèle PyTorch, et inutilisable
pour un modèle TensorFlow.

**Backend** — Couche logicielle traduisant les opérations d'un framework vers un matériel
donné : `cuda`, `cpu`, `mps`.

**Core ML** — Framework d'Apple pour exécuter un modèle **déjà entraîné**. Trois parties :
le format `.mlpackage`, un compilateur qui décide opération par opération quelle unité
l'exécute (CPU / GPU / ANE), un moteur d'exécution. N'entraîne pas.

**coremltools** — Paquet Python convertissant un modèle PyTorch ou TensorFlow vers Core ML.
Seul chemin d'accès à l'ANE. Paramètre `ComputeUnit` : `ALL`, `CPU_ONLY`, `CPU_AND_GPU`,
`CPU_AND_NE`.

**MLX** — Framework de calcul sur tableaux d'Apple (open source, fin 2023). Interface
proche de NumPy et PyTorch, conçu pour la mémoire unifiée, évaluation paresseuse.
N'utilise pas l'ANE. Sa niche réelle est l'exécution de grands modèles de langage en
local. Pour un projet bioacoustique, PyTorch reste le choix pragmatique : écosystème plus
fourni, `torchaudio`, et chemin de conversion Core ML mieux balisé.

**TensorFlow** — Framework de deep learning de Google. Point décisif sur MacBook Air M4 :
aucun chemin d'accélération matérielle. Pas de CUDA sur Apple Silicon, et le support Metal
est irrégulier.

**TFLite** — Runtime d'inférence embarqué dérivé de TensorFlow. Un fichier `.tflite` est
**figé** : on n'accède qu'aux tenseurs d'entrée et de sortie que l'auteur a explicitement
exposés. Ni gradients, ni couches intermédiaires, ni fine-tuning. C'est le format de
distribution de BirdNET, et la raison pour laquelle l'*attentive probing* y est
impossible.

**ONNX** — Format d'échange de modèles neutre vis-à-vis du framework, avec un runtime
optimisé. Permet de faire tourner un modèle TensorFlow sur CPU ARM sans installer
TensorFlow. Porte de sortie pour évaluer Perch 2.0 en local.

**Framework** *(cadriciel)* — Ensemble de bibliothèques et de conventions fournissant
l'ossature d'une application. Inversion du contrôle : c'est le framework qui appelle ton
code, pas l'inverse. Un framework de ML fournit un type tenseur, la différentiation
automatique, des couches et optimiseurs, et les backends matériels.

**Écarts numériques après conversion** — Un modèle converti en Core ML ne donne pas
exactement les mêmes sorties qu'en PyTorch. Causes cumulées : passage FP32 → FP16
(dominant), non-associativité de l'arithmétique flottante, algorithmes de convolution
différents (direct, im2col, Winograd), fusion d'opérations. Écarts typiques de 10⁻³ à
10⁻² sur les logits, accord top-1 généralement > 99 %. **Le vrai piège est ailleurs :
dans le prétraitement**, si le spectrogramme n'est pas calculé à l'identique des deux
côtés.

---

## 2. Traitement du signal audio

**Fréquence d'échantillonnage** *(sample rate)* — Nombre de mesures de pression par
seconde. 32 kHz est le standard en avifaune (BirdCLEF, BirdNET, Perch).

**Théorème de Nyquist-Shannon** — Une fréquence d'échantillonnage `sr` permet de
représenter les fréquences jusqu'à `sr/2`. À 32 kHz : plafond à 16 kHz. À 16 kHz
(fréquence de travail de NatureLM-audio) : plafond à 8 kHz — sans conséquence sur une
fondamentale à 4,75 kHz, limitant pour un paysage sonore complet.

**Duty cycle** *(cycle d'enregistrement)* — Fraction du temps où l'enregistreur est actif.
Ici 2 min par heure, soit 3,3 %. **À ne pas confondre avec la fréquence
d'échantillonnage.**

**Taux d'émission** *(call rate)* — Nombre de cris ou de notes par unité de temps. Chez les
anoures, souvent corrélé à la température.

**STFT** *(Short-Time Fourier Transform)* — Découpage du signal en fenêtres courtes qui se
chevauchent, transformée de Fourier sur chacune, empilement des résultats. Produit une
matrice complexe de dimensions `(n_fft/2 + 1) × nb_trames`.

**n_fft** — Longueur de la fenêtre d'analyse en échantillons. Fixe la résolution
fréquentielle : `Δf = sr / n_fft`. À 32 kHz : n_fft = 1024 → Δf = 31,3 Hz, fenêtre de 32 ms.

**hop_length** — Décalage entre deux fenêtres successives. Fixe la cadence des colonnes.
Typiquement `n_fft/4` (75 % de recouvrement). hop = 320 à 32 kHz → une colonne toutes les
10 ms.

**Fenêtre de Hann** — Fonction de fenêtrage s'annulant doucement aux bords, appliquée à
chaque tranche avant la transformée. Évite le *spectral leakage* causé par une coupure
franche. Défaut de `torchaudio`, aucune raison d'en changer.

**Magnitude / phase** — Chaque coefficient complexe de la STFT porte une amplitude et un
décalage. On ne garde que la magnitude en classification ; la phase n'est nécessaire que
pour reconstruire du son.

**Compromis temps-fréquence** — `Δt · Δf ≥ 1/(4π)` (inégalité de Heisenberg-Gabor).
Impossibilité mathématique, pas une limite technologique : une fenêtre courte localise
bien dans le temps mais mal en fréquence, et réciproquement. On répartit l'incertitude,
on ne la réduit pas.

| n_fft (à 32 kHz) | durée de fenêtre | Δf |
|---|---|---|
| 256 | 8 ms | 125 Hz |
| 512 | 16 ms | 62,5 Hz |
| 1024 | 32 ms | 31,3 Hz |
| 2048 | 64 ms | 15,6 Hz |

**Spectrogramme** — Représentation temps-fréquence : temps en abscisse, fréquence en
ordonnée, intensité en niveau. C'est une image, donc traitable par un CNN.

**Échelle mel** — Échelle perceptuelle calibrée sur l'audition **humaine** :
`m = 2595·log₁₀(1 + f/700)`. Quasi linéaire sous 1000 Hz, logarithmique au-dessus.
Fonctionne bien en avifaune par coïncidence de plage, mais ce n'est pas un argument de
principe — inadaptée pour chiroptères ou orthoptères.

**Banc de filtres mel** — 64 à 128 filtres triangulaires répartis uniformément en échelle
mel. Réduit les 513 raies fréquentielles à 128 bandes. Régler `f_min` (50 Hz coupe le
vent) et `f_max` (borne haute réelle des espèces).

**Log-mel** — `S = log(mel + ε)`. L'ordre de grandeur de ε fixe le plancher de bruit
(10⁻⁶ à 10⁻¹⁰ selon la normalisation).

**PCEN** *(Per-Channel Energy Normalization)* — Alternative au logarithme, conçue pour les
enregistrements de terrain (Wang et al. 2017, Lostanlen et al. 2019). Divise chaque valeur
par une moyenne glissante de la même bande de fréquence : un bruit stationnaire (vent,
pluie, route, insectes) se divise par lui-même et disparaît, un transitoire ressort.
`PCEN = (E/(ε+M)^α + δ)^r − δ^r`. Paramètres optionnellement apprenables.
Disponible dans `librosa.pcen`. **Meilleure défense contre le clustering par site.**

**Approche multi-résolution** — Calculer deux ou trois spectrogrammes à résolutions
différentes et les empiler comme canaux d'entrée `(3, H, W)`. Le réseau choisit
lui-même sur lequel s'appuyer selon l'espèce.

**Configuration de référence** — 32 kHz mono, n_fft = 1024, hop = 320, Hann, 128 mels,
50–14000 Hz, fenêtres de 3 à 5 s → tenseur `(128, 300)`. Conforme aux pratiques des
soumissions BirdCLEF+ 2026.

---

## 3. Réseaux de neurones — briques de base

**Tenseur d'activations** — Tableau à quatre dimensions `(N, C, H, W)` : N = taille du
lot, C = canaux, H et W = dimensions spatiales (fréquence et temps sur un spectrogramme).

**Canal** — Une grille `(H, W)` parmi la pile. En entrée : 1 pour un spectrogramme mono.
En profondeur : un **type de motif détecté**, un canal par noyau. Le canal perd tout sens
physique et devient la carte de réponse d'un détecteur appris.

**Convolution** — On fait glisser un noyau sur toutes les positions ; à chaque position,
multiplication terme à terme puis somme, ce qui produit **une** valeur.

**Noyau** *(kernel, filtre)* — Petit tableau de poids, typiquement 3×3. Il s'étend sur
**toute la profondeur des canaux d'entrée** : un noyau 3×3 avec C_in = 64 contient
3×3×64 = 576 poids et produit **un seul** canal de sortie.

**C_in / C_out** — Nombres de canaux, pas des grilles. La règle qui débloque tout :
**C_in est absorbé par chaque noyau, C_out est le nombre de noyaux.**
Tenseur de poids : `(C_out, C_in, k, k)`, soit `C_out × C_in × k²` paramètres + C_out biais.
En PyTorch : `nn.Conv2d(in_channels, out_channels, kernel_size)`.

**Partage des poids** — Le même noyau est appliqué à toutes les positions : un motif est
détecté où qu'il se trouve, et le nombre de paramètres ne dépend pas de la taille de
l'entrée. En passe arrière, cela devient une **accumulation** : le gradient d'un poids
somme les contributions de toutes les positions et de tous les exemples.

**Localité** — Chaque sortie ne dépend que d'un voisinage restreint.

**Neurone dans un CNN** — Une case de la sortie, c'est-à-dire un triplet (canal, temps,
fréquence). Une sortie 64 canaux en 4×4 compte 1024 neurones. **Un noyau n'est pas un
neurone** : c'est un jeu de poids partagé par tous les neurones d'un même canal.
Formulation utile : une couche convolutive est une couche dense contrainte par la
localité et le partage des poids.

**Padding** — Ajout de zéros autour de l'entrée pour conserver la taille spatiale.
Sans padding, un noyau 3×3 fait perdre une case de chaque côté.

**Stride** *(pas)* — Déplacement du noyau entre deux positions. Un pas de 2 divise par
deux la taille de sortie et double le champ réceptif effectif des couches suivantes.

**Pooling / max pooling** — Agrégation d'un bloc 2×2 en une valeur. Réduit la taille
spatiale. En passe arrière, le gradient est routé uniquement vers la position gagnante.

**Champ réceptif** — Zone d'entrée influençant une valeur de sortie. Croît par
composition : pour des noyaux 3×3 de pas 1, `RF_L = 2L + 1` (croissance linéaire, lente).
Avec des pas et du pooling : `RF_L = RF_{L−1} + (k_L − 1) × ∏(pas précédents)`, croissance
géométrique. **Règle de conception : le champ réceptif final doit être au moins de l'ordre
de la durée de l'événement cherché.**

**Activation** — Deux sens. (1) La *fonction* d'activation φ, non-linéarité appliquée
après la combinaison linéaire. (2) Les *activations*, tenseurs de valeurs intermédiaires
circulant entre couches, par opposition aux poids.

**Pré-activation** — `z = Σᵢ wᵢxᵢ + b`, avant application de φ.

**Somme d'activations** — La combinaison linéaire vue sous l'angle du coût numérique :
une convolution 3×3 sur 512 canaux accumule 4608 produits pour une seule sortie, ce qui
expose au débordement en FP16.

**ReLU** — `max(0, x)`. Dérivée 1 si x > 0, sinon 0. Sans non-linéarité, empiler des
couches est inutile : la composée d'opérations affines est affine, et dix blocs
Conv + BatchNorm sans ReLU se réduisent à une seule transformation affine.
Lecture en détection : ne conserve que la preuve positive.

**LeakyReLU / GELU** — Variantes laissant passer une pente négative (LeakyReLU) ou lisses
partout (GELU). Parades contre la ReLU morte.

**Sigmoid** — `σ(z) = 1/(1+e^(−z))`. Transforme un logit en probabilité, **indépendamment
des autres logits**.

**Softmax** — `pᵢ = e^(zᵢ)/Σⱼe^(zⱼ)`. Les probabilités se partagent un budget de 1, donc
les classes sont en **compétition**. Ne vaut 1 exactement que pour un logit infini.

**Logit** — Valeur brute en sortie de la dernière couche linéaire, avant toute activation.
Du terme *log-odds*, `log(p/(1−p))`. Le nombre de logits vaut K, le nombre de classes.

**Batch normalization** — Couche insérée entre convolution et activation. Pour chaque
canal, calcule moyenne μ et variance σ² **sur le lot et sur les dimensions spatiales**,
normalise, puis applique une transformation affine apprise γ, β (2C paramètres).
Normalise les **activations**, pas les poids.
- Par canal parce qu'un canal = un détecteur : ses valeurs sont comparables entre elles.
- À l'inférence, μ et σ² proviennent de **moyennes glissantes** figées, sinon la prédiction
  dépendrait des autres exemples du lot. D'où `model.eval()`.
- Se dégrade nettement avec de petits lots (N < 8) : préférer alors GroupNorm ou LayerNorm,
  qui normalisent à l'intérieur d'un exemple.
- À l'inférence, se fusionne exactement dans la convolution précédente.
- Justification historique (*internal covariate shift*) contestée depuis Santurkar et al.
  2018 ; efficacité empirique non contestée.

**Dropout** — Régularisation (Srivastava et al. 2014). Chaque neurone est mis à zéro avec
probabilité p à chaque passe d'entraînement, par tirage indépendant. Empêche la
co-adaptation. Variante *inverted dropout* : division par (1−p) à l'entraînement, donc
rien à faire à l'inférence — on désactive simplement. `model.train()` / `model.eval()`.

**Tirage** — Réalisation d'une variable aléatoire. Ici, une variable de Bernoulli par
neurone.

**Connexions résiduelles** — `y = x + f(x)`, donc `∂y/∂x = I + ∂f/∂x`. Le terme identité
offre au gradient un chemin direct de gain 1, ce qui empêche l'effondrement du produit des
jacobiennes. C'est ce qui rend possibles les réseaux à plus de cent couches.

---

## 4. Entraînement — la boucle

**Batch / mini-batch** — Sous-ensemble d'exemples sur lequel on calcule un gradient moyen
avant une mise à jour. Ce n'est **pas** un optimizer : deux axes indépendants.
- **Batch size** : 16 à 64 typiquement pour un CNN sur spectrogrammes.
- **Itération** *(step)* : une mise à jour des poids.
- **Époque** *(epoch)* : un passage complet sur les données, soit ⌈N/B⌉ itérations.

Trois effets de la taille de lot : mémoire (linéaire, contrainte dominante), qualité de
l'estimation (bruit en 1/√B), et **bruit régularisant** — un petit lot explore mieux et
généralise souvent un peu mieux. Ne jamais la changer seule : ajuster le learning rate en
conséquence.

**Passe avant** *(forward)* — Traversée du réseau de l'entrée vers la sortie.

**Inférence** — Régime d'exploitation : poids gelés, aucune comptabilité pour le gradient,
dropout désactivé, batch norm sur moyennes glissantes. `torch.no_grad()` + `model.eval()`.
Toute inférence est une passe avant, l'inverse est faux.

**Passe arrière / rétropropagation** *(backward)* — Calcul du gradient de la perte par
rapport à chaque paramètre. **Elle ne modifie rien** : elle remplit `.grad`. C'est
`optimizer.step()` qui corrige les poids.

**Gradient** — Vecteur des dérivées partielles `∂L/∂θᵢ`. Calculé pour P paramètres en
environ deux passes avant, quel que soit P.

**Règle de dérivation composée** *(chain rule)* — Le gradient est un produit de facteurs
locaux. Chaque couche n'a besoin que de sa propre dérivée et du gradient reçu de l'aval.

**Jacobienne** — Matrice `J_ij = ∂y_i/∂x_j`. `∂L/∂x = Jᵀ(∂L/∂y)`. On ne la construit
**jamais** explicitement : on implémente directement le produit vecteur-jacobienne (VJP).

**Mode inverse / mode direct** — Un produit de matrices s'évalue dans n'importe quel ordre,
mais le coût en dépend. Mode direct : une passe donne ∂(tout)/∂(une entrée) → P passes.
Mode inverse : une passe donne ∂(une sortie)/∂(tout) → une passe. Comme la perte est
**scalaire**, le mode inverse gagne d'un facteur P. D'où l'exigence d'un scalaire pour
`loss.backward()`.
Le mode direct reste utilisé pour : les dérivées par rapport aux **entrées** (PINNs, EDP),
les produits hessiens `Hv` (forward-over-reverse), les jacobiennes complètes de petits
modèles, et la recherche sur l'entraînement sans stockage d'activations.

**Graphe de calcul** — Enregistré dynamiquement pendant la passe avant pour chaque
opération sur un tenseur `requires_grad=True`. Reconstruit à chaque passe, ce qui autorise
boucles et branchements. Libéré après `backward()`.

**Règle d'addition** — Si un tenseur alimente plusieurs consommateurs, les gradients qui
lui reviennent **s'additionnent**. Fondement des connexions résiduelles.

**Règles locales par couche** (δ = ∂L/∂y) :
- Linéaire : `∂L/∂x = Wᵀδ`, `∂L/∂W = δxᵀ`, `∂L/∂b = δ`
- ReLU : `∂L/∂x = δ ⊙ 1[x>0]` (simple masque)
- Max pooling : gradient routé vers l'argmax
- Softmax + cross-entropy fusionnés : `∂L/∂z = p − y` (le gradient est l'erreur)
- Batch norm : le gradient d'un exemple dépend des autres exemples du lot
- Convolution : `∂L/∂x` est elle-même une convolution (noyau retourné de 180°) ;
  `∂L/∂W` accumule sur toutes les positions et tout le lot

**Mémoire d'entraînement** — Les activations de la passe avant doivent être conservées
jusqu'à la passe arrière. `mémoire ≈ Σ_couches (N × C × H × W) × 2 octets` en FP16.
Linéaire en N : quand ça déborde, réduire d'abord la batch size.

**Gradient checkpointing** — Ne conserver qu'une activation tous les k blocs et recalculer
les intermédiaires en passe arrière. ~30 % de calcul en plus, gain mémoire d'un facteur
proche de √L.

**Accumulation de gradient** — Enchaîner plusieurs passes avant/arrière avant un seul
`step()`, les gradients se cumulant dans `.grad`. Simule un grand lot à mémoire réduite.
**Diviser la loss par le nombre d'accumulations**, sinon le learning rate effectif est
multiplié d'autant. Ne gagne aucune vitesse, et ne reproduit pas les statistiques de
batch norm.

**Écrêtage du gradient** *(gradient clipping)* — Si `‖g‖ > c`, multiplier tout le vecteur
par `c/‖g‖`. La **direction est préservée**, seule la longueur du pas est bornée. Protège
contre un lot aberrant qui enverrait les poids dans une zone absurde.
`clip_grad_norm_(params, max_norm=1.0)` — renvoie la norme **avant** écrêtage, excellent
diagnostic à journaliser. Préférer la version sur la norme globale à `clip_grad_value_`,
qui déforme la direction.

**Disparition / explosion du gradient** — Le gradient à la couche l est un produit de
jacobiennes ; avec une valeur singulière typique s, il évolue en `s^L`. s < 1 → disparition
(la sigmoïde, dérivée ≤ 0,25, donnait 0,25¹⁰ ≈ 10⁻⁶) ; s > 1 → explosion, NaN.

**ReLU morte** — Une unité dont la pré-activation est toujours négative a un gradient
toujours nul : elle est définitivement perdue.

**Initialisation** — Conditionne s. **He** (pour ReLU) : variance `2/fan_in`.
**Xavier / Glorot** (pour tanh) : `1/fan_in`. Maintient la variance des activations et des
gradients constante d'une couche à l'autre.

**PyTorch — boucle canonique** :
```python
for x, y in dataloader:
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
```
- `zero_grad()` obligatoire : les gradients **s'accumulent** par défaut.
- `detach()` / `requires_grad_(False)` : gèle un sous-réseau.
- `torch.autograd.set_detect_anomaly(True)` : localise le premier NaN (lent, débogage).

---

## 5. Optimizers

**SGD nue** — `θ ← θ − η·g`. Aucune mémoire, même pas pour tous les paramètres.
Échoue dans une **ravine** (loss beaucoup plus raide dans une direction) : elle oscille
d'un flanc à l'autre sans avancer le long du fond de vallée.

**Conditionnement** — Rapport entre la plus grande et la plus petite valeur propre de la
hessienne. 10⁴ n'a rien d'exceptionnel en deep learning.

**Momentum** — `v ← βv + g` puis `θ ← θ − ηv`, β = 0,9 typiquement. Inertie : amortit les
oscillations (gradients de signes alternés qui se compensent) et accélère dans les
directions cohérentes (v tend vers `g/(1−β)`, soit ×10). Lecture de β : moyenne
exponentielle sur ~`1/(1−β)` itérations.

**Nesterov** — Évalue le gradient au point où l'inertie va emmener. Gain modeste mais réel.

**AdaGrad** — `G ← G + g²`, `θ ← θ − η·g/(√G + ε)`. Un pas par paramètre. Défaut : G croît
indéfiniment, le pas s'effondre. Plus utilisé.

**RMSProp** — Remplace la somme par une moyenne exponentielle : `s ← ρs + (1−ρ)g²`.
Corrige le défaut d'AdaGrad.

**Adam** — Momentum + adaptation par paramètre.
```
m ← β₁m + (1−β₁)g          moment d'ordre 1, la direction
v ← β₂v + (1−β₂)g²         moment d'ordre 2, l'amplitude
m̂ = m/(1−β₁ᵗ) ; v̂ = v/(1−β₂ᵗ)     bias correction
θ ← θ − η·m̂/(√v̂ + ε)
```
Défauts : β₁ = 0,9, β₂ = 0,999, ε = 10⁻⁸.

**Bias correction** — m et v démarrent à zéro, donc m₁ = 0,1·g₁ (dix fois trop petit) et
v₁ = 0,001·g₁². La division par (1−β₁ᵗ) restitue l'échelle exacte et s'éteint d'elle-même
quand t croît.

**Lecture du ratio m̂/√v̂** — Rapport signal sur bruit. Gradients cohérents → pas ample ;
gradients erratiques → pas réduit. Conséquence : le pas est approximativement borné par η
quelle que soit l'amplitude du gradient, ce qui rend Adam peu sensible au réglage.

**Coût mémoire d'Adam** — Poids 4P + gradients 4P + m 4P + v 4P = **16P octets** en FP32.
Pour un ResNet-50 (25 M paramètres) : 400 Mo avant toute activation. SGD + momentum : 300 Mo.

**L2 regularization vs weight decay** — **Ne sont pas la même chose.**
- L2 : `L' = L + (λ/2)‖θ‖²` donc `g' = g + λθ`.
- Weight decay : `θ ← θ − ηg − ηλθ`, appliqué directement aux poids.
Équivalents avec la SGD nue. **Avec Adam, non** : le terme λθ traverse la division par √v̂,
donc les poids à gros gradients historiques sont *moins* régularisés — l'inverse de ce
qu'on veut.

**AdamW** — Version découplée : `θ ← θ − η·m̂/(√v̂+ε) − ηλθ`. Utiliser AdamW, pas Adam.

**Parameter groups** — Le weight decay ne doit pas s'appliquer aux biais ni aux γ, β des
BatchNorm (les rétrécir revient à éteindre un canal). Critère usuel : `p.ndim <= 1`.
```python
opt = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
], lr=1e-3)
```

**Méthode de Newton** — Second ordre : `Δ = −H⁻¹g`, où H est la hessienne. Transforme
l'ellipse en sphère (multiplier par H⁻¹ divise chaque direction par sa courbure λᵢ),
converge en un pas sur une quadratique, invariante par changement de variable affine,
convergence quadratique près d'un minimum.
**Inapplicable en deep learning** pour quatre raisons :
1. Coût : H est P×P (2500 To pour 25 M paramètres), inversion en O(P³).
2. Non-convexité : converge vers un *point critique*. Dans une direction de courbure
   négative, 1/λᵢ < 0 donc Newton **remonte** la pente — il est attiré par les
   points-selles, qui dominent largement en haute dimension (Dauphin et al. 2014).
3. Stochasticité : H estimée sur un mini-batch est très bruitée, et l'inversion amplifie
   le bruit dans les directions de petite valeur propre.
4. On veut minimiser l'erreur de généralisation, pas la training loss.

**Adam comme méthode du second ordre dégradée** — v estime la **diagonale** d'une matrice
de courbure (Fisher / Gauss-Newton), et diviser par √v applique une approximation
diagonale de H^(−1/2). Adam connaît la courbure coordonnée par coordonnée, mais ignore
tous les termes croisés. Coût O(P) au lieu de O(P²).

**K-FAC, gradient naturel, Shampoo** — Approximations intermédiaires entre la diagonale et
la hessienne complète. Front de recherche actif, pas encore un remplaçant d'Adam en
pratique.

**Hessian-free** — Résout `HΔ = −g` par gradient conjugué en n'utilisant que des produits
`Hv`, calculables en ~2 passes arrière sans former H (astuce de Pearlmutter).

**BFGS / quasi-Newton** — Apprend H⁻¹ au fil des itérations via l'**équation sécante** :
entre deux pas, `s = Δθ` et `y = Δg` vérifient `y ≈ Hs`. Correction de rang 2, définie
positive (donc pas de point-selle). Coût O(P²) en mémoire, encore trop.

**L-BFGS** *(Limited-memory BFGS)* — Ne stocke aucune matrice, seulement les m dernières
paires (s, y), m = 5 à 20. La *two-loop recursion* reconstruit l'action de B sur un
vecteur. Coût O(mP).
**Exige un objectif déterministe** (les paires deviennent du bruit avec des mini-batches)
et une *line search*. Excellent en **full batch convexe** : c'est le solveur par défaut de
`LogisticRegression` dans scikit-learn, et le bon choix pour un linear probe.

---

## 6. Learning rate et schedules

**Learning rate** — Coefficient multipliant le gradient. L'hyperparamètre le plus
déterminant, loin devant l'architecture.

**Borne de stabilité** — Sur une quadratique de courbure λ, la convergence exige
`η < 2/λ`. Comme la condition vaut pour toutes les directions, `η_max = 2/λ_max`.
D'où le couplage : le pas maximal est fixé par la direction la plus raide, la vitesse par
la plus plate.

**Edge of stability** — Constat empirique (Cohen et al. 2021) : les réseaux profonds
s'entraînent juste à cette limite, λ_max augmentant spontanément jusqu'à ce que 2/λ_max
touche le η choisi. Le réseau adapte sa courbure au pas imposé.

**Learning rate range test** — Procédure de Leslie Smith (2015). Augmenter η
exponentiellement de 10⁻⁷ à 10 sur ~100 à 200 itérations, tracer la loss en fonction de η
en échelle log. La courbe montre un plateau, une descente, un minimum, une remontée
abrupte. **Ne pas prendre le minimum** (déjà à la limite de stabilité) mais le point de
pente la plus forte, ou le minimum divisé par dix. Outil : `torch-lr-finder`.

**Pourquoi faire varier η** — Au début, le vrai gradient domine le bruit : grand pas utile.
À la fin, le vrai gradient tend vers zéro mais le bruit reste, et on erre dans une boule de
rayon ∝ η/√B. **La seule façon de rétrécir cette boule est de réduire η.** D'où les chutes
brutales de loss après réduction : le modèle cesse de vibrer autour d'une solution déjà
trouvée. C'est un recuit simulé, et η/B joue le rôle d'une température.

**Step decay** — Division par 10 à des époques fixées. Historique, paliers arbitraires.

**Cosine annealing** — `η(t) = η_max · ½ · (1 + cos(π·t/T))`. Défaut actuel : décroissance
douce, un seul paramètre T. `CosineAnnealingLR`.

**One-cycle** — Warmup jusqu'au maximum sur ~30 % du budget, puis descente cosinus.
Associé à la *super-convergence*. `OneCycleLR`. **Le meilleur choix quand le budget
d'époques est contraint.**

**ReduceLROnPlateau** — Réduction quand la validation cesse de s'améliorer. Utile quand le
budget est inconnu, moins performant qu'un cosinus calibré.

**Warm restarts (SGDR)** — Cosinus redémarrant périodiquement, ce qui fait explorer
d'autres bassins. Support des **snapshot ensembles**.

**Warmup** — Montée progressive de η sur 3 à 5 époques (ou 5 % du budget). Deux raisons :
(1) les poids initiaux sont aléatoires et les gradients incohérents ; (2) **le v d'Adam est
mal calibré au début** — moyenne sur ~1000 itérations reposant sur une poignée
d'échantillons, avec une variance énorme, et il est au dénominateur. La bias correction
rétablit l'espérance, pas la variance. Non négociable sur les transformeurs et les grands
lots.

**Couplage LR / batch size** — Multiplier B par k : **règle linéaire** (η × k, Goyal et al.
2017) dans le régime des petits et moyens lots ; **règle en racine** (η × √k) aux très
grands lots, qui découle de maintenir η/√B constant. Aucune n'est un théorème.

**Layer-wise LR decay / discriminative fine-tuning** — Chaque couche reçoit le LR de la
suivante × α (0,65 à 0,9). Les couches basses portent des détecteurs génériques encore
valides, la tête est neuve.

**Entraînement en deux temps** — Plus simple et souvent suffisant. Phase 1 : backbone gelé,
seule la tête s'entraîne à 10⁻³. Phase 2 : dégel des couches hautes à 10⁻⁵.
**L'ordre compte** : dégeler avec une tête encore aléatoire envoie des gradients absurdes
dans le backbone.

**Diagnostic d'une courbe de loss**

| Observation | Interprétation |
|---|---|
| Explosion / NaN | LR trop grand, ou pas de warmup |
| Descente puis remontée durable | LR trop grand pour la phase |
| Descente très lente et régulière | LR trop petit |
| Plateau puis chute nette | normal, le schedule agit |
| Oscillations fortes | batch trop petit, ou LR à la limite |
| Train baisse, val remonte | overfitting |

Toujours journaliser le LR à côté de la loss, sinon les courbes sont ininterprétables.

---

## 7. Fonctions de perte

**Loss** — Le seul objectif que le modèle cherche à minimiser. S'il mesure la mauvaise
chose, il optimisera parfaitement la mauvaise chose sans qu'aucune courbe ne le signale.

**Cross-entropy** — `L = −log(p_correcte)`. C'est la *negative log-likelihood* : minimiser
cette loss équivaut à un maximum de vraisemblance, et à une constante près à la divergence
de Kullback-Leibler entre distribution prédite et cible.

| p_correcte | loss |
|---|---|
| 0,9 | 0,105 |
| 0,5 | 0,693 |
| 0,1 | 2,303 |
| 0,01 | 4,605 |

**Repère d'initialisation** — Un modèle qui n'a rien appris répond uniformément, donc
`loss = log(K)`. Pour 50 classes : ≈ 3,9. Un démarrage éloigné signale un bug.

**Single-label vs multi-label** — **La décision critique en éco-acoustique.**

| | single-label | multi-label |
|---|---|---|
| Activation | softmax (compétition) | sigmoid (K questions indépendantes) |
| Loss | `CrossEntropyLoss` | `BCEWithLogitsLoss` |
| Cible | un indice entier | vecteur de 0,0 et 1,0 de longueur K |
| Décision | argmax | un seuil **par classe** |
| Somme des probas | 1 | libre |

Sur les mêmes logits `[2,0 ; 1,5 ; −1,0]` : softmax donne `[0,60 ; 0,37 ; 0,03]` (somme 1,
il faut choisir) ; sigmoid donne `[0,88 ; 0,82 ; 0,27]` (somme 1,97, A et B présentes).

Deux raisons de choisir le multi-label en éco-acoustique :
1. **Le chevauchement** — plusieurs espèces chantent simultanément ; un softmax ne peut pas
   l'exprimer et apprendra à ignorer les espèces non dominantes.
2. **Le silence** — cas majoritaire en terrain. Avec sigmoid, « rien » se représente
   naturellement (toutes les sorties basses) ; avec softmax il faut inventer une classe
   « fond sonore » en compétition avec les vraies espèces.

**BCE** *(Binary Cross-Entropy)* — Cross-entropy appliquée K fois :
`L = −(1/K)·Σₖ [yₖ·log(pₖ) + (1−yₖ)·log(1−pₖ)]`. Le second terme punit les faux positifs.

**Label smoothing** — Remplacer la cible `[0,1,0]` par `[0,05 ; 0,90 ; 0,05]`.
**Pourquoi** : un softmax n'atteint 1 qu'avec des logits infinis, donc la cross-entropy
reste strictement positive tant que p < 1 et l'optimiseur pousse les logits indéfiniment.
Le modèle apprend que gonfler ses logits est toujours récompensé, et annonce 0,999 sur des
cas ambigus. Une cible lissée **est atteignable** avec des logits finis : il existe un
optimum, le gradient s'y annule, et les probabilités deviennent **calibrées**.
`nn.CrossEntropyLoss(label_smoothing=0.1)`. Pertinent quand l'annotation est incertaine.

**Class imbalance** — Avec 1 positif sur 500, répondre toujours « absent » donne 99,8 %
d'accuracy. Le déséquilibre n'abîme pas l'entraînement : il offre une solution paresseuse
qui réussit.

**pos_weight** — `BCEWithLogitsLoss(pos_weight=...)` multiplie le terme des positifs.
Valeur naturelle : `n_neg/n_pos` par classe, **plafonnée vers 30–50** (un rapport brut de
5000 sur trois exemples déstabiliserait tout l'entraînement).

**Focal loss** (Lin et al. 2017) — `focal = (1−p)^γ × BCE`, γ = 2 typiquement.
**Ce n'est pas une alternative à la BCE, c'en est une généralisation** : γ = 0 redonne
exactement la BCE.
Problème réglé : 10 000 silences faciles à 0,01 de loss totalisent 100, contre 35 pour
50 cas difficiles à 0,7 — le gradient est dominé par ce qu'il n'y a plus à apprendre.
Le facteur réduit un exemple à p = 0,95 d'un facteur 400, un exemple à p = 0,5 d'un
facteur 4.
**pos_weight rééquilibre les classes, la focal loss rééquilibre les difficultés.**
Elles se combinent.

**Pièges d'implémentation** (aucun ne lève d'exception) :
- **Double softmax** : `CrossEntropyLoss` et `BCEWithLogitsLoss` attendent des **logits**
  et appliquent l'activation en interne. La dernière couche du modèle est une linéaire nue.
- **Toujours `BCEWithLogitsLoss`, jamais `BCELoss` + sigmoid** : la version fusionnée
  utilise l'astuce du **log-sum-exp** (soustraction du maximum avant l'exponentielle) et
  reste numériquement exacte sur toute la plage.
- **Types de cible** : `CrossEntropyLoss` veut des entiers `(N,)` ;
  `BCEWithLogitsLoss` veut des flottants `(N, K)`.

**Losses de régression** — **MSE** (sensible aux aberrations, max de vraisemblance sous
bruit gaussien), **MAE** (robuste, gradient constant), **Huber** (quadratique près de zéro,
linéaire au-delà — le bon défaut pour des données de terrain).

---

## 8. Régularisation

**Overfitting** — Zhang et al. (2017) : un CNN standard atteint 100 % de justesse sur
CIFAR-10 aux labels **entièrement randomisés**. Il n'y a rien à apprendre et il apprend
quand même.

**Définition** (Goodfellow) — Toute modification de l'algorithme visant à réduire l'erreur
de généralisation mais pas l'erreur d'entraînement.

**L2 (ridge, weight decay)** — Pénalité `(λ/2)‖θ‖²`, gradient `λθ`. Force de rappel
**proportionnelle à la valeur du poids** : contraction multiplicative, jamais de zéro exact.
Dans la base propre de la hessienne, chaque composante i est multipliée par `λᵢ/(λᵢ+α)` :
les directions de forte courbure sont préservées, les directions plates écrasées.
**L2 supprime les degrés de liberté que les données ne contraignent pas.**
Lecture bayésienne : prior gaussien centré.
En pratique : un seul réglage, `weight_decay ≈ 0.01`, hors biais et paramètres de BatchNorm.

**L1 (lasso)** — Pénalité `λ‖θ‖₁`, gradient `λ·sign(θ)`. Force de rappel **constante**,
indépendante de la valeur : les poids faibles sont poussés à **exactement zéro**, d'où la
**sparsité**. Prior laplacien. Non dérivable en 0 → sous-gradient ou *soft thresholding*.
**Géométrie** : reformulée en contrainte, la région L2 est un disque lisse (tangence
générique), la région L1 un losange dont les **sommets sont sur les axes** — un coin touche
une ligne de niveau pour toute une plage d'orientations, d'où l'annulation de coordonnées.

**Pourquoi L1 est rare en deep learning** — Sparsité non structurée, dont un GPU ne tire
aucun parti ; redondance massive entre poids, donc pas d'interprétabilité comme en
régression linéaire ; mauvais mariage avec Adam. Pour accélérer réellement un modèle, on
fait du *structured pruning*.

**Elastic net** — Combinaison L1 + L2. Usuel en statistique, absent en deep learning.

**Structured pruning** — **Rien à voir avec le dropout.** Le dropout éteint des neurones au
hasard, **temporairement, pendant l'entraînement**, pour régulariser. Le pruning supprime
des canaux ou noyaux **définitivement, après l'entraînement**, sur critère d'utilité, pour
compresser et accélérer.

**Early stopping** — Arrêt quand la validation cesse de s'améliorer. La régularisation la
moins chère. Propriété élégante : dans le cas quadratique, s'arrêter après T itérations
équivaut approximativement à une L2 de coefficient `1/(ηT)`.

**Data augmentation** — La plus puissante. Transformations aléatoires préservant le label,
qui injectent une connaissance a priori et remplacent le jeu fini par une distribution
continue autour de lui.

**SpecAugment** (Park et al. 2019) — Augmentation opérant **directement sur le
spectrogramme**, donc sans recalcul de la transformée de Fourier — d'où son adoption
universelle.
- **Masquage fréquentiel** : annuler une bande de largeur f ∈ [0, F]. Simule un filtrage de
  canal ou un masquage par une autre source.
- **Masquage temporel** : annuler un intervalle de largeur t ∈ [0, T].
- **Time warping** : étirement local du temps. Faible gain pour un coût élevé, souvent omis.
Paramètres indicatifs : F = 15 bandes, T = 30 trames, deux masques par axe.
**Avertissement éco-acoustique** : si le signal cible occupe une bande étroite, un masque
trop large efface toute l'information discriminante en conservant le label — on apprend
alors au réseau à prédire une espèce à partir de rien.

**Augmentations sur la forme d'onde** — À combiner avec SpecAugment, dans l'ordre :
forme d'onde → spectrogramme → SpecAugment.
- **mixup** : combinaison linéaire de deux exemples et de leurs labels. En audio, cela
  superpose deux enregistrements, ce qui reproduit fidèlement le chevauchement naturel.
- **Bruit de fond réel** tiré de segments vides du corpus — la plus pertinente, elle
  apprend l'ambiance exacte des sites.
- Décalage temporel, décalage de hauteur, variation de gain.

**Negative mining** — Sélection délibérée d'exemples négatifs **difficiles** plutôt que
tirés au hasard. Ici : cris de congénères et stridulations d'orthoptères dans la bande
4,4–5,5 kHz, plutôt que du silence ou de la pluie. Même logique que la focal loss —
concentrer le signal d'apprentissage là où le modèle se trompe — mais appliquée à la
constitution du jeu de données plutôt qu'à la pondération de la perte.

**Ordre de priorité** — (1) data augmentation, (2) early stopping sur validation
partitionnée par site, (3) weight decay 0,01, (4) label smoothing 0,1, (5) dropout sur la
tête seulement. L1 : aucune raison ici.

---

## 9. Apprentissage par transfert

**Modèle pré-entraîné / backbone** — Réseau entraîné sur un grand corpus générique,
réutilisé comme extracteur de caractéristiques.

**Tête** *(head)* — Dernière ou dernières couches, spécifiques à la tâche, réentraînées.

**Embedding / représentation** — Vecteur de l'avant-dernière couche, de quelques centaines
à ~1280 dimensions, encodant la structure acoustique **indépendamment de toute liste
d'espèces**. À distinguer des **logits**, qui projettent sur le vocabulaire vu à
l'entraînement — inutilisables si l'espèce cible n'y figure pas.

**Probing** — On gèle l'encodeur pré-entraîné et on n'entraîne qu'une petite tête par-dessus
ses représentations.
- ***Linear probing*** : la tête est une simple régression logistique sur un vecteur unique
  par fenêtre. **La baseline de référence** : si le CNN maison ne la bat pas, il n'apporte
  rien.
- ***Attentive probing*** : la tête est un module d'attention qui apprend **où regarder**
  parmi les représentations *token-level*. Plus puissant, mais la contrainte est l'accès aux tokens avant agrégation donc **impossible sur un modèle figé au format TFLite**.

**Régression logistique** — `z = w·x + b`, `p = σ(z)`, loss BCE. Littéralement un neurone
unique suivi d'une sigmoïde, sans couche cachée : frontière de décision = hyperplan.
Sa loss est **convexe** (un seul minimum, pas de point-selle), donc L-BFGS trouve l'optimum
global exact sans learning rate à régler.

**Fine-tuning** — Réadaptation d'un modèle pré-entraîné sur un jeu spécifique plus petit.
Learning rate un à deux ordres de grandeur sous celui d'un entraînement from scratch
(10⁻⁴ à 10⁻⁵ contre 10⁻³).

**Gel** *(freezing)* — `requires_grad_(False)`. La passe arrière s'arrête à la première
couche gelée : on économise le calcul **et** le stockage des activations en amont.
Geler les couches basses en priorité, ce sont elles qui portent les plus grandes cartes.

**LoRA** *(Low-Rank Adaptation)* — Geler les poids d'origine et n'entraîner que de petites
matrices de rang faible ajoutées. Réduit d'un ou deux ordres de grandeur la mémoire.

**kNN** — Classement d'une donnée nouvelle par vote majoritaire parmi ses k plus proches
voisins stockés. Sans gradient.

**Sur MacBook Air — ce qui passe et ce qui ne passe pas**
- Extraction d'embeddings sur tout le corpus : oui, c'est de l'inférence pure.
- Classifieur léger sur ces vecteurs : trivial, CPU suffit.
- Fine-tuning des dernières couches sur un sous-ensemble : possible mais lent, chronométrer
  une époque avant de s'engager.
- Entraînement from scratch sur le corpus complet : non.
Ordre des remèdes mémoire : réduire la batch size → précision mixte (`torch.autocast`) →
gradient checkpointing.
**Test de cohérence MPS** : entraîner quelques dizaines d'itérations en `mps` et en `cpu`
sur un petit échantillon et vérifier que les pertes suivent des trajectoires comparables.

---

## 10. Ensembles

**Principe** — Faire voter plusieurs modèles. Ce qui compte n'est pas qu'ils soient tous
bons, c'est qu'ils **se trompent différemment**.

**Formule de la variance** — Avec M modèles d'erreur de variance σ² et de corrélation
moyenne ρ :
```
σ²_ens = ρ·σ² + (1 − ρ)·σ²/M
```
Le second terme s'annule quand M croît, **le premier reste** : la corrélation fixe un
plancher infranchissable.

| ρ | M = 3 | M = 5 | M = 20 | M = ∞ |
|---|---|---|---|---|
| 0,8 | 0,87 σ² | 0,84 σ² | 0,81 σ² | 0,80 σ² |
| 0,5 | 0,67 σ² | 0,60 σ² | 0,52 σ² | 0,50 σ² |
| 0,3 | 0,53 σ² | 0,44 σ² | 0,33 σ² | 0,30 σ² |

**Trois modèles vraiment différents valent mieux que vingt variantes du même.** Les gains
saturent entre 3 et 7 modèles. Ajouter un modèle nettement plus faible à poids égal
**dégrade** l'ensemble.

**Fabriquer de la diversité** — Faire varier l'architecture, le modèle pré-entraîné, la
résolution du spectrogramme, les augmentations, le découpage temporel. Changer seulement la
graine donne ρ très élevé.

**Score oracle** — Ce qu'on obtiendrait si un devin indiquait à chaque fois quel modèle
écouter. Borne supérieure = `1 − d/total` (d = les deux se trompent). L'écart entre le
score oracle et le score obtenu mesure le potentiel restant dans la règle de combinaison.

**Taux de désaccord** — `(b+c)/total` dans le tableau croisé des prédictions. Deux modèles
utiles à ensembler ont à la fois de bonnes performances individuelles **et** un désaccord
élevé.

**Règles de combinaison**
- **Soft voting** : moyenne des probabilités. Simple, robuste, exige des modèles calibrés.
- **Moyenne pondérée** : poids ajustés sur validation.
- **Stacking** : un petit modèle (régression logistique) apprend **quand faire confiance à
  qui**. À entraîner sur un fold distinct de ceux des modèles de base.
- **Sélection par classe** : en multi-label, la complémentarité est souvent *par espèce*.
  Prendre chaque modèle là où il gagne. Sans réglage, et directement interprétable.

**Concaténation d'embeddings** *(fusion précoce)* vs **ensemble** *(fusion tardive)* —
Deux mécanismes différents, et **la réponse est empirique**.

| | avantage concaténation | avantage ensemble |
|---|---|---|
| Peu de données annotées | non (2304 dims surapprend) | oui (chaque probe en faible dim) |
| Beaucoup de données | oui (interactions) | moins |
| Calibration hétérogène | non | oui |
| Pondération par classe | difficile | facile |
| Simplicité opérationnelle | oui | non |

Si concaténation : **normaliser chaque bloc d'embeddings** avant, et mettre du weight decay.

**Explosion combinatoire** — 2^M − 1 sous-ensembles. Choisir la meilleure combinaison parmi
mille en regardant la validation, c'est **surapprendre la validation**.
**Sélection gloutonne avec remise** (Caruana et al. 2004) : partir de l'ensemble vide,
ajouter à chaque étape le modèle qui améliore le plus le score de validation, autoriser les
répétitions (ce qui pondère implicitement), arrêter quand plus rien n'améliore. Coût
linéaire en M. Limiter à 5 ou 6 candidats choisis pour leur diversité.

**Snapshot ensemble** — Sauvegarder plusieurs checkpoints d'un **unique** entraînement à
warm restarts, juste avant chaque redémarrage. Plusieurs modèles pour le prix d'un.
La seule version d'ensemble tenant dans un budget de MacBook Air.

---

## 11. Évaluation et méthodologie

**Accuracy** — **À proscrire ici.** Avec 1 positif sur 500, répondre toujours « absent »
donne 99,8 % sans avoir détecté un seul oiseau. L'accuracy mesure essentiellement la
fréquence de la classe majoritaire.

**Matrice de confusion** — TP (vrai positif), FP (faux positif), FN (faux négatif),
TN (vrai négatif).

**Precision** — `TP/(TP+FP)`, la **ligne** des prédictions positives. « Quand j'annonce une
détection, à quelle fréquence ai-je raison ? »

**Recall** — `TP/(TP+FN)`, la **colonne** des cas réellement positifs. « Parmi les vrais
événements, combien j'en attrape ? » Identique au TPR.

Ni l'une ni l'autre ne fait intervenir TN, d'où leur robustesse au déséquilibre.

**F1** — `2PR/(P+R)`, moyenne **harmonique** : elle punit le déséquilibre. Avec P = 1,0 et
R = 0,02, l'arithmétique donnerait 0,51, l'harmonique donne 0,039.

**Courbe précision-rappel** — Balaie tous les seuils. Ne contient aucun TN, donc immune à
l'illusion du déséquilibre.

**AP** *(Average Precision)* — Aire sous la courbe précision-rappel. Une valeur par classe.
**Ce n'est pas le recall** : le recall est une coordonnée à un seuil donné, l'AP résume le
comportement sur **tous** les seuils. Référence « hasard » = la prévalence de la classe.

**mAP** — Moyenne des AP. **Macro** (chaque classe pèse autant) ou **micro** (toutes les
décisions dans le même sac, donc dominée par les espèces communes). Rapporter la macro, et
l'AP par classe en annexe.

**ROC** — Courbe TPR (= recall) en ordonnée contre FPR en abscisse, en balayant le seuil.
Du seuil 1 (0,0) au seuil 0 (1,1). Diagonale = hasard.

**FPR** — `FP/(FP+TN)`. Proportion des vrais négatifs signalés à tort.

**AUROC / ROC-AUC / AUC** — Trois noms pour la même chose : l'aire sous la courbe ROC.
1 = parfait, 0,5 = hasard, < 0,5 = labels probablement inversés.
**Interprétation exacte** : `AUROC = P(score d'un positif tiré au hasard > score d'un
négatif tiré au hasard)`. Égale la statistique U de Mann-Whitney normalisée. Elle mesure
donc la **qualité du classement**, pas la décision ni la calibration — seul l'ordre compte.

**La limite de l'AUROC, chiffrée** — 20 positifs, 10 000 négatifs, tous les positifs dans
les 200 premiers rangs : recall = 1,00, FPR = 180/10 000 = 0,018 (AUROC > 0,99), mais
precision = 0,10 — neuf alertes sur dix sont fausses. Le TN au dénominateur dilue les faux
positifs. La version **macro** corrige la pondération entre classes, pas ce phénomène.

| | AUROC | AP |
|---|---|---|
| Mesure | qualité du classement | precision atteignable à chaque recall |
| Référence hasard | 0,5 toujours | la prévalence |
| Sensible au déséquilibre | peu (parfois trop peu) | oui, fidèlement |
| Comparable entre espèces de rareté différente | oui | difficilement |
| Reflète le coût opérationnel | non | oui |
| Stabilité avec peu de positifs | correcte | très bruitée |

**Quoi rapporter** — Les deux, elles ne sont pas redondantes. **AUROC macro** pour comparer
les modèles entre eux et se situer par rapport à la littérature (métrique officielle de
BirdCLEF+ 2026). **AP macro + AP par classe** pour la difficulté réelle de chaque espèce.
**Precision et recall au seuil retenu** pour la description opérationnelle finale.

**Seuil de décision** — Un seuil **par classe**, jamais un seuil global, calibré **sur la
validation**, jamais sur le test. Critère aligné sur l'usage : si un humain vérifie,
privilégier le recall ; si la sortie alimente un indice automatique, privilégier la
precision. Formulation utile : « meilleur seuil garantissant 80 % de recall ».

**Data leakage** — Le point qui décide de la validité de tout. Canaux de fuite en
éco-acoustique, par gravité :
1. **Fenêtres du même enregistrement** — quasi identiques (même fond, même individu,
   autocorrélation temporelle).
2. **Même individu** — un oiseau a un répertoire idiosyncratique ; le modèle apprend
   l'individu, pas l'espèce.
3. **Même site** — ambiance, réverbération, cortège d'espèces. Souvent le canal dominant.
4. **Même enregistreur** — la réponse en fréquence d'un micro est une signature.
5. **Prétraitement calculé sur tout le corpus** — statistiques de normalisation, paramètres
   PCEN. À calculer sur le training fold seul.
6. **Métadonnées** — nom de fichier contenant l'espèce, ordre de tri non aléatoire.

**Règle** — Découper au niveau de l'unité la plus grossière susceptible de porter une
signature partagée : le **site**, parfois site × saison.
`StratifiedGroupKFold(n_splits=5)` avec `groups = df["site_id"]`. Jamais `train_test_split`.

**Ce qu'il faut dire** — Un split par site donnera un score **nettement inférieur** à un
split aléatoire (écarts de 15 à 20 points de mAP courants dans la littérature). Ce n'est
pas une régression, c'est la vraie performance qui apparaît. Formuler comme un résultat :
« avec un découpage aléatoire j'obtiens X, avec un découpage par site Y ; le second est le
seul qui prédit ce qui se passera sur un nouveau site, donc c'est celui que je rapporte ».

**k-fold** — Protocole d'**évaluation** : k entraînements, une part laissée de côté à
chaque tour, moyenne des scores. Le produit est un **chiffre**, pas un modèle. Distinct de
l'ensemble, qui est une méthode de **prédiction** — même si les k modèles peuvent ensuite
être recyclés en ensemble.

**Train / val / test** — Le train ajuste les poids, la validation porte **tous** les choix
(architecture, hyperparamètres, seuils, arrêt), le test ne sert **qu'une fois**. Le test
mesure ce qui reste après ton propre surapprentissage de la validation.

**Variance et graines** — Relancer l'entraînement final avec 3 à 5 graines, rapporter
moyenne ± écart-type. Un gain de 0,8 point n'existe pas si l'écart-type entre graines vaut
1,5 point.

**Baselines** — Deux. Une triviale (prédire la prévalence, dont la mAP n'est pas zéro) et
une sérieuse (linear probe sur embeddings gelés).

**Ce qu'il faut rapporter** — Stratégie de découpage explicite ; nombre de positifs par
classe dans val et test ; AP par classe et macro-mAP ; moyenne ± écart-type ; les
baselines ; **les classes exclues et le critère d'exclusion** (avec moins de ~10 positifs,
une AP est du bruit pur — l'écrire honnêtement).

**La loss n'est pas la métrique** — On optimise une BCE pondérée parce qu'elle est
différentiable, on évalue en AP parce que c'est ce qui a un sens écologique. Une BCE qui
baisse pendant que l'AP stagne est un cas possible qu'il faut savoir diagnostiquer.

---

## 12. Clustering (approche non supervisée)

**Principe** — Les embeddings sont produits qu'il y ait des annotations ou non. Ce qu'on en
fait ensuite est libre : régression logistique avec labels, clustering sans.
**Toujours prendre les embeddings, jamais les logits** : ces derniers projettent sur un
vocabulaire d'espèces qui peut ne pas contenir les espèces locales.

**Intérêt en milieu peu couvert** — Le clustering ne suppose pas de liste d'espèces
préétablie. Il peut isoler un type sonore récurrent que personne n'a étiqueté — insectes,
amphibiens, espèces néotropicales absentes des corpus d'entraînement.

**Préparation**
1. **Normaliser en norme 1** et travailler en **distance cosinus**. La norme d'un embedding
   encode plutôt l'intensité que l'identité : l'oublier produit des clusters qui séparent
   les sons forts des sons faibles.
2. **PCA vers 50–100 dimensions** avant clustering : réduit le bruit, accélère, et conserve
   la géométrie globale puisqu'elle est linéaire.

**Le piège UMAP / t-SNE** — Ce sont des outils de **visualisation**, pas de représentation.
Ils déforment les distances par construction et peuvent faire apparaître des amas
inexistants. La pratique HDBSCAN-sur-UMAP est répandue mais méthodologiquement contestable.
**Version défendable : clusteriser dans l'espace PCA, visualiser avec UMAP.**

**HDBSCAN** — Recommandé. Fondé sur la densité, détermine lui-même le nombre de clusters,
et possède une catégorie **bruit**. Décisif quand la majorité des fenêtres ne contiennent
rien : un algorithme qui assigne tout mélange le silence aux signaux.

**k-means** — Exige de fixer k, suppose des clusters sphériques de taille comparable,
assigne tout. Utile comme référence, insuffisant seul.

**Clustering agglomératif** — Produit un dendrogramme, donc une hiérarchie de similarité
acoustique. Parlant pour un rapport.

**Le problème de l'évaluation** — Un algorithme de clustering trouve **toujours** des
clusters, même sur du bruit pur. Rien dans sa sortie ne dit s'ils correspondent à des
espèces, à des sites ou à des conditions météo.
- **Métriques internes** (silhouette, Davies-Bouldin, Calinski-Harabasz) : ne mesurent que
  la compacité géométrique, favorisent mécaniquement les hypothèses de l'algorithme.
  Utiles pour choisir un nombre de clusters, **jamais pour valider**.
- **Métriques externes** (AMI, ARI) : comparent à une vérité terrain. Les seules
  informatives, et elles exigent des labels.
**Conclusion opérationnelle** : ne pas annoter pour *entraîner* ne doit pas signifier ne
rien annoter du tout. Quelques centaines de fenêtres de validation changent le statut du
résultat.

**Le piège qui va arriver** — **Les clusters sépareront d'abord les sites, pas les
espèces.** L'ambiance acoustique d'un site est une structure réelle et plus saillante que
la biologie. Trois parades :
1. **PCEN** en amont, qui supprime le fond stationnaire propre à chaque site.
2. **Tableau de contingence clusters × métadonnées** (site, date, heure). Un cluster à 95 %
   mono-site est un artefact. Ce tableau doit figurer dans le rapport.
3. **Stabilité inter-sites** : un cluster présent sur plusieurs sites indépendants a de
   bonnes chances d'être biologique.

**Cluster d'abord, annote ensuite** *(agile modeling, human-in-the-loop)* — Le compromis
qui domine :
1. Extraire les embeddings de tout le corpus.
2. Clusteriser → une quarantaine de groupes.
3. Écouter **5 à 10 exemples par cluster** (quelques centaines d'écoutes au lieu de
   dizaines de milliers).
4. Étiqueter **le cluster**, pas chaque fenêtre. Subdiviser les clusters mélangés, écarter
   le bruit.
5. Propager les labels → jeu d'entraînement.
Le clustering dit **où regarder** : annoter au hasard ferait écouter mille silences pour
trouver dix chants. Perch a été conçu explicitement pour ce type de flux.

---

## 13. Modèles pré-entraînés

**Rappel** — Toujours prendre les **embeddings** (avant-dernière couche), pas les
**logits**, qui projettent sur le vocabulaire d'espèces vu à l'entraînement. Voir §9.

**BirdNET** — Kahl, Wood, Eibl & Klinck (2021), *Ecological Informatics* 61:101236.
CNN multi-espèces sur fenêtres de 3 s, environ 6 500 espèces. Développé par le K. Lisa Yang
Center for Conservation Bioacoustics (Cornell) avec la Chemnitz University of Technology.
Un package Python officiel expose bien les embeddings : l'objection porte sur l'ergonomie
et le plafond de performance, pas sur la faisabilité.
**Licence CC BY-NC-SA 4.0** — non commercial et partage à l'identique, ce qui pose problème
pour un outil redistribué avec ses poids. Contrainte de conception, pas de détail juridique.

**Perch** — Modèle bioacoustique Google DeepMind / Google Research.
**Perch 2.0** (van Merriënboer et al., 2025, arXiv:2508.04665) : EfficientNet-B3,
~12 M paramètres, embeddings **1536-d**, fenêtres de **5 s à 32 kHz**, plus de
14 500 espèces **dont des amphibiens**. Embeddings conçus pour être linéairement séparables.
**Logits par espèce non calibrés** et peu fiables pour les espèces rares — les auteurs
recommandent d'ajuster ses propres seuils sur ses propres données, ce qui rejoint la règle
du §11 (un seuil par classe, calibré sur validation).
**Licence Apache 2.0.** Requiert actuellement TensorFlow 2.20 et un GPU, variante CPU
annoncée sans date — d'où l'intérêt du passage par ONNX.

**BEATs** — *Bidirectional Encoder representation from Audio Transformers*, Chen et al.,
Microsoft, ICML 2023, arXiv:2212.09058. Pré-entraînement audio auto-supervisé itératif :
un tokenizer acoustique et le modèle sont optimisés tour à tour, avec un objectif de
prédiction de labels discrets plutôt qu'une reconstruction. Entraîné sur AudioSet, donc
généraliste. PyTorch.

**BEATs_NLM** (aussi *NatureBEATs*) — L'encodeur BEATs extrait de NatureLM-audio, après
fine-tuning intégral sur un corpus bioacoustique. On jette le LLM, on garde l'encodeur.

**NatureLM-audio** — Earth Species Project, ICLR 2025, arXiv:2411.07186. Encodeur BEATs,
connecteur Q-Former, Llama-3.1-8B-Instruct. Répond en langage naturel à des questions sur
un clip audio, en zero-shot. **Fonctionne à 16 kHz**, d'où le plafond d'analyse à 8 kHz.

**BirdMAE (Bird-MAE)** — Rauch, Heinrich, Moummad, Joly, Sick & Scholz (2025), TMLR,
arXiv:2504.12880. *Masked autoencoder* spécialisé, pré-entraîné sur BirdSet. ViT-B,
92,9 M paramètres, embeddings 768-d, 5 s à 32 kHz. PyTorch, poids sur HuggingFace
(`DBD-research-group/Bird-MAE-{Base,Large,Huge}`).
Principe du MAE : on masque une grande partie du spectrogramme et on entraîne le modèle à
reconstruire ce qui manque — pas besoin d'étiquettes, la supervision vient du signal
lui-même.

**ProtoCLR** — Moummad, Serizel, Benetos & Farrugia (2024), hal-04696391.
IMT Atlantique / Lab-STICC. Apprentissage contrastif prototypique : au lieu de comparer les
exemples deux à deux, on les compare à des **prototypes de classe**, ce qui réduit fortement
le coût. Validé sur du transfert *few-shot* avec entraînement sur enregistrements focaux et
évaluation sur *soundscapes*, donc mesure explicite de la robustesse au décalage de domaine.

**PANNs / YAMNet** — Modèles d'incorporation audio généralistes, utiles comme comparaison
hors avifaune.

---

## 14. Outils et bibliothèques

**bacpipe** — *BioAcoustic Collection Pipeline*, Kather, Haupert, Ghani & Stowell,
arXiv:2604.11560 (avril 2026). `pip install bacpipe`, github.com/bioacoustic-ai/bacpipe.
Unifie l'accès à une vingtaine de modèles bioacoustiques : rééchantillonnage automatique
vers la fréquence propre à chaque modèle, segmentation, mise en lots, génération et
stockage des embeddings, puis évaluation par probing linéaire et kNN, clustering (AMI, ARI)
et visualisation UMAP dans une interface graphique. Conçu pour deux publics : écologues peu
à l'aise avec Python, et informaticiens.
Modèles embarqués : birdnet, perch_bird, perch_v2, avesecho_passt, aves_especies,
birdaves_especies, audiomae, beats, biolingual, birdmae, convnext_birdset, surfperch,
vggish, plus des modèles insectes et chiroptères. Réduction de dimension : PCA, sparse PCA,
t-SNE, UMAP. Classes bas niveau : `Loader`, `AudioHandler`, `Embedder`, `Classifier`.
**Trois avertissements** : (1) un MacBook Air ne fera pas tourner vingt modèles — en choisir
3 ou 4 ; (2) paquet jeune (v1.1.2), épingler la version pour la reproductibilité ;
(3) **vérifier en priorité si le probing intégré permet un découpage groupé par site**,
sinon ses chiffres seront optimistes pour la raison exposée au §11.

**scikit-maad** — Bibliothèque Python d'écoacoustique quantitative (MNHN / Paris-Saclay ;
Ulloa, Haupert, Latorre, Aubin, Sueur, *Methods in Ecology and Evolution*, 2021). Importée
sous `maad`, licence BSD. **Sylvain Haupert, co-encadrant du stage, en est co-auteur.**
Contient plus de 50 **indices acoustiques** (ACI complexité temporelle, ADI diversité
spectrale, BI activité biotique, NDSI rapport biophonie/anthropophonie, H entropie, RMS),
la détection de régions d'intérêt sur spectrogramme, et des spectrogrammes calibrés avec
mesure de niveau sonore en unités physiques.
**Quand s'en servir** : exploration de données, baseline peu coûteuse, couche
complémentaire dans le rapport (langage de l'écoacoustique classique).
**Réserve** : la validité écologique des indices acoustiques est activement débattue
(interprétabilité, reproductibilité, sensibilité au bruit non biologique). Les présenter
comme une description du signal, pas comme une mesure de biodiversité.

**torchaudio / librosa / scikit-learn** — Chaîne de traitement audio en torch
(`MelSpectrogram`, `AmplitudeToDB`), implémentation de référence du PCEN (`librosa.pcen`),
et outillage classique (`StratifiedGroupKFold`, `LogisticRegression` avec solveur `lbfgs`,
HDBSCAN, PCA).

**Streamlit** — Framework Python qui transforme un script en application web. Le script
entier est ré-exécuté à chaque interaction, l'état persistant passe par `st.session_state`.

**Gradio** — Même créneau, orienté démos de modèles, avec des composants audio natifs
(lecteur, forme d'onde) et une intégration directe aux HuggingFace Spaces. **Meilleur des
deux pour une boucle de réannotation audio**, donc pour le flux cluster-puis-annote du §12.

---

## 15. Organismes, concours et références

**Cornell** — *K. Lisa Yang Center for Conservation Bioacoustics*, Cornell Lab of
Ornithology. Développeur de BirdNET avec la Chemnitz University of Technology.
« La codebase de Cornell » désigne BirdNET-Analyzer.

**BirdCLEF / LifeCLEF** — Concours annuel de référence du domaine, annexe du lab LifeCLEF
(conférence CLEF), hébergé sur Kaggle. Identification multi-label d'espèces dans des
paysages sonores.
**BirdCLEF+ 2026** : 234 classes multi-taxons (oiseaux, amphibiens, insectes, reptiles),
zones humides du Pantanal (Brésil), évaluation en **ROC-AUC macro-moyennée** sur des
fenêtres de 5 s d'un test caché, budget d'inférence **CPU uniquement**. Données
d'entraînement issues de xeno-canto et iNaturalist, rééchantillonnées à 32 kHz, format ogg.
Les **working notes** des éditions récentes sont la meilleure source de pratiques concrètes
du domaine. Les solutions gagnantes sont presque toujours des **ensembles**.

**Prix Netflix** — Concours 2006-2009 doté d'un million de dollars pour améliorer de 10 % le
système de recommandation. Remporté par un agrégat de plus d'une centaine de modèles.
**Netflix n'a jamais déployé la solution gagnante**, jugée trop complexe pour le gain
obtenu. Leçon : un ensemble de sept modèles qui tourne vaut mieux qu'un ensemble de trente
qu'on ne pourra jamais exécuter sur le corpus complet.

---

## 16. Le signal cible — *Anomaloglossus blanci*

Source : Fouquet, Vacher, Courtois, Villette, Reizine, Gaucher, Jairam, Ouboter & Kok
(2018), *Zootaxa* 4379(1):1–23, doi:10.11646/zootaxa.4379.1.1.

- Cri à **note unique**, longueur **0,090 à 0,103 s**
- **Fréquence dominante 4,75 kHz** en moyenne, intervalle **4,48 à 5,41 kHz**
- Légère modulation ascendante d'environ 0,1 kHz
- **Notes isolées répétées**, et non trains de notes soudées — discriminant vis-à-vis de
  plusieurs congénères
- Activité : chant diurne depuis les berges de criques, pics à l'aube (6–7 h) et en fin
  d'après-midi (16–17 h)
- Habitat : petites criques sableuses ou rocheuses, 50 à 200 m d'altitude

Congénères sympatriques ou proches, **tous dans la même bande spectrale** :
*A. degranvillei*, *A. dewynteri*, *A. surinamensis*, *A. baeobatrachus*, et
*A. saramaka* au Suriname (trains de 6 à 11 notes de 0,028 s, dominante 4,84 kHz).

### Implications pour la chaîne de traitement

*Déductions à partir des paramètres ci-dessus, à vérifier sur les données réelles — ce ne
sont pas des valeurs issues de la littérature.*

**Le discriminant principal est temporel, pas fréquentiel.** Les congénères occupent la même
bande. Ce qui sépare *A. blanci* d'*A. saramaka*, c'est la structure des notes : isolées de
~95 ms contre trains de notes de 28 ms. Une note de 28 ms exige une résolution temporelle
franchement inférieure — avec hop = 320 à 32 kHz (10 ms par trame), une note de *saramaka*
couvre à peine 3 trames. **Comparer n_fft = 512 (16 ms) et n_fft = 1024 (32 ms)** plutôt que
retenir 1024 par défaut, ou empiler les deux résolutions en canaux (§2).

**`f_max` peut descendre bien en dessous de 14 kHz.** La fondamentale est à 4,75 kHz. Reste
à vérifier sur spectrogramme la présence d'harmoniques exploitables avant de fixer la borne.

**SpecAugment doit être recalibré.** Le signal occupe ~1 kHz. Sur une échelle mel de
50 à 14 000 Hz en 128 bandes, cela représente une dizaine de bandes seulement. Un masquage
fréquentiel à F = 15 effacerait l'intégralité de l'information discriminante tout en
conservant le label — exactement le cas nuisible signalé au §8. Réduire F en conséquence.

**Le duty cycle de 3,3 % contraint le volume de corpus** : 2 min par heure, soit 48 min par
enregistreur et par jour.

**L'heure est un prior fort et un risque de fuite.** Les pics d'activité à 6–7 h et 16–17 h
sont une information écologique réelle, mais un modèle ou un clustering peut s'en saisir
comme raccourci. À croiser systématiquement dans le tableau de contingence du §12, et à ne
pas fournir en entrée du modèle sans y avoir réfléchi.

**Negative mining** — Les congénères de la bande 4,4–5,5 kHz sont les négatifs à privilégier,
pas le silence.

---

## Annexe — points de vigilance récurrents

1. **Le découpage par site est une décision d'architecture de projet, pas un réglage.**
   Il doit être en place avant la première ligne d'entraînement.
2. **Les paramètres du spectrogramme plafonnent ce que le réseau pourra apprendre.**
   Regarder les spectrogrammes à l'œil avant d'entraîner quoi que ce soit.
3. **La chaîne de prétraitement doit être identique à l'entraînement et au déploiement.**
   L'écrire comme un `nn.Module` unique inclus dans le modèle exporté.
4. **Toujours la dernière couche en linéaire nue** — les losses de PyTorch incluent
   l'activation.
5. **`model.eval()` et `zero_grad()`** — les deux oublis les plus coûteux, et aucun ne
   lève d'erreur.
6. **Journaliser le learning rate à côté de la loss**, sinon les courbes ne s'interprètent
   pas.
7. **Les statistiques de normalisation se calculent sur le training fold seul.**
8. **Le test ne se regarde qu'une fois.**
9. **La licence de BirdNET (CC BY-NC-SA 4.0) contraint la conception** d'un outil
   redistribué avec ses poids. À trancher avant de choisir le modèle de base, pas après.
10. **Les logits de Perch ne sont pas calibrés** — ajuster ses propres seuils sur ses
    propres données, espèce par espèce.
11. **Un modèle TFLite est figé** : ni fine-tuning, ni attentive probing, ni accès aux
    couches intermédiaires.
12. **Le discriminant entre congénères est temporel**, pas fréquentiel. Calibrer la
    résolution et les augmentations en conséquence.

---

## À compléter

- **Corpus** — entrée laissée sans définition dans la version précédente du glossaire.
  À renseigner : volume, nombre de sites, nombre d'enregistreurs, période couverte,
  état d'annotation.
