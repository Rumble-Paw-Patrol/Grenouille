"""Tableaux des benchmarks du projet, en images PNG (demande de Léonard du 26/09/2026).

Les tableaux Markdown s'affichent mal dans Xcode : chaque benchmark a ici son tableau, rendu en
PNG, lisible partout. Une seule source : les données ci-dessous. Pour ajouter ou corriger un
tableau, modifier `TABLEAUX` puis relancer :

    uv run --group notebook python documentation/tableaux/generer.py

Rendu avec Pillow et les polices DejaVu fournies par matplotlib (groupes `notebook` ou `app`) :
rien à télécharger. Images en palette de 48 couleurs, pour rester légères.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ICI = Path(__file__).resolve().parent
POLICES = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

# Encre et surfaces : palette de référence du skill dataviz (clair).
FOND = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
ENCRE_3 = "#898781"
FILET = "#e1e0d9"
ENTETE = "#efeee9"
ZEBRE = "#f6f5f1"


def _teinte(hexa: str, part: float) -> str:
    """Couleur mêlée au blanc (part = part de la couleur)."""
    r, g, b = (int(hexa[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(255 - (255 - c) * part):02x}" for c in (r, g, b))


# Étiquettes des cellules : fond teinté + légende en bas de l'image. Le texte de la cellule dit
# toujours la même chose que la couleur (jamais la couleur seule).
ETIQUETTES = {
    "bon": (_teinte("#0ca30c", 0.18), "facile, déjà en place, à considérer"),
    "moyen": (_teinte("#fab219", 0.28), "moyen, à surveiller"),
    "dur": (_teinte("#d03b3b", 0.16), "difficile, écarté, risqué"),
    "ref": (_teinte("#52514e", 0.12), "référence"),
    "neuf": (_teinte("#2a6fdb", 0.13), "ajouté le 26/09/2026"),
    "attente": (_teinte("#8a5cd0", 0.13), "en discussion, pas programmé"),
}


@dataclass
class C:
    """Cellule : texte et étiquette de couleur (clé de `ETIQUETTES`)."""

    texte: str
    etiquette: str | None = None


@dataclass
class Colonne:
    titre: str
    largeur: float  # part relative
    code: bool = False  # police à chasse fixe (noms de têtes, commandes)


@dataclass
class Section:
    colonnes: list[Colonne]
    lignes: list[list[str | C]]
    titre: str = ""
    etiquettes_lignes: dict[int, str] = field(default_factory=dict)  # ligne → étiquette


@dataclass
class Tableau:
    fichier: str
    titre: str
    sous_titre: str
    sections: list[Section]
    notes: list[str] = field(default_factory=list)
    largeur: int = 2000


# --- Données ------------------------------------------------------------------------------------

ENCODEURS = Tableau(
    "encodeurs",
    "Encodeurs du projet",
    "bacpipe 1.3.5. Débit : i5-1145G7, CPU seul (DECISIONS n° 72), campagne = une semaine de pose,"
    " 575 h d'audio. Jetons et couches : relevés dans le code sans exécuter les modèles (n° 111).",
    [
        Section(
            [
                Colonne("Encodeur\nframework", 1.55, code=True),
                Colonne("f_e", 0.5),
                Colonne("Fenêtre", 0.55),
                Colonne("Dim.", 0.5),
                Colonne("Fen./s", 0.5),
                Colonne("Campagne", 0.74),
                Colonne("Mémoire", 0.6),
                Colonne("Ce que bacpipe rend", 1.6),
                Colonne("Jetons (avant agrégation)", 1.85),
                Colonne("Couches intermédiaires", 1.45),
            ],
            [
                [
                    "birdnet\nTensorFlow / Keras, v2.4",
                    "48 kHz",
                    "3 s",
                    "1 024",
                    "43,3",
                    "9 h",
                    "2,0 Go",
                    "sortie de l'avant-dernière couche (layers[-3])",
                    C(
                        "Facile : la carte avant l'agrégation globale, comme l'embedding"
                        " (tf.keras.Model). Taille à mesurer",
                        "bon",
                    ),
                    C(
                        "Facile : n'importe quelle couche. Licence non commerciale : référence"
                        " seulement",
                        "bon",
                    ),
                ],
                [
                    "protoclr\nPyTorch (CvT-13)",
                    "16 kHz",
                    "6 s",
                    "384",
                    "17,5",
                    "11 h",
                    "1,7 Go",
                    "moyenne des jetons du dernier étage",
                    C("Facile : hook sur le dernier étage", "bon"),
                    C("Facile : 3 étages, un hook chacun", "bon"),
                ],
                [
                    "perch_v2\nONNX Runtime, sans TensorFlow",
                    "32 kHz",
                    "5 s",
                    "1 536",
                    "7,0",
                    "33 h",
                    "4,1 Go",
                    "embedding, jetons, spectrogramme, logits des 14 795 classes",
                    C("Déjà utilisés : 16 temps × 4 fréquences × 1 536", "bon"),
                    C(
                        "Difficile : le graphe ONNX ne sort que 4 tenseurs, il faudrait le"
                        " modifier",
                        "dur",
                    ),
                ],
                [
                    "birdmae_base\nPyTorch, Hugging Face",
                    "32 kHz",
                    "5 s",
                    "768",
                    "3,7",
                    "62 h",
                    "1,5 Go",
                    "l'embedding agrégé (moyenne des patchs puis fc_norm), malgré le nom"
                    " last_hidden_state",
                    C(
                        "Facile : output_hidden_states=True, 1 jeton de classe + 32 temps"
                        " × 8 fréquences",
                        "bon",
                    ),
                    C("Facile : même option, 13 sorties", "bon"),
                ],
                [
                    "convnext_birdset\nPyTorch, Hugging Face",
                    "32 kHz",
                    "5 s",
                    "1 024",
                    "3,7",
                    "62 h",
                    "2,7 Go",
                    "pooler_output (moyenne de la dernière carte)",
                    C(
                        "Facile : output_hidden_states=True, dernière carte fréquence × temps",
                        "bon",
                    ),
                    C("Facile : les 4 étages, même option", "bon"),
                ],
                [
                    "beats\nPyTorch",
                    "16 kHz",
                    "5 s",
                    "768",
                    "2,9",
                    "81 h",
                    "2,2 Go",
                    "la moyenne des jetons, faite dans bacpipe",
                    C(
                        "Facile : avg_pooling = False, une ligne. Grille ≈ 8 fréquences × 31"
                        " temps, à mesurer",
                        "bon",
                    ),
                    C("Moyen : hook sur les couches du transformer", "moyen"),
                ],
                [
                    "naturebeats\nPyTorch (BEATs, poids NatureLM)",
                    "16 kHz",
                    "5 s",
                    "768",
                    "2,8",
                    "82 h",
                    "2,3 Go",
                    "comme BEATs",
                    C("Facile : comme BEATs", "bon"),
                    C("Moyen : comme BEATs", "moyen"),
                ],
                [
                    "perch_bird\nTensorFlow SavedModel (Perch v1)",
                    "32 kHz",
                    "5 s",
                    "1 280",
                    "1,8",
                    "129 h",
                    "2,4 Go",
                    "embedding, logits, spectrogramme",
                    C("Difficile : la signature n'expose pas la carte avant agrégation", "dur"),
                    C("Difficile : même raison", "dur"),
                ],
                [
                    "birdmae (Huge)\nPyTorch, Hugging Face",
                    "32 kHz",
                    "5 s",
                    "1 280",
                    C("0,5", "dur"),
                    C("494 h", "dur"),
                    "4,2 Go",
                    "comme Base",
                    C("Facile : comme Base (Huge à confirmer)", "bon"),
                    C("Facile : comme Base", "bon"),
                ],
            ],
        ),
        Section(
            [
                Colonne("Autres modèles\nde bacpipe", 1.55, code=True),
                Colonne("f_e", 0.5),
                Colonne("Fenêtre", 0.55),
                Colonne("Domaine", 1.45),
                Colonne("Jetons / couches", 2.2),
                Colonne("Intérêt ici", 2.45),
            ],
            [
                [
                    "audioprotopnet\nPyTorch, Hugging Face",
                    "32 kHz",
                    "5 s",
                    "oiseaux (BirdSet)",
                    "jetons déjà gardés par bacpipe",
                    C("À considérer : oiseaux, 32 kHz", "bon"),
                ],
                [
                    "birdnet_v3\nPyTorch",
                    "32 kHz",
                    "3 s",
                    "oiseaux",
                    "à lire",
                    C("À considérer si la licence le permet", "bon"),
                ],
                [
                    "aves_especies, birdaves_especies\nPyTorch (wav2vec2)",
                    "16 kHz",
                    "1 s",
                    "animaux",
                    "toutes les couches (extract_features)",
                    C("Moyen : fenêtre courte, à la taille d'une note", "moyen"),
                ],
                [
                    "avesecho_passt\nPyTorch (PaSST)",
                    "32 kHz",
                    "3 s",
                    "oiseaux",
                    "hook",
                    C("Moyen", "moyen"),
                ],
                [
                    "mix2\nPyTorch",
                    "16 kHz",
                    "3 s",
                    "amphibiens / insectes (à vérifier)",
                    "hook",
                    C("À lire", "moyen"),
                ],
                [
                    "rcl_fs_bsed\nPyTorch",
                    "22,05 kHz",
                    "0,2 s",
                    "détection d'événements, peu d'exemples",
                    "hook",
                    C("À lire : fenêtre de la taille d'une note", "moyen"),
                ],
                [
                    "surfperch\nTensorFlow (Perch)",
                    "32 kHz",
                    "5 s",
                    "récifs coralliens",
                    "comme Perch v1",
                    "faible",
                ],
                [
                    "biolingual\nPyTorch (CLAP)",
                    "48 kHz",
                    "10 s",
                    "bioacoustique + texte",
                    "couches du modèle audio",
                    "faible (fenêtre de 10 s)",
                ],
                ["audiomae\nPyTorch (ViT)", "16 kHz", "10 s", "audio général", "hook", "faible"],
                [
                    "vggish\nTensorFlow",
                    "16 kHz",
                    "1 s",
                    "audio général",
                    "—",
                    "faible (référence ancienne)",
                ],
                [
                    "insect66, insect459\nPyTorch",
                    "44,1 kHz",
                    "5,5 s",
                    "insectes",
                    "embeddings du modèle",
                    "faible (orthoptères = faux amis)",
                ],
                [
                    "bat, batdetect2_…\nPyTorch",
                    "256 kHz",
                    "1 s",
                    "chauves-souris",
                    "—",
                    C("Aucun : ultrasons", "dur"),
                ],
                [
                    "google_whale, hbdet\nTensorFlow",
                    "24 / 2 kHz",
                    "2–4 s",
                    "cétacés",
                    "—",
                    C("Aucun", "dur"),
                ],
            ],
            titre="Hors projet",
        ),
    ],
    [
        "Facile : une option ou un hook, puis la vérification de notes.md (comparer l'embedding"
        " à la moyenne, au maximum et au jeton de classe). Moyen : lire le code du modèle pour"
        " choisir la couche. Difficile : modifier ou reconstruire le modèle exporté.",
        "birdmae (Huge) : hors de portée de l'i5 (×1,2 le temps réel) ; birdmae_base est la"
        " variante retenue pour le benchmark (n° 72).",
    ],
)

TETES = Tableau(
    "tetes",
    "Benchmark des têtes",
    "blanci heads --encoder <stock> : toutes les têtes sur les mêmes fenêtres et les mêmes plis"
    " groupés par micro ; comparées à la logistique par bootstrap apparié (n° 92–93).",
    [
        Section(
            [
                Colonne("Tête", 1.35, code=True),
                Colonne("Famille", 0.8),
                Colonne("Score d'une fenêtre", 2.4),
                Colonne("Apprentissage", 0.95),
                Colonne("Réglages", 1.35),
            ],
            [
                [
                    "exemplar_medoid",
                    "similarité",
                    "cosinus au positif le plus central : un seul exemple",
                    "aucun",
                    "—",
                ],
                [
                    "exemplar",
                    "similarité",
                    "cosinus au positif le plus proche",
                    "aucun",
                    "k (R39, exemplar:k=3)",
                ],
                [
                    "knn",
                    "similarité",
                    "plus proche positif − plus proche négatif",
                    "aucun",
                    "k, pondération (R39)",
                ],
                [
                    "simple_prototype",
                    "prototype",
                    "cosinus à μ₊, la moyenne des positifs",
                    "aucun",
                    "—",
                ],
                [
                    "prototype",
                    "prototype",
                    "μ₊ − μ₋ avec des négatifs appariés : retire le fond sonore commun",
                    "aucun",
                    "—",
                ],
                [
                    C("logistic", "ref"),
                    C("linéaire", "ref"),
                    C("régression logistique L2, classes équilibrées : la référence", "ref"),
                    C("oui", "ref"),
                    C("C par validation groupée", "ref"),
                ],
                [
                    "logistic_to_prototype",
                    "linéaire",
                    "R30 : logistique tirée vers le prototype, ½‖w − w₀‖²",
                    "oui",
                    "C",
                ],
                [
                    "lda_shrunk",
                    "linéaire",
                    "R31 : LDA à covariance rétrécie (Ledoit-Wolf)",
                    "forme fermée",
                    "aucun",
                ],
                [
                    "loss:<nom>",
                    "linéaire",
                    "R34, R35 : même tête, autre perte (tableau des pertes)",
                    "oui",
                    "C",
                ],
                [
                    "logistic:<pooling>",
                    "linéaire",
                    "logistique sur les jetons résumés autrement (tableau des poolings)",
                    "oui",
                    "C, pooling",
                ],
                [
                    "gated",
                    "non linéaire",
                    "R85 : poids de chaque dimension selon la fenêtre, porte σ(B·A·x + c)"
                    " de rang 8",
                    "torch",
                    "weight decay, époques (R40, R42, R46)",
                ],
                [
                    "attentive",
                    "attention",
                    "une requête apprise pondère les jetons avant le classement",
                    "torch",
                    "weight decay, époques (R40–R42, R45–R47)",
                ],
                [
                    "cascade",
                    "2 étages",
                    "logistic trie tout, attentive reclasse les 20 % meilleurs",
                    "torch",
                    "C, fraction",
                ],
            ],
        )
    ],
    [
        "Groupes pour --methods : losses (pertes), neighbors (têtes par similarité). Une"
        " régularisation s'ajoute au nom : logistic+R19, attentive+R41+R42 (tableau des"
        " régularisations).",
        "blanci heads-curve : AP sur un site cible selon le nombre k d'enregistrements positifs"
        " de ce site à l'entraînement (k = 0, 1, 2, 5, 10, 20) — prototype contre linear probe"
        " quand un site est peu annoté.",
    ],
)

POOLINGS = Tableau(
    "poolings",
    "Poolings des jetons",
    "logistic:<pooling> : la même logistique, sur les jetons de la fenêtre résumés autrement que"
    " par l'embedding par défaut (n° 92). perch_v2 : 16 temps × 4 fréquences.",
    [
        Section(
            [
                Colonne("Pooling", 1.1, code=True),
                Colonne("Calcul, par dimension", 2.2),
                Colonne("Ce qu'il garde d'une note de 0,1 s", 2.4),
            ],
            [
                [
                    C("mean", "ref"),
                    C("moyenne des jetons", "ref"),
                    C("la dilue dans le fond (≈ l'embedding par défaut)", "ref"),
                ],
                ["max", "maximum des jetons", "la garde, mais sensible à un bruit isolé"],
                [
                    "meanmax",
                    "moyenne et maximum, concaténés (dimension × 2)",
                    "les deux, au prix de deux fois plus de coefficients",
                ],
                ["meanstd", "moyenne et écart-type, concaténés", "la variabilité dans la fenêtre"],
                [
                    "topk",
                    "moyenne des k plus fortes valeurs (k = 2)",
                    "une note dans un ou deux jetons, entre moyenne et maximum",
                ],
                [
                    "gem, gem2, gem5",
                    "R22 : (moyenne des x^p)^(1/p), p = 3 par défaut",
                    "p = 1 : moyenne ; p → ∞ : maximum. p fixé, comparé, pas appris",
                ],
                [
                    "mean_f_max_t",
                    "moyenne en fréquence, puis maximum en temps",
                    "l'instant de la note (grille temps × fréquence seulement)",
                ],
                [
                    "max_f_mean_t",
                    "maximum en fréquence, puis moyenne en temps",
                    "la bande de la note, moyennée dans le temps",
                ],
            ],
        )
    ],
    [
        "Échantillon (n° 113) : sur les jetons de perch_v2, max fait moins bien que la moyenne"
        " (−0,25 d'AP, intervalle [−0,42 ; 0,00]) ; à confirmer au benchmark réel."
    ],
)

PERTES = Tableau(
    "pertes",
    "Benchmark des pertes (R34, R35)",
    "blanci heads --methods losses : la même tête linéaire (standardisation, classes équilibrées,"
    " L2, C par validation groupée), seule la perte change (n° 112). m = marge, p = probabilité"
    " donnée à la bonne classe.",
    [
        Section(
            [
                Colonne("Tête", 1.25, code=True),
                Colonne("Perte", 1.35),
                Colonne("Exemple bien classé (m ≫ 0)", 1.6),
                Colonne("Exemple très mal classé (m ≪ 0)", 1.6),
                Colonne("Ce qu'elle change", 1.9),
            ],
            [
                [
                    C("logistic", "ref"),
                    C("log(1 + e^−m)", "ref"),
                    C("poids qui décroît avec p, jamais nul", "ref"),
                    C("coût ≈ −m, sans limite", "ref"),
                    C("référence", "ref"),
                ],
                [
                    "loss:hinge",
                    "max(0, 1 − m)",
                    C("ignoré au-delà de la marge (m > 1)", "bon"),
                    C("coût ≈ −m, sans limite", "dur"),
                    "SVM : la frontière est tracée par les cas limites (vecteurs de support)",
                ],
                [
                    "loss:squared_hinge",
                    "max(0, 1 − m)²",
                    C("ignoré au-delà de la marge", "bon"),
                    C("coût ≈ m², le plus sensible", "dur"),
                    "SVM lisse (défaut de LinearSVC)",
                ],
                [
                    "loss:least_squares",
                    "(1 − m)²",
                    C("pénalisé s'il est « trop » bien classé (m > 1)", "moyen"),
                    C("coût ≈ m²", "dur"),
                    "moindres carrés sur ±1, cousine de la LDA (R31)",
                ],
                [
                    "loss:focal",
                    "−(1 − p)^γ log p, γ = 2",
                    C("écrasé par (1 − p)^γ, jamais nul", "bon"),
                    C("coût ≈ −m, comme la logistique", "dur"),
                    "insiste sur les difficiles : utile contre les faux amis, nuisible si les"
                    " difficiles sont des étiquettes fausses",
                ],
                [
                    "loss:gce",
                    "(1 − p^q) / q, q = 0,7",
                    "poids faible",
                    C("plafonné à 1/q ≈ 1,4", "bon"),
                    "robuste aux étiquettes fausses",
                ],
                [
                    "loss:sce",
                    "0,1·(−log p) + 4·(1 − p)",
                    "poids faible",
                    C("coût ≈ 0,1·(−m) + 4 : croît 10 fois moins vite", "moyen"),
                    "entropie croisée symétrique, robuste",
                ],
                [
                    "loss:sigmoid",
                    "1 − p",
                    "poids faible",
                    C("plafonné à 1", "bon"),
                    "bornée : une étiquette fausse coûte au plus 1",
                ],
            ],
        )
    ],
    [
        "Pourquoi les pertes bornées : les négatifs présumés sont contaminés (n° 106 : 80 % des"
        " négatifs nearest détectés par BlanciNet). Avec la logistique, un vrai chant étiqueté"
        " négatif et noté haut coûte sans limite et tire la frontière vers lui.",
        "hinge et focal se ressemblent : toutes deux négligent les exemples faciles et se"
        " concentrent sur les difficiles, et aucune ne plafonne le coût d'une étiquette fausse."
        " Seules gce, sce et sigmoid le font.",
        "Échantillon (n° 113, 10 positifs) : loss:hinge −0,10 d'AP [−0,25 ; −0,01] face à la"
        " logistique ; les autres pertes ne s'en distinguent pas à cet effectif.",
    ],
)

VOISINS = Tableau(
    "voisins",
    "Têtes par similarité (R39)",
    "blanci heads --methods neighbors : aucun apprentissage, seulement des cosinus entre"
    " embeddings. Les seules têtes qui marchent avec 1 à 5 exemples : l'amorçage d'un site"
    " (n° 115).",
    [
        Section(
            [
                Colonne("Tête", 1.2, code=True),
                Colonne("Exemples", 0.9),
                Colonne("Score d'une fenêtre", 2.5),
                Colonne("Quand", 1.9),
            ],
            [
                [
                    "exemplar_medoid",
                    "1 positif",
                    "cosinus au positif le plus central",
                    "premier chant validé sur un site",
                ],
                [
                    "exemplar",
                    "positifs",
                    "cosinus au positif le plus proche",
                    "« trouve-moi des fenêtres comme celles-ci » (récolte, §5)",
                ],
                [
                    C("exemplar:k=3, :k=5", "neuf"),
                    "positifs",
                    "moyenne des cosinus aux k positifs les plus proches",
                    "moins sensible à un positif bizarre",
                ],
                [
                    "knn",
                    "positifs et négatifs",
                    "plus proche positif − plus proche négatif",
                    "rabaisse une fenêtre qui ressemble encore plus à un faux ami",
                ],
                [
                    C("knn:k=3, knn:k=5", "neuf"),
                    "positifs et négatifs",
                    "moyenne des k plus proches positifs − moyenne des k plus proches négatifs",
                    "dans la liste par défaut ; plus lisse, plus flou",
                ],
                [
                    C("knn:k=10", "neuf"),
                    "positifs et négatifs",
                    "idem, k = 10",
                    "site déjà un peu annoté",
                ],
                [
                    C("knn:k=5:w, …:w", "neuf"),
                    "positifs et négatifs",
                    "moyenne pondérée par 1 / distance : les très proches comptent plus",
                    "entre k = 1 et la moyenne des k",
                ],
            ],
        )
    ],
    [
        "Les k voisins sont pris dans chaque classe séparément : un vote parmi les k plus"
        " proches toutes classes confondues serait écrasé par les ~20 négatifs par positif.",
        "Faiblesse : l'ambiance d'un micro domine le cosinus (le problème que corrige le"
        " prototype différentiel) ; à tester aussi avec R19 : knn:k=5+R19.",
        "Courbe selon le nombre d'annotations (heads-curve) : exemplar:k=3, knn et knn:k=3 y sont"
        " ajoutées.",
    ],
)

REGULARISATIONS = Tableau(
    "regularisations",
    "Régularisations des têtes et de la fusion",
    "Numéros de documentation/regularisation.md. Coupées par défaut ; une tête les active dans"
    " son nom : blanci heads --methods logistic,logistic+R19,logistic+R18=16 (« =v » remplace"
    " le réglage ◆).",
    [
        Section(
            [
                Colonne("R", 0.45, code=True),
                Colonne("Nom", 1.45),
                Colonne("Où", 0.75),
                Colonne("Têtes", 1.35),
                Colonne("Réglage ◆ (défaut)", 1.2),
                Colonne("État", 1.35),
            ],
            [
                [
                    "R13",
                    "chaque micro pèse autant dans sa classe",
                    "poids",
                    "logistic, loss:, R30, cascade",
                    "—",
                    C("programmée (n° 108)", "bon"),
                ],
                [
                    "R15",
                    "négatifs annotés surpondérés face aux présumés",
                    "poids",
                    "idem",
                    "hard_weight (3)",
                    C("programmée (n° 108)", "bon"),
                ],
                [
                    "R17",
                    "embeddings ramenés à la norme 1",
                    "fenêtre",
                    "toutes (embedding)",
                    "—",
                    C("programmée (n° 108)", "bon"),
                ],
                [
                    "R18",
                    "ACP avant la tête",
                    "pli",
                    "toutes (embedding)",
                    "components (32)",
                    C("programmée (n° 108)", "bon"),
                ],
                [
                    "R19",
                    "centrage par micro (− moyenne du stock)",
                    "fenêtre",
                    "embedding par défaut",
                    "—",
                    C("programmée (n° 108)", "bon"),
                ],
                [
                    "R20",
                    "AdaBN : centrage et réduction par micro",
                    "fenêtre",
                    "embedding par défaut",
                    "eps",
                    C("programmée (n° 108)", "bon"),
                ],
                [
                    "R21",
                    "retrait des directions du micro",
                    "pli",
                    "toutes (embedding)",
                    "means ou inlp",
                    C("programmée (n° 108)", "bon"),
                ],
                [
                    "R22",
                    "pooling gem",
                    "jetons",
                    "logistic:gem",
                    "p (3)",
                    C("tête (n° 108)", "bon"),
                ],
                ["R26", "L2 (ridge)", "pénalité", "logistic", "C", C("en place", "ref")],
                [
                    "R27",
                    "L1 (lasso)",
                    "pénalité",
                    "logistic, cascade",
                    "—",
                    C("programmée (n° 108)", "bon"),
                ],
                [
                    "R28",
                    "Elastic Net",
                    "pénalité",
                    "logistic, cascade",
                    "l1_ratio (0,5)",
                    C("programmée (n° 108)", "bon"),
                ],
                [
                    "R30",
                    "rétrécissement vers le prototype",
                    "tête",
                    "logistic_to_prototype",
                    "C",
                    C("tête (n° 108)", "bon"),
                ],
                [
                    "R31",
                    "LDA à covariance rétrécie",
                    "tête",
                    "lda_shrunk",
                    "—",
                    C("tête (n° 108)", "bon"),
                ],
                [
                    "R33",
                    "norme maximale ‖w‖ ≤ c",
                    "—",
                    "—",
                    "—",
                    C("écartée : équivaut à la L2 pour une tête linéaire", "dur"),
                ],
                [
                    "R34, R35",
                    "autres pertes (SVM, focale, robustes)",
                    "tête",
                    "loss:<nom>",
                    "C",
                    C("têtes (n° 112)", "bon"),
                ],
                [
                    "R36",
                    "poids des classes (n / 2·n_classe)^power",
                    "poids",
                    "logistic, loss:, R30, cascade",
                    "power (0 : aucun)",
                    C("programmée", "neuf"),
                ],
                [
                    "R37",
                    "un biais par micro, a priori N(0, σ²)",
                    "fenêtre",
                    "logistic, loss:, cascade",
                    "scale σ (3, provisoire)",
                    C("programmée", "neuf"),
                ],
                [
                    "R38",
                    "modèle mixte : écarts de w par micro",
                    "—",
                    "—",
                    "—",
                    C("en discussion (ACP d'abord)", "attente"),
                ],
                [
                    "R39",
                    "k plus proches voisins",
                    "tête",
                    "knn:k=…, exemplar:k=…",
                    "k, :w",
                    C("têtes", "neuf"),
                ],
                [
                    "R40",
                    "weight decay par validation groupée",
                    "entraînement",
                    "attentive, gated",
                    "grid (1e-4 … 0,1)",
                    C("programmée", "neuf"),
                ],
                [
                    "R41",
                    "AdamW au lieu de la L2 d'Adam",
                    "entraînement",
                    "attentive",
                    "weight_decay (0,01)",
                    C("programmée", "neuf"),
                ],
                [
                    "R42",
                    "arrêt précoce",
                    "entraînement",
                    "attentive, gated",
                    "R42=ap ou R42=loss (ap, provisoire)",
                    C("programmée", "neuf"),
                ],
                [
                    "R43",
                    "pénalité d'entropie de l'attention",
                    "—",
                    "—",
                    "—",
                    C("en discussion", "attente"),
                ],
                ["R44", "température de l'attention", "—", "—", "—", C("en discussion", "attente")],
                [
                    "R45",
                    "dropout des jetons",
                    "entraînement",
                    "attentive",
                    "p (0,2)",
                    C("programmée", "neuf"),
                ],
                [
                    "R46",
                    "dropout des dimensions agrégées",
                    "entraînement",
                    "attentive, gated",
                    "p (0,2)",
                    C("programmée", "neuf"),
                ],
                [
                    "R47",
                    "rétrécissement vers la logistique",
                    "entraînement",
                    "attentive",
                    "strength λ (0,01)",
                    C("programmée", "neuf"),
                ],
                [
                    "R48",
                    "rang faible pour query et weight",
                    "—",
                    "—",
                    "—",
                    C("en discussion", "attente"),
                ],
                ["R49", "attention multi-têtes", "—", "—", "—", C("en discussion", "attente")],
                [
                    "R50",
                    "C de la fusion par validation groupée",
                    "fusion",
                    "logistic+R50",
                    "fusion.C_grid",
                    C("programmée (n° 120)", "neuf"),
                ],
                [
                    "R51",
                    "contraintes de signe (poids ≥ 0)",
                    "fusion",
                    "—",
                    "—",
                    C("en discussion", "attente"),
                ],
                [
                    "R52",
                    "combinaison convexe (poids ≥ 0, somme 1)",
                    "fusion",
                    "weighted, weight_grid",
                    "—",
                    C("en place pour ces deux méthodes", "ref"),
                ],
                [
                    "R53",
                    "sélection L1 des descripteurs",
                    "fusion",
                    "—",
                    "—",
                    C("en discussion", "attente"),
                ],
                [
                    "R54",
                    "descripteurs en classes, marches monotones",
                    "fusion",
                    "—",
                    "—",
                    C("en discussion", "attente"),
                ],
                [
                    "R55",
                    "modèle additif (GAM), courbes lissées",
                    "fusion",
                    "—",
                    "—",
                    C("en discussion", "attente"),
                ],
                [
                    "R56",
                    "pas de veto : plafond des descripteurs",
                    "fusion",
                    "—",
                    "—",
                    C("en discussion", "attente"),
                ],
                [
                    "R57",
                    "hors-pli strict au niveau 1",
                    "fusion",
                    "toutes",
                    "—",
                    C("en place", "ref"),
                ],
                [
                    "R58",
                    "phénologie en a priori",
                    "décision",
                    "—",
                    "—",
                    C("en discussion", "attente"),
                ],
                ["R85", "sonde à portes", "tête", "gated", "rang (8)", C("tête (n° 110)", "bon")],
            ],
        )
    ],
    [
        "Ordre d'application : R19/R20 → R17 → R18 → R21 → R37 → tête (poids R13/R15/R36,"
        " pénalité R27/R28). Groupe de R13, R19–R21, R37 : regularization.by (point = site/micro).",
        "Aucune conclusion avant la base complète : les mesures sur données simulées ou sur"
        " l'échantillon vérifient le code, elles ne classent rien ; les défauts marqués"
        " « provisoire » se choisiront sur la base (n° 119).",
        "R19, R20, R21 se jugent sur un site tenu à l'écart, pas sur les plis de Mataroni"
        " (n° 109).",
    ],
)

FUSION = Tableau(
    "fusion",
    "Benchmark de la fusion",
    "blanci fusion-bench : score de la tête (hors-pli) + descripteurs du module séquentiel +"
    " autres sources ; méthodes × emplacements, mêmes plis (n° 94–95, 103).",
    [
        Section(
            [
                Colonne("Méthode", 1.0, code=True),
                Colonne("Combinaison", 1.8),
                Colonne("Pondération", 1.6),
            ],
            [
                [
                    C("logistic", "ref"),
                    C("régression logistique (stacking)", "ref"),
                    C("apprise, C fixé à 1", "ref"),
                ],
                [
                    C("logistic+R50", "neuf"),
                    "régression logistique (stacking)",
                    "apprise, C choisi par validation groupée dans chaque pli",
                ],
                ["weighted", "somme pondérée", "fixée à la main (fusion.weights)"],
                ["weight_grid", "somme pondérée", "cherchée sur une grille (pas 0,1)"],
                ["mean", "moyenne des entrées", "égale"],
                ["rank_mean", "moyenne des rangs", "égale, insensible aux échelles"],
                ["max", "maximum : une entrée forte suffit (règle OU)", "—"],
                ["min", "minimum : toutes doivent être fortes (règle ET)", "—"],
            ],
            titre="Méthodes",
        ),
        Section(
            [
                Colonne("Emplacement", 1.0, code=True),
                Colonne("Ce que le module séquentiel y fait", 3.4),
            ],
            [
                ["[]", "absent : la tête seule"],
                [
                    "upstream",
                    "avant l'encodeur : transformations du son et portes (tableau du"
                    " seuillage en amont)",
                ],
                [
                    "parallel",
                    "descripteurs de rythme de la fenêtre (onsets, intervalles entre"
                    " notes) fusionnés avec le score",
                ],
                [
                    "downstream",
                    "descripteurs de persistance sur l'enregistrement (fraction de"
                    " fenêtres positives, plus longue série…)",
                ],
                [C("parallel, downstream", "ref"), C("défaut actuel", "ref")],
                ["upstream, parallel, downstream", "les trois"],
            ],
            titre="Emplacements",
        ),
    ],
    [
        "Au plus 4 descripteurs (~10 enregistrements positifs par coefficient) ; jamais de veto"
        " en parallèle ni en aval."
    ],
)

AMONT = Tableau(
    "seuillage-amont",
    "Seuillage en amont",
    "blanci upstream-bench : pour chaque porte et chaque seuil, fenêtres arrêtées (calcul"
    " économisé) contre positifs perdus (n° 90, 103). Tout est coupé par défaut.",
    [
        Section(
            [
                Colonne("Élément", 1.1, code=True),
                Colonne("Type", 0.9),
                Colonne("Effet", 2.2),
                Colonne("Mesure (n° 107)", 2.0),
            ],
            [
                [
                    "bandpass",
                    "transformation",
                    "filtre passe-bande 3–7 kHz avant l'encodeur : un autre stock d'embeddings",
                    "—",
                ],
                ["denoise", "transformation", "soustraction du fond médian", "—"],
                ["band_energy", "porte", "pic d'énergie en bande 4,4–5,5 kHz ≥ 6 dB", "—"],
                [
                    "band_contrast",
                    "porte",
                    "contraste bande / bandes voisines ≥ 3 dB",
                    C(
                        "12 % des négatifs arrêtés, 92 % des enregistrements positifs gardés",
                        "moyen",
                    ),
                ],
                [
                    "notes",
                    "porte",
                    "au moins un début de note de durée compatible",
                    C("83 % arrêtés mais 31 % des positifs gardés : inutilisable en l'état", "dur"),
                ],
                [
                    "rhythm",
                    "porte",
                    "au moins un intervalle entre notes d'A. blanci",
                    C("idem notes", "dur"),
                ],
            ],
        )
    ],
    [
        "Détection des notes trop stricte sur les données réelles (seuil médiane + 4 MAD) : à"
        " reprendre avant toute porte de rythme."
    ],
)

NEGATIFS = Tableau(
    "negatifs-apparies",
    "Négatifs appariés présumés",
    "benchmark.pairing : comment tirer les ~20 négatifs par positif, toujours sur le même micro"
    " (n° 88, 101). Contamination mesurée : fenêtres détectées par BlanciNet ≥ 0,5 (n° 106).",
    [
        Section(
            [
                Colonne("Stratégie", 0.9, code=True),
                Colonne("Tirage", 2.4),
                Colonne("Risque", 1.5),
                Colonne("BlanciNet ≥ 0,5", 1.0),
            ],
            [
                [
                    C("nearest", "ref"),
                    C(
                        "les fenêtres les plus proches dans le temps, dans l'enregistrement positif"
                        " d'abord, puis le même jour, puis un autre jour (défaut)",
                        "ref",
                    ),
                    C("A. blanci chante souvent tout l'enregistrement", "ref"),
                    C("80 % (dans l'enregistrement positif)", "dur"),
                ],
                [
                    "same_day",
                    "même jour, les enregistrements les plus proches, à ≥ 30 min",
                    "meilleur témoin du fond, mais chant par épisodes",
                    C("46 %", "dur"),
                ],
                [
                    "other_day",
                    "même créneau horaire (± 30 min), un autre jour",
                    "elle chante aux mêmes heures d'un jour à l'autre",
                    C("43 %", "dur"),
                ],
                ["mixed", "moitié autre jour, moitié même jour", "—", "—"],
            ],
        )
    ],
    [
        "BlanciNet n'est pas la vérité (faux amis) ; l'écoute tranchera (notebook"
        " 02_negatifs_apparies). Parades : pertes bornées (R35), R15, négatifs annotés."
    ],
)

BASELINES = Tableau(
    "baselines",
    "Baselines sans encodeur",
    "blanci baselines : lisent l'audio, n'encodent rien ; jugées comme les encodeurs (mêmes"
    " fenêtres, mêmes plis, mêmes métriques). Go/no-go de P0 (§3).",
    [
        Section(
            [
                Colonne("Score", 1.1, code=True),
                Colonne("Calcul", 3.0),
                Colonne("Apprentissage", 1.0),
            ],
            [
                [
                    "band_energy",
                    "pic d'énergie en bande 4,4–5,5 kHz au-dessus de sa médiane,"
                    " lissé sur la durée d'une note : le seuillage spectral classique",
                    "aucun",
                ],
                [
                    "band_contrast",
                    "idem, rapporté aux bandes voisines (3,0–4,2 et 5,7–7,0 kHz)"
                    " : un son large bande ne compte pas",
                    "aucun",
                ],
                ["notes", "nombre de débuts de notes en bande de durée compatible", "aucun"],
                ["rhythm", "nombre d'intervalles entre notes dans la plage d'A. blanci", "aucun"],
                [
                    "template_mean",
                    "corrélation maximale avec le gabarit moyen des notes des"
                    " positifs d'entraînement",
                    "gabarit par pli",
                ],
                [
                    "template_max",
                    "corrélation maximale avec le meilleur de 30 notes prises une à une",
                    "gabarits par pli",
                ],
            ],
        )
    ],
    [
        "Si aucun encodeur n'atteint AP ≥ 0,5 et rappel ≥ 0,8 à précision ≥ 0,1, le détecteur"
        " primaire sera « DSP + template matching »."
    ],
)

ENSEMBLES = Tableau(
    "ensembles",
    "Ensembles de modèles",
    "Combiner deux modèles ou plus (n° 96) ; un ensemble n'est jugé que contre sa meilleure"
    " source seule, par bootstrap apparié.",
    [
        Section(
            [Colonne("Façon", 1.4), Colonne("Commande", 1.6, code=True), Colonne("Principe", 3.0)],
            [
                [
                    "Tardive, par enregistrement",
                    "blanci ensemble --sources a,b",
                    "n'importe quelles sources hors-pli ramenées au score maximal par"
                    " enregistrement, combinées par les méthodes de la fusion",
                ],
                [
                    "Par fenêtre",
                    "fusion.sources: [head:<encodeur>]",
                    "la tête d'un autre encodeur ramenée sur la grille du principal ; la fusion"
                    " décide fenêtre par fenêtre (production possible)",
                ],
                [
                    "Concaténation",
                    "blanci ensemble --concat a,b",
                    "embeddings bout à bout (même grille) ; coûte les deux encodages, d grandit"
                    " (R24 : régulariser davantage)",
                ],
            ],
        )
    ],
)

PROTOCOLE = Tableau(
    "protocole",
    "Comment se lit un benchmark",
    "Mêmes règles pour tous les benchmarks du projet (§6, n° 91).",
    [
        Section(
            [Colonne("Élément", 1.3), Colonne("Règle", 3.6)],
            [
                [
                    "Plis",
                    "groupés par micro (point = site/micro), communs à tous les modèles :"
                    " un micro n'est jamais des deux côtés d'un pli",
                ],
                [
                    "Scores",
                    "hors-pli : chaque score vient d'un modèle entraîné sans son micro"
                    " (stock data/oof, réutilisé par la fusion et les ensembles)",
                ],
                [
                    "Niveaux",
                    "fenêtre, et enregistrement (score maximal de ses fenêtres) : seul"
                    " l'enregistrement compare des grilles de 3 s et 5 s",
                ],
                ["AP", "précision moyenne, intervalle par bootstrap sur les enregistrements"],
                ["Rappel", "aux précisions plancher 0,1 et 0,5, intervalle de Wilson"],
                [
                    "Comparaison",
                    "appariée (mêmes enregistrements) contre la référence :"
                    " « meilleur » seulement si l'intervalle exclut zéro",
                ],
                ["Égalité", "écart d'AP < 0,1 : licence, vitesse et prise en main départagent"],
                ["Jeu gelé", "exclu de l'entraînement, consulté une seule fois (R80)"],
            ],
            titre="Règles",
        ),
        Section(
            [
                Colonne("Benchmark", 1.3),
                Colonne("Commande", 1.8, code=True),
                Colonne("Tableau", 1.8, code=True),
            ],
            [
                ["Encodeurs", "blanci benchmark / throughput", "encodeurs.png"],
                ["Têtes", "blanci heads, heads-curve", "tetes.png"],
                ["Poolings", "blanci heads --methods logistic:max,…", "poolings.png"],
                ["Pertes", "blanci heads --methods losses", "pertes.png"],
                ["Similarité", "blanci heads --methods neighbors", "voisins.png"],
                ["Régularisations", "blanci heads --methods logistic+R19,…", "regularisations.png"],
                ["Fusion", "blanci fusion-bench", "fusion.png"],
                ["Seuillage en amont", "blanci upstream-bench", "seuillage-amont.png"],
                ["Négatifs appariés", "benchmark.pairing", "negatifs-apparies.png"],
                ["Baselines", "blanci baselines", "baselines.png"],
                ["Ensembles", "blanci ensemble", "ensembles.png"],
                ["Tout : toutes les sources, mêmes enregistrements", "blanci benchmark-all", "—"],
                ["Détecteurs audio (distillé, maison)", "blanci detector-bench", "—"],
                ["AnuraSet (pré-benchmark, autres anoures)", "blanci anuraset-benchmark", "—"],
                ["Échantillon : 66 clips, sans le disque", "blanci echantillon", "—"],
            ],
            titre="Les benchmarks",
        ),
    ],
)

TABLEAUX = [
    PROTOCOLE,
    ENCODEURS,
    TETES,
    POOLINGS,
    PERTES,
    VOISINS,
    REGULARISATIONS,
    FUSION,
    AMONT,
    NEGATIFS,
    BASELINES,
    ENSEMBLES,
]


# --- Rendu --------------------------------------------------------------------------------------


class Polices:
    def __init__(self, taille: int):
        def charge(nom: str, t: int) -> ImageFont.FreeTypeFont:
            return ImageFont.truetype(str(POLICES / nom), t)

        self.texte = charge("DejaVuSans.ttf", taille)
        self.gras = charge("DejaVuSans-Bold.ttf", taille)
        self.code = charge("DejaVuSansMono.ttf", taille - 1)
        self.code_gras = charge("DejaVuSansMono-Bold.ttf", taille - 1)
        self.petit = charge("DejaVuSans.ttf", taille - 3)
        self.titre = charge("DejaVuSans-Bold.ttf", taille + 12)
        self.section = charge("DejaVuSans-Bold.ttf", taille + 3)


def couper(texte: str, police: ImageFont.FreeTypeFont, largeur: float) -> list[str]:
    """Lignes de `texte` tenant dans `largeur` pixels (retours à la ligne gardés)."""
    lignes: list[str] = []
    for paragraphe in texte.split("\n"):
        courante = ""
        for mot in paragraphe.split(" "):
            essai = f"{courante} {mot}".strip()
            if police.getlength(essai) <= largeur:
                courante = essai
                continue
            if courante:
                lignes.append(courante)
            while police.getlength(mot) > largeur:  # mot trop long : coupé au caractère
                n = len(mot)
                while n > 1 and police.getlength(mot[:n]) > largeur:
                    n -= 1
                lignes.append(mot[:n])
                mot = mot[n:]
            courante = mot
        lignes.append(courante)
    return lignes


def lignes_cellule(
    texte: str, code: bool, premiere: bool, p: Polices, largeur: float
) -> list[tuple[str, ImageFont.FreeTypeFont, str]]:
    """(ligne, police, encre) d'une cellule. Colonne de code : le premier paragraphe est le nom
    (chasse fixe, gras), les suivants une précision (discrète, plus petite en 1re colonne)."""
    if not code:
        return [(ligne, p.texte, ENCRE) for ligne in couper(texte, p.texte, largeur)]
    nom, *reste = texte.split("\n")
    out = [(ligne, p.code_gras, ENCRE) for ligne in couper(nom, p.code_gras, largeur)]
    police = p.petit if premiere else p.texte
    for paragraphe in reste:
        out += [(ligne, police, ENCRE_2) for ligne in couper(paragraphe, police, largeur)]
    return out


def rendre(tableau: Tableau, taille: int = 19) -> Image.Image:
    p = Polices(taille)
    marge, pad, interligne = 36, 10, round(taille * 1.35)
    largeur_utile = tableau.largeur - 2 * marge

    y = marge
    titre_h = round((taille + 12) * 1.3)
    sous_titre = couper(tableau.sous_titre, p.texte, largeur_utile)
    y += titre_h + 6 + len(sous_titre) * interligne + 18
    mises_en_page = []
    for section in tableau.sections:
        total = sum(c.largeur for c in section.colonnes)
        largeurs = [largeur_utile * c.largeur / total for c in section.colonnes]
        section_h = round((taille + 3) * 1.5) if section.titre else 0
        entete = [
            couper(c.titre, p.gras, w - 2 * pad)
            for c, w in zip(section.colonnes, largeurs, strict=True)
        ]
        entete_h = max(len(e) for e in entete) * interligne + 2 * pad
        lignes = []
        for rang in section.lignes:
            cellules = []
            for j, cellule in enumerate(rang):
                c = cellule if isinstance(cellule, C) else C(cellule)
                code = section.colonnes[j].code
                texte = lignes_cellule(c.texte, code, j == 0, p, largeurs[j] - 2 * pad)
                cellules.append((c, texte))
            h = max(len(t) for _, t in cellules) * interligne + 2 * pad
            lignes.append((cellules, h))
        mises_en_page.append((section, largeurs, section_h, entete, entete_h, lignes))
        y += section_h + entete_h + sum(h for _, h in lignes) + 28
    etiquettes = sorted(
        {
            (cellule.etiquette if isinstance(cellule, C) else None)
            for s in tableau.sections
            for rang in s.lignes
            for cellule in rang
        }
        - {None},
        key=list(ETIQUETTES).index,
    )
    notes = [couper("• " + n, p.petit, largeur_utile) for n in tableau.notes]
    y += sum(len(n) for n in notes) * round((taille - 3) * 1.4) + (8 if notes else 0)
    y += (interligne + 14) if etiquettes else 0
    hauteur = y + marge

    image = Image.new("RGB", (tableau.largeur, hauteur), FOND)
    d = ImageDraw.Draw(image)
    y = marge
    d.text((marge, y), tableau.titre, font=p.titre, fill=ENCRE)
    y += titre_h + 6
    for ligne in sous_titre:
        d.text((marge, y), ligne, font=p.texte, fill=ENCRE_2)
        y += interligne
    y += 18
    for section, largeurs, section_h, entete, entete_h, lignes in mises_en_page:
        if section.titre:
            d.text((marge, y), section.titre, font=p.section, fill=ENCRE)
            y += section_h
        x = marge
        d.rectangle([marge, y, marge + largeur_utile, y + entete_h], fill=ENTETE)
        for w, texte in zip(largeurs, entete, strict=True):
            for k, ligne in enumerate(texte):
                d.text((x + pad, y + pad + k * interligne), ligne, font=p.gras, fill=ENCRE)
            x += w
        y += entete_h
        for i, (cellules, h) in enumerate(lignes):
            x = marge
            fond_ligne = ZEBRE if i % 2 else FOND
            for (c, texte), w in zip(cellules, largeurs, strict=True):
                fond = ETIQUETTES[c.etiquette][0] if c.etiquette else fond_ligne
                d.rectangle([x, y, x + w, y + h], fill=fond)
                for k, (ligne, police, encre) in enumerate(texte):
                    d.text((x + pad, y + pad + k * interligne), ligne, font=police, fill=encre)
                x += w
            d.line([marge, y + h, marge + largeur_utile, y + h], fill=FILET, width=1)
            y += h
        x = marge
        for w in largeurs[:-1]:  # filets verticaux discrets
            x += w
            d.line([x, y - sum(h for _, h in lignes), x, y], fill=FILET, width=1)
        y += 28
    for n in notes:
        for ligne in n:
            d.text((marge, y), ligne, font=p.petit, fill=ENCRE_2)
            y += round((taille - 3) * 1.4)
    if etiquettes:
        y += 8 + 6
        x = marge
        for cle in etiquettes:
            couleur, sens = ETIQUETTES[cle]
            d.rectangle([x, y + 2, x + 26, y + interligne - 4], fill=couleur, outline=FILET)
            d.text((x + 34, y), sens, font=p.petit, fill=ENCRE_2)
            x += 34 + p.petit.getlength(sens) + 36
    return image


def verifier(tableau: Tableau) -> None:
    for section in tableau.sections:
        for rang in section.lignes:
            if len(rang) != len(section.colonnes):
                raise ValueError(
                    f"{tableau.fichier} : {len(rang)} cellules au lieu de "
                    f"{len(section.colonnes)} dans {rang[0]!r}"
                )
            for cellule in rang:
                if isinstance(cellule, C) and cellule.etiquette not in (None, *ETIQUETTES):
                    raise ValueError(f"{tableau.fichier} : étiquette {cellule.etiquette!r}")


def main(noms: list[str]) -> None:
    for tableau in TABLEAUX:
        if noms and tableau.fichier not in noms:
            continue
        verifier(tableau)
        chemin = ICI / f"{tableau.fichier}.png"
        image = rendre(tableau).quantize(colors=48, method=Image.Quantize.MEDIANCUT)
        image.save(chemin, optimize=True)
        print(f"{chemin.name} : {image.width} × {image.height}, {chemin.stat().st_size // 1024} Ko")


if __name__ == "__main__":
    main(sys.argv[1:])
