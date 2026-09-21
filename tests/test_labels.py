import pandas as pd
import pytest

from blanci.labels import LABELS, column_key, detect_columns, parse_comment, parse_offset

# Commentaires des faux amis relevés à l'annotation (notes, H14) → label et espèce attendus.
NEGATIVES = [
    ("chants d'oiseau (Fourmilier tacheté) au moment des détections", "bird", "Fourmilier tacheté"),
    ('"cris" d\'amphibien de contact', "amphibian_contact_call", None),
    ("Oiseau (Fourmilier tacheté)", "bird", "Fourmilier tacheté"),
    ("oiseau Fourmilier tacheté et autres", "bird", "Fourmilier tacheté"),
    ("oiseau Moucherolle manakin", "bird", "Moucherolle manakin"),
    ("orthoptère (grillon sur les sessions +", "orthoptera", None),
    ("micro dans sac", "artefact_in_bag", None),
    ("A. andreae", "amphibian", "Adenomera andreae"),
    ("A andreae", "amphibian", "Adenomera andreae"),
    ("H. cappellei", "amphibian", "Hyalinobatrachium cappellei"),
    ("oiseaux divers bien audibles", "bird", None),
    ("un chant unique suspect mais probablement pas blanci", "uncertain", None),
    ("aucun chant audible", "background", None),
    ("oiseaux bien chanteurs", "bird", None),
    ("oiseau indét. (Martinet ?)", "bird", "Martinet"),
    ("Hyalinobatrachium", "amphibian", "Hyalinobatrachium sp."),
    ("pluie mais aucun chant distinguable à l'oreille", "rain", None),
    ("psittacidé", "bird", "Psittacidae"),
    ("oiseau Tangara mordoré", "bird", "Tangara mordoré"),
    ("oiseau (pigeon plombé)", "bird", "Pigeon plombé"),
    ("dans le sac", "artefact_in_bag", None),
    ("Oiseau indet", "bird", None),
    (
        "chants de fin similaire à l'écoute mais probablement autre chose avec la cadence "
        "précédente (Pic à cou rouge)",
        "bird",
        "Pic à cou rouge",
    ),
    ("A. hahneli similaire A. aff baeo", "amphibian", "Ameerega hahneli"),
    ("Hyalinobatrachium cappellei", "amphibian", "Hyalinobatrachium cappellei"),
    ("Amazophrynella teko", "amphibian", "Amazophrynella teko"),
    (
        "Amazophrynella teko (Oiseaux pïau et Tangara mordoré)",
        "amphibian",
        "Amazophrynella teko; Piauhau hurleur; Tangara mordoré",
    ),
    ("oiseau Evèque de Rothschild", "bird", "Évêque de Rothschild"),
    ("Hyalinobatrachium (mondolfii)", "amphibian", "Hyalinobatrachium mondolfii"),
    ("A femoralis sur la séquence", "amphibian", "Allobates femoralis"),
    ("oiseau myrmidon", "bird", "Myrmidon"),
    ("a priori rien d'audible mais bcp de bruits parasites", "background", None),
    ("oiseau Sclérure à bec court", "bird", "Sclérure à bec court"),
    ("oiseau (Sclérue obscure)", "bird", "Sclérure obscure"),
    (
        "Hyalinobatrachium mondolfii (et H. iaspidiense)",
        "amphibian",
        "Hyalinobatrachium mondolfii; Hyalinobatrachium iaspidiense",
    ),
    ("Amazophrynella teko (Otophryne)", "amphibian", "Amazophrynella teko; Otophryne sp."),
    ("Allobates femoralis", "amphibian", "Allobates femoralis"),
    ("Amphibien probable, pas un chant et pas A. blanci", "amphibian", None),
    ("oiseaux dont Fourmilier tacheté", "bird", "Fourmilier tacheté"),
    ("Otophryne", "amphibian", "Otophryne sp."),
    ("oiseau (psittacidés) au moment de la sequence +", "bird", "Psittacidae"),
    ("cri d'interaction d'amphibien", "amphibian_contact_call", None),
    ("Oiseau (Evèque de Rothschild)", "bird", "Évêque de Rothschild"),
]


@pytest.mark.parametrize("comment, label, species", NEGATIVES)
def test_false_friend_comments(comment, label, species):
    parsed = parse_comment(comment, "negative")
    assert (parsed.label, parsed.species) == (label, species)
    assert parsed.label in LABELS
    assert parsed.conditions["comment"] == comment


@pytest.mark.parametrize(
    "comment, quality, tags, co_occurring",
    [
        ("chants audibles en second plan", "C", ["second_plan"], None),
        ("chant lointain", "C", ["distant"], None),
        ("chants audibles malgré la pluie", "B", ["rain"], None),
        ("chant audible ET présence du fourmilier tacheté", "B", [], ["Fourmilier tacheté"]),
        ("", "A", [], None),
    ],
)
def test_positive_comments(comment, quality, tags, co_occurring):
    parsed = parse_comment(comment, "positive")
    assert parsed.label == "blanci" and parsed.species == "Anomaloglossus blanci"
    assert parsed.quality == quality and parsed.conditions["quality_inferred"]
    assert parsed.conditions["tags"] == tags
    assert parsed.conditions.get("co_occurring") == co_occurring


def test_explicit_quality_wins_over_inference():
    parsed = parse_comment("chant lointain", "positive", quality="A")
    assert parsed.quality == "A" and "quality_inferred" not in parsed.conditions


def test_missing_comment_is_other_for_negatives():
    assert parse_comment(None, "negative").label == "other"
    assert parse_comment(float("nan"), "negative").label == "other"


@pytest.mark.parametrize(
    "value, unit, expected",
    [
        (36, "seconds", 36.0),
        ("1,5", "seconds", 1.5),
        ("01:30", "seconds", 90.0),
        ("0:01:30.5", "seconds", 90.5),
        (12, "window_index", 36.0),
    ],
)
def test_parse_offset(value, unit, expected):
    assert parse_offset(value, unit, 3.0) == expected


def test_detect_columns_ignores_case_accents_and_units():
    df = pd.DataFrame(columns=["Fichier", "Début (s)", "Commentaires", "Qualité"])
    found = detect_columns(
        df,
        {
            "file": ["fichier"],
            "offset_s": ["debut"],
            "comment": ["commentaire", "commentaires"],
            "quality": ["qualite"],
            "site": ["site"],
        },
    )
    assert found == {
        "file": "Fichier",
        "offset_s": "Début (s)",
        "comment": "Commentaires",
        "quality": "Qualité",
    }
    assert column_key("Nom du fichier") == "nom_du_fichier"
