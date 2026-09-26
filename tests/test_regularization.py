"""Régularisations des têtes (DECISIONS n° 108), numérotées comme la liste du 25/09/2026."""

import numpy as np
import pytest

from blanci.evaluate import average_precision
from blanci.head import (
    differential_prototype,
    fit_and_score,
    fit_logistic,
    fit_logistic_to_prototype,
    lda_shrunk_scores,
    oof_scores,
)
from blanci.pooling import gem, pool
from blanci.regularization import (
    Context,
    Regularizer,
    canonical,
    distillation_loss,
    domain_statistics,
    dropout,
    group_indicators,
    grouped_search,
    head_name,
    l2_sp_penalty,
    mean_directions,
    nuisance_directions,
    parse_head,
    regularizer_for,
    sample_weights,
    validate,
)

DIM = 16


def mic_corpus(n_mics=6, n_per=40, seed=0, shared_axes=0):
    """Chaque micro a son fond (décalage propre, hors de la direction de l'espèce) ; A. blanci
    ajoute la direction 0. `shared_axes` > 0 : les fonds vivent dans un même petit sous-espace
    (gain, distance, bruit d'eau communs à tous les micros) au lieu d'être indépendants."""
    rng = np.random.default_rng(seed)
    axes = rng.normal(0, 1, (shared_axes, DIM)) if shared_axes else None
    if axes is not None:
        axes[:, 0] = 0.0
    X, y, groups = [], [], []
    for m in range(n_mics):
        if axes is None:
            background = rng.normal(0, 2.0, DIM)
        else:
            background = rng.normal(0, 2.0, shared_axes) @ axes
        background[0] = 0.0
        for label in (0, 1):
            block = background + rng.normal(0, 0.5, (n_per, DIM))
            block[:, 0] += 1.5 * label
            X.append(block)
            y += [label] * n_per
            groups += [f"M{m}"] * n_per
    return np.vstack(X).astype(np.float32), np.array(y), np.array(groups)


# --- Noms --------------------------------------------------------------------------------------


def test_head_names_are_canonical():
    assert parse_head("logistic:max+R18=16+R13") == ("logistic:max", {13: None, 18: 16.0})
    assert canonical("logistic+R18=16+R13") == "logistic+R13+R18=16"
    assert head_name("logistic", {28: 0.25}) == "logistic+R28=0.25"
    assert canonical("prototype") == "prototype"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("logistic+R99", "R99"),
        ("logistic+R19+R20", "R20"),
        ("logistic+R27+R28", "R27"),
        ("prototype+R27", "pénalité"),
        ("knn+R13", "poids"),
        ("prototype+R36", "poids"),
        ("prototype+R37", "biais"),
        ("logistic_to_prototype+R37", "biais"),
        ("logistic:max+R19", "défaut"),
        ("attentive+R17", "jetons"),
        ("logistic+R18=abc", "illisible"),
    ],
)
def test_meaningless_regularizations_are_refused(spec, message):
    with pytest.raises(ValueError, match=message):
        validate(*parse_head(spec))


def test_R42_takes_its_criterion_as_a_word():
    """Les deux critères de l'arrêt précoce se comparent dans un même run (n° 119)."""
    assert parse_head("attentive+R42=loss") == ("attentive", {42: "loss"})
    assert canonical("attentive+R42=ap+R41") == "attentive+R41+R42=ap"
    context = Context(np.array(["a"]))
    for word in ("ap", "loss"):
        options = regularizer_for(f"attentive+R42={word}", {}, context)[2].torch_options()
        assert options["monitor"] == word and options["early_stopping"]
    cfg = {"regularization": {"R42": {"monitor": "loss"}}}
    default = regularizer_for("attentive+R42", cfg, context)[2]
    assert default.torch_options()["monitor"] == "loss"
    with pytest.raises(ValueError, match="valeurs possibles"):
        parse_head("attentive+R42=auc")
    with pytest.raises(ValueError, match="illisible"):
        parse_head("logistic+R37=abc")


def test_a_plain_head_has_no_regularizer():
    assert regularizer_for("logistic", {}, None) == ("logistic", "logistic", None)


# --- R13, R15 : poids ----------------------------------------------------------------------------


def test_R13_gives_every_mic_the_same_weight_within_a_class():
    y = np.array([1, 1, 1, 1, 0, 0])
    groups = np.array(["a", "a", "a", "b", "a", "b"])
    w = sample_weights(y, groups, None, {13: None})
    assert w[:3].sum() == pytest.approx(w[3])  # 3 positifs de a = 1 positif de b
    assert w[y == 1].mean() == pytest.approx(1.0) and w[y == 0].mean() == pytest.approx(1.0)


def test_R15_upweights_annotated_negatives_only():
    y = np.array([1, 0, 0, 0])
    hard = np.array([False, True, False, False])
    w = sample_weights(y, np.array(list("aaaa")), hard, {15: None}, hard_weight=4.0)
    assert w[1] == pytest.approx(4 * w[2]) and w[0] == 1.0
    assert w[y == 0].mean() == pytest.approx(1.0)


def test_R36_power_goes_from_balanced_to_unweighted():
    """Les têtes pondèrent chaque classe par n / (2·n_classe) ; R36 élève ce poids à la
    puissance `class_power` : 0 = chaque fenêtre compte 1, 1 = inchangé."""
    y = np.array([1, 0, 0, 0, 0])
    balanced = len(y) / (2.0 * np.bincount(y))  # négatif 0,625, positif 2,5
    groups = np.array(list("aaaaa"))
    none = sample_weights(y, groups, None, {36: None}, class_power=0.0)
    assert none * balanced[y] == pytest.approx(np.ones(5))
    assert sample_weights(y, groups, None, {36: None}, class_power=1.0) == pytest.approx(1.0)
    half = sample_weights(y, groups, None, {36: 0.5}, class_power=0.5) * balanced[y]
    assert half[0] / half[1] == pytest.approx(2.0)  # √(2,5 / 0,625) au lieu de 4


def test_R36_reaches_the_heads_through_the_regularizer():
    X, y, groups = mic_corpus()
    keep = (y == 0) | (np.arange(len(y)) % 5 == 0)  # 1 positif pour 5 négatifs
    X, y, groups = X[keep], y[keep], groups[keep]
    reg = Regularizer({36: None}, {"R36": {"power": 0.0}}, Context(groups))
    _, w, _ = reg.prepare(X, y, np.arange(len(y)))
    assert w is not None and w[y == 1].mean() < w[y == 0].mean()
    for method in ("logistic", "loss:focal"):
        out = oof_scores(X, y, groups, n_splits=3, method=method, regularizer=reg).values
        assert np.isfinite(out).all() and average_precision(y, out) > 0.6


def test_no_weights_without_R13_or_R15():
    assert sample_weights(np.array([0, 1]), np.array(["a", "b"]), None, {18: None}) is None


# --- R19, R20 : par micro ------------------------------------------------------------------------


def context_with_domain(X, groups, y=None):
    stats = domain_statistics(X, groups)
    hard = None if y is None else np.zeros(len(groups), dtype=bool)
    return Context(groups, hard, stats, stats.rows(groups))


def test_R19_centres_each_mic_and_R20_also_scales_it():
    X, _, groups = mic_corpus()
    ctx = context_with_domain(X, groups)
    centred = Regularizer({19: None}, {}, ctx).window_transform(X)
    scaled = Regularizer({20: None}, {}, ctx).window_transform(X)
    for g in np.unique(groups):
        assert np.abs(centred[groups == g].mean(axis=0)).max() < 1e-4
        assert np.allclose(scaled[groups == g].std(axis=0), 1.0, atol=1e-3)


def test_R19_removes_the_mic_background_for_the_simple_prototype():
    """Sans négatifs, le prototype simple est trompé par le fond ; centré par micro, non."""
    X, y, groups = mic_corpus(seed=1)
    ctx = context_with_domain(X, groups)
    raw = oof_scores(X, y, groups, n_splits=3, method="simple_prototype").values
    reg = Regularizer({19: None}, {}, ctx)
    centred = oof_scores(
        X, y, groups, n_splits=3, method="simple_prototype", regularizer=reg
    ).values
    assert average_precision(y, centred) > average_precision(y, raw) + 0.1


def test_domain_stats_refuse_an_unknown_mic():
    stats = domain_statistics(np.ones((2, 3)), np.array(["a", "a"]))
    with pytest.raises(ValueError, match="statistiques"):
        stats.rows(np.array(["b"]))


# --- R17, R18, R21 ------------------------------------------------------------------------------


def test_R17_normalises_every_window():
    X, _, groups = mic_corpus()
    out = Regularizer({17: None}, {}, Context(groups)).window_transform(X)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_R18_projects_on_the_requested_number_of_components():
    X, y, groups = mic_corpus()
    reg = Regularizer({18: 4.0}, {"R18": {"components": 32}}, Context(groups))
    Xp, weights, options = reg.prepare(X, y, np.arange(len(y)))
    assert Xp.shape == (len(y), 4) and weights is None and options == {}


def mic_accuracy(X, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    return cross_val_score(LogisticRegression(max_iter=1000), X, groups).mean()


def test_R21_makes_the_mic_unpredictable_but_keeps_the_species_axis():
    X, y, groups = mic_corpus(seed=2)
    negatives = y == 0
    Q = mean_directions(X[negatives], groups[negatives])
    Xp = X - (X @ Q) @ Q.T
    assert Q.shape[1] == 5  # 6 micros → 5 directions
    assert mic_accuracy(X[negatives], groups[negatives]) > 0.9
    assert mic_accuracy(Xp[negatives], groups[negatives]) < 0.4
    assert average_precision(y, Xp[:, 0]) > 0.9  # la direction de l'espèce survit


def shortcut_corpus(seed=0, n_mics=12, n_per=80):
    """Micros « riches » (fond d'un côté d'un axe commun) : 50 % de positifs ; « pauvres » :
    5 %. Les 4 derniers micros (nouveau site) inversent la relation : le fond n'y dit plus
    rien de la présence."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(0, 1, DIM)
    axis[0] = 0.0
    axis /= np.linalg.norm(axis)
    X, y, groups, held = [], [], [], []
    for m in range(n_mics):
        rich, new_site = m % 2 == 0, m >= n_mics - 4
        rate = (0.05 if rich else 0.5) if new_site else (0.5 if rich else 0.05)
        labels = (rng.random(n_per) < rate).astype(int)
        block = rng.normal(0, 0.5, (n_per, DIM)) + (1.5 if rich else -1.5) * axis
        block += rng.normal(0, 0.3, DIM)
        block[:, 0] += 0.8 * labels
        X.append(block)
        y += list(labels)
        groups += [f"M{m}"] * n_per
        held += [new_site] * n_per
    return np.vstack(X).astype(np.float32), np.array(y), np.array(groups), np.array(held)


def test_R21_protects_a_new_site_from_a_mic_shortcut():
    """Là où le fond du micro prédit la présence, la logistique apprend ce raccourci ; sur un
    site où il ne tient plus, R21 (qui l'efface) fait bien mieux. Mesuré aussi : en validation
    croisée sur un seul site, où le raccourci reste vrai, R21 perd — ne pas le juger là."""
    X, y, groups, held = shortcut_corpus()
    train, test = np.flatnonzero(~held), np.flatnonzero(held)
    kw = {"C_grid": [0.01, 0.1, 1.0], "n_splits": 4}
    plain = fit_and_score("logistic", X, y, groups, train, test, **kw)
    reg = Regularizer({21: None}, {}, Context(groups))
    erased = fit_and_score("logistic", X, y, groups, train, test, regularizer=reg, **kw)
    assert average_precision(y[test], erased) > average_precision(y[test], plain) + 0.1


def test_R21_inlp_option_also_hides_the_mic():
    X, y, groups = mic_corpus(seed=2)
    negatives = y == 0
    Q = nuisance_directions(X[negatives], groups[negatives], iterations=10)
    Xp = X - (X @ Q) @ Q.T
    assert mic_accuracy(Xp[negatives], groups[negatives]) < 0.4
    reg = Regularizer({21: None}, {"R21": {"method": "inlp"}}, Context(groups))
    assert reg.prepare(X, y, np.arange(len(y)))[0].shape == X.shape
    with pytest.raises(ValueError, match="méthode"):
        Regularizer({21: None}, {"R21": {"method": "magie"}}, Context(groups)).prepare(
            X, y, np.arange(len(y))
        )


def test_R21_needs_two_mics():
    assert nuisance_directions(np.ones((4, 3)), np.array(list("aaaa"))).shape == (3, 0)
    assert mean_directions(np.ones((4, 3)), np.array(list("aaaa"))).shape == (3, 0)


# --- R37 : un biais par micro ---------------------------------------------------------------------


def test_R37_indicators_leave_unseen_mics_at_the_common_bias():
    groups = np.array(["a", "b", "a", "c"])
    ind = group_indicators(groups, np.array([0, 1, 2]))
    assert ind.tolist() == [[1, 0], [0, 1], [1, 0], [0, 0]]  # c : absent de l'entraînement


def test_R37_biases_absorb_the_mic_level_instead_of_w():
    """Même raccourci que pour R21 : le fond du micro prédit la présence. Avec un biais par
    micro, ce niveau passe dans les biais et w s'appuie moins sur l'axe du fond ; sur le
    nouveau site (biais inconnus, donc communs), le classement tient mieux. Mesuré, AP sur le
    nouveau site, 3 tirages : sans R37 0,25–0,37 ; σ = 0,3 : 0,24–0,42 (rien) ; σ = 1 :
    0,42–0,57 ; σ = 3 : 0,56–0,68 ; σ = 10 : 0,61–0,71 ; R21 : 0,50–0,62."""
    X, y, groups, held = shortcut_corpus()
    train, test = np.flatnonzero(~held), np.flatnonzero(held)
    reg = Regularizer({37: None}, {}, Context(groups))
    Xr, _, options = reg.prepare(X, y, train)
    assert options == {"bias_columns": 8, "bias_scale": 3.0}
    assert Xr.shape == (len(y), DIM + 8) and not Xr[test, DIM:].any()
    kw = {"C_grid": [0.01, 0.1, 1.0], "n_splits": 4}
    plain = fit_and_score("logistic", X, y, groups, train, test, **kw)
    biased = fit_and_score("logistic", X, y, groups, train, test, regularizer=reg, **kw)
    assert average_precision(y[test], biased) > average_precision(y[test], plain) + 0.1
    head = fit_logistic(Xr[train], y[train], 1.0, **options)
    assert head.meta["group_biases"] == 8
    focal = fit_and_score("loss:focal", X, y, groups, train, test, regularizer=reg, **kw)
    assert np.isfinite(focal).all()


def test_R37_scale_controls_how_far_the_biases_move():
    X, y, groups, held = shortcut_corpus()
    train = np.flatnonzero(~held)
    spread = {}
    for scale in (0.01, 3.0):
        reg = Regularizer({37: scale}, {}, Context(groups))
        Xr, _, options = reg.prepare(X, y, train)
        head = fit_logistic(Xr[train], y[train], 1.0, **options)
        spread[scale] = np.ptp(head.coef[DIM:] / head.scale[DIM:])  # biais b = w / échelle
    assert spread[0.01] < 0.2 * spread[3.0]


# --- R26–R28 : pénalités --------------------------------------------------------------------------


def test_R27_l1_zeroes_most_coefficients():
    X, y, _ = mic_corpus()
    l2 = fit_logistic(X, y, C=0.05)
    l1 = fit_logistic(X, y, C=0.05, l1_ratio=1.0)
    assert (np.abs(l1.coef) < 1e-6).sum() > (np.abs(l2.coef) < 1e-6).sum()
    assert np.argmax(np.abs(l1.coef)) == 0 and l1.meta["l1_ratio"] == 1.0


def test_R28_option_reaches_the_logistic():
    reg = Regularizer({28: 0.3}, {"R28": {"l1_ratio": 0.5}}, Context(np.array(["a"])))
    assert reg.fit_options() == {"l1_ratio": 0.3}
    assert Regularizer({27: None}, {}, Context(np.array(["a"]))).fit_options() == {"l1_ratio": 1.0}


def test_weights_and_penalty_flow_through_oof_scores():
    X, y, groups = mic_corpus()
    reg = Regularizer({13: None, 15: None, 28: None}, {}, Context(groups, y == 0))
    out = oof_scores(
        X, y, groups, n_splits=3, method="logistic", C_grid=[0.1, 1.0], regularizer=reg
    )
    assert not np.isnan(out.values).any() and average_precision(y, out.values) > 0.8


# --- R30, R31 : têtes entre prototype et logistique -----------------------------------------------


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_R30_is_the_prototype_at_small_C_and_the_logistic_at_large_C():
    X, y, _ = mic_corpus(seed=3)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    prototype = Z[y == 1].mean(axis=0) - Z[y == 0].mean(axis=0)
    tight = fit_logistic_to_prototype(X, y, C=1e-6)
    loose = fit_logistic_to_prototype(X, y, C=1e3)
    free = fit_logistic(X, y, C=1e3)
    assert cosine(tight.coef, prototype) > 0.999
    assert cosine(loose.coef, free.coef) > cosine(tight.coef, free.coef)
    assert tight.meta["shrink_to"] == "differential_prototype"


def test_R30_and_R31_rank_positives_first():
    X, y, groups = mic_corpus()
    for method in ("logistic_to_prototype", "lda_shrunk"):
        out = oof_scores(X, y, groups, n_splits=3, method=method, C_grid=[0.1, 1.0]).values
        assert average_precision(y, out) > 0.8, method


def test_R31_scores_follow_the_prototype_direction_on_isotropic_data():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (400, DIM))
    y = (np.arange(400) < 200).astype(int)
    X[y == 1, 0] += 2.0
    w, _ = differential_prototype(X[y == 1], X[y == 0])
    scores = lda_shrunk_scores(X, y, X)
    assert np.corrcoef(scores, X @ w)[0, 1] > 0.9


# --- R22 : GeM ---------------------------------------------------------------------------------


def test_R22_gem_goes_from_mean_to_max():
    rng = np.random.default_rng(5)
    tokens = rng.normal(0, 1, (3, 16, 4))
    assert np.allclose(gem(tokens, 1.0), tokens.mean(axis=1), atol=1e-5)
    assert np.allclose(gem(tokens, 200.0), tokens.max(axis=1), atol=0.1)
    middle = gem(tokens, 3.0)
    assert (middle >= tokens.mean(axis=1) - 1e-5).all()
    assert (middle <= tokens.max(axis=1) + 1e-5).all()
    assert np.allclose(pool(tokens, "gem"), middle) and np.allclose(
        pool(tokens, "gem1"), gem(tokens, 1.0)
    )


# --- Mécanique commune : recherche groupée, réseaux (R26, R40, R50, R62, R63) ---------------------


def test_grouped_search_keeps_the_best_value_on_held_out_mics():
    X, y, groups = mic_corpus()

    def score(C, train, test):
        head = fit_logistic(X[train], y[train], C)
        return average_precision(y[test], head.decision(X[test]))

    best, results = grouped_search(score, [1e-6, 1.0], y, groups, n_splits=3)
    assert best == max(results, key=results.get) and set(results) == {1e-6, 1.0}
    assert grouped_search(score, [1.0], y, np.repeat("a", len(y)))[0] is None  # un seul micro


def test_network_helpers_for_the_models_to_come():
    torch = pytest.importorskip("torch")
    w = torch.tensor([1.0, 2.0])
    assert float(l2_sp_penalty(torch, [w], [torch.tensor([1.0, 0.0])], 0.5)) == pytest.approx(1.0)
    logits = torch.tensor([3.0, -2.0])
    same = distillation_loss(torch, logits, logits, temperature=2.0)
    other = distillation_loss(torch, -logits, logits, temperature=2.0)
    assert float(other) > float(same)  # l'élève qui contredit l'enseignant paie plus
    x = torch.ones(1000)
    assert torch.equal(dropout(torch, x, 0.5, train=False), x)
    kept = dropout(torch, x, 0.5, train=True)
    assert 0.35 < float((kept == 0).float().mean()) < 0.65 and float(kept.max()) == 2.0
