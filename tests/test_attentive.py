"""Attentive probing (§3) : l'attention retrouve une note que la moyenne des jetons dilue."""

import numpy as np
import pytest

pytest.importorskip("torch")

from blanci.attentive import AttentiveHead, TokenStore, fit_attentive  # noqa: E402
from blanci.evaluate import average_precision  # noqa: E402
from blanci.head import fit_logistic, oof_scores  # noqa: E402
from blanci.regularization import (  # noqa: E402
    Context,
    Regularizer,
    fit_with_options,
    regularizer_for,
)

N_TOKENS, DIM = 16, 8


def windows(n_per_class=120, strength=3.0, seed=0):
    """Fond propre à chaque fenêtre ; chez les positifs, un seul jeton porte la « note »."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    tokens = rng.normal(0, 1.0, (n, N_TOKENS, DIM)) + rng.normal(0, 1.0, (n, 1, DIM))
    y = np.r_[np.ones(n_per_class), np.zeros(n_per_class)].astype(int)
    where = rng.integers(N_TOKENS, size=n_per_class)
    tokens[np.arange(n_per_class), where, 0] += strength
    groups = np.array([f"m{i % 6}" for i in range(n)])
    return tokens.astype(np.float32), y, groups, where


def test_attention_beats_the_mean_of_tokens_on_a_brief_note():
    tokens, y, groups, _ = windows()
    attentive = oof_scores(tokens, y, groups, n_splits=3, method="attentive")
    pooled = oof_scores(tokens.mean(axis=1), y, groups, n_splits=3, method="logistic")
    ap_attentive = average_precision(y, attentive.values)
    ap_pooled = average_precision(y, pooled.values)  # ~0,5 : la note est diluée dans 16 jetons
    assert ap_attentive > ap_pooled + 0.1


def test_attention_points_at_the_note():
    tokens, y, _, where = windows()
    head = fit_attentive(tokens, y)
    weights = head.attention(tokens[y == 1])
    assert (weights.argmax(axis=1) == where).mean() > 0.7


def test_saved_head_gives_the_same_scores(tmp_path):
    tokens, y, _, _ = windows(n_per_class=40)
    head = fit_attentive(tokens, y, epochs=50)
    head.save(tmp_path / "att")
    again = AttentiveHead.load(tmp_path / "att")
    assert np.allclose(head.decision(tokens), again.decision(tokens), atol=1e-5)
    assert sorted(p.name for p in (tmp_path / "att").iterdir()) == ["manifest.json", "weights.npz"]


def test_token_store_replaces_and_refuses_partial_loads(tmp_path):
    store = TokenStore(tmp_path, "perch_v2-1")
    assert store.load(["a"]) is None
    store.write(["a", "b"], np.zeros((2, 3, 4)))
    store.write(["b", "c"], np.ones((2, 3, 4)))
    got = store.load(["c", "a", "b"])
    assert got.shape == (3, 3, 4) and got[0].max() == 1 and got[1].max() == 0 and got[2].max() == 1
    assert store.load(["a", "z"]) is None  # une fenêtre manque : pas de sonde attentive


def test_probe_table_adds_the_attentive_probe_when_tokens_exist():
    from blanci.benchmark import probe_table

    tokens, y, groups, _ = windows(n_per_class=60)
    recordings = np.array([f"r{i // 3}" for i in range(len(y))])
    table, scores = probe_table(
        tokens.mean(axis=1), y, groups, recordings, n_splits=3, n_boot=10, tokens=tokens
    )
    assert "attentive" in set(table["probe"]) and "attentive" in scores


# --- Régularisations de l'attentive (R40–R42, R45–R47, DECISIONS n° 117) ------------------------


def overfitting_windows(seed=0):
    """Peu de fenêtres, beaucoup de dimensions de bruit : la tête finit par apprendre le bruit."""
    rng = np.random.default_rng(seed)
    n, t, d = 60, 8, 64
    tokens = rng.normal(0, 1.0, (n, t, d)).astype(np.float32)
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    tokens[: n // 2, 0, 0] += 1.0  # une note faible, toujours dans le premier jeton
    groups = np.array([f"m{i % 6}" for i in range(n)])
    return tokens, y, groups


def test_R42_stops_when_the_validation_loss_rises():
    tokens, y, groups = overfitting_windows()
    val = groups == "m0"
    head = fit_attentive(tokens[~val], y[~val], validation=(tokens[val], y[val]), patience=10)
    assert head.meta["best_epoch"] < 300


def test_R42_picks_the_epoch_of_the_best_mean_validation_curve():
    """Courbe moyenne des plis groupés, puis réentraînement pour ce nombre d'époques. Mesuré
    ici : la perte de validation est au plus bas dès la 1re époque (la tête devient vite trop
    sûre d'elle), alors que l'AP de validation progresse jusqu'à ~185 époques."""
    tokens, y, groups, _ = windows()
    by_ap = fit_with_options(fit_attentive, tokens, y, groups, early_stopping=True)
    stop = by_ap.meta["early_stopping"]
    assert stop["monitor"] == "ap" and stop["folds"] == 3
    assert by_ap.meta["epochs"] == stop["best_epoch"]
    by_loss = fit_with_options(
        fit_attentive, tokens, y, groups, early_stopping=True, monitor="loss"
    )
    assert by_loss.meta["early_stopping"]["best_epoch"] < stop["best_epoch"]


def test_R40_picks_the_weight_decay_by_grouped_validation():
    tokens, y, groups = overfitting_windows()
    head = fit_with_options(
        fit_attentive, tokens, y, groups, weight_decays=[1e-4, 1.0], epochs=40, n_splits=3
    )
    assert set(head.meta["weight_decay_cv"]) == {"0.0001", "1.0"}
    assert head.meta["weight_decay"] in (1e-4, 1.0)


def test_R41_R45_R46_are_recorded_and_still_find_the_note():
    tokens, y, groups, _ = windows()
    out = oof_scores(
        tokens,
        y,
        groups,
        n_splits=3,
        method="attentive",
        regularizer=Regularizer({41: None, 45: 0.2, 46: 0.1}, {}, Context(groups)),
    ).values
    plain = oof_scores(tokens, y, groups, n_splits=3, method="attentive").values
    assert average_precision(y, out) > average_precision(y, plain) - 0.05
    head = fit_attentive(tokens, y, optimizer="adamw", token_dropout=0.2, dim_dropout=0.1)
    assert head.meta["optimizer"] == "adamw" and head.meta["token_dropout"] == 0.2


def test_R47_keeps_w_near_the_logistic_on_the_mean_of_tokens():
    tokens, y, _, _ = windows(n_per_class=40)
    free = fit_attentive(tokens, y, epochs=100)
    tied = fit_attentive(tokens, y, epochs=100, shrink=1e3)
    x = (tokens - tied.mean) / tied.scale
    logistic = fit_logistic(x.mean(axis=1), y, 1.0)
    w0 = logistic.coef / logistic.scale
    assert np.linalg.norm(tied.weight - w0) < 0.05 * np.linalg.norm(w0)
    assert np.linalg.norm(free.weight - w0) > np.linalg.norm(tied.weight - w0)


def test_torch_regularizations_are_refused_elsewhere():
    assert regularizer_for("attentive+R42+R41", {}, Context(np.array(["a"])))[0] == (
        "attentive+R41+R42"
    )
    for spec, message in [
        ("logistic+R42", "attentive et gated"),
        ("gated+R41", "R41 sans objet"),
        ("gated+R45", "R45 sans objet"),
        ("attentive+R13", "jetons"),
    ]:
        with pytest.raises(ValueError, match=message):
            regularizer_for(spec, {}, Context(np.array(["a"])))


def test_R42_also_stops_the_gated_probe():
    from blanci.gated import fit_gated

    tokens, y, groups = overfitting_windows()
    head = fit_with_options(fit_gated, tokens.mean(axis=1), y, groups, early_stopping=True)
    assert head.meta["early_stopping"]["best_epoch"] <= 300


def test_R59_warmup_and_clipping_are_recorded_and_change_the_training():
    tokens, y, groups, _ = windows(n_per_class=40)
    plain = fit_attentive(tokens, y, epochs=40)
    tamed = fit_attentive(tokens, y, epochs=40, warmup=20, clip_norm=0.1)
    assert tamed.meta["warmup"] == 20 and tamed.meta["clip_norm"] == 0.1
    assert not np.allclose(plain.weight, tamed.weight)
    options = Regularizer({59: 5}, {}, Context(groups)).torch_options()
    assert options == {"warmup": 5, "clip_norm": 1.0}
    out = oof_scores(
        tokens,
        y,
        groups,
        n_splits=3,
        method="attentive",
        regularizer=Regularizer({59: None}, {}, Context(groups)),
    ).values
    assert np.isfinite(out).all()


def test_R64_weight_averaging_keeps_a_mean_of_the_last_weights():
    tokens, y, groups, _ = windows(n_per_class=40)
    plain = fit_attentive(tokens, y, epochs=60)
    for method in ("ema", "swa"):
        averaged = fit_attentive(tokens, y, epochs=60, average=method, swa_start=0.5)
        assert averaged.meta["average"] == method
        assert not np.allclose(plain.weight, averaged.weight)
    options = Regularizer({64: "swa"}, {}, Context(groups)).torch_options()
    assert options["average"] == "swa"
    out = oof_scores(
        tokens,
        y,
        groups,
        n_splits=3,
        method="attentive",
        regularizer=Regularizer({64: None, 42: None}, {"R42": {"max_epochs": 40}}, Context(groups)),
    ).values
    assert np.isfinite(out).all()
    with pytest.raises(ValueError, match="R64"):
        fit_attentive(tokens, y, epochs=5, average="magie")


def test_R61_R62_fine_tuning_helpers_on_a_small_network():
    import torch

    from blanci.regularization import (
        l2_sp_model_penalty,
        layerwise_lr_groups,
        snapshot,
        unfreeze_top,
        unfreezing_schedule,
    )

    layers = [torch.nn.Linear(4, 4) for _ in range(3)]
    groups = layerwise_lr_groups(layers, lr=1.0, decay=0.5)
    assert [g["lr"] for g in groups] == [0.25, 0.5, 1.0]  # du bas vers le haut
    assert unfreezing_schedule(4, 6, every=2) == [1, 1, 2, 2, 3, 3]
    unfreeze_top(layers, 1)
    assert [layers[i].weight.requires_grad for i in range(3)] == [False, False, True]
    model = torch.nn.Sequential(*layers)
    unfreeze_top(layers, 3)
    reference = snapshot(model)
    assert float(l2_sp_model_penalty(torch, model, reference, alpha=1.0)) == 0.0
    with torch.no_grad():
        layers[2].weight += 1.0  # 16 poids écartés de 1
    assert float(l2_sp_model_penalty(torch, model, reference, alpha=1.0)) == pytest.approx(8.0)
    head = torch.nn.Sequential(model, torch.nn.Linear(4, 1))  # une tête nouvelle
    pretrained = {k: v for k, v in snapshot(head).items() if not k.startswith("1.")}
    penalty = l2_sp_model_penalty(torch, head, pretrained, alpha=0.0, beta=2.0)
    new = head[1]
    expected = float((new.weight**2).sum() + (new.bias**2).sum())
    assert float(penalty) == pytest.approx(expected, rel=1e-5)


def test_R43_entropy_penalty_sharpens_or_spreads_the_attention():
    tokens, y, groups, _ = windows(n_per_class=60)

    def mean_entropy(head):
        a = head.attention(tokens)
        return float(-(a * np.log(a + 1e-12)).sum(axis=1).mean())

    plain = fit_attentive(tokens, y, epochs=150)
    peaky = fit_attentive(tokens, y, epochs=150, entropy=0.5)
    diffuse = fit_attentive(tokens, y, epochs=150, entropy=-0.5)
    assert mean_entropy(peaky) < mean_entropy(plain) < mean_entropy(diffuse)
    assert peaky.meta["entropy"] == 0.5
    options = Regularizer({43: -0.01}, {}, Context(groups)).torch_options()
    assert options["entropy"] == -0.01
    assert regularizer_for("attentive+R43=-0.01", {}, Context(groups))[0] == "attentive+R43=-0.01"
