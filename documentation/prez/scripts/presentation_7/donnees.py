# ruff: noqa: E501
"""Toutes les données de la prez PNA (lecture seule) : `donnees.json` + `courbe.png`.

    uv run python documentation/prez/scripts/presentation_7/donnees.py

Entrées (écrites par extraire.py, extraire_labels.py, figures_audio.py) : `echantillon.npz`
(fenêtres Perch v2 tirées au hasard, ≈ 600 par site), `annotees.npz` (fenêtres étiquetées par
`training_set`), `enregistrements.npz` (signature moyenne de 80 enregistrements par site),
`envelope.npz`.

1. Ce qui organise les groupes : k-moyennes (k = nombre de sites) sur les vecteurs normés, accord
   (ARI) avec le site, le point d'écoute, le moment de la journée, le mois ; au niveau fenêtre et
   au niveau enregistrement ; intervalle à 95 % par bootstrap (200 tirages des points).
   Répartition des fenêtres positives entre les groupes (centre le plus proche).
2. Simulation d'apprentissage actif, tête linéaire (régression logistique, C = 1, classes
   équilibrées) : les micros sont coupés en deux (70 % apprentissage, 30 % évaluation, graine
   fixe). Départ : 40 fenêtres annotées tirées au hasard dans les micros d'apprentissage ; à
   chaque tour, 30 fenêtres de plus, choisies 60-20-20 (incertaines, meilleurs scores, hasard)
   ou toutes au hasard ; on « révèle » leur label (déjà connu : c'est un rejeu). Note : AP sur
   les fenêtres des micros d'évaluation, jamais vues. 10 répétitions (graines), moyenne et
   enveloppe min-max.
3. Le nuage affiché : 70 fenêtres au hasard par site (Kaw, Molokoi, Trésor, Mataroni) + des
   fenêtres annotées de ces sites (dont toutes celles révélées aux tours 0 à 4 de la répétition
   0) ; vues ACP → direction de la tête finale ; score de chaque version de la tête.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ICI = Path(__file__).resolve().parents[4] / "rapports" / "presentation_7"  # données, jamais committées
ICI.mkdir(parents=True, exist_ok=True)
SITES4 = ["Kaw", "Molokoi", "Tresor", "Mataroni"]
SEED = 0
OUT: dict = {}


def norm(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


con = sqlite3.connect("file:C:/Users/ll34272/base_blanci/blanci.sqlite?mode=ro", uri=True)
recs = pd.read_sql_query("SELECT recording_id, site, mic_id, start_utc FROM recordings", con).set_index("recording_id")
e, a, r = (np.load(ICI / f, allow_pickle=False) for f in ("echantillon.npz", "annotees.npz", "enregistrements.npz"))
Xe, Xa, Xr = norm(e["X"]), norm(a["X"]), norm(r["X"])
ya, sa, pa = a["y"], a["site"], a["point"]
print("annotées :", len(ya), "dont positives", int(ya.sum()))
OUT["annotees"] = (
    pd.DataFrame({"site": sa, "y": ya}).groupby("site")["y"].agg(fenetres="size", positives="sum").reset_index().to_dict("records")
)
OUT["n_annotees"], OUT["n_positives"] = int(len(ya)), int(ya.sum())
OUT["n_enr_positifs"] = int(pd.Series(a["recording_id"][ya == 1]).nunique())
OUT["n_micros_annotes"] = int(pd.Series(pa).nunique())


# ------------------------------------------------------------ 1. ce qui organise les groupes
def facteurs(ids):
    m = recs.loc[ids].reset_index()
    h = (pd.to_datetime(m["start_utc"], utc=True) - pd.Timedelta(hours=3)).dt.hour
    moment = np.select([(h >= 5) & (h < 8), (h >= 8) & (h < 17), (h >= 17) & (h < 20)], ["aube", "jour", "crépuscule"], "nuit")
    return {
        "site": m["site"].to_numpy(str),
        "point d'écoute": (m["site"] + "/" + m["mic_id"]).to_numpy(str),
        "moment de la journée": moment,
        "mois": pd.to_datetime(m["start_utc"], utc=True).dt.strftime("%m").to_numpy(str),
    }


def accords(X, ids, k, n_boot=200):
    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    fac = facteurs(ids)
    rng = np.random.default_rng(SEED)
    res = {}
    for nom, f in fac.items():
        val = adjusted_rand_score(f, lab)
        boot = []
        for _ in range(n_boot):
            i = rng.integers(0, len(lab), len(lab))
            boot.append(adjusted_rand_score(f[i], lab[i]))
        res[nom] = {"ari": round(val, 3), "lo": round(float(np.percentile(boot, 2.5)), 3), "hi": round(float(np.percentile(boot, 97.5)), 3)}
    return res, lab


k = len(np.unique(e["site"]))
OUT["clusters"] = {"k": k, "n_fenetres": int(len(Xe)), "n_enregistrements": int(len(Xr))}
OUT["clusters"]["fenetre"], lab_e = accords(Xe, e["recording_id"], k)
OUT["clusters"]["enregistrement"], _ = accords(Xr, r["recording_id"], k)
# robustesse au nombre de groupes (site seulement)
OUT["clusters"]["k_robustesse"] = {
    niveau: {kk: round(adjusted_rand_score(s, KMeans(kk, n_init=10, random_state=SEED).fit_predict(X)), 3) for kk in (4, 8, 12)}
    for niveau, X, s in (("fenetre", Xe, e["site"]), ("enregistrement", Xr, r["site"]))
}
km = KMeans(k, n_init=10, random_state=SEED).fit(Xe)
grp_pos = km.predict(Xa[ya == 1])
part = np.bincount(grp_pos, minlength=k) / len(grp_pos)
part_hasard = np.bincount(km.labels_, minlength=k) / len(km.labels_)
OUT["clusters"]["positifs_par_groupe"] = [round(float(x), 3) for x in part]
OUT["clusters"]["hasard_par_groupe"] = [round(float(x), 3) for x in part_hasard]
grp_a = km.predict(Xa)
OUT["clusters"]["annotees_par_groupe"] = [[int(((grp_a == g) & (ya == 1)).sum()), int((grp_a == g).sum())] for g in range(k)]
OUT["clusters"]["sites_par_groupe_pos"] = [sorted(set(sa[(grp_a == g) & (ya == 1)].tolist())) for g in range(k)]
print(json.dumps(OUT["clusters"], ensure_ascii=False, indent=1))

# ------------------------------------------------------------ 2. simulation d'apprentissage actif
points = np.unique(pa)
rng0 = np.random.default_rng(SEED)
pts_pos = np.unique(pa[ya == 1])
eval_pts = set(rng0.choice(pts_pos, max(1, int(0.3 * len(pts_pos))), replace=False)) | set(
    rng0.choice(np.setdiff1d(points, pts_pos), int(0.3 * len(np.setdiff1d(points, pts_pos))), replace=False)
)
is_eval = np.isin(pa, list(eval_pts))
tr, te = np.flatnonzero(~is_eval), np.flatnonzero(is_eval)
print("évaluation :", len(te), "fenêtres,", int(ya[te].sum()), "positives,", len(eval_pts), "micros")
N0, LOT, TOURS = 40, 30, 8


def tete(idx):
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Xa[idx], ya[idx])


def simule(graine, mode):
    rng = np.random.default_rng(graine)
    while True:
        lab = list(rng.choice(tr, N0, replace=False))
        if 3 <= ya[lab].sum() <= N0 - 3:
            break
    hist = [list(lab)]
    aps, n = [], []
    for t in range(TOURS + 1):
        m = tete(lab)
        aps.append(average_precision_score(ya[te], m.predict_proba(Xa[te])[:, 1]))
        n.append(len(lab))
        if t == TOURS:
            break
        reste = np.setdiff1d(tr, lab)
        p = m.predict_proba(Xa[reste])[:, 1]
        if mode == "hasard":
            new = list(rng.choice(reste, LOT, replace=False))
        else:
            n_inc, n_top = int(0.6 * LOT), int(0.2 * LOT)
            inc = reste[np.argsort(np.abs(p - 0.5))[:n_inc]]
            top = [i for i in reste[np.argsort(-p)] if i not in set(inc)][:n_top]
            autres = np.setdiff1d(reste, np.concatenate([inc, top]))
            new = list(inc) + list(top) + list(rng.choice(autres, LOT - n_inc - n_top, replace=False))
        lab = lab + [int(i) for i in new]
        hist.append([int(i) for i in new])
    return np.array(n), np.array(aps), hist


courbes = {}
for mode in ("actif", "hasard"):
    runs = [simule(g, mode) for g in range(10)]
    A = np.stack([ap for _, ap, _ in runs])
    courbes[mode] = {"n": runs[0][0].tolist(), "moy": A.mean(0).round(3).tolist(), "min": A.min(0).round(3).tolist(), "max": A.max(0).round(3).tolist()}
ap_tout = average_precision_score(ya[te], tete(tr).predict_proba(Xa[te])[:, 1])
OUT["simulation"] = {
    "courbes": courbes, "ap_toutes_annotations": round(float(ap_tout), 3), "n_train": int(len(tr)),
    "n_eval": int(len(te)), "n_eval_pos": int(ya[te].sum()), "n_eval_micros": len(eval_pts),
    "prevalence_eval": round(float(ya[te].mean()), 3), "N0": N0, "lot": LOT, "tours": TOURS,
}
print(json.dumps(OUT["simulation"], ensure_ascii=False))

fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
for mode, coul, nom in (("actif", "#1E6B45", "file 60-20-20"), ("hasard", "#9AA39E", "au hasard")):
    c = courbes[mode]
    ax.fill_between(c["n"], c["min"], c["max"], color=coul, alpha=0.15, lw=0)
    ax.plot(c["n"], c["moy"], "-o", color=coul, ms=4, lw=2, label=nom)
ax.axhline(ap_tout, color="#5B6B63", ls=":", lw=1.2)
ax.text(c["n"][-1], ap_tout + 0.01, f"toutes mes annotations ({len(tr)} fenêtres)", ha="right", va="bottom", fontsize=8, color="#5B6B63")
ax.axhline(OUT["simulation"]["prevalence_eval"], color="#C8553D", ls="--", lw=1)
ax.text(c["n"][0], OUT["simulation"]["prevalence_eval"] + 0.01, "modèle qui tire au hasard", fontsize=8, color="#C8553D")
ax.set_xlabel("fenêtres annotées pour l'entraînement", color="#5B6B63")
ax.set_ylabel("précision moyenne (AP)", color="#5B6B63")
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="lower right", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors="#5B6B63")
fig.tight_layout()
fig.savefig(ICI / "courbe.png", transparent=True)

# ------------------------------------------------------------ 3. le nuage affiché
n0, ap0, hist = simule(0, "actif")
tour_rev = np.full(len(ya), -1)
for t, idx in enumerate(hist):
    tour_rev[idx] = t
cat1 = {}  # catégorie de chaque fenêtre de la file du tour 1
for j, i in enumerate(hist[1]):
    cat1[i] = "incertaine" if j < int(0.6 * LOT) else "meilleur score" if j < int(0.8 * LOT) else "hasard"
dans4 = np.isin(sa, SITES4)
rev = np.flatnonzero(dans4 & (tour_rev >= 0) & (tour_rev <= 4))
rng = np.random.default_rng(SEED + 1)
jamais = np.flatnonzero(dans4 & (tour_rev < 0) & ~is_eval)
sel_a = np.concatenate([rev, rng.choice(jamais, max(0, 140 - len(rev)), replace=False)])
idx_e = np.concatenate([rng.choice(np.flatnonzero(e["site"] == s), 70, replace=False) for s in SITES4])
X = np.vstack([Xe[idx_e], Xa[sel_a]])
site = np.concatenate([e["site"][idx_e], sa[sel_a]])
y = np.concatenate([np.full(len(idx_e), -1), ya[sel_a]])
tour = np.concatenate([np.full(len(idx_e), -1), tour_rev[sel_a]])
cat = [""] * len(idx_e) + [cat1.get(int(i), "") for i in sel_a]
# fenêtre « héros » : une fenêtre positive de l'enregistrement de Molokoi des figures audio
hero_rec = "39313dd7512432b3"
cand = np.flatnonzero((a["recording_id"] == hero_rec) & (ya == 1))
hero = int(cand[np.argmin(np.abs(a["offset_s"][cand] - 25))]) if len(cand) else int(np.flatnonzero((sa == "Molokoi") & (ya == 1))[0])
sel_a = sel_a[sel_a != hero]
X = np.vstack([Xe[idx_e], Xa[sel_a]])
site = np.concatenate([e["site"][idx_e], sa[sel_a]])
y = np.concatenate([np.full(len(idx_e), -1), ya[sel_a]])
tour = np.concatenate([np.full(len(idx_e), -1), tour_rev[sel_a]])
cat = [""] * len(idx_e) + [cat1.get(int(i), "") for i in sel_a]
OUT["hero"] = {"recording_id": str(a["recording_id"][hero]), "offset_s": float(a["offset_s"][hero]), "site": str(sa[hero])}
X = np.vstack([X, Xa[hero : hero + 1]])
site, y = np.append(site, sa[hero]), np.append(y, ya[hero])
tour, cat = np.append(tour, -2), cat + [""]  # -2 : le héros, annoté d'emblée
ordre = rng.permutation(len(X) - 1)
ordre = np.append(ordre, len(X) - 1)  # le héros reste le dernier (au-dessus)
X, site, y, tour, cat = X[ordre], site[ordre], y[ordre], tour[ordre], [cat[i] for i in ordre]

# versions de la tête : tour t = entraînée sur les fenêtres révélées jusqu'au tour t (rejeu 0)
lab_t, versions = [], []
for idx in hist:
    lab_t = lab_t + list(idx)
    versions.append(tete(lab_t))
finale = tete(np.concatenate([tr, [hero]]))
scores = [v.predict_proba(X)[:, 1].round(3).tolist() for v in versions]

Xc = X - X.mean(0)
a1, a2 = PCA(2, random_state=SEED).fit(Xc).components_
b1 = finale.coef_[0] / np.linalg.norm(finale.coef_[0])
b2 = a1 - (a1 @ b1) * b1
b2 /= np.linalg.norm(b2)
TS = [0.0, 0.25, 0.5, 0.75, 1.0]
vues, boussole = [], []
for t in TS:
    v1 = (1 - t) * a1 + t * b1
    v1 /= np.linalg.norm(v1)
    v2 = (1 - t) * a2 + t * b2
    v2 -= (v2 @ v1) * v1
    v2 /= np.linalg.norm(v2)
    vues.append([(Xc @ v1).round(4).tolist(), (Xc @ v2).round(4).tolist()])
    boussole.append({nom: [round(float(d @ v1), 3), round(float(d @ v2), 3)] for nom, d in (("ACP 1", a1), ("ACP 2", a2), ("tête", b1))})
seuil_u = float((-finale.intercept_[0] - X.mean(0) @ finale.coef_[0]) / np.linalg.norm(finale.coef_[0]))
groupes = KMeans(4, n_init=10, random_state=SEED).fit_predict(X).tolist()
# bande de 48 nombres de la fenêtre héros et poids de la tête finale sur ces 48 nombres
w = finale.coef_[0]
top = np.argsort(-np.abs(w))[:8]
dims = np.sort(np.concatenate([top, rng.choice(np.setdiff1d(np.arange(len(w)), top), 40, replace=False)]))
OUT["bande"] = {"dims": dims.tolist(), "brut": a["X"][hero][dims].round(3).tolist(), "poids": (np.abs(w[dims]) / np.abs(w).max()).round(3).tolist()}
OUT["nuage"] = {
    "site": site.tolist(), "y": y.astype(int).tolist(), "tour": tour.astype(int).tolist(), "cat1": cat,
    "groupe": groupes, "vues": vues, "ts": TS, "boussole": boussole, "seuil_u": seuil_u, "scores": scores,
    "ap_rejeu": [round(float(x), 3) for x in ap0], "n_rejeu": n0.tolist(),
}
env = np.load(ICI / "envelope.npz")
OUT["enveloppe"] = (env["large"] / env["large"].max()).round(4).tolist()
(ICI / "donnees.json").write_text(json.dumps(OUT, ensure_ascii=False), encoding="utf-8")
print("nuage :", len(X), "points ; révélés tours 0-4 :", len(rev), "; héros", OUT["hero"])
print("AP rejeu 0 :", OUT["nuage"]["ap_rejeu"])
