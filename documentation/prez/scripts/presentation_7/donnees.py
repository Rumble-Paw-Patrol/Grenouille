# ruff: noqa: E501
"""Toutes les données de la prez PNA v2, BirdNET 2.4 (lecture seule) : `donnees.json` + `courbe.png`.

    uv run python documentation/prez/scripts/presentation_7/donnees.py

Entrées (rapports/presentation_7/, écrites par extraire_birdnet.py) : `birdnet_hasard.npz`
(fenêtres de 3 s tirées au hasard, ≈ 600 par site), `birdnet_annotees.npz` (fenêtres
étiquetées par `training_set`).

1. Ce qui rend deux fenêtres semblables : part de la variance des vecteurs (normés) expliquée par
   chaque facteur pris seul, sur les fenêtres écoutées (somme des carrés entre groupes / somme
   totale, comme `heads/cluster.variance_partition`) : site ; point d'écoute dans son site
   (part du point moins part du site) ; moment de la journée ; mois ; présence d'A. blanci.
   Variabilité : 100 moitiés des enregistrements tirées sans remise (2,5 et 97,5 centiles). Contrôle : les quatre premiers
   facteurs sur les fenêtres tirées au hasard. Contrôle aussi : k-moyennes (k = 8) et part des
   fenêtres avec chant dans chaque groupe.
2. Simulation d'apprentissage actif, tête linéaire (régression logistique, C = 1, classes
   équilibrées) : les micros sont coupés en deux (70 % apprentissage, 30 % évaluation, graine
   fixe). Départ : 170 fenêtres annotées tirées au hasard dans les micros d'apprentissage ; à
   chaque tour, 40 fenêtres de plus, choisies 60-20-20 (incertaines, meilleurs scores, hasard)
   ou toutes au hasard ; on « révèle » leur label (déjà connu : c'est un rejeu). Note : AP sur
   les fenêtres des micros d'évaluation, jamais vues. 10 répétitions (graines), moyenne et
   enveloppe min-max.
3. Le nuage affiché : 120 fenêtres au hasard par site (Kaw, Molokoi, Trésor, Mataroni) + des
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
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ICI = Path(__file__).resolve().parents[4] / "rapports" / "presentation_7"  # données, jamais committées
ICI.mkdir(parents=True, exist_ok=True)
SITES4 = ["Kaw", "Molokoi", "Tresor", "Mataroni"]
SEED = 0
OUT: dict = {}


def norm(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


con = sqlite3.connect("file:C:/Users/ll34272/base_blanci/blanci.sqlite?mode=ro", uri=True)
recs = pd.read_sql_query("SELECT recording_id, site, mic_id, start_utc FROM recordings", con).set_index("recording_id")
e, a = (np.load(ICI / f, allow_pickle=False) for f in ("birdnet_hasard.npz", "birdnet_annotees.npz"))
Xe, Xa = norm(e["X"]), norm(a["X"])
ya, sa, pa = a["y"], a["site"], a["point"]
print("annotées :", len(ya), "dont positives", int(ya.sum()), "; dimension", Xa.shape[1])
OUT["dim"] = int(Xa.shape[1])
OUT["annotees"] = (
    pd.DataFrame({"site": sa, "y": ya}).groupby("site")["y"].agg(fenetres="size", positives="sum").reset_index().to_dict("records")
)
OUT["n_annotees"], OUT["n_positives"] = int(len(ya)), int(ya.sum())
OUT["n_enr_positifs"] = int(pd.Series(a["recording_id"][ya == 1]).nunique())
OUT["n_micros_annotes"] = int(pd.Series(pa).nunique())
OUT["n_sites_annotes"] = int(pd.Series(sa).nunique())


# ------------------------------------------------------------ 1. ce qui rend deux fenêtres semblables
def facteurs(ids, y=None):
    m = recs.loc[ids].reset_index()
    h = (pd.to_datetime(m["start_utc"], utc=True) - pd.Timedelta(hours=3)).dt.hour
    f = {
        "site": m["site"].to_numpy(str),
        "point": (m["site"] + "/" + m["mic_id"]).to_numpy(str),
        "moment de la journée": np.select([(h >= 5) & (h < 8), (h >= 8) & (h < 17), (h >= 17) & (h < 20)], ["aube", "jour", "crépuscule"], "nuit"),
        "mois": pd.to_datetime(m["start_utc"], utc=True).dt.strftime("%m").to_numpy(str),
    }
    if y is not None:
        f["présence d'A. blanci"] = np.asarray(y).astype(str)
    return f


def eta2(X, lab):
    """Part de la somme des carrés entre groupes : Σ n_g ‖m_g − m‖² / Σ ‖x − m‖²."""
    mu = X.mean(axis=0, dtype=np.float64)
    tot = float((X.astype(np.float64) ** 2).sum() - len(X) * (mu**2).sum())
    _, inv = np.unique(lab, return_inverse=True)
    M = csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))))
    sums = np.asarray(M @ X, dtype=np.float64)
    n = np.bincount(inv).astype(np.float64)
    return float(((sums**2).sum(axis=1) / n).sum() - len(X) * (mu**2).sum()) / tot


def parts(X, fac):
    r = {k: eta2(X, v) for k, v in fac.items()}
    r["point d'écoute (dans son site)"] = r.pop("point") - r["site"]
    return r


fac_a = facteurs(a["recording_id"], ya)
val = parts(Xa, fac_a)
rng_b = np.random.default_rng(SEED)
rec_u, rec_inv = np.unique(a["recording_id"], return_inverse=True)
par_rec = [np.flatnonzero(rec_inv == i) for i in range(len(rec_u))]
boot = []
for _ in range(100):  # moitiés d'enregistrements tirées sans remise (un doublon gonflerait les facteurs fins)
    idx = np.concatenate([par_rec[i] for i in rng_b.choice(len(rec_u), len(rec_u) // 2, replace=False)])
    boot.append(parts(Xa[idx], {k: v[idx] for k, v in fac_a.items()}))
OUT["variance"] = sorted(
    [{"facteur": k, "part": round(v, 4), "lo": round(float(np.percentile([b[k] for b in boot], 2.5)), 4), "hi": round(float(np.percentile([b[k] for b in boot], 97.5)), 4)} for k, v in val.items()],
    key=lambda d: -d["part"],
)
OUT["variance_hasard"] = {k: round(v, 4) for k, v in parts(Xe, facteurs(e["recording_id"])).items()}
OUT["n_enr_annotes"] = int(len(rec_u))
OUT["n_hasard"] = int(len(Xe))
print(json.dumps(OUT["variance"], ensure_ascii=False), OUT["variance_hasard"])

k = 8
km = KMeans(k, n_init=10, random_state=SEED).fit(Xe)
grp_a = km.predict(Xa)
OUT["clusters"] = {
    "k": k, "hasard_par_groupe": [round(float(x), 3) for x in np.bincount(km.labels_, minlength=k) / len(km.labels_)],
    "positifs_par_groupe": [round(float(x), 3) for x in np.bincount(grp_a[ya == 1], minlength=k) / max(1, (ya == 1).sum())],
    "annotees_par_groupe": [[int(((grp_a == g) & (ya == 1)).sum()), int((grp_a == g).sum())] for g in range(k)],
}
print(json.dumps(OUT["clusters"], ensure_ascii=False))

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
N0, LOT, TOURS = 170, 40, 8


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
ax.text(c["n"][-1], ap_tout + 0.01, f"toutes mes annotations ({len(tr):,} fenêtres)".replace(",", " "), ha="right", va="bottom", fontsize=8, color="#5B6B63")
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
sel_a = np.concatenate([rev, rng.choice(jamais, max(0, 220 - len(rev)), replace=False)])
idx_e = np.concatenate([rng.choice(np.flatnonzero(e["site"] == s), 120, replace=False) for s in SITES4])
X = np.vstack([Xe[idx_e], Xa[sel_a]])
site = np.concatenate([e["site"][idx_e], sa[sel_a]])
y = np.concatenate([np.full(len(idx_e), -1), ya[sel_a]])
tour = np.concatenate([np.full(len(idx_e), -1), tour_rev[sel_a]])
cat = [""] * len(idx_e) + [cat1.get(int(i), "") for i in sel_a]
# fenêtre « héros » : une fenêtre positive de l'enregistrement de Molokoi des figures audio
hero_rec = "39313dd7512432b3"
cand = np.flatnonzero((a["recording_id"] == hero_rec) & (ya == 1))
hero = int(cand[np.argmin(np.abs(a["offset_s"][cand] - 24))]) if len(cand) else int(np.flatnonzero((sa == "Molokoi") & (ya == 1))[0])
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
# repère XYZ : toutes les vues sont des projections du même espace à 3 dimensions,
# engendré par l'ACP 1 (X), l'ACP 2 (Y) et la direction de la tête rendue orthogonale aux deux (Z)
e3 = b1 - (b1 @ a1) * a1 - (b1 @ a2) * a2
e3 /= np.linalg.norm(e3)
REPERE = {"X": a1, "Y": a2, "Z": e3}
cube = float(max(np.percentile(np.abs(Xc @ d), 85) for d in REPERE.values()))
vues, boussole, triedre = [], [], []
for t in TS:
    v1 = (1 - t) * a1 + t * b1
    v1 /= np.linalg.norm(v1)
    v2 = (1 - t) * a2 + t * b2
    v2 -= (v2 @ v1) * v1
    v2 /= np.linalg.norm(v2)
    vues.append([(Xc @ v1).round(4).tolist(), (Xc @ v2).round(4).tolist()])
    boussole.append({nom: [round(float(d @ v1), 3), round(float(d @ v2), 3)] for nom, d in (("ACP 1", a1), ("ACP 2", a2), ("tête", b1))})
    triedre.append({nom: [round(float(d @ v1), 4), round(float(d @ v2), 4)] for nom, d in REPERE.items()})
seuil_u = float((-finale.intercept_[0] - X.mean(0) @ finale.coef_[0]) / np.linalg.norm(finale.coef_[0]))
groupes = KMeans(4, n_init=10, random_state=SEED).fit_predict(X).tolist()
# bande de 48 nombres de la fenêtre héros et poids de la tête finale sur ces 48 nombres
w = finale.coef_[0]
top = np.argsort(-np.abs(w))[:8]
dims = np.sort(np.concatenate([top, rng.choice(np.setdiff1d(np.arange(len(w)), top), 40, replace=False)]))
OUT["bande"] = {"dims": dims.tolist(), "brut": a["X"][hero][dims].round(3).tolist(), "poids": (np.abs(w[dims]) / np.abs(w).max()).round(3).tolist()}
OUT["nuage"] = {
    "site": site.tolist(), "y": y.astype(int).tolist(), "tour": tour.astype(int).tolist(), "cat1": cat,
    "groupe": groupes, "vues": vues, "ts": TS, "boussole": boussole, "triedre": triedre, "cube": cube, "seuil_u": seuil_u, "scores": scores,
    "ap_rejeu": [round(float(x), 3) for x in ap0], "n_rejeu": n0.tolist(),
}
brut = a["X"][hero]
OUT["vecteur"] = {"haut": brut[:8].round(2).tolist(), "bas": brut[-8:].round(2).tolist(), "poids": np.concatenate([w[:8], w[-8:]]).round(2).tolist(), "poids_max": round(float(np.abs(w).max()), 2), "min": round(float(brut.min()), 3), "max": round(float(brut.max()), 3),
                  "corpus_min": round(float(np.percentile(e["X"], 0.5)), 3), "corpus_max": round(float(np.percentile(e["X"], 99.5)), 3)}
(ICI / "donnees.json").write_text(json.dumps(OUT, ensure_ascii=False), encoding="utf-8")
print("nuage :", len(X), "points ; révélés tours 0-4 :", len(rev), "; héros", OUT["hero"])
print("AP rejeu 0 :", OUT["nuage"]["ap_rejeu"])
