"""Têtes de CAMPAIGN_HEADS sur une espèce et un encodeur (un pli par site) ; garde aussi les
scores hors-pli (pour l'amorçage d'un site). Usage : bench.py <encoder_id> <ESPECE> <sortie>"""

import sys
import time
from pathlib import Path

import numpy as np

import blanci.heads.head as head
from blanci.core.config import config_path, load_config, project_path
from blanci.core.db import connect
from blanci.evaluation.anuraset import CAMPAIGN_HEADS, read_strong_labels, run_anuraset_heads

enc, sp, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
cfg = load_config(Path("anuraset/anuraset.yaml"))
cfg["regularization"]["R37"]["glmm_grid"] = [0.3, 1.0, 3.0]  # comme le benchmark 01
captured = []
_orig = head.oof_scores
depth = [0]


def spy(inputs, y, groups, **kw):
    depth[0] += 1
    try:
        r = _orig(inputs, y, groups, **kw)
    finally:
        depth[0] -= 1
    if depth[0] == 0:
        captured.append((np.asarray(y), np.asarray(groups), np.asarray(r.values)))
    return r


head.oof_scores = spy
t = time.time()
con = connect(config_path(cfg, "db"))
calls = read_strong_labels(project_path(cfg["anuraset"]["labels"]))
res = run_anuraset_heads(con, cfg, enc, [sp], calls, CAMPAIGN_HEADS)
for k, v in res.items():
    v.to_csv(out / f"{sp}_{k}.csv", index=False)
names = res["table"][res["table"]["level"] == "window"]["head"].tolist()
assert len(names) == len(captured), (len(names), len(captured))
y, g = captured[0][0], captured[0][1]
np.savez_compressed(
    out / f"{sp}_scores.npz",
    y=y,
    site=g,
    heads=np.array(names),
    scores=np.stack([c[2] for c in captured]),
)
(out / f"{sp}_duree_s.txt").write_text(f"{time.time() - t:.0f}")
print("FINI", sp, f"{time.time() - t:.0f} s")
