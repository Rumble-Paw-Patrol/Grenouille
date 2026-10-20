"""Encode AnuraSet (sélection du n° 141) avec les encodeurs donnés, l'un après l'autre."""

import sys
import time
from pathlib import Path

from blanci.cli import upstream_chain
from blanci.core.config import config_path, load_config, project_path
from blanci.core.db import connect
from blanci.embedding.embed import embed_recordings, select_recordings
from blanci.embedding.encoders import get_encoder
from blanci.embedding.grid import overlap_from_cfg
from blanci.evaluation.anuraset import campaign_recordings, read_strong_labels

cfg = load_config(Path("anuraset/anuraset.yaml"))
acfg = cfg["anuraset"]
con = connect(config_path(cfg, "db"))
calls = read_strong_labels(project_path(acfg["labels"]))
wanted = campaign_recordings(
    select_recordings(con, exclude_flags=()), calls, acfg.get("weak_labels") or None
)
print(f"{len(wanted)} enregistrements", flush=True)
for name in sys.argv[1:]:
    t = time.time()
    model = get_encoder(name, cfg, upstream_chain(cfg, "none"))
    done = embed_recordings(
        con,
        model,
        wanted,
        config_path(cfg, "raw"),
        config_path(cfg, "embeddings"),
        overlap=overlap_from_cfg(cfg),
        channel=cfg["audio"]["channel"],
        resample=cfg["encoders"].get("resample", "recording"),
    )
    print(
        f"FINI {done.encoder_id} : {done.recordings} encodés, {done.skipped} sautés, "
        f"{done.windows_per_s:.1f} f/s, {time.time() - t:.0f} s",
        flush=True,
    )
