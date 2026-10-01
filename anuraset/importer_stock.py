"""Importe le stock d'un encodeur depuis sa branche de données, sans écraser la base locale :
les fichiers du stock sont extraits, et son modèle (table `models`) et ses fenêtres (table
`windows`) sont ajoutés à `data/db/anuraset.sqlite`, qui doit déjà exister (anuraset prepare
ou un stock importé avant).

Usage : importer_stock.py <encodeur> [<branche>]   (branche : donnees-anuraset-<encodeur> ;
perch_v2 : donnees-anuraset)
"""

import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

name = sys.argv[1]
branch = sys.argv[2] if len(sys.argv) > 2 else f"donnees-anuraset-{name}"
subprocess.run(["git", "fetch", "origin", branch], check=True)
archive = subprocess.run(
    ["git", "archive", f"origin/{branch}", "data/embeddings_anuraset"],
    check=True,
    capture_output=True,
)
subprocess.run(["tar", "-x"], input=archive.stdout, check=True)
db = Path("data/db/anuraset.sqlite")
if not db.exists():
    raise SystemExit("data/db/anuraset.sqlite absent : lancer anuraset prepare d'abord")
with tempfile.TemporaryDirectory() as tmp:
    theirs = Path(tmp) / "anuraset.sqlite"
    theirs.write_bytes(
        subprocess.run(
            ["git", "show", f"origin/{branch}:data/db/anuraset.sqlite"],
            check=True,
            capture_output=True,
        ).stdout
    )
    con, other = sqlite3.connect(db), sqlite3.connect(theirs)
    mine = con.execute("SELECT recording_id, path FROM recordings ORDER BY recording_id")
    if (
        mine.fetchall()
        != other.execute(
            "SELECT recording_id, path FROM recordings ORDER BY recording_id"
        ).fetchall()
    ):
        raise SystemExit("inventaires différents : enregistrements non alignés, rien importé")
    rows = other.execute("SELECT window_id, recording_id, offset_s, dur_s FROM windows").fetchall()
    con.executemany("INSERT OR IGNORE INTO windows VALUES (?, ?, ?, ?)", rows)
    models = other.execute("SELECT * FROM models WHERE kind = 'encoder'").fetchall()
    width = len(models[0]) if models else 0
    con.executemany(f"INSERT OR REPLACE INTO models VALUES ({','.join('?' * width)})", models)
    con.commit()
print(f"importé depuis {branch} : {[m[0] for m in models]}")
