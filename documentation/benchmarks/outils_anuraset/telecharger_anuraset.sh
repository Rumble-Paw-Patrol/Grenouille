#!/usr/bin/env bash
# Télécharge raw_data.zip d'AnuraSet (Zenodo 8342596, 7,2 Go) par 12 plages d'octets en
# parallèle, reprenable, puis assemble sans doubler la place. À lancer depuis la racine du dépôt.
# Les étiquettes (strong_labels.zip, weak_labels.csv) sont sur la branche resultats-anuraset-07 :
#   git archive origin/resultats-anuraset-07 data | tar -x
set -u
DEST=data/external/anuraset
mkdir -p "$DEST/parts" && cd "$DEST"
URL="https://zenodo.org/api/records/8342596/files/raw_data.zip/content"
S=$(curl -sS "https://zenodo.org/api/records/8342596" | python3 -c \
  "import sys, json; print(next(f['size'] for f in json.load(sys.stdin)['files'] if f['key'] == 'raw_data.zip'))")
K=12
CHUNK=$(( (S + K - 1) / K ))
fetch() {
  local i=$1 start=$(( $1 * CHUNK )) end=$(( ($1 + 1) * CHUNK - 1 ))
  [ $end -ge $S ] && end=$(( S - 1 ))
  local want=$(( end - start + 1 )) f=$(printf "parts/p%02d" $i)
  for attempt in $(seq 1 50); do
    local have=0
    [ -f "$f" ] && have=$(stat -c %s "$f")
    [ $have -ge $want ] && return 0
    curl -sS -L --max-time 1800 -r $(( start + have ))-$end "$URL" >> "$f" || sleep 3
  done
}
for i in $(seq 0 $((K - 1))); do fetch $i & done
wait
for f in $(ls parts/p* | sort | tail -n +2); do cat "$f" >> parts/p00 && rm "$f"; done
mv parts/p00 raw_data.zip && rmdir parts
echo "assemblé : $(stat -c %s raw_data.zip) / $S octets"
