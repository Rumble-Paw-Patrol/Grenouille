# Encodage du corpus entériné (contrôle qualité clos le 20/11/2026, manifeste annotations\corpus_20261120\) : perch_v2, birdnet et birdnet_v3 encodent tout ce qui manque à leur stock (dont ce qui vient d'en être retiré pour mauvaise piste), puis audioprotopnet tout le corpus, sur F: (data\embeddings : C: n'a pas la place, ≈ 9,3 Go).
# Avant : 1. uv run python scripts/encodage/file_pistes.py (listes <modèle>.csv dans rapports\encodage_pistes\) ; 2. retirer des stocks ce qui est encodé sur une autre piste que la règle (blanci pistes ... --apply, voir ENCODAGE.txt) : sans ça, `blanci embed` refuse le stock et ce script s'arrête.
# Les deux disques doivent rester branchés : BLANCI_2023-24 (audio 2023, Kaw D) et BLANCI_2025-26 (F: : audio 2026, stocks perch_v2 et audioprotopnet sous data\embeddings).
# Usage : encode_pistes.ps1 [-Depuis <modèle>]   (-Depuis birdnet_v3 : reprend à ce modèle ; chaque modèle saute ce qui est déjà fait)
param([ValidateSet('perch_v2', 'birdnet', 'birdnet_v3', 'audioprotopnet')][string]$Depuis = 'perch_v2')
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$modele = Join-Path $PSScriptRoot 'encode_modele.ps1'
$listes = Join-Path $repo 'rapports\encodage_pistes'
$file = @(
  @{ Encoder = 'perch_v2'; Recordings = "$listes\perch_v2.csv"; Embeddings = 'data\embeddings' },
  @{ Encoder = 'birdnet'; Recordings = "$listes\birdnet.csv" },
  @{ Encoder = 'birdnet_v3'; Recordings = "$listes\birdnet_v3.csv" },
  @{ Encoder = 'audioprotopnet'; Embeddings = 'data\embeddings' }  # tout le corpus, sur F:
)
$parti = $false
foreach ($etape in $file) {
  if ($etape.Encoder -eq $Depuis) { $parti = $true }
  if (-not $parti) { continue }
  if ($etape.Recordings -and -not (Test-Path $etape.Recordings)) { throw "liste absente : $($etape.Recordings) (lancer file_pistes.py)" }
  & $modele @etape
  if ($LASTEXITCODE -ne 0) { throw "$($etape.Encoder) : abandon, voir rapports\embed_$($etape.Encoder).log" }
}
