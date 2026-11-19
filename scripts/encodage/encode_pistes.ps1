# File « bonnes pistes » : réencode sur la bonne piste les trois groupes (piste 0 étouffée, Kaw D au micro 0 sourd, gain inversé) avec perch_v2, birdnet et birdnet_v3, puis encode tout le corpus avec audioprotopnet, déjà sur les bonnes pistes.
# Avant : 1. uv run python scripts/encodage/file_pistes.py (listes dans rapports\encodage_pistes\) ; 2. retirer des stocks ce qui est encodé sur la mauvaise piste (blanci pistes ... --apply, voir ENCODAGE.txt) : sans ça, `blanci embed` refuse le stock et ce script s'arrête.
# Les deux disques doivent rester branchés : BLANCI_2023-24 (audio 2023, Kaw D) et BLANCI_2025-26 (F: : audio 2026, stock perch_v2 sous data\embeddings).
# Usage : encode_pistes.ps1 [-Depuis <modèle>]   (-Depuis birdnet_v3 : reprend à ce modèle ; chaque modèle saute ce qui est déjà fait)
param([ValidateSet('perch_v2', 'birdnet', 'birdnet_v3', 'audioprotopnet')][string]$Depuis = 'perch_v2')
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$modele = Join-Path $PSScriptRoot 'encode_modele.ps1'
$listes = Join-Path $repo 'rapports\encodage_pistes'
$file = @(
  @{ Encoder = 'perch_v2'; Recordings = "$listes\perch_v2_pistes.csv"; Embeddings = 'data\embeddings' },
  @{ Encoder = 'birdnet'; Recordings = "$listes\birdnet_pistes.csv" },
  @{ Encoder = 'birdnet_v3'; Recordings = "$listes\birdnet_v3_pistes.csv" },
  @{ Encoder = 'audioprotopnet' }  # tout le corpus
)
$parti = $false
foreach ($etape in $file) {
  if ($etape.Encoder -eq $Depuis) { $parti = $true }
  if (-not $parti) { continue }
  if ($etape.Recordings -and -not (Test-Path $etape.Recordings)) { throw "liste absente : $($etape.Recordings) (lancer file_pistes.py)" }
  & $modele @etape
  if ($LASTEXITCODE -ne 0) { throw "$($etape.Encoder) : abandon, voir rapports\embed_$($etape.Encoder).log" }
}
