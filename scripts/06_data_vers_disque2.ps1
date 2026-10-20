# Déplace tout data\ du PC vers F:\data (BLANCI_2025-26) et laisse à la place une jonction data -> F:\data :
# les chemins relatifs « data/... » du code et de la configuration restent valables. Relance ensuite BirdNET.
$repo = Split-Path $PSScriptRoot -Parent
$log = Join-Path $repo 'rapports\weekend.log'
function Note($m) { "{0:yyyy-MM-dd HH:mm} {1}" -f (Get-Date), $m | Add-Content $log -Encoding utf8 }
$fs = Get-Volume -DriveLetter F -ErrorAction Stop
if ($fs.FileSystem -ne 'NTFS' -or $fs.FileSystemLabel -ne 'BLANCI_2025-26') { Note "data : F: n'est pas BLANCI_2025-26, arrêt"; exit 1 }
$data = Join-Path $repo 'data'
if ((Get-Item $data).LinkType -eq 'Junction') { Note "data : déjà une jonction"; exit 0 }

Note "data : déplacement vers F:\data : début"
robocopy $data 'F:\data' /E /MOVE /COPY:DAT /DCOPY:DAT /R:2 /W:5 /NP /NFL /NDL "/UNILOG+:$repo\rapports\data_deplacement.log"
$left = @(Get-ChildItem $data -Recurse -File -ErrorAction SilentlyContinue).Count
if ($LASTEXITCODE -ge 8 -or $left -gt 0) { Note "data : $left fichiers non déplacés (code $LASTEXITCODE), arrêt"; exit 1 }
Remove-Item $data -Recurse -Force
New-Item -ItemType Junction -Path $data -Target 'F:\data' | Out-Null
Note "data : jonction data -> F:\data créée"

# Les embeddings suivent data/ : on retire le chemin direct ajouté par 05.
$cfg = Join-Path $repo 'config\local.yaml'
$t = [IO.File]::ReadAllText($cfg) -replace '(?m)^  embeddings: "F:/embeddings"\r?\n', ''
[IO.File]::WriteAllText($cfg, $t, (New-Object Text.UTF8Encoding $false))
if (Test-Path 'F:\data\embeddings') { Remove-Item 'F:\data\embeddings' -Recurse -Force }  # dossier vide laissé par 05
Move-Item 'F:\embeddings' 'F:\data\embeddings'
Note "data : embeddings rangés dans F:\data\embeddings"

& (Join-Path $PSScriptRoot 'weekend.ps1')
