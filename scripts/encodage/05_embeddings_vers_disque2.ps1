# Déplace data\embeddings du PC vers F:\embeddings (BLANCI_2025-26), pointe paths.embeddings dessus,
# puis relance l'encodage BirdNET (weekend.ps1). Les encodages en cours doivent être arrêtés avant.
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$log = Join-Path $repo 'rapports\weekend.log'
function Note($m) { "{0:yyyy-MM-dd HH:mm} {1}" -f (Get-Date), $m | Add-Content $log -Encoding utf8 }
$fs = Get-Volume -DriveLetter F -ErrorAction Stop
if ($fs.FileSystem -ne 'NTFS' -or $fs.FileSystemLabel -ne 'BLANCI_2025-26') { Note "embeddings : F: n'est pas BLANCI_2025-26, arrêt"; exit 1 }

Note "embeddings : déplacement vers F:\embeddings : début"
robocopy (Join-Path $repo 'data\embeddings') 'F:\embeddings' /E /MOVE /COPY:DAT /DCOPY:DAT /R:2 /W:5 /NP /NFL /NDL "/UNILOG+:$repo\rapports\embeddings_deplacement.log"
if ($LASTEXITCODE -ge 8) { Note "embeddings : erreurs robocopy (code $LASTEXITCODE), arrêt"; exit 1 }
Note "embeddings : déplacement : fin"

$cfg = Join-Path $repo 'config\local.yaml'
$t = [IO.File]::ReadAllText($cfg)
if ($t -notmatch '(?m)^  embeddings:') { $t = $t -replace '(?m)^(  raw: "D:/")', "`$1`r`n  embeddings: `"F:/embeddings`"" }
[IO.File]::WriteAllText($cfg, $t, (New-Object Text.UTF8Encoding $false))

& (Join-Path $PSScriptRoot 'weekend.ps1')
