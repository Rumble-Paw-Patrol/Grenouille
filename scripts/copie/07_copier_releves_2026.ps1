# Copie des relevés 2026 arrivés après le 20/11/2026 (disque Seagate Basic de Biophonia, lu
# seulement) -> F: BLANCI_2025-26. Relançable : robocopy ne recopie que ce qui manque, ne
# supprime rien, ne touche pas la source. La lettre du Seagate se repère par son étiquette.
param([string[]]$Seulement)
$ErrorActionPreference = 'Stop'
$fs = Get-Volume | Where-Object FileSystemLabel -eq 'BLANCI_2025-26'
if (-not $fs -or $fs.FileSystem -ne 'NTFS') { throw "Disque BLANCI_2025-26 introuvable. Arrêt." }
$src = Get-Volume | Where-Object FileSystemLabel -eq 'Seagate Basic'
if (-not $src) { throw "Disque Seagate Basic introuvable. Arrêt." }
$Dest = "$($fs.DriveLetter):\audio"
$Root = "$($src.DriveLetter):\"
$Rapports = Join-Path $PSScriptRoot '..\..\rapports'
New-Item -ItemType Directory -Force $Rapports | Out-Null
$log = Join-Path $Rapports ("copie_releves_2026_{0:yyyyMMdd_HHmm}.log" -f (Get-Date))
# Dossier source -> dossier normalisé sur BLANCI_2025-26\audio
$Releves = [ordered]@{
  'Enregistrements blanci Mataroni mai 2026'        = 'R7_2026-05_Mataroni'
  # Gabaret (GAB_164, GAB_166) et sa sous-section Pai (PAI_124), 11/02 au 15/04/2026
  'Enregistrements blanci GAB RHOZACE fev-mars2026' = 'R4_2026-02_RHOZACE'
}
$opts = '/E', '/COPY:DAT', '/DCOPY:DAT', '/R:2', '/W:5', '/NP', '/XF', 'Thumbs.db'
foreach ($s in $Releves.Keys) {
  if ($Seulement -and $Releves[$s] -notin $Seulement) { continue }
  robocopy (Join-Path $Root $s) (Join-Path $Dest $Releves[$s]) @opts /UNILOG+:$log
  if ($LASTEXITCODE -ge 8) { throw "robocopy : échec ($LASTEXITCODE) pour $s, voir $log" }
}
"{0:yyyy-MM-dd HH:mm} copie relevés 2026 : fin ($log)" -f (Get-Date)
exit 0  # robocopy rend 1 à 7 quand tout va bien (fichiers copiés, sautés…)
