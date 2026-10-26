# Copie D: -> F: (disque 1). Relancable : robocopy ne recopie que ce qui manque ou diffère.
. "$PSScriptRoot\config.ps1"
$fs = (Get-Volume -DriveLetter F -ErrorAction Stop)
if ($fs.FileSystem -ne 'NTFS' -or $fs.FileSystemLabel -ne 'BLANCI_2023-24') { throw "F: n'est pas le disque BLANCI_2023-24 en NTFS. Arrêt." }
New-Item -ItemType Directory -Force $Rapports | Out-Null
$log = Join-Path $Rapports ("copie_{0:yyyyMMdd_HHmm}.log" -f (Get-Date))

foreach ($src in $Releves.Keys) {
  $dst = Join-Path $Dest ("audio\" + $Releves[$src])
  robocopy (Join-Path $DRoot $src) $dst /E /COPY:DAT /DCOPY:DAT /R:2 /W:5 /NP /XF 'Thumbs.db' /UNILOG+:$log /TEE
  if ($LASTEXITCODE -ge 8) { Write-Warning "robocopy a signalé des erreurs pour $src (code $LASTEXITCODE)" }
}
# Documentation : Analyses Biophonia (inclut le summary_byrec.xlsx, identique à celui de E:)
robocopy (Join-Path $DRoot 'Analyses Biophonia') (Join-Path $Dest 'documentation\Analyses_Biophonia') /E /COPY:DAT /R:2 /W:5 /NP /UNILOG+:$log /TEE
"Journal : $log"
