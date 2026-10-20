# Lecture seule : compare E:\Pheno_blanci_Molokoi avec les dossiers Molokoi de D:
. "$PSScriptRoot\config.ps1"
New-Item -ItemType Directory -Force $Rapports | Out-Null
$rows = @()
foreach ($e in Get-ChildItem $EMolo -Directory) {
  $dDir = Get-ChildItem $DRoot -Directory -Recurse -Depth 2 | Where-Object Name -eq $e.Name | Select-Object -First 1
  if (-not $dDir) { $rows += [pscustomobject]@{Dossier=$e.Name;Statut='DOSSIER_ABSENT_DE_D';Fichier='';TailleE=''}; continue }
  $dIdx = @{}
  Get-ChildItem $dDir.FullName -File -Recurse -Force | ForEach-Object { $dIdx[$_.FullName.Substring($dDir.FullName.Length)] = $_.Length }
  Get-ChildItem $e.FullName -File -Recurse -Force | ForEach-Object {
    $rel = $_.FullName.Substring($e.FullName.Length)
    if (-not $dIdx.ContainsKey($rel)) { $st='SEULEMENT_SUR_E' }
    elseif ($dIdx[$rel] -ne $_.Length) { $st='TAILLE_DIFFERENTE' } else { $st='IDENTIQUE' }
    $rows += [pscustomobject]@{Dossier=$e.Name;Statut=$st;Fichier=$rel;TailleE=$_.Length}
  }
}
$rows | Export-Csv "$Rapports\comparaison_molokoi_E_vs_D.csv" -NoTypeInformation -Encoding UTF8
$rows | Group-Object Dossier,Statut | ForEach-Object { '{0,-45} {1,6}' -f $_.Name,$_.Count }
