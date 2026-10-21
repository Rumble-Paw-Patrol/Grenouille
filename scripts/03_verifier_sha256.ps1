# Vérifie la copie : SHA-256 de chaque fichier source et destination, puis comparaison.
# Écrit F:\manifeste_sha256.csv et un rapport de contrôle. Long (plusieurs heures).
. "$PSScriptRoot\config.ps1"
$pairs = @()
foreach ($src in $Releves.Keys) { $pairs += ,@((Join-Path $DRoot $src), (Join-Path $Dest ("audio\" + $Releves[$src]))) }
$pairs += ,@((Join-Path $DRoot 'Analyses Biophonia'), (Join-Path $Dest 'documentation\Analyses_Biophonia'))

$manifest = New-Object System.Collections.Generic.List[object]
$bad = New-Object System.Collections.Generic.List[object]
$n = 0
foreach ($p in $pairs) {
  $s = $p[0]; $d = $p[1]
  foreach ($f in Get-ChildItem $s -File -Recurse -Force) {
    $rel = $f.FullName.Substring($s.Length)
    $t = Join-Path $d $rel
    if (-not (Test-Path -LiteralPath $t)) { $bad.Add([pscustomobject]@{Fichier=$f.FullName;Probleme='MANQUANT'}); continue }
    $h1 = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash
    $h2 = (Get-FileHash -LiteralPath $t -Algorithm SHA256).Hash
    if ($h1 -ne $h2) { $bad.Add([pscustomobject]@{Fichier=$f.FullName;Probleme='HASH_DIFFERENT'}) }
    $manifest.Add([pscustomobject]@{Chemin=$t.Substring(3);Taille=$f.Length;SHA256=$h2})
    if ((++$n) % 1000 -eq 0) { Write-Host "$n fichiers vérifiés" }
  }
}
$manifest | Export-Csv (Join-Path $Dest 'manifeste_sha256.csv') -NoTypeInformation -Encoding UTF8
$bad | Export-Csv (Join-Path $Rapports 'verification_problemes.csv') -NoTypeInformation -Encoding UTF8
"Vérifiés : $n | Problèmes : $($bad.Count)"
