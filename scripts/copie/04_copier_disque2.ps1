# Copie D: (+ 18 fichiers de E:) -> F: BLANCI_2025-26 (disque 2). Relancable : robocopy ne recopie que ce qui manque.
# Les fichiers de RELEVE 3 déjà présents dans RELEVE 1 (même nom, même taille : enregistreurs non vidés) sont écartés.
$fs = Get-Volume -DriveLetter F -ErrorAction Stop
if ($fs.FileSystem -ne 'NTFS' -or $fs.FileSystemLabel -ne 'BLANCI_2025-26') { throw "F: n'est pas le disque BLANCI_2025-26 en NTFS. Arrêt." }
$Root = 'D:\Projet blanci 2025'
$Rapports = Join-Path $PSScriptRoot '..\..\rapports'
$log = Join-Path $Rapports ("copie2_{0:yyyyMMdd_HHmm}.log" -f (Get-Date))
$Releves = [ordered]@{
  'RELEVE 1 CDR - 21-27 décembre 2025'         = 'R1_2025-12_CDR'
  'RELEVE 2 Patawa - 30 dec 2025 -04 janv 2026' = 'R2_2025-12_Patawa'
  'RELEVE 3 Mataroni - 06-13 janv 2026'         = 'R3_2026-01_Mataroni'
  'RELEVE 5 RNRT - 9-16 fev 2026'               = 'R5_2026-02_RNRT'
  'RELEVE 6 Patawa Est - 20 fev-3 mars 2026'    = 'R6_2026-02_PatawaEst'
}
$opts = '/E', '/COPY:DAT', '/DCOPY:DAT', '/R:2', '/W:5', '/NP', '/XF', 'Thumbs.db'

$r1 = @{}
Get-ChildItem (Join-Path $Root ($Releves.Keys | Select -First 1)) -Recurse -File | % { $r1["$($_.Name)|$($_.Length)"] = 1 }

foreach ($src in $Releves.Keys) {
  $s = Join-Path $Root $src; $d = "F:\audio\" + $Releves[$src]
  if ($src -like 'RELEVE 3*') {
    foreach ($sub in Get-ChildItem $s -Directory) {
      $dup = Get-ChildItem $sub.FullName -Recurse -File | ? { $r1.ContainsKey("$($_.Name)|$($_.Length)") } | % Name
      robocopy $sub.FullName (Join-Path $d $sub.Name) @opts @dup /UNILOG+:$log
    }
    robocopy $s $d /LEV:1 /COPY:DAT /R:2 /W:5 /NP /UNILOG+:$log
  } else {
    robocopy $s $d @opts /UNILOG+:$log
  }
}
robocopy 'E:\Enregistrements blanci Mataroni mai 2026' 'F:\audio\R7_2026-05_Mataroni' @opts /UNILOG+:$log
"{0:yyyy-MM-dd HH:mm} copie disque 2 : fin" -f (Get-Date) | Add-Content (Join-Path $Rapports 'weekend.log') -Encoding utf8
