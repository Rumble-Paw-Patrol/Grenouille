# Chaîne du week-end : copie D: -> F:, puis encodage perch_v2, puis birdnet (tout est reprenable).
# Empêche la mise en veille tant que ce script tourne (sans toucher aux réglages de Windows).
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power]::SetThreadExecutionState(0x80000001)
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$log = Join-Path $repo 'rapports\weekend.log'
function Note($m) { "{0:yyyy-MM-dd HH:mm} {1}" -f (Get-Date), $m | Add-Content $log -Encoding utf8 }

Set-Location $repo
foreach ($enc in 'birdnet', 'birdnet_v3') {
  # Plantages mémoire de TensorFlow toutes les 4 à 6 h : on relance (la reprise saute le déjà fait).
  # Arrêt après 3 échecs de moins de 10 min d'affilée (erreur autre que la mémoire).
  $echecs = 0
  while ($true) {
    $debut = Get-Date
    Note "encodage $enc : début"
    # Code figé dans une copie propre (Grenouille_encodage) : la fusion en cours dans le dépôt ne
    # doit pas arrêter l'encodage. Environnement Python : celui du dépôt.
    $env:PYTHONPATH = 'C:\Users\ll34272\Grenouille_encodage'
    & (Join-Path $repo '.venv\Scripts\blanci.exe') --config config/local.yaml embed --encoder $enc *>> (Join-Path $repo "rapports\embed_$enc.log")
    $code = $LASTEXITCODE
    Note "encodage $enc : fin (code $code)"
    if ($code -eq 0) { break }
    if (((Get-Date) - $debut).TotalMinutes -lt 10) { $echecs++ } else { $echecs = 0 }
    if ($echecs -ge 3) { Note "encodage $enc : abandon (3 échecs rapides)"; break }
    Start-Sleep 30
  }
}


