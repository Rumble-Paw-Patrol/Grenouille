# Boucle d'encodage BirdNET 3.0 qui saute les enregistrements déjà encodés. Le script empêche la mise en veille automatique. Ne pas fermer le clapet, ni VS Code, ni le terminal
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power]::SetThreadExecutionState([uint32]2147483651)  # 0x80000003 : système + écran, contre la veille moderne
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$log = Join-Path $repo 'rapports\weekend.log'
function Note($m) { "{0:yyyy-MM-dd HH:mm} {1}" -f (Get-Date), $m | Add-Content $log -Encoding utf8 }

Set-Location $repo
$enc = 'birdnet_v3'
$echecs = 0
while ($true) {
  $debut = Get-Date
  Note "encodage $enc : début"
  $env:PYTHONPATH = 'C:\Users\ll34272\Grenouille_encodage'
  & (Join-Path $repo '.venv\Scripts\blanci.exe') --config config/local.yaml embed --encoder $enc *>> (Join-Path $repo "rapports\embed_$enc.log")
  $code = $LASTEXITCODE
  Note "encodage $enc : fin (code $code)"
  if ($code -eq 0) { break }
  if (((Get-Date) - $debut).TotalMinutes -lt 10) { $echecs++ } else { $echecs = 0 }
  if ($echecs -ge 3) { Note "encodage $enc : abandon (3 échecs rapides)"; break }
  Start-Sleep 30
}
