# Modèle de script d'encodage : boucle qui relance `blanci embed` pour un encodeur jusqu'à la fin, en sautant les enregistrements déjà encodés. Le script empêche la mise en veille automatique. Ne pas fermer le clapet, ni VS Code, ni le terminal.
# Usage : encode_modele.ps1 -Encoder <nom de l'encodeur>   (copier ce fichier pour un besoin particulier)
param([Parameter(Mandatory)][string]$Encoder)
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power]::SetThreadExecutionState([uint32]2147483651)  # 0x80000003 : système + écran, contre la veille moderne
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$log = Join-Path $repo 'rapports\encodage.log'
function Note($m) { "{0:yyyy-MM-dd HH:mm} {1}" -f (Get-Date), $m | Add-Content $log -Encoding utf8 }

Set-Location $repo
$env:HF_HUB_OFFLINE = '1'  # modèles Hugging Face lus dans le cache : pas de code retéléchargé en cours d'encodage
$echecs = 0
while ($true) {
  $debut = Get-Date
  Note "encodage $Encoder : début"
  & (Join-Path $repo '.venv\Scripts\blanci.exe') --config config/local.yaml embed --encoder $Encoder *>> (Join-Path $repo "rapports\embed_$Encoder.log")
  $code = $LASTEXITCODE
  Note "encodage $Encoder : fin (code $code)"
  if ($code -eq 0) { break }
  # Arrêt après 3 échecs de moins de 10 min d'affilée (erreur autre qu'un plantage mémoire ponctuel).
  if (((Get-Date) - $debut).TotalMinutes -lt 10) { $echecs++ } else { $echecs = 0 }
  if ($echecs -ge 3) { Note "encodage $Encoder : abandon (3 échecs rapides)"; break }
  Start-Sleep 30
}
