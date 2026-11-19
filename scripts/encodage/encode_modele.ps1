# Modèle de script d'encodage : boucle qui relance `blanci embed` pour un encodeur jusqu'à la fin, en sautant les enregistrements déjà encodés. Le script empêche la mise en veille automatique. Ne pas fermer le clapet, ni VS Code, ni le terminal.
# Usage : encode_modele.ps1 -Encoder <nom de l'encodeur> [-Recordings <liste.csv>] [-Embeddings <dossier des stocks>]
#   -Recordings : n'encoder que les enregistrements de cette liste (colonne recording_id), ex. rapports\encodage_pistes\birdnet_pistes.csv
#   -Embeddings : dossier des stocks, si ce n'est pas paths.embeddings de config/local.yaml (ex. data\embeddings pour perch_v2, sur F:)
# Sortie : code 0 si l'encodage est allé au bout, 1 après 3 échecs rapides.
param([Parameter(Mandatory)][string]$Encoder, [string]$Recordings, [string]$Embeddings)
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power]::SetThreadExecutionState([uint32]2147483651)  # 0x80000003 : système + écran, contre la veille moderne
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$log = Join-Path $repo 'rapports\encodage.log'
function Note($m) { "{0:yyyy-MM-dd HH:mm} {1}" -f (Get-Date), $m | Add-Content $log -Encoding utf8 }

Set-Location $repo
$env:HF_HUB_OFFLINE = '1'  # modèles Hugging Face lus dans le cache : pas de code retéléchargé en cours d'encodage
$options = @('--encoder', $Encoder)
if ($Recordings) { $options += @('--recordings', $Recordings) }
if ($Embeddings) { $options += @('--embeddings', $Embeddings) }
$echecs = 0
while ($true) {
  $debut = Get-Date
  Note "encodage $Encoder : début $Recordings"
  & (Join-Path $repo '.venv\Scripts\blanci.exe') --config config/local.yaml embed @options *>> (Join-Path $repo "rapports\embed_$Encoder.log")
  $code = $LASTEXITCODE
  Note "encodage $Encoder : fin (code $code)"
  if ($code -eq 0) { exit 0 }
  # Arrêt après 3 échecs de moins de 10 min d'affilée (erreur autre qu'un plantage mémoire ponctuel).
  if (((Get-Date) - $debut).TotalMinutes -lt 10) { $echecs++ } else { $echecs = 0 }
  if ($echecs -ge 3) { Note "encodage $Encoder : abandon (3 échecs rapides)"; exit 1 }
  Start-Sleep 30
}
