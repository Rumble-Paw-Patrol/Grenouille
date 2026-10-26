# Configuration commune (disque 1 : BLANCI_2023-24)
$Dest   = 'F:\'
$DRoot  = 'D:\Projet Phénologie blanci'
$EMolo  = 'E:\Pheno_blanci_Molokoi'
$Rapports = Join-Path $PSScriptRoot '..\..\rapports'

# Dossier source D: -> dossier de destination normalisé sur F:\audio
$Releves = [ordered]@{
  'Premier relevé_décembre 2023'    = 'R1_2023-12_decembre'
  'Second relevé_février 2024'      = 'R2_2024-02_fevrier'
  'Troisième relevé_avril 2024'     = 'R3_2024-04_avril'
  'Quatrième relevé_juin 2024'      = 'R4_2024-06_juin'
  'Cinquième relevé_septembre 2024' = 'R5_2024-09_septembre'
  'Sixième relevé_novembre 2024'    = 'R6_2024-11_novembre'
}
