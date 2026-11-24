# Copie

Les scripts PowerShell (Windows) de copie des enregistrements vers les disques externes. Tous sont relançables : `robocopy` ne recopie que ce qui manque. Les journaux vont dans `rapports/` (ignoré par git).

- `config.ps1` : la configuration commune (lettres de disque, dossiers sources, correspondance relevé vers dossier normalisé).
- `01_comparer_doublons.ps1` : compare en lecture seule les dossiers Molokoi de E: et de D:.
- `02_copier_disque1.ps1` : copie D: vers le disque 1 (`BLANCI_2023-24`).
- `03_verifier_sha256.ps1` : vérifie la copie par SHA-256 de chaque fichier source et destination (long, plusieurs heures).
- `04_copier_disque2.ps1` : copie D: et quelques fichiers de E: vers le disque 2 (`BLANCI_2025-26`).
- `06_data_vers_disque2.ps1` : déplace `data/` du PC vers le disque 2 et laisse une jonction à la place.
- `07_copier_releves_2026.ps1` : copie les relevés 2026 arrivés après le 20/11/2026 depuis le disque Seagate Basic de Biophonia (lu seulement, repéré par son étiquette) vers `BLANCI_2025-26\audio` : `R7_2026-05_Mataroni` et `R4_2026-02_RHOZACE` (Gabaret et Pai). `-Seulement R4_2026-02_RHOZACE` pour un seul relevé. Le 23/11 : 11 689 fichiers, 269 Go, 0 échec, comptes et octets identiques des deux côtés.
