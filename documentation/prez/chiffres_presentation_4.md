# Chiffres de la présentation n° 4, rejoués

Rejoué le 19/11/2026 à 17 h 56 par `uv run python scripts/verification/chiffres_presentation_4.py`, code au commit `d00fe7e`. Chaque chiffre vient de la fonction citée, dans `blanci/exploration/inventaire.py` (fiches : `blanci/exploration/inventaire.md`).

Base : `blanci.sqlite`, schéma v6, 96 349 lignes dans `recordings`, empreinte de l'inventaire `92b0ecc4b7569d7b`, des drapeaux `303db0347b5b3c22`, dernier extrait écouté le 2026-11-19T19:34:51Z. Même empreinte = même base : les chiffres doivent être les mêmes.

Rapports lus : `pistes/echantillon_pistes.csv`, `ingest_duplicates_2026_Mataroni.csv`. Absents : aucun.

| Diapo | Chiffre | Annoncé | Recalculé | Statut | Fonction |
|---|---|---|---|---|---|
| 3 | enregistrements (jeux 2023 et 2026) | 96 310 | 96 310 | ok | `corpus_totals(load_recordings(con, cfg))['recordings']` |
| 3 | heures | 3 207 | 3 206,752 | ok | `corpus_totals(rec)['hours']` |
| 3 | points d'écoute (jeu, site, micro) | 164 | 164 | ok | `corpus_totals(rec)['points']` |
| 3 | enregistreurs | 95 | 95 | ok | `corpus_totals(rec)['recorders']` |
| 3 | lieux (relevés empilés sur leur site) | 4 | 4 | ok | `rec.site.map(lieux).nunique()` |
| — | lignes hors des jeux 2023 et 2026 (non comptées) | — | 39 | recalculé (pas de chiffre annoncé) | `dataset_counts(con)` |
| 5 bis | taille de la base (Go) | 6,7 | 6,677 | ok | `database_fingerprint(con)['size_bytes']` |
| 3 | enregistrements, programme 5 h → 19 h 30 | 74 007 | 74 007 | ok | `programme_counts(rec, recording_programme(rec))` |
| 3 | enregistrements, programme 6 h → 18 h 30 | 22 289 | 22 289 | ok | `programme_counts(rec, recording_programme(rec))` |
| 3 | enregistrements de points sans programme lu | — | 14 | recalculé (pas de chiffre annoncé) | `programme_counts(rec, recording_programme(rec))` |
| 10 | Mataroni, 30-31/12/2025 : enregistrements | 26 | 26 | ok | `counts_by(rec[Mataroni, 30-31/12], 'site')` |
| 10 | Mataroni, 30-31/12/2025 : micros | 20 | 20 | ok | `counts_by(...)` |
| 4 | dont dans la campagne (trou de moins de 7 j) : enregistrements | 25 | 25 | ok | `near_campaign_orphans(rec, cfg['qc'])` |
| 4 | dont dans la campagne : micros | 12 | 19 | différent | `near_campaign_orphans(rec, cfg['qc'])` |
| 6 | points réglés 6 / 18 dB | 143 | 143 | ok | `gain_settings(echantillon_pistes)` |
| 6 | points réglés 18 / 6 dB | 21 | 21 | ok | `gain_settings(echantillon_pistes)` |
| 6 | dont enregistreurs 2MA | 18 | 18 | ok | `gain_settings(echantillon_pistes)` |
| 6 | dont enregistreurs SMA | 3 | 3 | ok | `gain_settings(echantillon_pistes)` |
| 6 | enregistrements lus (toutes lignes de l'échantillon) | 3 607 | 3 607 | ok | `len(echantillon_pistes.csv)` |
| 6 | dont un par micro et par jour (le reste : bascules d'état) | — | 3 466 | recalculé (pas de chiffre annoncé) | `sample_recordings(rec, channels=2) ≈ motif « jour »` |
| 6 | points de l'échantillon | 164 | 164 | ok | `echantillon_pistes.csv` |
| 20 | points à plus de 6 dB de l'écart réglé | 5 | 5 | ok | `track_gap_by_point(echantillon_pistes, gain_settings(...), 'level_diff_db')` |
| 7 | jours-micros intérieurs | 3 137 | 3 137 | ok | `completeness_summary(mic_days(rec, prog))` |
| 7 | jours-micros complets | 3 128 | 3 128 | ok | `completeness_summary(...)` |
| 7 | part complète (%) | 99,7 | 99,713 | ok | `completeness_summary(...)` |
| 7 | jours partiels | 9 | 9 | ok | `completeness_summary(...)` |
| 7 | jours-micros 2023 | 2 217 | 2 217 | ok | `completeness_summary(...)` |
| 7 | part complète 2023 (%) | 99,8 | 99,82 | ok | `completeness_summary(...)` |
| 7 | jours-micros 2026 | 920 | 920 | ok | `completeness_summary(...)` |
| 7 | part complète 2026 (%) | 99,5 | 99,457 | ok | `completeness_summary(...)` |
| 7 | Molokoï E : plus long trou (jours) | 54 | 54 | ok | `empty_days(rec, 'station', stations)` |
| 7 | Kaw D : plus long trou (jours) | 5 | 5 | ok | `empty_days(rec, 'station', stations)` |
| 7 | RNRT : micros sans rien le 10/02/2026 | 5 | 5 | ok | `empty_days(rec, 'mic')` |
| 9 | enregistrements à l'horloge corrigée | 1 521 | 1 521 | ok | `clock_corrected_counts(rec)` |
| 9 | corrélation à −1 h | 0,57 | 0,566 | ok | `clock_lag_correlation(rec, 'SMA14636', 'SMA14163', site='Molokoi')` |
| 9 | corrélation sans décalage | 0,34 | 0,342 | ok | `clock_lag_correlation(...)` |
| 9 | décalage de corrélation maximale (h) | -1 | -1 | ok | `clock_lag_correlation(...)` |
| 9 | créneaux communs à −1 h | 1 521 | 1 521 | ok | `clock_lag_correlation(...)` |
| 9 ter | écartés | 681 | 695 | différent | `flag_counts(rec)['excluded']` |
| 24 | encodables (diapo : 95 611 sur 96 292) | 95 611 | 95 615 | différent | `flag_counts(rec)['kept']` |
| 24 | inventaire du bilan (diapo : 96 292) | 96 292 | 96 310 | différent | `flag_counts(rec)['recordings']` |
| 9 ter | écartés pour son étouffé | 506 | 521 | différent | `flag_counts(rec)['causes']` |
| 9 ter | dont étouffé par le calcul (deux pistes) | 399 | 436 | différent | `flag_counts(rec)['causes']` |
| 9 ter | écartés pour la durée | 173 | 172 | différent | `flag_counts(rec)['causes']` |
| 9 ter | hors relevé seul | 2 | 2 | ok | `flag_counts(rec)['causes']` |
| 9 ter | hors relevé et durée | 86 | 86 | ok | `flag_counts(rec)['causes']` |
| 10 | blocs | 238 | 238 | ok | `campaign_blocks(rec, cfg['qc'])` |
| 10 | petits blocs (hors relevé) | 74 | 74 | ok | `campaign_blocks(...)` |
| 10 | enregistrements hors relevé (levés) | 106 | 106 | ok | `campaign_blocks(...)` |
| 10 | dont de 2 min | 20 | 20 | ok | `campaign_blocks(...)` |
| 10 | hors relevé écartés | 88 | 88 | ok | `campaign_blocks(...)` |
| 10 | fichiers de durée anormale | 173 | 173 | ok | `duration_classes(rec, cfg['qc'])` |
| 10 | dont tests (< 30 s) | 125 | 125 | ok | `duration_classes(...)` |
| 10 | dont coupés (30 à 119 s) | 34 | 34 | ok | `duration_classes(...)` |
| 10 | dont longs (> 121 s) | 14 | 14 | ok | `duration_classes(...)` |
| 10 | enregistrements 2026 scorés par BlanciNet | 8 349 | 8 349 | ok | `recording_detections(con, 'blancinet')` |
| 10 | coupés (30 à 119 s) à score BlanciNet ≥ 0,5 | 3 | 3 | ok | `duration_classes(...) ⋈ recording_detections(con, 'blancinet')` |
| nb 06 | écartés avec une détection BlanciNet | 108 | 116 | différent | `excluded_with_detections(rec, recording_detections(con, 'blancinet'))` |
| nb 06 | dont score ≥ 0,5 | 19 | 20 | différent | `excluded_with_detections(...)` |
| 8 | doublons écartés à l'inventaire (rapport) | 3 673 | 3 673 | ok | `duplicate_report_summary(ingest_duplicates_*.csv)` |
| 17 | deux pistes retenues, écartés | 399 | 436 | différent | `track_flags_summary(rec, qc)` |
| 17 | piste 0 retenue seule, gardés (encodés sur la piste 1) | 407 | 486 | différent | `track_flags_summary(...)` |
| 17 | piste 0 retenue seule, écartés (piste 1 douteuse) | 85 | 28 | différent | `track_flags_summary(...)` |
| 17 | piste 1 retenue seule (notes : « une quarantaine ») | 40 | 41 | différent | `track_flags_summary(...)` |
| 17 | candidats isolés de la piste 0 (aucune piste retenue) | 268 | 253 | différent | `track_flags_summary(...)` |
| 19 | écartés portés par les six points les plus touchés (tous à CDR) | 452 | 467 | différent | `counts_by(rec[rec.excluded], ['site', 'mic_id'])` |
| — | drapeaux d'inventaire périmés en base (à recalculer par blanci flag) | 0 | 0 | ok | `stale_flags(load_recordings(con, cfg, None), cfg)` |
| 11 | enregistrements où A. blanci a été entendu, écartés (règle : aucun) | 0 | 0 | ok | `protected_positives(rec, positive_recordings(con))` |
