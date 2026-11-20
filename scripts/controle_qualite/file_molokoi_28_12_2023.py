"""File d'écoute `molokoi_28_12_2023` : le fichier de SMA14157 du 28/12/2023 à 11 h (heure locale).

Molokoï E : SMA14157 s'arrête le 21/11/2023 et SMA14636 reprend le 15/01/2024. Entre les deux, ce
seul fichier de 120 s, rangé dans le relevé 2 le jour de la pose (28/12), coupe le trou de 54 jours
en 36 et 18 (diapo 7). Il porte `confirmed_off_campaign` (posé quand la règle des 7 jours le
mettait hors relevé) alors que les dates déclarées (`qc.surveys`) le gardent dans le relevé. À
écouter : un vrai enregistrement de terrain (le micro était là le 28/12 : le trou de Molokoï E
serait alors de deux trous) ou un essai de réglage à la pose (à écarter, `confirmed_off_campaign`
tient) ? Enregistrement entier.

Lecture seule de la base ; écrit la file dans `paths.queues`.

    uv run python scripts/controle_qualite/file_molokoi_28_12_2023.py
"""

from __future__ import annotations

from files_apres_prez4 import entiers

from blanci.annotation.selection import write_queue
from blanci.core.config import default_user_config, load_config
from blanci.exploration import inventaire as inv


def main() -> None:
    cfg = load_config(default_user_config())
    rec = inv.load_recordings(inv.open_database(cfg), cfg)
    f = rec[(rec["site"] == "Molokoi") & (rec["mic_id"] == "SMA14157")
            & (rec["start_utc"] == "2023-12-28T14:00:00Z")]
    assert len(f) == 1, f"{len(f)} enregistrements trouvés"
    file = entiers(f, "molokoi_28_12_2023")
    write_queue(cfg, file, "molokoi_28_12_2023", {
        "origine": "scripts/controle_qualite/file_molokoi_28_12_2023.py (suite des dates de "
                   "relevé déclarées, 20/11/2026)",
        "contenu": "SMA14157 à Molokoï, 28/12/2023 11 h locale, 120 s, seul fichier entre le 21/11 "
                   "et le 15/01 : il coupe le trou de Molokoï E (54 → 36 jours) ; "
                   "`confirmed_off_campaign` mais dans le relevé 2 depuis les dates déclarées ; "
                   "enregistrement entier ; vrai terrain (lever le drapeau confirmé, `keep`) ou "
                   "essai de réglage (le laisser)"})
    print(file[["recording_id", "path", "start_utc", "dur_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
