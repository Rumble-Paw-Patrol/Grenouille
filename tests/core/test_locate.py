"""Recherche des enregistrements sur les disques branchés (DECISIONS n° 195)."""

from blanci.core import locate as loc

REL = "Projet/Relevé avril/SM E/Data/SMA_20240403_163000.wav"


def _put(root, rel=REL):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")
    return path


def test_found_under_the_configured_root(tmp_path):
    path = _put(tmp_path / "disque")
    assert loc.locate(REL, tmp_path / "disque") == path


def test_found_on_any_mounted_disk_and_one_level_below(tmp_path, monkeypatch):
    disks = [tmp_path / "Vide", tmp_path / "Disque1", tmp_path / "Disque2"]
    for disk in disks:
        disk.mkdir()
    monkeypatch.setattr(loc, "mounted_disks", lambda: disks)
    a = _put(disks[1], "a/x.wav")
    b = _put(disks[2] / "Sauvegarde ONF", "b/y.wav")
    assert loc.locate("a/x.wav", tmp_path / "absent") == a
    assert loc.locate("b/y.wav", tmp_path / "absent") == b
    assert loc._FOUND[0] == disks[2] / "Sauvegarde ONF"


def test_windows_separators_and_missing_file(tmp_path):
    path = _put(tmp_path)
    assert loc.locate(REL.replace("/", "\\"), tmp_path) == path
    assert loc.locate("rien/du/tout.wav", tmp_path) is None


def test_pasted_roots_are_remembered_next_to_the_db(tmp_path):
    db = tmp_path / "db" / "blanci.sqlite"
    loc.remember_root(db, tmp_path / "A")
    loc.remember_root(db, tmp_path / "B")
    loc.remember_root(db, tmp_path / "A")
    assert loc.remembered_roots(db) == [tmp_path / "A", tmp_path / "B"]
    path = _put(tmp_path / "B")
    assert loc.locate(REL, None, loc.remembered_roots(db)) == path
