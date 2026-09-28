"""Configuration : sections vides du fichier utilisateur, chemins relatifs (DECISIONS n° 144)."""

from pathlib import Path

from blanci.config import PROJECT_ROOT, config_path, load_config, project_path


def test_an_empty_user_section_keeps_the_default_one(tmp_path):
    user = tmp_path / "local.yaml"
    user.write_text("qc:\n  # rien à changer pour l'instant\npaths:\n  raw: D:/audio\n", "utf-8")
    cfg = load_config(user)
    assert isinstance(cfg["qc"], dict) and cfg["qc"] == load_config()["qc"]
    assert cfg["paths"]["raw"] == "D:/audio"


def test_relative_paths_start_from_the_project_not_the_current_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # lancé d'ailleurs (D:\ par exemple)
    cfg = load_config()
    assert config_path(cfg, "db") == PROJECT_ROOT / cfg["paths"]["db"]
    assert project_path(tmp_path / "x") == tmp_path / "x"  # absolu : tel quel
    assert project_path("data") == PROJECT_ROOT / "data"
    assert not Path(cfg["paths"]["db"]).is_absolute()
