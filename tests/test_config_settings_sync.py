import json


def _use_temp_settings_path(tmp_path, monkeypatch):
    from src import config

    path = tmp_path / "app_settings.json"
    monkeypatch.setattr(config, "_SETTINGS_PATH", path)
    config.invalidate_cache()
    return config, path


def test_load_settings_hydrates_missing_defaults_into_app_file(tmp_path, monkeypatch):
    config, settings_path = _use_temp_settings_path(tmp_path, monkeypatch)
    settings_path.write_text(
        json.dumps(
            {
                "season": "2024-25",
                "optuna_top_n_validation": 120,
            }
        ),
        encoding="utf-8",
    )

    loaded = config.load_settings()

    assert loaded["season"] == "2024-25"
    assert loaded["optuna_top_n_validation"] == 120
    assert (
        loaded["optimizer_objective_val_probe_slices"]
        == config._DEFAULTS["optimizer_objective_val_probe_slices"]
    )

    persisted = json.loads(settings_path.read_text(encoding="utf-8"))
    assert persisted["season"] == "2024-25"
    assert persisted["optuna_top_n_validation"] == 120
    assert (
        persisted["optimizer_objective_val_probe_slices"]
        == config._DEFAULTS["optimizer_objective_val_probe_slices"]
    )


def test_load_settings_bootstraps_file_when_missing(tmp_path, monkeypatch):
    config, settings_path = _use_temp_settings_path(tmp_path, monkeypatch)

    assert not settings_path.exists()
    loaded = config.load_settings()

    assert settings_path.exists()
    persisted = json.loads(settings_path.read_text(encoding="utf-8"))
    assert persisted["season"] == loaded["season"]
    assert persisted["optuna_top_n_validation"] == loaded["optuna_top_n_validation"]
