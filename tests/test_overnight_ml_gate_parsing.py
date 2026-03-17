from overnight import RichOvernightConsole


def test_overnight_parser_tracks_applied_ml_gate_deltas():
    tui = RichOvernightConsole(max_hours=1.0)
    tui._parse_message("--- Pass 2: Optimization Loop ---")
    tui._parse_message("Optimizing fundamentals")
    tui._parse_message(
        "ML underdog diagnostics: "
        "brier 0.2200->0.2050 (lift +0.0150), "
        "logloss 0.6400->0.6200 (lift +0.0200), "
        "gate PASS (min +0.0025)"
    )

    info = tui.pass_ml_gate[2]["fundamentals"]
    assert info["status"] == "applied"
    assert info["passed"] is True
    assert info["brier_lift"] == 0.015
    assert info["logloss_lift"] == 0.02


def test_overnight_parser_tracks_skipped_ml_gate():
    tui = RichOvernightConsole(max_hours=1.0)
    tui._parse_message("--- Pass 3: Optimization Loop ---")
    tui._parse_message("Optimizing sharp")
    tui._parse_message(
        "ML underdog diagnostics: skipped "
        "(insufficient validation upset samples (12 < 60))"
    )

    info = tui.pass_ml_gate[3]["sharp"]
    assert info["status"] == "skip"
    assert info["passed"] is True
    assert "insufficient validation upset samples" in info["reason"]
