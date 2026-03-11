import json


def test_extract_save_gate_state_compacts_details():
    from src.analytics import pipeline

    result = {
        "improved": False,
        "save_gate_reason": "x" * 260,
        "save_gate_details": {
            "weight_delta": 0.0,
            "use_roi_gate": False,
            "compression_ok": True,
            "nested": {"ignore": True},
        },
    }

    snapshot = pipeline._extract_save_gate_state(result)
    assert isinstance(snapshot, dict)
    assert snapshot["saved"] is False
    assert snapshot["weight_delta"] == 0.0
    assert snapshot["use_roi_gate"] is False
    assert snapshot["compression_ok"] is True
    assert "nested" not in snapshot
    assert snapshot["reason"].endswith("...")
    assert len(snapshot["reason"]) <= 220


def test_run_pipeline_persists_save_gate_metadata(tmp_path, monkeypatch):
    from src.analytics import pipeline

    state_path = tmp_path / "pipeline_state.json"
    monkeypatch.setattr(pipeline, "PIPELINE_STATE_PATH", str(state_path))

    def fake_optimize_step(callback=None, is_cancelled=None):
        return {
            "improved": False,
            "save_gate_reason": "loss gate failed due regression",
            "save_gate_details": {
                "weight_delta": 0.0,
                "min_weight_delta": 0.0001,
                "weight_change_ok": False,
                "candidate_ml_roi_lb95": -0.4,
            },
        }

    monkeypatch.setattr(
        pipeline,
        "PIPELINE_STEPS",
        [("optimize_fundamentals", fake_optimize_step)],
    )

    result = pipeline.run_pipeline(is_cancelled_fn=lambda: False)
    assert "optimize_fundamentals" in result

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    step_state = state["step_optimize_fundamentals"]
    assert step_state["status"] == "completed"
    assert step_state["save_gate"]["saved"] is False
    assert step_state["save_gate"]["reason"] == "loss gate failed due regression"
    assert step_state["save_gate"]["weight_delta"] == 0.0
    assert step_state["save_gate"]["weight_change_ok"] is False
