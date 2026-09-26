"""Protects the video: both demo beats run from the committed data/processed/ (no network) and carry every label
the screens show (ESTIMATED, PLACEHOLDER FEED, "exposure ranking, county-validated", config version)."""
import json

from src.config import load_config
from src.decisions.common import EXPOSURE_LABEL
from src.decisions.run import run
from src.llm import template
from src.store import record

CFG_VERSION = load_config()["_version"]


def _beat(tmp_path, storm, when):
    conn = record.connect(tmp_path / "r.db")
    outs = {o["decision"]: o for o in run(storm, when, conn=conn)}
    for o in outs.values():
        assert record.get_record(conn, o["record_id"]) is not None  # written to the store
    return {k: o["record"] for k, o in outs.items()}


def test_beat_a_milton_t72(tmp_path):
    r = _beat(tmp_path, "milton", "T-72h")
    a = r["A"]
    assert a["recommendation"]["plan"]["mutual_aid_crews"] > 0
    est = json.dumps(a["estimates"])
    assert EXPOSURE_LABEL in est and "PLACEHOLDER FEED" in est
    assert a["config_version"] == CFG_VERSION
    brief = template.briefing(a, r.get("B"))
    assert CFG_VERSION in brief and EXPOSURE_LABEL in brief and "PLACEHOLDER FEED" in brief


def test_beat_b_ian_t12(tmp_path):
    r = _beat(tmp_path, "ian", "T-12h")
    assert set(r) == {"A", "B"}
    b = r["B"]
    assert any(s["prob_source"] == "PSURGE" for s in b["estimates"]["substations"])
    assert any(s["recommendation"] == "DE-ENERGIZE" for s in b["recommendation"]["substations"])
    assert b["height_source"] == "ESTIMATED" and "ESTIMATED" in json.dumps(b["estimates"])
    assert b["config_version"] == CFG_VERSION and b["sensor_value"] is None
    brief = template.briefing(r["A"], b)
    assert CFG_VERSION in brief and "ESTIMATED" in brief and "PLACEHOLDER FEED" in brief
