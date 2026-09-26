"""NFR reproducibility: same advisory snapshot + same configuration version -> same output (0 difference)."""
from src.decisions.run import replay, run
from src.store import record


def test_rerun_same_inputs_same_record(tmp_path):
    conn = record.connect(tmp_path / "r.db")
    first = run("milton", "T-72h", conn=conn)
    second = run("milton", "T-72h", conn=conn)
    assert [o["record_id"] for o in first] == [o["record_id"] for o in second]
    assert [o["output_hash"] for o in first] == [o["output_hash"] for o in second]
    assert all(o["created"] for o in first) and not any(o["created"] for o in second)


def test_replay_identical_both_decisions(tmp_path):
    conn = record.connect(tmp_path / "r.db")
    outs = run("ian", "T-12h", conn=conn)
    assert {o["decision"] for o in outs} == {"A", "B"}
    for o in outs:
        r = replay(o["record_id"], conn=conn)
        assert r["identical"], r


def test_crew_hours_entered_changes_record(tmp_path):
    conn = record.connect(tmp_path / "r.db")
    a = run("milton", "T-72h", conn=conn)[0]
    b = run("milton", "T-72h", own_crew_hours=50_000, conn=conn)[0]
    assert a["record_id"] != b["record_id"]
    assert b["record"]["crew_hours_source"] == "entered by P1"
    assert b["record"]["recommendation"]["plan"]["mutual_aid_crews"] < a["record"]["recommendation"]["plan"]["mutual_aid_crews"]
    assert replay(b["record_id"], conn=conn)["identical"]
