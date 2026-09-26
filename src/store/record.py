"""Decision record (PRD FR25, FR23, FR24, FR28-FR30; NFR audit logging and reproducibility). SQLite.

records  one row per decision per advisory: the FR25 fields plus output_hash. Append-only: triggers refuse any
         change to what was decided (inputs, estimates, recommendation, hash) and any delete.
actions  append-only human actions: APPROVE / EDIT / OVERRIDE (P1, reason required for edit and override),
         SIGN_OFF / DECLINE (P2, per substation), DECLARE_STORM_OPS / END_STORM_OPS (FR29), WITHDRAW_MODEL (FR30).
configs  the full YAML of every configuration version a record used, so any record can be replayed exactly.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ROOT

DEFAULT_DB = Path(os.environ.get("SGW_RECORD_DB", ROOT / "var" / "record.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS configs (
  config_version TEXT PRIMARY KEY,
  yaml           TEXT NOT NULL,
  created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
  record_id           TEXT PRIMARY KEY,
  storm_id            TEXT NOT NULL,
  advisory_time       TEXT NOT NULL,
  hours_to_landfall   REAL,
  resolution          TEXT NOT NULL,
  decision            TEXT NOT NULL CHECK (decision IN ('A', 'B')),
  config_version      TEXT NOT NULL REFERENCES configs(config_version),
  model_version       TEXT NOT NULL,
  inputs_json         TEXT NOT NULL,
  estimates_json      TEXT NOT NULL,
  recommendation_json TEXT NOT NULL,
  output_hash         TEXT NOT NULL,
  override_reason     TEXT,
  crew_hours_entered  REAL,
  crew_hours_source   TEXT,
  height_source       TEXT,
  prob_source         TEXT,
  sensor_value        REAL,
  sensor_note         TEXT,
  status              TEXT NOT NULL DEFAULT 'PROPOSED',
  created_at          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS actions (
  action_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  record_id    TEXT REFERENCES records(record_id),
  action       TEXT NOT NULL,
  actor        TEXT NOT NULL,
  reason       TEXT,
  payload_json TEXT,
  created_at   TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS records_immutable BEFORE UPDATE OF
  record_id, storm_id, advisory_time, decision, config_version, model_version, inputs_json, estimates_json,
  recommendation_json, output_hash, crew_hours_entered, height_source, prob_source, sensor_value ON records
BEGIN SELECT RAISE(ABORT, 'decision records are append-only'); END;
CREATE TRIGGER IF NOT EXISTS records_no_delete BEFORE DELETE ON records
BEGIN SELECT RAISE(ABORT, 'decision records are append-only'); END;
CREATE TRIGGER IF NOT EXISTS actions_immutable BEFORE UPDATE ON actions
BEGIN SELECT RAISE(ABORT, 'actions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS actions_no_delete BEFORE DELETE ON actions
BEGIN SELECT RAISE(ABORT, 'actions are append-only'); END;
"""

REASON_REQUIRED = {"EDIT", "OVERRIDE", "DECLINE", "WITHDRAW_MODEL", "END_STORM_OPS"}
STATUS_FOR = {"APPROVE": "APPROVED", "EDIT": "EDITED", "OVERRIDE": "OVERRIDDEN", "SIGN_OFF": "REVIEWED", "DECLINE": "REVIEWED"}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_config(conn, version: str, yaml_text: str) -> None:
    conn.execute("INSERT OR IGNORE INTO configs VALUES (?, ?, ?)", (version, yaml_text, now()))
    conn.commit()


def load_config_text(conn, version: str) -> str | None:
    r = conn.execute("SELECT yaml FROM configs WHERE config_version = ?", (version,)).fetchone()
    return r["yaml"] if r else None


def insert_record(conn, rec: dict) -> bool:
    """Insert a record; identical inputs give the same record_id, so a rerun does not duplicate. True if new."""
    cols = ["record_id", "storm_id", "advisory_time", "hours_to_landfall", "resolution", "decision", "config_version",
            "model_version", "inputs_json", "estimates_json", "recommendation_json", "output_hash", "crew_hours_entered",
            "crew_hours_source", "height_source", "prob_source", "sensor_value", "sensor_note", "created_at"]
    vals = [rec.get(c) for c in cols[:-1]] + [now()]
    cur = conn.execute(f"INSERT OR IGNORE INTO records ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
    conn.commit()
    return cur.rowcount == 1


def add_action(conn, action: str, actor: str, record_id: str | None = None, reason: str | None = None,
               payload: dict | None = None) -> int:
    action = action.upper()
    if action in REASON_REQUIRED and not (reason and reason.strip()):
        raise ValueError(f"{action} requires a logged reason (FR23)")
    cur = conn.execute("INSERT INTO actions (record_id, action, actor, reason, payload_json, created_at) VALUES (?,?,?,?,?,?)",
                       (record_id, action, actor, reason, json.dumps(payload, sort_keys=True) if payload else None, now()))
    if record_id and action in STATUS_FOR:
        conn.execute("UPDATE records SET status = ?, override_reason = COALESCE(?, override_reason) WHERE record_id = ?",
                     (STATUS_FOR[action], reason if action in ("EDIT", "OVERRIDE") else None, record_id))
    conn.commit()
    return cur.lastrowid


def get_record(conn, record_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM records WHERE record_id = ?", (record_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    for k in ("inputs_json", "estimates_json", "recommendation_json"):
        d[k.replace("_json", "")] = json.loads(d[k])
    return d


def list_records(conn, storm_id: str | None = None) -> pd.DataFrame:
    q = ("SELECT record_id, decision, storm_id, advisory_time, hours_to_landfall, resolution, config_version, "
         "model_version, output_hash, status, override_reason, crew_hours_entered, height_source, prob_source, "
         "sensor_value, created_at FROM records")
    args = ()
    if storm_id:
        q += " WHERE storm_id = ?"
        args = (storm_id,)
    return pd.read_sql_query(q + " ORDER BY created_at DESC, record_id", conn, params=args)


def list_actions(conn, record_id: str | None = None) -> pd.DataFrame:
    q, args = "SELECT * FROM actions", ()
    if record_id:
        q, args = q + " WHERE record_id = ?", (record_id,)
    return pd.read_sql_query(q + " ORDER BY action_id DESC", conn, params=args)


def storm_ops_state(conn) -> dict:
    r = conn.execute("SELECT action, actor, created_at FROM actions WHERE action IN ('DECLARE_STORM_OPS', 'END_STORM_OPS') "
                     "ORDER BY action_id DESC LIMIT 1").fetchone()
    return {"declared": bool(r and r["action"] == "DECLARE_STORM_OPS"), **(dict(r) if r else {})}


def withdrawn_models(conn) -> list[str]:
    rows = conn.execute("SELECT payload_json FROM actions WHERE action = 'WITHDRAW_MODEL'").fetchall()
    return [json.loads(r["payload_json"])["model_kind"] for r in rows if r["payload_json"]]
