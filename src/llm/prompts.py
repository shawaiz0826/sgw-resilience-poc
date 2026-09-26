"""C6 prompts and the context built from the record (PRD C6, FR33-FR35). The LLM sees only the record."""
from __future__ import annotations

import json
import re

SYSTEM = """You draft storm briefings and answer questions for a utility's emergency management lead.
You work only from the DECISION RECORD provided in the user message. Rules you must follow:
- Copy every number exactly as it appears in the record. Never compute, round, convert or estimate a number.
  Outage counts are "customers (meters)"; never call them people or residents.
- Cite the record IDs you used, in square brackets, e.g. [A-milton-2024100700-3973db77].
- Do not recommend, change or second-guess any decision, threshold or setting. Report what the record says.
- If the answer is not in the record, reply exactly: Not in the record.
- Keep "exposure ranking, county-validated" wherever you mention asset scores, and keep labels such as
  ESTIMATED, STATIC, PLACEHOLDER FEED and LOW CONFIDENCE where the record carries them."""

BRIEFING_TASK = """Draft a situational briefing for regulators, county EOCs and the board, under 250 words, with four
short sections: expected outages; mutual aid and staging (with any Storm Director action and reason); substations
(Decision B); pumping stations to receive generators. End with the record IDs used."""


def record_context(a: dict, b: dict | None, actions_a: list[dict]) -> str:
    plan = a["recommendation"]["plan"]
    ctx = {
        "record_A": {
            "record_id": a["record_id"], "advisory": a["inputs"], "config_version": a["config_version"],
            "model_version": a["model_version"], "banner": a["recommendation"]["banner"], "plan": plan,
            "study_counties": [c for c in a["estimates"]["counties"] if c["study_zone"]],
            "top_counties_statewide_by_p50": sorted(a["estimates"]["counties"], key=lambda c: -c["out_p50"])[:5],
            "pumping_stations": [d for d in a["estimates"]["dependents"] if d["type"] == "pumping"][:8],
            "storm_director_actions": actions_a[:3],
        },
    }
    if b:
        est = {r["asset_id"]: r for r in b["estimates"]["substations"]}
        ctx["record_B"] = {
            "record_id": b["record_id"], "prob_source": b["recommendation"]["prob_source"],
            "threshold_in_force": b["recommendation"]["threshold_in_force"], "height_source": "ESTIMATED",
            "sensor_values": "BLANK (no historian)",
            "substations": [{**est[r["asset_id"]], **r} for r in b["recommendation"]["substations"]],
        }
    return json.dumps(ctx, indent=1, default=str)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def unsupported_numbers(output: str, context: str) -> list[str]:
    """FR35 guard: numbers in the LLM output that do not appear anywhere in the record context."""
    ctx = _numbers(context)
    ctx |= {n.split(".")[0] for n in ctx}  # allow integer rendering of a recorded value (e.g. 0.65 -> no; 36962.0 -> 36962)
    return sorted(n for n in _numbers(output) if n not in ctx and not re.fullmatch(r"[1-4]", n))
