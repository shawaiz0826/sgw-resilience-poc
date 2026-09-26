"""Configuration loading and versioning (FR28). The version is content-addressed, so any edit is a new version."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "v1.yaml"


def config_version(path: Path = CONFIG_PATH) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{path.stem}-{digest[:12]}"


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg["_version"] = config_version(path)
    return cfg
