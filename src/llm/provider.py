"""C6 provider switch (PRD C6, A7, U6). One setting, LLM_PROVIDER, picks the backend:

  template  (default) no LLM: fixed template + retrieval (FR36). Zero keys.
  anthropic Claude via the official `anthropic` SDK (pip install -r requirements-llm.txt), ANTHROPIC_API_KEY.
            Model ANTHROPIC_MODEL, default claude-opus-5, with server-side refusal fallback ("default" routing).
  ollama    self-hosted open-weights model (the production path: nothing leaves SGW's environment).
  groq / gemini   hosted open or third-party models through their OpenAI-compatible endpoints.

The prototype holds only public data, which is why a hosted API is acceptable here (PRD Appendix A); in
production LLM_PROVIDER=ollama (or another self-hosted server) is the one-setting switch.
The LLM never sees anything but the decision record, and it has no write path (FR35).
"""
from __future__ import annotations

import os

import requests

from src.config import ROOT

OPENAI_COMPATIBLE = {
    "ollama": {"base": os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/v1", "key_env": None,
               "model_env": "OLLAMA_MODEL", "model": "llama3.1:8b"},
    "groq": {"base": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY", "model_env": "GROQ_MODEL",
             "model": "llama-3.3-70b-versatile"},
    "gemini": {"base": "https://generativelanguage.googleapis.com/v1beta/openai", "key_env": "GEMINI_API_KEY",
               "model_env": "GEMINI_MODEL", "model": "gemini-2.5-flash"},
}


def load_dotenv() -> None:
    """Minimal .env reader (KEY=VALUE lines); real environment variables win."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def provider_name() -> str:
    load_dotenv()
    return os.environ.get("LLM_PROVIDER", "template").strip().lower() or "template"


class LLMError(RuntimeError):
    pass


def describe() -> str:
    name = provider_name()
    if name == "anthropic":
        return f"anthropic ({os.environ.get('ANTHROPIC_MODEL', 'claude-opus-5')})"
    if name in OPENAI_COMPATIBLE:
        p = OPENAI_COMPATIBLE[name]
        return f"{name} ({os.environ.get(p['model_env'], p['model'])})"
    return "template (no LLM)"


def complete(system: str, user: str) -> str:
    name = provider_name()
    if name == "anthropic":
        return _anthropic(system, user)
    if name in OPENAI_COMPATIBLE:
        return _openai_compatible(name, system, user)
    raise LLMError("LLM_PROVIDER=template: no LLM configured (FR36 template mode)")


def _anthropic(system: str, user: str) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise LLMError("pip install -r requirements-llm.txt to use LLM_PROVIDER=anthropic") from e
    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},  # re-run a policy decline on Anthropic's recommended fallback model
        )
    except anthropic.AuthenticationError as e:
        raise LLMError("Anthropic: invalid or missing API key") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Anthropic: rate limited, try again shortly") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Anthropic: network error") from e
    if response.stop_reason == "refusal":
        raise LLMError("The model declined this request; use the template briefing.")
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _openai_compatible(name: str, system: str, user: str) -> str:
    p = OPENAI_COMPATIBLE[name]
    headers = {"Content-Type": "application/json"}
    if p["key_env"]:
        key = os.environ.get(p["key_env"])
        if not key:
            raise LLMError(f"{p['key_env']} is not set")
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": os.environ.get(p["model_env"], p["model"]), "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    try:
        r = requests.post(f"{p['base']}/chat/completions", json=body, headers=headers, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        raise LLMError(f"{name}: {e.__class__.__name__}: {e}") from e
